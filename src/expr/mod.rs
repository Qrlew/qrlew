//! # `Expr` definition and manipulation
//!
//! `Expr` combine values and columns with functions and aggregations.
//!
//! `Expr` propagate data types and ranges.
//!
#[macro_use]
pub mod dsl;
pub mod aggregate;
pub mod dot;
pub mod function;
pub mod identifier;
pub mod implementation;
pub mod split;
pub mod sql;
pub mod transforms;

use itertools::Itertools;
use paste::paste;
use std::{
    cmp,
    collections::BTreeMap,
    convert::identity,
    error, fmt, hash,
    ops::{Add, BitAnd, BitOr, BitXor, Deref, Div, Mul, Neg, Not, Rem, Sub},
    sync::Arc,
    result,
};

use crate::{
    data_type::{self, function::Function as _, value, DataType, DataTyped, Variant as _},
    hierarchy::Hierarchy,
    visitor::{self, Acceptor},
};

pub use identifier::Identifier;
pub use split::{Map, Reduce, Split};

/*
TODO
- Maybe function defnition is redundant
- Remove
*/

// Error management

#[derive(Debug, Clone)]
pub enum Error {
    InvalidExpression(String),
    InvalidConversion(String),
    Other(String),
}

impl Error {
    pub fn invalid_expression(expr: impl fmt::Display) -> Error {
        Error::InvalidExpression(format!("{} is invalid", expr))
    }
    pub fn invalid_conversion(from: impl fmt::Display, to: impl fmt::Display) -> Error {
        Error::InvalidConversion(format!("Invalid conversion from {} to {}", from, to))
    }
    pub fn other<T: fmt::Display>(desc: T) -> Error {
        Error::Other(desc.to_string())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidExpression(desc) => writeln!(f, "InvalidExpression: {}", desc),
            Error::InvalidConversion(desc) => writeln!(f, "InvalidConversion: {}", desc),
            Error::Other(err) => writeln!(f, "{}", err),
        }
    }
}

impl error::Error for Error {}

impl From<data_type::Error> for Error {
    fn from(err: data_type::Error) -> Self {
        Error::Other(err.to_string())
    }
}
impl From<value::Error> for Error {
    fn from(err: value::Error) -> Self {
        Error::Other(err.to_string())
    }
}
impl From<data_type::function::Error> for Error {
    fn from(err: data_type::function::Error) -> Self {
        Error::Other(err.to_string())
    }
}

impl From<Error> for data_type::function::Error {
    fn from(err: Error) -> Self {
        data_type::function::Error::Other(err.to_string())
    }
}

pub type Result<T> = result::Result<T, Error>;

/// Each expression variant must comply with this trait
pub trait Variant:
    TryFrom<Expr> + Into<Expr> + Clone + fmt::Debug + fmt::Display + hash::Hash + cmp::PartialEq
{
}

/// A column expression
pub type Column = Identifier;

impl Variant for Column {}

/// A value expression
pub type Value = value::Value;

impl Variant for Value {}

/// A function expression
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Function {
    /// Operator
    function: function::Function,
    /// Argumants
    arguments: Vec<Arc<Expr>>,
}

impl Function {
    /// Basic constructor
    pub fn new(function: function::Function, arguments: Vec<Arc<Expr>>) -> Function {
        Function {
            function,
            arguments,
        }
    }

    pub fn function(&self) -> function::Function {
        self.function
    }

    pub fn arguments(&self) -> Vec<Expr> {
        self.arguments.iter().map(|x| x.as_ref().clone()).collect()
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.function.clone().style() {
            function::Style::UnaryOperator => {
                write!(f, "({} {})", self.function, self.arguments[0])
            }
            function::Style::BinaryOperator => write!(
                f,
                "({} {} {})",
                self.arguments[0], self.function, self.arguments[1]
            ),
            function::Style::Function => write!(
                f,
                "{}({})",
                self.function,
                self.arguments
                    .iter()
                    .map(|expr| expr.to_string())
                    .join(", ")
            ),
        }
    }
}

impl Variant for Function {}

/// Implemant random function constructor (same thing but no macro here)
impl Function {
    pub fn random(n: usize) -> Function {
        Function::new(function::Function::Random(n), vec![])
    }
}

/// Implemant random expression constructor (same thing but no macro here)
impl Expr {
    pub fn random(n: usize) -> Expr {
        Expr::from(Function::random(n))
    }

    pub fn filter_column(
        name: &str,
        min: Option<data_type::value::Value>,
        max: Option<data_type::value::Value>,
        possible_values: Vec<data_type::value::Value>,
    ) -> Option<Expr> {
        let column = Expr::col(name.to_string());
        let mut p = None;
        if let Some(m) = min {
            let expr = Expr::gt(column.clone(), Expr::val(m));
            p = Some(p.map_or(expr.clone(), |x| Expr::and(x, expr)))
        }
        if let Some(m) = max {
            let expr = Expr::lt(column.clone(), Expr::val(m));
            p = Some(p.map_or(expr.clone(), |x| Expr::and(x, expr)))
        };
        if !possible_values.is_empty() {
            let expr = Expr::in_list(column.clone(), Expr::list(possible_values));
            p = Some(p.map_or(expr.clone(), |x| Expr::and(x, expr)))
        }
        p
    }

    pub fn and_iter<I: IntoIterator<Item = Expr>>(exprs: I) -> Expr {
        exprs
            .into_iter()
            .reduce(|f, p| Expr::and(f, p))
            .unwrap_or(Expr::val(true))
    }

    /// Returns an `Expr` for filtering the columns
    ///
    /// # Arguments
    /// - `columns`: `Vec<(column_name, minimal_value, maximal_value, possible_values)>`
    ///
    /// For example,
    /// - `filter(vec![("my_col", Value::float(2.), Value::float(10.), vec![])])`
    ///         ≡ `(my_col > 2.) and (my_col < 10)`
    /// - `filter(vec![("my_col", None, Value::float(10.), vec![Value::integer(1), Value::integer(2), Value::integer(5)])])`
    ///         ≡ `(my_col < 10.) and (my_col in (1, 2, 5))`
    /// - `filter(vec![("my_col1", None, Value::integer(10), vec![]), ("my_col2", Value::float(1.), None, vec![])])])`
    ///         ≡ `(my_col1 < 10) and (my_col2 > 1.)`
    pub fn filter(
        columns: BTreeMap<
            &str,
            (
                Option<data_type::value::Value>,
                Option<data_type::value::Value>,
                Vec<data_type::value::Value>,
            ),
        >,
    ) -> Expr {
        let predicates: Vec<Expr> = columns
            .into_iter()
            .filter_map(|(name, (min, max, values))| Expr::filter_column(name, min, max, values))
            .collect();
        Self::and_iter(predicates)
    }
}

/// Implement unary function constructors
macro_rules! impl_unary_function_constructors {
    ($( $Function:ident ),*) => {
        impl Function {
            paste! {
                $(pub fn [<$Function:snake>]<E: Into<Expr>>(expr: E) -> Function {
                    Function::new(function::Function::$Function, vec![Arc::new(expr.into())])
                }
                )*
            }
        }

        impl Expr {
            paste! {
                $(pub fn [<$Function:snake>]<E: Into<Expr>>(expr: E) -> Expr {
                    Expr::from(Function::[<$Function:snake>](expr))
                }
                )*
            }
        }
    };
}

impl_unary_function_constructors!(
    Opposite,
    Not,
    Exp,
    Ln,
    Log,
    Abs,
    Sin,
    Cos,
    Sqrt,
    Md5,
    Lower,
    Upper,
    CharLength,
    CastAsText,
    CastAsInteger,
    CastAsFloat,
    CastAsDateTime
); // TODO Complete that

/// Implement binary function constructors
macro_rules! impl_binary_function_constructors {
    ($( $Function:ident ),*) => {
        impl Function {
            paste! {
                $(
                    pub fn [<$Function:snake>]<L: Into<Expr>, R: Into<Expr>>(left: L, right: R) -> Function {
                        Function::new(function::Function::$Function, vec![Arc::new(left.into()), Arc::new(right.into())])
                    }
                )*
            }
        }

        impl Expr {
            paste! {
                $(
                    pub fn [<$Function:snake>]<L: Into<Expr>, R: Into<Expr>>(left: L, right: R) -> Expr {
                        Expr::from(Function::[<$Function:snake>](left, right))
                    }
                )*
            }
        }
    };
}

impl_binary_function_constructors!(
    Plus,
    Minus,
    Multiply,
    Divide,
    Modulo,
    StringConcat,
    Gt,
    Lt,
    GtEq,
    LtEq,
    Eq,
    NotEq,
    And,
    Or,
    Xor,
    BitwiseOr,
    BitwiseAnd,
    BitwiseXor,
    Pow,
    Position,
    InList,
    Least,
    Greatest
);

/// Implement ternary function constructors
macro_rules! impl_ternary_function_constructors {
    ($( $Function:ident ),*) => {
        impl Function {
            paste! {
                $(
                    pub fn [<$Function:snake>]<F: Into<Expr>, S: Into<Expr>, T: Into<Expr>>(first: F, second: S, third: T) -> Function {
                        Function::new(function::Function::$Function, vec![Arc::new(first.into()), Arc::new(second.into()), Arc::new(third.into())])
                    }
                )*
            }
        }

        impl Expr {
            paste! {
                $(
                    pub fn [<$Function:snake>]<F: Into<Expr>, S: Into<Expr>, T: Into<Expr>>(first: F, second: S, third: T) -> Expr {
                        Expr::from(Function::[<$Function:snake>](first, second, third))
                    }
                )*
            }
        }
    };
}

impl_ternary_function_constructors!(Case);

/// Implement nary function constructors
macro_rules! impl_nary_function_constructors {
    ($( $Function:ident ),*) => {
        impl Function {
            paste! {
                $(
                    pub fn [<$Function:snake>]<E: Into<Expr>>(args: Vec<E>) -> Function {
                        Function::new(function::Function::$Function(args.len()), args.into_iter().map(|e| Arc::new(e.into())).collect())
                    }
                )*
            }
        }

        impl Expr {
            paste! {
                $(
                    pub fn [<$Function:snake>]<E: Into<Expr>>(args: Vec<E>) -> Expr {
                        Expr::from(Function::[<$Function:snake>](args))
                    }
                )*
            }
        }
    };
}

impl_nary_function_constructors!(Concat);

/// An aggregate function expression
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Aggregate {
    /// Operator
    aggregate: aggregate::Aggregate,
    /// Argument
    argument: Arc<Expr>,
}

impl Aggregate {
    /// Basic constructor
    pub fn new(aggregate: aggregate::Aggregate, argument: Arc<Expr>) -> Aggregate {
        Aggregate {
            aggregate,
            argument,
        }
    }
    /// Get aggregate
    pub fn aggregate(&self) -> aggregate::Aggregate {
        self.aggregate
    }
    /// Get argument
    pub fn argument(&self) -> &Expr {
        self.argument.as_ref()
    }
    /// Get argument
    pub fn argument_column(&self) -> Result<&Column> {
        match self.argument.as_ref() {
            Expr::Column(col) => Ok(col),
            _ => Err(Error::other("Cannot return the argument column")),
        }
    }
    /// Get the argument name
    pub fn argument_name(&self) -> Result<&str> {
        Ok(self.argument_column()?.last()?)
    }
}

impl fmt::Display for Aggregate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.aggregate, self.argument)
    }
}

impl Variant for Aggregate {}

/// Implement unary function constructors
macro_rules! impl_aggregation_constructors {
    ($( $Aggregate:ident ),*) => {
        impl Aggregate {
            paste! {
                $(pub fn [<$Aggregate:snake>]<E: Into<Expr>>(expr: E) -> Aggregate {
                    Aggregate::new(aggregate::Aggregate::$Aggregate, Arc::new(expr.into()))
                }
                )*
            }
        }

        impl Expr {
            paste! {
                $(pub fn [<$Aggregate:snake>]<E: Into<Expr>>(expr: E) -> Expr {
                    Expr::from(Aggregate::[<$Aggregate:snake>](expr))
                }
                )*
            }
        }

        impl AggregateColumn {
            paste! {
                $(pub fn [<$Aggregate:snake>]<S: Into<String>>(col: S) -> AggregateColumn {
                    AggregateColumn::new(aggregate::Aggregate::$Aggregate, Column::from(col.into()))
                }
                )*
            }
        }
    };
}

impl_aggregation_constructors!(First, Last, Min, Max, Count, Mean, Sum, Var, Std);

/// An aggregate function expression
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Struct {
    /// Fields
    fields: Vec<(Identifier, Arc<Expr>)>,
}

impl Struct {
    /// Basic constructor
    pub fn new(fields: Vec<(Identifier, Arc<Expr>)>) -> Struct {
        Struct { fields }
    }
}

impl fmt::Display for Struct {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{{ {} }}",
            self.fields
                .iter()
                .map(|(i, e)| format!("{i}: {e}"))
                .join(", ")
        )
    }
}

impl<S: Into<Identifier>, E: Into<Arc<Expr>>> FromIterator<(S, E)> for Struct {
    fn from_iter<I: IntoIterator<Item = (S, E)>>(iter: I) -> Self {
        Struct::new(
            iter.into_iter()
                .map(|(s, e)| (s.into(), e.into()))
                .collect(),
        )
    }
}

impl Variant for Struct {}

/// A Expr enum
/// inspired by: https://docs.rs/sqlparser/latest/sqlparser/ast/enum.Expr.html
/// and mostly: https://docs.rs/polars/latest/polars/prelude/enum.Expr.html
/// or https://docs.rs/polars-lazy/latest/polars_lazy/dsl/enum.Expr.html

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Expr {
    Column(Column),
    Value(Value),
    Function(Function),
    Aggregate(Aggregate),
    Struct(Struct),
}

/// Basic constructors
/// They are short because they are supposed to be the primary API for the module
impl Expr {
    pub fn col<S: Into<String>>(field: S) -> Expr {
        Expr::Column(Column::from_name(field))
    }

    pub fn qcol<S: Into<String>>(relation: S, field: S) -> Expr {
        Expr::Column(Column::from_qualified_name(relation, field))
    }

    pub fn val<V: Into<Value>>(value: V) -> Expr {
        Expr::Value(value.into())
    }

    pub fn list<L: IntoIterator<Item = V>, V: Into<Value>>(values: L) -> Expr {
        Expr::Value(Value::list(
            values.into_iter().map(|v| v.into()).collect::<Vec<Value>>(),
        ))
    }

    pub fn structured<S: Clone + Into<String>, E: Clone + Into<Arc<Expr>>, F: AsRef<[(S, E)]>>(
        fields: F,
    ) -> Expr {
        Expr::Struct(
            fields
                .as_ref()
                .iter()
                .map(|(s, e)| (s.clone().into(), e.clone().into()))
                .collect(),
        )
    }

    // pub fn all<E: Clone+Into<Expr>, F: AsRef<[E]>>(
    //     factors: F,
    // ) -> Expr {
    //     let factors = factors.as_ref();
    //     match factors.split_first() {
    //         Some((head, tail)) => Expr::and(head.clone(), Expr::all(tail)),
    //         None => Expr::val(true),
    //     }
    // }

    pub fn all<F: IntoIterator<Item = Expr>>(factors: F) -> Expr {
        let mut factors = factors.into_iter();
        match factors.next() {
            Some(head) => Expr::and(head, Expr::all(factors)),
            None => Expr::val(true),
        }
    }
}

/// Implement basic Variant conversions
macro_rules! impl_conversions {
    ( $Variant:ident ) => {
        impl From<$Variant> for Expr {
            fn from(v: $Variant) -> Self {
                Expr::$Variant(v)
            }
        }

        impl TryFrom<Expr> for $Variant {
            type Error = Error;

            fn try_from(expr: Expr) -> Result<Self> {
                if let Expr::$Variant(v) = expr {
                    Ok(v)
                } else {
                    Err(Error::invalid_conversion(expr, stringify!($Variant)))
                }
            }
        }
    };
}

/// Implement Expr traits
macro_rules! impl_traits {
    ( $( $Variant:ident ),* ) => {
        impl fmt::Display for Expr {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self {
                    $(Expr::$Variant(variant) => variant.fmt(f),)*
                }
            }
        }
    }
}

impl_traits!(Column, Value, Function, Aggregate, Struct);
impl_conversions!(Column);
impl_conversions!(Function);
impl_conversions!(Aggregate);
impl_conversions!(Struct);

impl From<Value> for Expr {
    fn from(v: Value) -> Self {
        Expr::Value(v)
    }
}
impl TryFrom<Expr> for Value {
    type Error = Error;
    fn try_from(expr: Expr) -> Result<Self> {
        let v = TryInto::<Vec<Value>>::try_into(expr.co_domain())?;
        if v.len() == 1 {
            Ok(v[0].clone())
        } else {
            Err(Error::invalid_conversion(expr, "Value"))
        }
    }
}

impl Variant for Expr {}

// Implement ops
impl Add for Expr {
    type Output = Expr;

    fn add(self, rhs: Self) -> Self::Output {
        Expr::plus(self, rhs)
    }
}

impl BitAnd for Expr {
    type Output = Expr;

    fn bitand(self, rhs: Self) -> Self::Output {
        Expr::and(self, rhs)
    }
}

impl BitOr for Expr {
    type Output = Expr;

    fn bitor(self, rhs: Self) -> Self::Output {
        Expr::or(self, rhs)
    }
}

impl BitXor for Expr {
    type Output = Expr;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Expr::xor(self, rhs)
    }
}

impl Div for Expr {
    type Output = Expr;

    fn div(self, rhs: Self) -> Self::Output {
        Expr::divide(self, rhs)
    }
}

impl Mul for Expr {
    type Output = Expr;

    fn mul(self, rhs: Self) -> Self::Output {
        Expr::multiply(self, rhs)
    }
}

impl Neg for Expr {
    type Output = Expr;

    fn neg(self) -> Self::Output {
        Expr::opposite(self)
    }
}

impl Not for Expr {
    type Output = Expr;

    fn not(self) -> Self::Output {
        Expr::not(self)
    }
}

impl Rem for Expr {
    type Output = Expr;

    fn rem(self, rhs: Self) -> Self::Output {
        Expr::modulo(self, rhs)
    }
}

impl Sub for Expr {
    type Output = Expr;

    fn sub(self, rhs: Self) -> Self::Output {
        Expr::minus(self, rhs)
    }
}

/// Implement the Acceptor trait
impl<'a> Acceptor<'a> for Expr {
    fn dependencies(&'a self) -> visitor::Dependencies<'a, Self> {
        match self {
            Expr::Column(_) => visitor::Dependencies::empty(),
            Expr::Value(_) => visitor::Dependencies::empty(),
            Expr::Function(f) => f.arguments.iter().map(|e| &**e).collect(),
            Expr::Aggregate(a) => visitor::Dependencies::from([&(*a.argument)]),
            Expr::Struct(s) => s.fields.iter().map(|(_, e)| &**e).collect(),
        }
    }
}

impl<'a> IntoIterator for &'a Expr {
    type Item = &'a Expr;
    type IntoIter = visitor::Iter<'a, Expr>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// An aggregate column expr
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct AggregateColumn {
    aggregate: aggregate::Aggregate,
    column: Column,
    expr: Expr,
}

impl AggregateColumn {
    pub fn new(aggregate: aggregate::Aggregate, column: Column) -> Self {
        AggregateColumn {
            aggregate,
            column: column.clone(),
            expr: Expr::Aggregate(Aggregate::new(aggregate, Arc::new(Expr::Column(column)))),
        }
    }
    /// Access aggregate
    pub fn aggregate(&self) -> &aggregate::Aggregate {
        &self.aggregate
    }
    /// Access column
    pub fn column(&self) -> &Column {
        &self.column
    }
    /// Access column name
    pub fn column_name(&self) -> Result<&str> {
        Ok(&self.column.last()?)
    }
    /// A constructor
    pub fn col<S: Into<String>>(field: S) -> AggregateColumn {
        AggregateColumn::first(field)
    }
}

impl Deref for AggregateColumn {
    type Target = Expr;

    fn deref(&self) -> &Self::Target {
        &self.expr
    }
}

impl From<AggregateColumn> for Expr {
    fn from(value: AggregateColumn) -> Self {
        value.expr
    }
}

impl TryFrom<Expr> for AggregateColumn {
    type Error = Error;

    fn try_from(value: Expr) -> result::Result<Self, Self::Error> {
        match value {
            Expr::Column(column) => Ok(column.into()),
            Expr::Aggregate(Aggregate {
                aggregate,
                argument,
            }) => {
                if let Expr::Column(column) = argument.as_ref() {
                    Ok(AggregateColumn::new(aggregate, column.clone()))
                } else {
                    Err(Error::invalid_conversion(argument, "Column"))
                }
            }
            _ => Err(Error::invalid_conversion(value, "AggregateColumn")),
        }
    }
}

impl From<Column> for AggregateColumn {
    fn from(value: Column) -> Self {
        AggregateColumn::new(aggregate::Aggregate::First, value)
    }
}

impl<S: Into<String>> From<S> for AggregateColumn {
    fn from(value: S) -> Self {
        AggregateColumn::new(aggregate::Aggregate::First, Column::from(value.into()))
    }
}

// Visitors

/// A Visitor for the type Expr
pub trait Visitor<'a, T: Clone> {
    fn column(&self, column: &'a Column) -> T;
    fn value(&self, value: &'a Value) -> T;
    fn function(&self, function: &'a function::Function, arguments: Vec<T>) -> T;
    fn aggregate(&self, aggregate: &'a aggregate::Aggregate, argument: T) -> T;
    fn structured(&self, fields: Vec<(Identifier, T)>) -> T;
}

/// Implement a specific visitor to dispatch the dependencies more easily
impl<'a, T: Clone, V: Visitor<'a, T>> visitor::Visitor<'a, Expr, T> for V {
    fn visit(&self, acceptor: &'a Expr, dependencies: visitor::Visited<'a, Expr, T>) -> T {
        match acceptor {
            Expr::Column(c) => self.column(c),
            Expr::Value(v) => self.value(v),
            Expr::Function(f) => self.function(
                &f.function,
                f.arguments
                    .iter()
                    .map(|a| dependencies.get(&**a).clone())
                    .collect(),
            ),
            Expr::Aggregate(a) => {
                self.aggregate(&a.aggregate, dependencies.get(&a.argument).clone())
            }
            Expr::Struct(s) => self.structured(
                s.fields
                    .iter()
                    .map(|(i, e)| (i.clone(), dependencies.get(&**e).clone()))
                    .collect(),
            ),
        }
    }
}

// Some useful visitors

/// Visit the expression to display it
#[derive(Clone, Debug)]
pub struct DisplayVisitor;

impl<'a> Visitor<'a, String> for DisplayVisitor {
    fn column(&self, column: &'a Column) -> String {
        format!("{column}")
    }

    fn value(&self, value: &'a Value) -> String {
        format!("{value}")
    }

    fn function(&self, function: &'a function::Function, arguments: Vec<String>) -> String {
        match function.style() {
            function::Style::UnaryOperator => format!("{} {}", function, arguments[0]),
            function::Style::BinaryOperator => {
                format!("{} {} {}", arguments[0], function, arguments[1])
            }
            function::Style::Function => format!("{}({})", function, arguments.join(", ")),
        }
    }

    fn aggregate(&self, aggregate: &'a aggregate::Aggregate, argument: String) -> String {
        format!("{}({})", aggregate, argument)
    }

    fn structured(&self, fields: Vec<(Identifier, String)>) -> String {
        format!(
            "{{ {} }}",
            fields.iter().map(|(i, e)| format!("{i}: {e}")).join(", ")
        )
    }
}

// Implement the data_type::function::Function trait with visitors

/// A visitor to compute the domain
#[derive(Clone, Debug)]
pub struct DomainVisitor;

impl<'a> Visitor<'a, DataType> for DomainVisitor {
    fn column(&self, column: &'a Column) -> DataType {
        let (col_name, path) = column.split_last().unwrap();
        path.iter().rev().fold(
            DataType::structured([(&col_name, DataType::Any)]),
            |acc, name| DataType::structured([(name, acc)]),
        )
    }

    fn value(&self, _value: &'a Value) -> DataType {
        DataType::unit()
    }

    fn function(&self, _function: &'a function::Function, arguments: Vec<DataType>) -> DataType {
        DataType::product(arguments)
    }

    fn aggregate(&self, _aggregate: &'a aggregate::Aggregate, argument: DataType) -> DataType {
        argument
    }

    fn structured(&self, fields: Vec<(Identifier, DataType)>) -> DataType {
        DataType::product(fields.into_iter().map(|(_, data_type)| data_type))
    }
}

/// A visitor to compute the super_image
#[derive(Clone, Debug)]
pub struct SuperImageVisitor<'a>(&'a DataType);

impl<'a> Visitor<'a, Result<DataType>> for SuperImageVisitor<'a> {
    fn column(&self, column: &'a Column) -> Result<DataType> {
        Ok(self.0[column.clone()].clone())
    }

    fn value(&self, value: &'a Value) -> Result<DataType> {
        Ok(value.data_type())
    }

    fn function(
        &self,
        function: &'a function::Function,
        arguments: Vec<Result<DataType>>,
    ) -> Result<DataType> {
        let sets: Result<Vec<DataType>> = arguments.into_iter().collect();
        function.super_image(&sets?)
    }

    fn aggregate(
        &self,
        aggregate: &'a aggregate::Aggregate,
        argument: Result<DataType>,
    ) -> Result<DataType> {
        aggregate.super_image(&argument?)
    }

    fn structured(&self, fields: Vec<(Identifier, Result<DataType>)>) -> Result<DataType> {
        let fields: Result<Vec<(String, DataType)>> = fields
            .into_iter()
            .map(|(i, data_type)| Ok((i.split_last()?.0, data_type?)))
            .collect();
        Ok(DataType::structured(fields?))
    }
}

/// A visitor to compute the value
#[derive(Clone, Debug)]
pub struct ValueVisitor<'a>(&'a Value);

impl<'a> Visitor<'a, Result<Value>> for ValueVisitor<'a> {
    fn column(&self, column: &'a Column) -> Result<Value> {
        Ok(self.0[column.clone()].clone())
    }

    fn value(&self, value: &'a Value) -> Result<Value> {
        Ok(value.clone())
    }

    fn function(
        &self,
        function: &'a function::Function,
        arguments: Vec<Result<Value>>,
    ) -> Result<Value> {
        let args: Result<Vec<Value>> = arguments.into_iter().collect();
        function.value(&args?)
    }

    fn aggregate(
        &self,
        aggregate: &'a aggregate::Aggregate,
        argument: Result<Value>,
    ) -> Result<Value> {
        aggregate.value(&argument?)
    }

    fn structured(&self, fields: Vec<(Identifier, Result<Value>)>) -> Result<Value> {
        let fields: Result<Vec<(String, Value)>> = fields
            .into_iter()
            .map(|(ident, value)| Ok((ident.split_last()?.0, value?)))
            .collect();
        Ok(value::Value::structured(fields?))
    }
}

impl data_type::function::Function for Expr {
    fn domain(&self) -> DataType {
        self.accept(DomainVisitor)
    }

    fn super_image(&self, set: &DataType) -> result::Result<DataType, data_type::function::Error> {
        Ok(self.accept(SuperImageVisitor(set))?)
    }

    fn value(
        &self,
        arg: &value::Value,
    ) -> result::Result<value::Value, data_type::function::Error> {
        Ok(self.accept(ValueVisitor(arg))?)
    }
}

/// A visitor to collect column
#[derive(Clone, Debug)]
pub struct ColumnsVisitor;

impl<'a> Visitor<'a, Vec<&'a Column>> for ColumnsVisitor {
    fn column(&self, column: &'a Column) -> Vec<&'a Column> {
        vec![column]
    }

    fn value(&self, _value: &'a Value) -> Vec<&'a Column> {
        vec![]
    }

    fn function(
        &self,
        _function: &'a function::Function,
        arguments: Vec<Vec<&'a Column>>,
    ) -> Vec<&'a Column> {
        arguments
            .into_iter()
            .flat_map(|c| c.into_iter())
            .unique()
            .collect()
    }

    fn aggregate(
        &self,
        _aggregate: &'a aggregate::Aggregate,
        argument: Vec<&'a Column>,
    ) -> Vec<&'a Column> {
        argument
    }

    fn structured(&self, fields: Vec<(Identifier, Vec<&'a Column>)>) -> Vec<&'a Column> {
        fields
            .into_iter()
            .flat_map(|(_, c)| c.into_iter())
            .unique()
            .collect()
    }
}

impl Expr {
    /// Collect all columns in an expression
    pub fn columns(&self) -> Vec<&Column> {
        self.accept(ColumnsVisitor)
    }
}

/// A visitor to test the presence of column
#[derive(Clone, Debug)]
pub struct HasColumnVisitor;

impl<'a> Visitor<'a, bool> for HasColumnVisitor {
    fn column(&self, _column: &'a Column) -> bool {
        true
    }

    fn value(&self, _value: &'a Value) -> bool {
        false
    }

    fn function(&self, _function: &'a function::Function, arguments: Vec<bool>) -> bool {
        arguments.into_iter().any(identity)
    }

    fn aggregate(&self, _aggregate: &'a aggregate::Aggregate, argument: bool) -> bool {
        argument
    }

    fn structured(&self, fields: Vec<(Identifier, bool)>) -> bool {
        fields.into_iter().any(|(_, value)| value)
    }
}

impl Expr {
    pub fn has_column(&self) -> bool {
        self.accept(HasColumnVisitor)
    }
}

/// A visitor to test the presence of an aggregate
#[derive(Clone, Debug)]
pub struct HasAggregateVisitor;

impl<'a> Visitor<'a, bool> for HasAggregateVisitor {
    fn column(&self, _column: &'a Column) -> bool {
        false
    }

    fn value(&self, _value: &'a Value) -> bool {
        false
    }

    fn function(&self, _function: &'a function::Function, arguments: Vec<bool>) -> bool {
        arguments.into_iter().any(identity)
    }

    fn aggregate(&self, _aggregate: &'a aggregate::Aggregate, argument: bool) -> bool {
        true
    }

    fn structured(&self, fields: Vec<(Identifier, bool)>) -> bool {
        fields.into_iter().any(|(_, value)| value)
    }
}

impl Expr {
    pub fn has_aggregate(&self) -> bool {
        self.accept(HasAggregateVisitor)
    }
}

/// Rename the columns with the namer
#[derive(Clone, Debug)]
pub struct RenameVisitor<'a>(&'a Hierarchy<Identifier>);

impl<'a> Visitor<'a, Expr> for RenameVisitor<'a> {
    fn column(&self, column: &'a Column) -> Expr {
        self.0
            .get(column)
            .map(|identifier| Expr::Column(identifier.clone()))
            .unwrap_or_else(|| Expr::Column(column.clone()))
    }

    fn value(&self, value: &'a Value) -> Expr {
        Expr::Value(value.clone())
    }

    fn function(&self, function: &'a function::Function, arguments: Vec<Expr>) -> Expr {
        let arguments: Vec<Arc<Expr>> = arguments.into_iter().map(|a| Arc::new(a)).collect();
        Expr::Function(Function::new(function.clone(), arguments))
    }

    fn aggregate(&self, aggregate: &'a aggregate::Aggregate, argument: Expr) -> Expr {
        Expr::Aggregate(Aggregate::new(aggregate.clone(), Arc::new(argument)))
    }

    fn structured(&self, fields: Vec<(Identifier, Expr)>) -> Expr {
        let fields: Vec<(Identifier, Arc<Expr>)> =
            fields.into_iter().map(|(i, e)| (i, Arc::new(e))).collect();
        Expr::Struct(Struct::from_iter(fields))
    }
}

impl Expr {
    pub fn rename<'a>(&'a self, columns: &'a Hierarchy<Identifier>) -> Expr {
        self.accept(RenameVisitor(columns))
    }
}

/* Replace
 */

/// A visitor to replace sub-expressions with expressions
#[derive(Clone, Debug)]
pub struct ReplaceVisitor(Vec<(Expr, Expr)>);

impl<'a> visitor::Visitor<'a, Expr, (Expr, Vec<(Expr, Expr)>)> for ReplaceVisitor {
    fn visit(
        &self,
        acceptor: &'a Expr,
        dependencies: visitor::Visited<'a, Expr, (Expr, Vec<(Expr, Expr)>)>,
    ) -> (Expr, Vec<(Expr, Expr)>) {
        self.0
            .iter()
            .find(|(pattern, _)| acceptor == pattern)
            .map_or_else(
                || {
                    match acceptor {
                        Expr::Function(f) => {
                            let (arguments, matched): (Vec<Arc<Expr>>, Vec<&Vec<(Expr, Expr)>>) = f
                                .arguments
                                .iter()
                                .map(|a| {
                                    let (argument, matched) = dependencies.get(&**a);
                                    (Arc::new(argument.clone()), matched)
                                })
                                .unzip();
                            (
                                Expr::Function(Function::new(f.function.clone(), arguments)),
                                matched
                                    .into_iter()
                                    .flat_map(|m| m.into_iter().cloned())
                                    .collect(),
                            )
                        }
                        Expr::Aggregate(a) => {
                            let (argument, matched) = dependencies.get(&a.argument);
                            (
                                Expr::Aggregate(Aggregate::new(
                                    a.aggregate.clone(),
                                    Arc::new(argument.clone()),
                                )),
                                matched.clone(),
                            )
                        }
                        Expr::Struct(s) => {
                            let (fields, matched): (
                                Vec<(Identifier, Arc<Expr>)>,
                                Vec<&Vec<(Expr, Expr)>>,
                            ) = s
                                .fields
                                .iter()
                                .map(|(i, e)| {
                                    let (argument, matched) = dependencies.get(&**e);
                                    ((i.clone(), Arc::new(argument.clone())), matched)
                                })
                                .unzip();
                            (
                                Expr::Struct(Struct::new(fields)),
                                matched
                                    .into_iter()
                                    .flat_map(|m| m.into_iter().cloned())
                                    .collect(),
                            )
                        }
                        // No replacement
                        e => (e.clone(), vec![]),
                    }
                },
                |(pattern, replacement)| {
                    (
                        replacement.clone(),
                        vec![(pattern.clone(), replacement.clone())],
                    )
                },
            )
    }
}

impl Expr {
    /// Replace matched left expressions by corresponding right expressions
    pub fn replace(&self, map: Vec<(Expr, Expr)>) -> (Expr, Vec<(Expr, Expr)>) {
        self.accept(ReplaceVisitor(map))
    }
    /// Alias expressions by name
    pub fn alias(&self, named_exprs: Vec<(String, Expr)>) -> (Expr, Vec<(String, Expr)>) {
        let map = named_exprs
            .into_iter()
            .map(|(name, expr)| (expr, Expr::col(name)))
            .collect();
        let (expr, matched) = self.replace(map);
        (
            expr,
            matched
                .into_iter()
                .filter_map(|(p, r)| {
                    if let Expr::Column(c) = r {
                        Some((c.last().ok()?.to_string(), p))
                    } else {
                        None
                    }
                })
                .collect(),
        )
    }
    /// Transform an expression into an aggregation
    pub fn into_aggregate(self) -> Expr {
        match self {
            Expr::Aggregate(_) => self,
            _ => Expr::first(self), //TODO maybe change this default behavior
        }
    }
}

impl DataType {
    pub fn filter(&self, predicate: &Expr) -> DataType {
        match predicate {
            Expr::Column(c) => self.filter_by_column(c),
            Expr::Value(v) => self.filter_by_value(v),
            Expr::Function(f) => self.filter_by_function(f),
            Expr::Aggregate(_) | Expr::Struct(_) => self.clone(),
        }
    }

    /// Returns a new `DataType` clone of the current `DataType`
    /// filtered by the predicate `Identifier`
    /// TODO
    fn filter_by_column(&self, predicate: &Identifier) -> DataType {
        self.clone()
    }

    /// Returns a new `DataType` clone of the current `DataType`
    /// filtered by the predicate `Value`
    /// TODO
    fn filter_by_value(&self, predicate: &Value) -> DataType {
        self.clone()
    }

    /// Returns a new `DataType` clone of the current `DataType`
    /// filtered by the predicate `Function`
    ///
    /// Note: for the moment, we support only:
    /// - `Gt`, `GtEq`, `Lt`, `LtEq` functions comparing a column to a float or an integer value,
    /// - `Eq` function comparing a column to any value,
    /// - `And` and `Or` function between two supported Expr::Function,
    /// - 'InList` test if a column value belongs to a list
    fn filter_by_function(&self, predicate: &Function) -> DataType {
        let mut datatype = self.clone();

        match (predicate.function(), predicate.arguments().as_slice()) {
            (function::Function::And, [left, right]) => {
                let dt1 = self.filter(right).filter(left);
                let dt2 = self.filter(left).filter(right);
                datatype = dt1.super_intersection(&dt2).unwrap_or(datatype)
            }
            (function::Function::Or, [left, right]) => {
                let dt1 = self.filter(right);
                let dt2 = self.filter(left);
                datatype = dt1.super_union(&dt2).unwrap_or(datatype)
            }
            // Set min or max
            (function::Function::Gt, [left, right])
            | (function::Function::GtEq, [left, right])
            | (function::Function::Lt, [right, left])
            | (function::Function::LtEq, [right, left]) => {
                let left_dt = left.super_image(&datatype).unwrap();
                let left_dt = if let DataType::Optional(o) = left_dt {
                    o.data_type().clone()
                } else {
                    left_dt
                };
                let right_dt = right.super_image(&datatype).unwrap();
                let right_dt = if let DataType::Optional(o) = right_dt {
                    o.data_type().clone()
                } else {
                    right_dt
                };
                let set = DataType::structured_from_data_types([left_dt.clone(), right_dt.clone()]);
                if let Expr::Column(col) = left {
                    let dt = data_type::function::greatest()
                        .super_image(&set)
                        .unwrap()
                        .super_intersection(&left_dt)
                        .unwrap();
                    datatype = datatype.replace(col, dt)
                }
                if let Expr::Column(col) = right {
                    let dt = data_type::function::least()
                        .super_image(&set)
                        .unwrap()
                        .super_intersection(&right_dt)
                        .unwrap();
                    datatype = datatype.replace(col, dt)
                }
            }
            (function::Function::Eq, [left, right]) => {
                let left_dt = left.super_image(&datatype).unwrap();
                let right_dt = right.super_image(&datatype).unwrap();
                let dt = left_dt.super_intersection(&right_dt).unwrap();
                if let Expr::Column(col) = left {
                    datatype = datatype.replace(&col, dt.clone())
                }
                if let Expr::Column(col) = right {
                    datatype = datatype.replace(&col, dt)
                }
            }
            (function::Function::InList, [Expr::Column(col), Expr::Value(Value::List(l))]) => {
                let dt = DataType::from_iter(l.to_vec().clone())
                    .super_intersection(&datatype[col.as_slice()])
                    .unwrap_or(datatype.clone());
                datatype = datatype.replace(col, dt)
            }
            _ => (),
        }
        datatype
    }

    pub fn replace(&self, name: &Identifier, dt: DataType) -> DataType {
        let name = Identifier::from(
            self.hierarchy()
                .get_key_value(&name.to_vec())
                .unwrap()
                .0
                .into_iter()
                .cloned()
                .collect::<Vec<String>>(),
        );
        match self {
            DataType::Struct(st) => {
                let (head, tail) = name.split_head().unwrap();
                DataType::structured(
                    st.iter()
                        .map(|(s, d)| {
                            if &head == s {
                                (s, (**d).clone().replace(&tail, dt.clone()))
                            } else {
                                (s, (**d).clone())
                            }
                        })
                        .collect::<Vec<_>>(),
                )
            }
            DataType::Union(u) => {
                let (head, tail) = name.split_head().unwrap();
                DataType::union(
                    u.iter()
                        .map(|(s, d)| {
                            if &head == s {
                                (s, (**d).clone().replace(&tail, dt.clone()))
                            } else {
                                (s, (**d).clone())
                            }
                        })
                        .collect::<Vec<_>>(),
                )
            }
            _ => {
                assert_eq!(name.len(), 0);
                dt
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        builder::{Ready, With},
        relation::{schema::Schema, Relation},
    };

    #[test]
    fn test_column() {
        let sub_dict = DataType::unit() & ("age", DataType::integer_range(10..=200));
        let dict = DataType::unit() & ("height", DataType::float()) & ("table_1", sub_dict);
        println!("dict = {}", &dict);
        let col = Expr::col("height");
        println!("col = {}", col);
        println!(
            "col.super_image(dict) = {}",
            col.super_image(&dict).unwrap()
        );
        assert_eq!(col.super_image(&dict).unwrap(), DataType::float());
        let qual_col = Expr::qcol("table_1", "age");
        println!("qual_col = {}", qual_col);
        println!(
            "qual_col.super_image(dict) = {}",
            qual_col.super_image(&dict).unwrap()
        );
        assert_eq!(
            qual_col.super_image(&dict).unwrap(),
            DataType::integer_range(10..=200)
        );
    }

    #[test]
    fn test_bin_op() {
        let dict = DataType::unit()
            & ("0", DataType::float_interval(-5., 2.))
            & ("1", DataType::float_interval(-1., 2.));
        let left = Expr::col("0");
        let right = Expr::col("1");
        // Sum
        let sum = left.clone() + right.clone();
        println!(
            "{left}: {} + {right}: {} = sum: {}",
            left.co_domain(),
            right.co_domain(),
            sum.co_domain()
        );
        assert_eq!(
            sum.super_image(&dict).unwrap(),
            DataType::float_interval(-6., 4.)
        );
        // Prod
        let prod = left.clone() * right.clone();
        println!(
            "{left}: {} * {right}: {} = prod: {}",
            left.co_domain(),
            right.co_domain(),
            prod.co_domain()
        );
        assert_eq!(
            prod.super_image(&dict).unwrap(),
            DataType::float_interval(-10., 5.)
        );
    }

    #[test]
    fn test_bool_ops() {
        let dict = DataType::unit()
            & ("left", DataType::boolean())
            & ("right", DataType::boolean_value(true));
        let left = Expr::col("left");
        let right = Expr::col("right");
        // And
        let conj = left.clone() & right.clone();
        println!(
            "{left}: {} && {right}: {} = and: {}",
            left.co_domain(),
            right.co_domain(),
            conj.co_domain()
        );
        assert_eq!(conj.super_image(&dict).unwrap(), DataType::boolean());
        // Or
        let disj = left.clone() | right.clone();
        println!(
            "{left}: {} || {right}: {} = or: {}",
            left.co_domain(),
            right.co_domain(),
            disj.co_domain()
        );
        assert_eq!(
            disj.super_image(&dict).unwrap(),
            DataType::boolean_value(true)
        );
        // Not Or
        let ndisj = !Expr::or(left.clone(), right.clone());
        println!(
            "not ({left}: {} || {right}: {}) = not_or: {}",
            left.co_domain(),
            right.co_domain(),
            ndisj.co_domain()
        );
        assert_eq!(
            ndisj.super_image(&dict).unwrap(),
            DataType::boolean_value(false)
        );
    }

    #[test]
    fn test_value() {
        let dict = DataType::unit()
            & ("left", DataType::float_interval(-1., 2.))
            & ("hello", DataType::float_value(5.))
            & ("world", DataType::float_value(5.));
        let left = Expr::col("left");
        let right = Expr::val(5.);
        // Sum
        let sum = Expr::plus(left.clone(), right.clone());
        println!(
            "{left}: {} + {right}: {} = sum: {}",
            left.co_domain(),
            right.co_domain(),
            sum.co_domain()
        );
        assert_eq!(
            sum.super_image(&dict).unwrap(),
            DataType::float_interval(4., 7.)
        );
        // Prod
        let prod = Expr::multiply(left.clone(), right.clone());
        println!(
            "{left}: {} * {right}: {} = prod: {}",
            left.co_domain(),
            right.co_domain(),
            prod.co_domain()
        );
        assert_eq!(
            prod.super_image(&dict).unwrap(),
            DataType::float_interval(-5., 10.)
        );
        // Sin
        let expr = Expr::sin(Expr::multiply(Expr::val(0.2), Expr::col("hello")));
        println!("{expr}: {}", expr.super_image(&dict).unwrap());
        // Sin
        let expr = Expr::sin(Expr::multiply(Expr::val(0.2), Expr::col("world")));
        println!("{expr}: {}", expr.super_image(&dict).unwrap());
    }

    #[test]
    fn test_display_visitor() {
        let a = Expr::col("a");
        let b = Expr::col("b");
        let x = Expr::col("x");
        let expr = Expr::exp(Expr::sin(Expr::plus(Expr::multiply(a, x), b)));
        println!("{}", expr.accept(DisplayVisitor));
        assert_eq!(
            expr.accept(DisplayVisitor),
            "exp(sin(a * x + b))".to_string()
        );
    }

    #[test]
    fn test_iter() {
        let a = Expr::col("a");
        let b = Expr::col("b");
        let x = Expr::col("x");
        let expr = Expr::exp(Expr::sin(Expr::plus(Expr::multiply(a, x), b)));
        for e in expr.iter() {
            println!("Iter -> {e}");
        }
        assert_eq!(expr.iter().collect_vec().len(), 7);
    }

    #[test]
    fn test_ref_iterable() {
        let a = Expr::col("a");
        let b = Expr::col("b");
        let x = Expr::col("x");
        let expr = Expr::exp(Expr::sin(a * x + b));
        for e in &expr {
            println!("Iter -> {e}");
        }
        assert_eq!(expr.into_iter().collect_vec().len(), 7);
    }

    #[test]
    fn test_dsl() {
        let _rel: Arc<Relation> = Arc::new(
            Relation::table()
                .schema(
                    Schema::builder()
                        .with(("a", DataType::float_range(0.0..=1.0)))
                        .with(("b", DataType::float_range(0.0..=2.0)))
                        .with(("c", DataType::float_range(0.0..=5.0)))
                        .with(("d", DataType::float_range(0.0..=10.0)))
                        .with(("x", DataType::float_range(0.0..=20.0)))
                        .with(("z", DataType::float_range(0.0..=50.0)))
                        .with(("t", DataType::float_range(0.0..=100.0)))
                        .build(),
                )
                .build(),
        );
        let x = expr! { exp(a*b + cos(2*z)*d - 2*z + t*sin(c+3*x)) };
        println!("expr = {x}");
        println!("expr.data_type() = {}", x.data_type());
    }

    #[test]
    fn test_expr_domain_co_domain() {
        let x = expr! { exp(a*b + cos(2*z)*d - 2*z + t*sin(c+3*x)) };
        println!("expr = {x}");
        println!("expr.domain() = {}", x.domain());
        println!("expr.co_domain() = {}", x.co_domain());
        println!("expr.data_type() = {}", x.data_type());
    }

    #[test]
    fn test_expr_super_image() {
        let x = expr! { exp(a*b + cos(2*z)*d - 2*z + t*sin(c+3*x)) };
        println!("expr = {x}");
        println!(
            "expr.super_image() = {}",
            x.super_image(
                &(DataType::unit()
                    & ("a", DataType::float_interval(0.0, 0.1))
                    & ("b", DataType::float_interval(0.0, 0.1))
                    & ("z", DataType::float_interval(0.0, 0.1))
                    & ("d", DataType::integer_interval(-2, 2))
                    & ("t", DataType::float_interval(0.0, 0.1))
                    & ("c", DataType::float())
                    & ("x", DataType::integer()))
            )
            .unwrap()
        );
    }

    #[test]
    fn test_expr_value() {
        let x = expr! { exp(a*b + cos(2*z)*d - 2*z + t*sin(c+3*x)) };
        println!("expr = {x}");
        let val = x
            .value(
                &(Value::structured([
                    ("a", Value::float(0.1)),
                    ("b", Value::float(0.1)),
                    ("z", Value::float(0.1)),
                    ("d", Value::integer(0)),
                    ("t", Value::float(0.1)),
                    ("c", Value::float(0.0)),
                    ("x", Value::float(0.0)),
                ])),
            )
            .unwrap();
        println!("expr.value() = {}", val);
        assert_eq!(
            val,
            Value::from(f64::exp(
                0.1 * 0.1 + f64::cos(2.0 * 0.1) * 0.0 - 2.0 * 0.1 + 0.1 * f64::sin(0.0 + 3.0 * 0.0)
            ))
        );
    }

    #[test]
    fn test_expr_visitor() {
        let x = expr!(a + 2 * x - exp(c));
        println!("x = {}", x);
        println!("x.accept = {}", x.accept(DisplayVisitor));
        for (x, s) in x.iter_with(DisplayVisitor) {
            println!("(expr, str) = ({}, {})", x, s);
        }
        for s in &x {
            println!("expr = {}", s);
        }
        let typ = DataType::structured([
            ("a", DataType::float_interval(0.1, 0.5)),
            ("x", DataType::float_interval(-0.1, 2.0)),
            ("c", DataType::float_values([-1., 2.])),
        ]);
        for (x, t) in x.iter_with(SuperImageVisitor(&typ)) {
            println!("(expr, type) = ({}, {})", x, t.unwrap());
        }
    }

    #[test]
    fn test_aggregate() {
        let x = expr!(sum(a));
        println!("x = {}", x);
        let typ = DataType::structured([("a", DataType::integer_interval(3, 190))]);
        for (x, t) in x.iter_with(SuperImageVisitor(&typ)) {
            println!("(expr, type) = ({}, {})", x, t.unwrap());
        }
    }

    #[test]
    fn test_structured() {
        let x = Expr::structured([
            ("a", expr!(exp(a + 2 * sum(c)) + b * count(d))),
            ("b", expr!(ln(a + 2 * sum(c)) - b * count(d))),
        ]);
        println!("x = {x}");
    }

    #[test]
    fn test_columns() {
        let x = expr!(exp(a * b + cos(2 * z) * d - 2 * z + t * sin(c + 3 * x)));
        println!("x = {x}");
        let columns = x.columns();
        println!(
            "columns = {}",
            columns.into_iter().map(|c| format!("{c}")).join(", ")
        );
    }

    #[test]
    fn test_rename() {
        let x = expr!(exp(a * b + cos(2 * z) * d - 2 * z + t * sin(c + 3 * x)));
        println!("x = {x}");
        let names: Hierarchy<Identifier> = Hierarchy::from([(["a"], format!("A").into())]);
        let renamed = x.rename(&names);
        println!("renamed x = {renamed} ({names})");
    }

    #[test]
    fn test_replace() {
        let x =
            expr!(exp(a * b + cos(2 * z) * d - 2 * z + t * sin(c + 3 * x)) + cos(2 * z) - 2 * z);
        println!("x = {x}");
        let (a, m) = x.replace(vec![(expr!(2 * z), expr!(R))]);
        println!("a = {a}\nmatched = {m:?}");
        let (b, n) = x.replace(vec![
            (expr!(2 * z), expr!(R)),
            (expr!(cos(2 * z) * d - 2 * z), expr!(S)),
        ]);
        println!("b = {b}\nmatched = {n:?}");
    }

    #[test]
    fn test_sqrt() {
        let expression = expr!(sqrt(x + 1));
        println!("expression = {}", expression);
        println!("expression data type = {}", expression.data_type());
        println!(
            "expression super image = {}",
            expression
                .super_image(&DataType::structured([(
                    "x",
                    DataType::float_interval(1., 100.)
                ),]))
                .unwrap()
        );
    }

    #[test]
    fn test_pow() {
        let expression = expr!(pow(x, y));
        println!("expression = {}", expression);
        println!("expression data type = {}", expression.data_type());
        println!(
            "expression super image = {}",
            expression
                .super_image(&DataType::structured([
                    ("x", DataType::float_interval(1., 10.)),
                    ("y", DataType::float_values([-2., 0.5])),
                ]))
                .unwrap()
        );
    }

    #[test]
    fn test_case() {
        let expression = expr!(case(gt(x, 5), x, y));
        println!("\nexpression = {}", expression);
        println!("expression data type = {}", expression.data_type());
        let set = DataType::structured([
            ("x", DataType::float_interval(1., 10.)),
            ("y", DataType::float_values([-2., 0.5])),
        ]);
        println!(
            "expression super image = {}",
            expression.super_image(&set).unwrap()
        );

        let expression = Expr::case(
            Expr::gt(Expr::col(stringify!(x)), Expr::val(5)),
            Expr::col("x"),
            Expr::Value(Value::unit()),
        );
        println!("\nexpression = {}", expression);
        println!("expression data type = {}", expression.data_type());
        let set = DataType::structured([("x", DataType::float_interval(1., 10.))]);
        println!(
            "expression super image = {}",
            expression.super_image(&set).unwrap()
        );

        let expression = expr!(case(gt(x, 1), x, 1));
        println!("\nexpression = {}", expression);
        println!("expression data type = {}", expression.data_type());
        println!(
            "expression super image = {}",
            expression
                .super_image(&DataType::structured([(
                    "x",
                    DataType::float_interval(0., 2.)
                ),]))
                .unwrap()
        );

        let expression = expr!(gt(x, 1) * x + lt_eq(x, 1));
        println!("\nexpression = {}", expression);
        println!("expression data type = {}", expression.data_type());
        println!(
            "expression super image = {}",
            expression
                .super_image(&DataType::structured([(
                    "x",
                    DataType::float_interval(0., 2.)
                ),]))
                .unwrap()
        );
    }

    #[test]
    fn test_in_list_integer() {
        // a IN (1, 2, 3)
        let expression = Expr::in_list(Expr::col("a"), Expr::list([1, 2, 3]));
        println!("\nexpression = {}", expression);
        println!("expression data type = {}", expression.data_type());

        // a ∈ integer([1, 100])
        let set = DataType::structured([("a", DataType::integer_interval(1, 100))]);
        assert_eq!(expression.super_image(&set).unwrap(), DataType::boolean());
        assert_eq!(
            expression
                .value(&Value::structured([("a", Value::integer(1)),]))
                .unwrap(),
            Value::boolean(true)
        );
        assert_eq!(
            expression
                .value(&Value::structured([("a", Value::integer(20)),]))
                .unwrap(),
            Value::boolean(false)
        );

        // a ∈ integer([10, 100])
        let set = DataType::structured([("a", DataType::integer_interval(10, 100))]);
        assert_eq!(
            expression.super_image(&set).unwrap(),
            DataType::from(Value::from(false))
        );

        // a ∈ float([1, 100])
        let set = DataType::structured([("a", DataType::float_interval(1., 100.))]);
        assert_eq!(expression.super_image(&set).unwrap(), DataType::boolean());
        assert_eq!(
            expression
                .value(&Value::structured([("a", Value::float(1.)),]))
                .unwrap(),
            Value::boolean(true)
        );
        assert_eq!(
            expression
                .value(&Value::structured([("a", Value::float(20.5)),]))
                .unwrap(),
            Value::boolean(false)
        );

        // a ∈ text()
        let set = DataType::structured([(
            "a",
            DataType::text_values(["1".to_string(), "a".to_string()]),
        )]);
        assert_eq!(expression.super_image(&set).unwrap(), DataType::boolean());
        assert_eq!(
            expression
                .value(&Value::structured([("a", Value::text("1".to_string())),]))
                .unwrap(),
            Value::boolean(true)
        );
        assert_eq!(
            expression
                .value(&Value::structured([("a", Value::text("a".to_string())),]))
                .unwrap(),
            Value::boolean(false)
        );
    }

    #[test]
    fn test_in_list_float() {
        // a IN (10.5, 2.)
        let expression = Expr::in_list(Expr::col("a"), Expr::list([10.5, 2.]));
        println!("\nexpression = {}", expression);
        println!("expression data type = {}", expression.data_type());

        // a ∈ float([1, 100])
        let set = DataType::structured([("a", DataType::float_interval(1., 100.))]);
        assert_eq!(expression.super_image(&set).unwrap(), DataType::boolean());
        assert_eq!(
            expression
                .value(&Value::structured([("a", Value::float(10.5)),]))
                .unwrap(),
            Value::boolean(true)
        );
        assert_eq!(
            expression
                .value(&Value::structured([("a", Value::float(20.)),]))
                .unwrap(),
            Value::boolean(false)
        );

        // a ∈ float([100., 150])
        let set = DataType::structured([("a", DataType::float_interval(100., 150.))]);
        assert_eq!(
            expression.super_image(&set).unwrap(),
            DataType::boolean_value(false)
        );

        // a ∈ integer([1, 100])
        let set = DataType::structured([("a", DataType::integer_interval(1, 100))]);
        assert_eq!(expression.super_image(&set).unwrap(), DataType::boolean());
        assert_eq!(
            expression
                .value(&Value::structured([("a", Value::integer(2)),]))
                .unwrap(),
            Value::boolean(true)
        );
        assert_eq!(
            expression
                .value(&Value::structured([("a", Value::integer(20)),]))
                .unwrap(),
            Value::boolean(false)
        );

        // a ∈ text()
        let set = DataType::structured([(
            "a",
            DataType::text_values(["1".to_string(), "a".to_string()]),
        )]);
        assert_eq!(expression.super_image(&set).unwrap(), DataType::boolean());
        assert_eq!(
            expression
                .value(&Value::structured(
                    [("a", Value::text("10.5".to_string())),]
                ))
                .unwrap(),
            Value::boolean(true)
        );
        assert_eq!(
            expression
                .value(&Value::structured([("a", Value::text("a".to_string())),]))
                .unwrap(),
            Value::boolean(false)
        );
    }

    #[test]
    fn test_in_list_text() {
        // a IN ("a", "10", "2.")
        let expression = Expr::in_list(
            Expr::col("a"),
            Expr::list(["a".to_string(), "10".to_string()]),
        );
        println!("\nexpression = {}", expression);
        println!("expression data type = {}", expression.data_type());

        // a ∈ text()
        let set = DataType::structured([(
            "a",
            DataType::text_values(["1".to_string(), "a".to_string()]),
        )]);
        assert_eq!(expression.super_image(&set).unwrap(), DataType::boolean());
        assert_eq!(
            expression
                .value(&Value::structured([("a", Value::text("a".to_string())),]))
                .unwrap(),
            Value::boolean(true)
        );
        assert_eq!(
            expression
                .value(&Value::structured([("a", Value::text("1".to_string())),]))
                .unwrap(),
            Value::boolean(false)
        );

        // a ∈ float([1, 100])
        let set = DataType::structured([("a", DataType::float_interval(1., 100.))]);
        assert_eq!(expression.super_image(&set).unwrap(), DataType::boolean());
        assert_eq!(
            expression
                .value(&Value::structured([("a", Value::float(10.)),]))
                .unwrap(),
            Value::boolean(true)
        );
        assert_eq!(
            expression
                .value(&Value::structured([("a", Value::float(20.)),]))
                .unwrap(),
            Value::boolean(false)
        );

        // a ∈ integer([1, 100])
        let set = DataType::structured([("a", DataType::integer_interval(1, 100))]);
        assert_eq!(expression.super_image(&set).unwrap(), DataType::boolean());
        assert_eq!(
            expression
                .value(&Value::structured([("a", Value::integer(10)),]))
                .unwrap(),
            Value::boolean(true)
        );
        assert_eq!(
            expression
                .value(&Value::structured([("a", Value::integer(20)),]))
                .unwrap(),
            Value::boolean(false)
        );
    }
    #[test]
    fn test_std() {
        let expression = expr!(std(x));
        println!("expression = {}", expression);
        println!("expression data type = {}", expression.data_type());
        println!(
            "expression super image = {}",
            expression
                .super_image(&DataType::structured([(
                    "x",
                    DataType::list(DataType::float_interval(-1., 10.), 1, 50)
                ),]))
                .unwrap()
        );
    }

    #[test]
    fn test_md5() {
        let expression = expr!(md5(x));
        println!("expression = {}", expression);
        println!("expression data type = {}", expression.data_type());
        println!(
            "expression super image = {}",
            expression
                .super_image(&DataType::structured([(
                    "x",
                    DataType::text_values(["foo".into(), "bar".into()])
                ),]))
                .unwrap()
        );
        println!(
            "expression value = {}",
            expression
                .value(&Value::structured([("x", Value::text("foo")),]))
                .unwrap()
        );
    }

    #[test]
    fn test_concat() {
        let expression = Expr::concat(vec![Expr::col("x"), Expr::col("y"), Expr::col("x")]);
        println!("expression = {}", expression);
        println!("expression data type = {}", expression.data_type());
        println!(
            "expression super image = {}",
            expression
                .super_image(&DataType::structured([
                    ("x", DataType::text_values(["foo".into(), "bar".into()])),
                    ("y", DataType::text_values(["hello".into(), "world".into()]))
                ]))
                .unwrap()
        );
        println!(
            "expression value = {}",
            expression
                .value(&Value::structured([
                    ("x", Value::text("foo")),
                    ("y", Value::float(0.5432)),
                ]))
                .unwrap()
        );
    }

    #[test]
    fn test_filter_qualified_columns() {
        let dt = DataType::union([
            (
                "table1",
                DataType::structured([
                    ("a", DataType::float_interval(-10., 10.)),
                    ("x", DataType::float_interval(-20., 5.)),
                ]),
            ),
            (
                "table2",
                DataType::structured([("x", DataType::float_interval(-15., 3.))]),
            ),
        ]);

        // table1.a < table1.x
        let x = Expr::lt(Expr::qcol("table1", "a"), Expr::qcol("table1", "x"));
        println!("{}", x);
        let filtered_dt = dt.filter(&x);
        let true_dt = DataType::union([
            (
                "table1",
                DataType::structured([
                    ("a", DataType::float_interval(-10., 5.)),
                    ("x", DataType::float_interval(-10., 5.)),
                ]),
            ),
            (
                "table2",
                DataType::structured([("x", DataType::float_interval(-15., 3.))]),
            ),
        ]);
        println!("{true_dt}\n{filtered_dt}");
        println!("{}", true_dt[["table1"]] == filtered_dt[["table1"]]);
        assert_eq!(filtered_dt, true_dt);

        // a < table1.x
        let x = Expr::lt(Expr::col("a"), Expr::qcol("table1", "x"));
        println!("{}", x);
        let filtered_dt = dt.filter(&x);
        let true_dt = DataType::union([
            (
                "table1",
                DataType::structured([
                    ("a", DataType::float_interval(-10., 5.)),
                    ("x", DataType::float_interval(-10., 5.)),
                ]),
            ),
            (
                "table2",
                DataType::structured([("x", DataType::float_interval(-15., 3.))]),
            ),
        ]);
        println!("{true_dt}\n{filtered_dt}");
        println!("{}", true_dt[["table1"]] == filtered_dt[["table1"]]);
        assert_eq!(filtered_dt, true_dt);

        // a < table2.x
        let x = Expr::lt(Expr::col("a"), Expr::qcol("table2", "x"));
        println!("{}", x);
        let filtered_dt = dt.filter(&x);
        let true_dt = DataType::union([
            (
                "table1",
                DataType::structured([
                    ("a", DataType::float_interval(-10., 3.)),
                    ("x", DataType::float_interval(-20., 5.)),
                ]),
            ),
            (
                "table2",
                DataType::structured([("x", DataType::float_interval(-10., 3.))]),
            ),
        ]);
        println!("{true_dt}\n{filtered_dt}");
        println!("{}", true_dt[["table1"]] == filtered_dt[["table1"]]);
        assert_eq!(filtered_dt, true_dt);
    }

    #[test]
    fn test_filter_simple() {
        let dt = DataType::structured([
            ("a", DataType::float_interval(-10., 10.)),
            ("b", DataType::integer_interval(0, 8)),
            ("c", DataType::float()),
        ]);

        // (a > 5)
        let x = expr!(gt(a, 5));
        println!("{}", x);
        let filtered_dt = dt.filter(&x);
        println!("{}", filtered_dt);
        let true_dt = DataType::structured([
            ("a", DataType::float_interval(5., 10.)),
            ("b", DataType::integer_interval(0, 8)),
            ("c", DataType::float()),
        ]);
        assert_eq!(filtered_dt, true_dt);

        // ((((a > 5) and (b < 4)) and ((9 >= a) and (2 <= b))) and (c = 0.99))
        let x = expr!(and(
            and(and(gt(a, 5), lt(b, 4.)), and(gt_eq(9., a), lt_eq(2, b))),
            eq(c, 0.99)
        ));
        println!("{}", x);
        let filtered_dt = dt.filter(&x);
        let true_dt = DataType::structured([
            ("a", DataType::float_interval(5., 9.)),
            ("b", DataType::integer_interval(2, 4)),
            ("c", DataType::float_value(0.99)),
        ]);
        assert_eq!(filtered_dt, true_dt);

        // ((a = 45) and (b = 3.5) and (0 = c))
        let x = expr!(and(eq(a, 45), and(eq(b, 3.5), eq(0, c))));
        println!("{}", x);
        let filtered_dt = dt.filter(&x);
        let true_dt = DataType::structured([
            ("a", DataType::Null),
            ("b", DataType::Null),
            ("c", DataType::float_value(0.)),
        ]);
        assert_eq!(filtered_dt, true_dt);

        // (a in (-1, 3, 4.5)) and (b in (-1, 3, 4.5))
        let val = Expr::list([-1., 3., 4.5]);
        let a = Expr::in_list(Expr::col("a"), val.clone());
        let b = Expr::in_list(Expr::col("b"), val.clone());
        let x = Expr::and(a, b);
        println!("{}", x);
        let filtered_dt = dt.filter(&x);
        let true_dt = DataType::structured([
            ("a", DataType::float_values([-1., 3., 4.5])),
            ("b", DataType::integer_value(3)),
            ("c", DataType::float()),
        ]);
        assert_eq!(filtered_dt, true_dt);

        // (b = exp(a))
        let x = expr!(eq(b, exp(a)));
        let dt = DataType::structured([
            ("a", DataType::float_interval(-1., 1.)),
            ("b", DataType::float()),
        ]);
        let filtered_dt = dt.filter(&x);
        println!("{} -> {}", x, filtered_dt);
        let true_dt = DataType::structured([
            ("a", DataType::float_interval(-1., 1.)),
            (
                "b",
                DataType::float_interval((-1. as f64).exp(), (1. as f64).exp()),
            ),
        ]);
        assert_eq!(filtered_dt, true_dt);

        // Or
        let dt = DataType::structured([
            ("a", DataType::float_interval(-20., 20.)),
            ("b", DataType::integer_interval(0, 15)),
        ]);

        //  a > 0 or a < -10
        let x1 = Expr::lt(Expr::col("a"), Expr::val(-10));
        let x2 = Expr::gt(Expr::col("a"), Expr::val(0));
        let x = Expr::or(x1, x2);
        let filtered_dt = dt.filter(&x);
        println!("{} -> {}", x, filtered_dt);
        let true_dt = DataType::structured([
            (
                "a",
                DataType::from(data_type::Float::from_intervals([[0., 20.], [-20., -10.]])),
            ),
            ("b", DataType::integer_interval(0, 15)),
        ]);
        assert_eq!(filtered_dt, true_dt);

        let x1 = Expr::lt(Expr::col("a"), Expr::val(-8));
        let x2 = expr!(gt(b, 5));
        let x3 = expr!(gt_eq(a, 2 * b));
        println!("x1 = {}, x2 = {}, x3 = {}", x1, x2, x3);
    }

    #[test]
    fn test_filter_with_simple_column_deps() {
        let dt = DataType::structured([
            ("a", DataType::float_interval(-10., 10.)),
            ("b", DataType::integer_interval(0, 20)),
        ]);
        // (b < a)
        let x = expr!(lt(b, a));
        let filtered_dt = dt.filter(&x);
        println!("{} -> {}", x, filtered_dt);
        let true_dt = DataType::structured([
            ("a", DataType::float_interval(-0., 10.)),
            ("b", DataType::integer_interval(0, 10)),
        ]);
        assert_eq!(filtered_dt, true_dt);
        // (a > b)
        let x = expr!(gt(a, b));
        let filtered_dt = dt.filter(&x);
        println!("{} -> {}", x, filtered_dt);
        assert_eq!(filtered_dt, true_dt);

        // (b = a)
        let x = expr!(eq(b, a));
        let filtered_dt = dt.filter(&x);
        println!("{} -> {}", x, filtered_dt);
        let true_dt = DataType::structured([
            ("a", DataType::integer_interval(0, 10)),
            ("b", DataType::integer_interval(0, 10)),
        ]);
        assert_eq!(filtered_dt, true_dt);
        // (a = b)
        let x = expr!(eq(a, b));
        let filtered_dt = dt.filter(&x);
        println!("{} -> {}", x, filtered_dt);
        assert_eq!(filtered_dt, true_dt);
    }

    #[test]
    fn test_filter_with_column_deps() {
        let dt = DataType::structured([
            ("a", DataType::float_interval(-10., 10.)),
            ("b", DataType::integer_interval(0, 18)),
            ("c", DataType::float()),
        ]);

        // ((b < 2) and (b = c))
        let x = expr!(and(lt(b, 2), eq(b, c)));
        let filtered_dt = dt.filter(&x);
        println!("{} -> {}", x, filtered_dt);
        let true_dt = DataType::structured([
            ("a", DataType::float_interval(-10., 10.)),
            ("b", DataType::integer_values([0, 1, 2])),
            ("c", DataType::float_values([0., 1., 2.])),
        ]);
        assert_eq!(filtered_dt, true_dt);

        // ((b = c) and (b < 2))
        let x = expr!(and(lt(b, 2), eq(b, c)));
        let filtered_dt = dt.filter(&x);
        println!("{} -> {}", x, filtered_dt);
        let true_dt = DataType::structured([
            ("a", DataType::float_interval(-10., 10.)),
            ("b", DataType::integer_values([0, 1, 2])),
            ("c", DataType::float_values([0., 1., 2.])),
        ]);
        assert_eq!(filtered_dt, true_dt);

        // ((((a > 5) and (b < 14)) and ((b >= a) and (2 <= b))) and (a = c))
        let x = expr!(and(
            and(and(gt(a, 5), lt(b, 14.)), and(gt_eq(b, a), lt_eq(2, b))),
            eq(a, c)
        ));
        let filtered_dt = dt.filter(&x);
        println!("{} -> {}", x, filtered_dt);
        let true_dt = DataType::structured([
            ("a", DataType::float_interval(5., 10.)),
            (
                "b",
                DataType::integer_values([5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
            ),
            ("c", DataType::float_interval(5., 10.)),
        ]);
        assert_eq!(filtered_dt, true_dt);

        // (a >= (2 * b))
        let dt = DataType::structured([
            ("a", DataType::float_interval(-10., 10.)),
            ("b", DataType::integer_interval(0, 8)),
        ]);
        let x = expr!(gt_eq(a, 2 * b));
        let filtered_dt = dt.filter(&x);
        println!("{} -> {}", x, filtered_dt);
        let true_dt = DataType::structured([
            ("a", DataType::float_interval(0., 10.)),
            ("b", DataType::integer_interval(0, 8)),
        ]);
        assert_eq!(filtered_dt, true_dt);

        // (a <= (2 * b))
        let dt = DataType::structured([
            ("a", DataType::float_interval(-10., 10.)),
            ("b", DataType::integer_interval(0, 2)),
        ]);
        let x = expr!(lt_eq(a, 2 * b));
        let filtered_dt = dt.filter(&x);
        println!("{} -> {}", x, filtered_dt);
        let true_dt = DataType::structured([
            ("a", DataType::float_interval(-10., 4.)),
            ("b", DataType::integer_interval(0, 2)),
        ]);
        assert_eq!(filtered_dt, true_dt);
    }

    #[test]
    fn test_filter_composed() {
        let dt = DataType::structured([
            ("a", DataType::float_interval(-10., 10.)),
            ("b", DataType::integer_interval(0, 8)),
        ]);

        let x1 = expr!(lt(a, (3 * 5 - 8)));
        let x2 = expr!(gt(b, ((5 / 2 - 1) + 2)));
        let x3 = expr!(gt_eq(a, 2 * b));
        println!("x1 = {}, x2 = {}, x3 = {}", x1, x2, x3);

        // (a < ((3 * 5) - 8))
        let filtered_dt = dt.filter(&x1);
        println!("x1 = {} -> {}", x1, filtered_dt);
        let true_dt = DataType::structured([
            ("a", DataType::float_interval(-10., 7.)),
            ("b", DataType::integer_interval(0, 8)),
        ]);
        assert_eq!(filtered_dt, true_dt);

        // (b > (((5 / 2) - 1) + 2))
        let filtered_dt = dt.filter(&x2);
        println!("x2 = {} -> {}", x2, filtered_dt);
        let true_dt = DataType::structured([
            ("a", DataType::float_interval(-10., 10.)),
            ("b", DataType::integer_interval(3, 8)),
        ]);
        assert_eq!(filtered_dt, true_dt);

        // (a >= (2 * b))
        let filtered_dt = dt.filter(&x3);
        println!("x3 = {} -> {}", x3, filtered_dt);
        let true_dt = DataType::structured([
            ("a", DataType::float_interval(0., 10.)),
            ("b", DataType::integer_interval(0, 8)),
        ]);
        assert_eq!(filtered_dt, true_dt);

        // And
        let true_dt = DataType::structured([
            ("a", DataType::float_interval(6.0, 7.)),
            ("b", DataType::integer_interval(3, 8)),
        ]);

        //  (x1 and (x2 and x3))
        let x = Expr::and(x1.clone(), Expr::and(x2.clone(), x3.clone()));
        let filtered_dt = dt.filter(&x);
        println!("{} -> {}", x, filtered_dt);
        assert_eq!(filtered_dt, true_dt);

        //  (x3 and (x1 and x2))
        let x = Expr::and(x3.clone(), Expr::and(x1.clone(), x2.clone()));
        let filtered_dt = dt.filter(&x);
        println!("{} -> {}", x, filtered_dt);
        assert_eq!(filtered_dt, true_dt);

        //  (x2 and (x3 and x1))
        let x = Expr::and(x2.clone(), Expr::and(x3.clone(), x1.clone()));
        let filtered_dt = dt.filter(&x);
        println!("{} -> {}", x, filtered_dt);
        assert_eq!(filtered_dt, true_dt);

        // ((x1 and (x3 and x2))
        let x = Expr::and(x1.clone(), Expr::and(x3.clone(), x2.clone()));
        let filtered_dt = dt.filter(&x);
        println!("{} -> {}", x, filtered_dt);
        assert_eq!(filtered_dt, true_dt);

        // ((x2 and (x1 and x3))
        let x = Expr::and(x2.clone(), Expr::and(x1.clone(), x3.clone()));
        let filtered_dt = dt.filter(&x);
        println!("{} -> {}", x, filtered_dt);
        assert_eq!(filtered_dt, true_dt);

        // ((x3 and (x2 and x1))
        let x = Expr::and(x2.clone(), Expr::and(x1.clone(), x3.clone()));
        let filtered_dt = dt.filter(&x);
        println!("{} -> {}", x, filtered_dt);
        assert_eq!(filtered_dt, true_dt);
    }

    #[test]
    fn test_filter_optional() {
        let dt =
            DataType::structured([("a", DataType::optional(DataType::float_interval(-10., 10.)))]);

        // (a > 1)
        let x = expr!(gt(a, 1));
        let filtered_dt = dt.filter(&x);
        println!("{} -> {}", x, filtered_dt);
        let true_dt = DataType::structured([("a", DataType::float_interval(1., 10.))]);
        assert_eq!(filtered_dt, true_dt);

        // (a < 1)
        let x = expr!(lt(a, 1));
        let filtered_dt = dt.filter(&x);
        println!("{} -> {}", x, filtered_dt);
        let true_dt = DataType::structured([("a", DataType::float_interval(-10., 1.))]);
        assert_eq!(filtered_dt, true_dt);
    }

    #[test]
    fn test_filter_column() {
        let x = Expr::filter_column(
            "col1",
            Some(1.into()),
            Some(10.into()),
            vec![1.into(), 4.into(), 5.into()],
        )
        .unwrap();
        let true_expr = Expr::and(
            Expr::and(
                Expr::gt(Expr::col("col1"), Expr::val(1)),
                Expr::lt(Expr::col("col1"), Expr::val(10)),
            ),
            Expr::in_list(Expr::col("col1"), Expr::list([1, 4, 5])),
        );
        assert_eq!(x, true_expr)
    }

    #[test]
    fn test_filter() {
        let columns = [
            (
                "col1",
                (
                    Some(Value::integer(1)),
                    Some(Value::integer(10)),
                    vec![
                        Value::integer(1),
                        Value::integer(3),
                        Value::integer(6),
                        Value::integer(7),
                    ],
                ),
            ),
            ("col2", (None, Some(Value::float(10.0)), vec![])),
            ("col3", (Some(Value::float(0.0)), None, vec![])),
            (
                "col4",
                (
                    None,
                    None,
                    vec![Value::text("a"), Value::text("b"), Value::text("c")],
                ),
            ),
        ]
        .into_iter()
        .collect();
        let col1_expr = Expr::and(
            Expr::and(
                Expr::gt(Expr::col("col1"), Expr::val(1)),
                Expr::lt(Expr::col("col1"), Expr::val(10)),
            ),
            Expr::in_list(Expr::col("col1"), Expr::list([1, 3, 6, 7])),
        );
        let col2_expr = Expr::lt(Expr::col("col2"), Expr::val(10.));
        let col3_expr = Expr::gt(Expr::col("col3"), Expr::val(0.));
        let col4_expr = Expr::in_list(
            Expr::col("col4"),
            Expr::list(["a".to_string(), "b".to_string(), "c".to_string()]),
        );

        let true_expr = Expr::and(
            Expr::and(Expr::and(col1_expr, col2_expr), col3_expr),
            col4_expr,
        );
        assert_eq!(Expr::filter(columns), true_expr);
    }

    #[test]
    fn test_greatest() {
        let dt = DataType::union([
            (
                "table1",
                DataType::structured([
                    ("x", DataType::float_interval(1., 4.)),
                    ("b", DataType::integer_interval(2, 7)),
                ]),
            ),
            (
                "table2",
                DataType::structured([
                    ("x", DataType::float_interval(1., 4.)),
                    ("y", DataType::float_interval(3.4, 7.1)),
                ]),
            ),
        ]);
        let value = Value::structured([
            (
                "table1",
                Value::structured([("x", Value::float(2.3)), ("b", Value::integer(5))]),
            ),
            (
                "table2",
                Value::structured([("x", Value::float(3.5)), ("y", Value::float(4.3))]),
            ),
        ]);

        // greatest(table1.x, y)
        let expression = Expr::greatest(Expr::qcol("table1", "x"), Expr::col("y"));
        println!("\nexpression = {}", expression);
        assert_eq!(
            expression.domain(),
            DataType::unit()
                & ("y", DataType::Any)
                & ("table1", DataType::structured([("x", DataType::Any)]))
        );
        println!("expression co_domain = {}", expression.co_domain());
        println!("expression data type = {}", expression.data_type());
        assert_eq!(
            expression.super_image(&dt).unwrap(),
            DataType::float_interval(3.4, 7.1)
        );
        assert_eq!(expression.value(&value).unwrap(), Value::float(4.3));

        // greatest(b, y)
        let expression = Expr::greatest(Expr::col("b"), Expr::col("y"));
        println!("\nexpression = {}", expression);
        assert_eq!(
            expression.domain(),
            DataType::unit() & ("b", DataType::Any) & ("y", DataType::Any)
        );
        println!("expression co_domain = {}", expression.co_domain());
        println!("expression data type = {}", expression.data_type());
        assert_eq!(
            expression.super_image(&dt).unwrap(),
            DataType::float_interval(3.4, 7.1)
        );
        assert_eq!(expression.value(&value).unwrap(), Value::float(5.0));

        // greatest(table1.x, table1.b)
        let expression = Expr::greatest(Expr::qcol("table1", "x"), Expr::qcol("table1", "b"));
        println!("\nexpression = {}", expression);
        println!("{}", expression.domain(),);
        assert_eq!(
            expression.domain(),
            DataType::unit()
                & (
                    "table1",
                    DataType::unit() & ("b", DataType::Any) & ("y", DataType::Any)
                )
        );
        println!("expression co_domain = {}", expression.co_domain());
        println!("expression data type = {}", expression.data_type());
        assert_eq!(
            expression.super_image(&dt).unwrap(),
            DataType::float_interval(2., 4.)
                .super_union(&DataType::integer_interval(5, 7))
                .unwrap()
        );
        assert_eq!(expression.value(&value).unwrap(), Value::float(5.));
    }

    #[test]
    fn test_replace_datatype() {
        let dt = DataType::union([(
            "table1",
            DataType::structured([("a", DataType::float()), ("b", DataType::integer())]),
        )]);
        let correct_dt = DataType::union([(
            "table1",
            DataType::structured([("a", DataType::integer()), ("b", DataType::integer())]),
        )]);
        let name: Identifier = vec!["table1".to_string(), "a".to_string()].into();
        let new_dt = dt.replace(&name, DataType::integer());
        assert_eq!(new_dt, correct_dt);

        let name: Identifier = vec!["a".to_string()].into();
        let new_dt = dt.replace(&name, DataType::integer());
        assert_eq!(new_dt, correct_dt);

        let new_dt = DataType::float().replace(&Identifier::empty(), DataType::integer());
        assert_eq!(new_dt, DataType::integer());
    }

    #[test]
    fn test_conversion() {
        let x = Expr::val(5);
        let v = Value::try_from(x).unwrap();
        assert_eq!(v, Value::from(5));

        let x = expr!(3 * 5);
        let v = Value::try_from(x).unwrap();
        assert_eq!(v, Value::from(15));

        let x = Expr::val(-5.);
        let v = Value::try_from(x).unwrap();
        assert_eq!(v, Value::from(-5.));
    }
}
