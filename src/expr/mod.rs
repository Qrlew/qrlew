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
    convert::identity,
    error, fmt, hash,
    ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Rem, Sub},
    rc::Rc,
    result,
};

use crate::{
    data_type::{self, value, DataType, DataTyped, Variant as _},
    hierarchy::Hierarchy,
    namer::{self, FIELD},
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
    pub arguments: Vec<Rc<Expr>>,
}

impl Function {
    /// Basic constructor
    pub fn new(function: function::Function, arguments: Vec<Rc<Expr>>) -> Function {
        Function {
            function,
            arguments,
        }
    }

    /// Returns the `DataType` of a column filterd by the current `Function`
    ///
    /// # Arguments:
    /// * `column` - The `Column` to be filtered
    /// * `datatype` - The `DataType` of `column`
    ///
    /// For example, we consider the `Column` `my_col` with `datatype = float[0 10]`.
    ///     -  `self = gt(my_col, 5)` returns `DataType` `float[5 10]`,
    ///     -  `self = gt(my_col, -5)` returns `DataType` `float[0 10]`,
    ///     -  `self = gt(another_col, 5)` returns `DataType` `float[0 10]`
    ///
    /// Note: for the moment, we support only `Function` made of the composition of:
    /// - `Gt`, `GtEq`, `Lt`, `LtEq` functions comparing a column to a float or an integer value,
    /// - `Eq` function comparing a column to any value,
    /// - `And` function between two supported Expr::Function.
    pub fn filter_column_data_type(&self, column: &Column, datatype: &DataType) -> DataType {
        let args: Vec<&Expr> = self.arguments.iter().map(|x| x.as_ref()).collect();
        match (self.function, args.as_slice()) {
            // And
            (function::Function::And, [Expr::Function(left), Expr::Function(right)]) => {
                left.filter_column_data_type(
                    column,
                    datatype
                ).super_intersection(
                    &right.filter_column_data_type(
                        column,
                        datatype
                    )
                ).unwrap_or(datatype.clone())
            },
            // Float, set min
            (function::Function::Gt, [Expr::Column(col), Expr::Value(Value::Float(f))])
            | (function::Function::GtEq, [Expr::Column(col), Expr::Value(Value::Float(f))])
            | (function::Function::Lt, [Expr::Value(Value::Float(f)), Expr::Column(col)])
            | (function::Function::LtEq, [Expr::Value(Value::Float(f)), Expr::Column(col)])
            if col == column => {
                DataType::float_min(**f).super_intersection(&datatype).unwrap_or(datatype.clone())
            },
            // Float, set max
            (function::Function::Lt, [Expr::Column(col), Expr::Value(Value::Float(f))])
            | (function::Function::LtEq, [Expr::Column(col), Expr::Value(Value::Float(f))])
            | (function::Function::Gt, [Expr::Value(Value::Float(f)), Expr::Column(col)])
            | (function::Function::GtEq, [Expr::Value(Value::Float(f)), Expr::Column(col)])
            if col == column => {
                DataType::float_max(**f).super_intersection(&datatype).unwrap_or(datatype.clone())
            },
            // Integer, set min
            (function::Function::Gt, [Expr::Column(col), Expr::Value(Value::Integer(i))])
            | (function::Function::GtEq, [Expr::Column(col), Expr::Value(Value::Integer(i))])
            | (function::Function::Lt, [Expr::Value(Value::Integer(i)), Expr::Column(col)])
            | (function::Function::LtEq, [Expr::Value(Value::Integer(i)), Expr::Column(col)])
            if col == column => {
                DataType::integer_min(**i).super_intersection(&datatype).unwrap_or(datatype.clone())
            },
            // Integer, set max
            (function::Function::Lt, [Expr::Column(col), Expr::Value(Value::Integer(i))])
            | (function::Function::LtEq, [Expr::Column(col), Expr::Value(Value::Integer(i))])
            | (function::Function::Gt, [Expr::Value(Value::Integer(i)), Expr::Column(col)])
            | (function::Function::GtEq, [Expr::Value(Value::Integer(i)), Expr::Column(col)])
            if col == column => {
                DataType::integer_max(**i).super_intersection(&datatype).unwrap_or(datatype.clone())
            },
            // Eq
            (function::Function::Eq, [Expr::Column(col), Expr::Value(val)])
            if col == column => {
                DataType::from(val.clone()).super_intersection(&datatype).unwrap_or(datatype.clone())
            },
            _ => datatype.clone(),
        }
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.function.style() {
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
            function::Style::Case => {
                write!(
                    f,
                    "CASE WHEN {} THEN {} ELSE {} END",
                    self.arguments[0], self.arguments[1], self.arguments[2]
                )
            }
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
}

/// Implement unary function constructors
macro_rules! impl_unary_function_constructors {
    ($( $Function:ident ),*) => {
        impl Function {
            paste! {
                $(pub fn [<$Function:snake>]<E: Into<Expr>>(expr: E) -> Function {
                    Function::new(function::Function::$Function, vec![Rc::new(expr.into())])
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
    Opposite, Not, Exp, Ln, Log, Abs, Sin, Cos, Sqrt, Md5, Lower, Upper, CharLength
); // TODO Complete that

/// Implement binary function constructors
macro_rules! impl_binary_function_constructors {
    ($( $Function:ident ),*) => {
        impl Function {
            paste! {
                $(
                    pub fn [<$Function:snake>]<L: Into<Expr>, R: Into<Expr>>(left: L, right: R) -> Function {
                        Function::new(function::Function::$Function, vec![Rc::new(left.into()), Rc::new(right.into())])
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
    Position
);

/// Implement ternary function constructors
macro_rules! impl_ternary_function_constructors {
    ($( $Function:ident ),*) => {
        impl Function {
            paste! {
                $(
                    pub fn [<$Function:snake>]<F: Into<Expr>, S: Into<Expr>, T: Into<Expr>>(first: F, second: S, third: T) -> Function {
                        Function::new(function::Function::$Function, vec![Rc::new(first.into()), Rc::new(second.into()), Rc::new(third.into())])
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
                        Function::new(function::Function::$Function(args.len()), args.into_iter().map(|e| Rc::new(e.into())).collect())
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
    argument: Rc<Expr>,
}

impl Aggregate {
    /// Basic constructor
    pub fn new(aggregate: aggregate::Aggregate, argument: Rc<Expr>) -> Aggregate {
        Aggregate {
            aggregate,
            argument,
        }
    }

    pub fn aggregate(&self) -> aggregate::Aggregate {
        self.aggregate
    }

    pub fn argument_name(&self) -> Result<&String> {
        match self.argument.as_ref() {
            Expr::Column(col) => Ok(col.last().unwrap()),
            _ => Err(Error::other("Cannot return the argument_name")),
        }
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
                    Aggregate::new(aggregate::Aggregate::$Aggregate, Rc::new(expr.into()))
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
    };
}

impl_aggregation_constructors!(First, Last, Min, Max, Count, Mean, Sum, Var, Std);

/// An aggregate function expression
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Struct {
    /// Fields
    fields: Vec<(Identifier, Rc<Expr>)>,
}

impl Struct {
    /// Basic constructor
    pub fn new(fields: Vec<(Identifier, Rc<Expr>)>) -> Struct {
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

impl<S: Into<Identifier>, E: Into<Rc<Expr>>> FromIterator<(S, E)> for Struct {
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

    pub fn structured<S: Clone + Into<String>, E: Clone + Into<Rc<Expr>>, F: AsRef<[(S, E)]>>(
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

        $(impl_conversions!($Variant);)*
    }
}

impl_traits!(Column, Value, Function, Aggregate, Struct);

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
            function::Style::Case => {
                format!(
                    "( CASE WHEN {} THEN {} ELSE {} END )",
                    arguments[0], arguments[1], arguments[2]
                )
            }
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
        if column.len() == 1 {
            DataType::structured([(&column.head().unwrap(), DataType::Any)]) // TODO fix this
        } else {
            DataType::Any
        }
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

/// Rename the columns with the namer
#[derive(Clone, Debug)]
pub struct RenameVisitor<'a>(&'a Hierarchy<Identifier>);

impl<'a> Visitor<'a, Expr> for RenameVisitor<'a> {
    fn column(&self, column: &'a Column) -> Expr {
        self.0
            .get(column)
            .map(|name| Expr::Column(name.clone()))
            .unwrap_or_else(|| Expr::Column(column.clone()))
    }

    fn value(&self, value: &'a Value) -> Expr {
        Expr::Value(value.clone())
    }

    fn function(&self, function: &'a function::Function, arguments: Vec<Expr>) -> Expr {
        let arguments: Vec<Rc<Expr>> = arguments.into_iter().map(|a| Rc::new(a)).collect();
        Expr::Function(Function::new(function.clone(), arguments))
    }

    fn aggregate(&self, aggregate: &'a aggregate::Aggregate, argument: Expr) -> Expr {
        Expr::Aggregate(Aggregate::new(aggregate.clone(), Rc::new(argument)))
    }

    fn structured(&self, fields: Vec<(Identifier, Expr)>) -> Expr {
        let fields: Vec<(Identifier, Rc<Expr>)> =
            fields.into_iter().map(|(i, e)| (i, Rc::new(e))).collect();
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
                            let (arguments, matched): (Vec<Rc<Expr>>, Vec<&Vec<(Expr, Expr)>>) = f
                                .arguments
                                .iter()
                                .map(|a| {
                                    let (argument, matched) = dependencies.get(&**a);
                                    (Rc::new(argument.clone()), matched)
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
                                    Rc::new(argument.clone()),
                                )),
                                matched.clone(),
                            )
                        }
                        Expr::Struct(s) => {
                            let (fields, matched): (
                                Vec<(Identifier, Rc<Expr>)>,
                                Vec<&Vec<(Expr, Expr)>>,
                            ) = s
                                .fields
                                .iter()
                                .map(|(i, e)| {
                                    let (argument, matched) = dependencies.get(&**e);
                                    ((i.clone(), Rc::new(argument.clone())), matched)
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
                        e => (e.clone(), Vec::new()),
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
                        Some((c.last()?.clone(), p))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        builder::{Ready, With},
        data_type::function::Function as _,
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
        // // Sum
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
        let _rel: Rc<Relation> = Rc::new(
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
    fn test_cmp() {
        let dict = DataType::unit()
            & ("a", DataType::float_interval(-5., 2.))
            & ("1", DataType::float_interval(-1., 2.));
        let left = Expr::col("a");
        let right = Expr::val(0);
        // gt
        let ex = Expr::gt(left.clone(), right.clone());
        println!("{}", ex);
        // assert_eq!(
        //     sum.super_image(&dict).unwrap(),
        //     DataType::float_interval(-6., 4.)
        // );
    }

    #[test]
    fn test_filter_column_data_type_float() {
        // set min value
        let col = Column::from("MyCol");
        let datatype = DataType::float();
        let value = Expr::val(5.);

        let func = Function::gt(col.clone(), value.clone());
        assert_eq!(
            func.filter_column_data_type(&col, &datatype),
            DataType::float_min(5.)
        );

        let func = Function::gt_eq(col.clone(), value.clone());
        assert_eq!(
            func.filter_column_data_type(&col, &datatype),
            DataType::float_min(5.)
        );

        let func = Function::lt(value.clone(), col.clone());
        assert_eq!(
            func.filter_column_data_type(&col, &datatype),
            DataType::float_min(5.)
        );

        let func = Function::lt_eq(value.clone(), col.clone());
        assert_eq!(
            func.filter_column_data_type(&col, &datatype),
            DataType::float_min(5.)
        );

        // columns do not match
        let col = Column::from("MyCol");
        let datatype = DataType::float();
        let func = Function::gt(Expr::col("NotMyCol"), Expr::val(5.));
        assert_eq!(
            func.filter_column_data_type(&col, &datatype),
            DataType::float()
        );

        // set max value
        let col = Column::from("MyCol");
        let datatype = DataType::float();
        let value = Expr::val(5.);

        let func = Function::lt(col.clone(), value.clone());
        assert_eq!(
            func.filter_column_data_type(&col, &datatype),
            DataType::float_max(5.)
        );

        let func = Function::lt_eq(col.clone(), value.clone());
        assert_eq!(
            func.filter_column_data_type(&col, &datatype),
            DataType::float_max(5.)
        );

        let func = Function::gt(value.clone(), col.clone());
        assert_eq!(
            func.filter_column_data_type(&col, &datatype),
            DataType::float_max(5.)
        );

        let func = Function::gt_eq(value.clone(), col.clone());
        assert_eq!(
            func.filter_column_data_type(&col, &datatype),
            DataType::float_max(5.)
        );

        // eq
        let col = Column::from("MyCol");
        let datatype = DataType::float();
        let value = Expr::val(5.);

        let func = Function::eq(col.clone(), value.clone());
        assert_eq!(
            func.filter_column_data_type(&col, &datatype),
            DataType::float_value(5.)
        );
    }

    #[test]
    fn test_filter_column_data_type_integer() {
        // set min value
        let col = Column::from("MyCol");
        let datatype = DataType::integer_interval(0, 10);
        let value = Expr::val(5);

        let func = Function::gt(col.clone(), value.clone());
        assert_eq!(
            func.filter_column_data_type(&col, &datatype),
            DataType::integer_interval(5, 10)
        );

        let func = Function::gt_eq(col.clone(), value.clone());
        assert_eq!(
            func.filter_column_data_type(&col, &datatype),
            DataType::integer_interval(5, 10)
        );

        let func = Function::lt(value.clone(), col.clone());
        assert_eq!(
            func.filter_column_data_type(&col, &datatype),
            DataType::integer_interval(5, 10)
        );

        let func = Function::lt_eq(value.clone(), col.clone());
        assert_eq!(
            func.filter_column_data_type(&col, &datatype),
            DataType::integer_interval(5, 10)
        );

        // columns do not match
        let col = Column::from("MyCol");
        let datatype = DataType::integer();
        let func = Function::gt(Expr::col("NotMyCol"), Expr::val(5));
        assert_eq!(
            func.filter_column_data_type(&col, &datatype),
            datatype.clone()
        );

        // set max value
        let col = Column::from("MyCol");
        let datatype: DataType = DataType::integer();
        let value = Expr::val(5);

        let func = Function::lt(col.clone(), value.clone());
        assert_eq!(
            func.filter_column_data_type(&col, &datatype),
            DataType::integer_max(5)
        );

        let func = Function::lt_eq(col.clone(), value.clone());
        assert_eq!(
            func.filter_column_data_type(&col, &datatype),
            DataType::integer_max(5)
        );

        let func = Function::gt(value.clone(), col.clone());
        assert_eq!(
            func.filter_column_data_type(&col, &datatype),
            DataType::integer_max(5)
        );

        let func = Function::gt_eq(value.clone(), col.clone());
        assert_eq!(
            func.filter_column_data_type(&col, &datatype),
            DataType::integer_max(5)
        );

        // eq
        let col = Column::from("MyCol");
        let datatype = DataType::integer();
        let value = Expr::val(5);

        let func = Function::eq(col.clone(), value.clone());
        assert_eq!(
            func.filter_column_data_type(&col, &datatype),
            DataType::integer_value(5)
        );
    }

    #[test]
    fn test_filter_column_data_type_and() {
        // set min value
        let col = Column::from("MyCol");
        let datatype = DataType::float();

        let func = Function::and(
            Function::gt(col.clone(), Expr::val(5.)),
            Function::lt(col.clone(), Expr::val(7.)),
        );
        assert_eq!(
            func.filter_column_data_type(&col, &datatype),
            DataType::float_interval(5., 7.)
        );
    }
}
