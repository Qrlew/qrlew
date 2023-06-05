//! This module defines the `Relation` struct
//! A `Relation` is the lazy representation of a computation that can be compiled into DP
//! It is rather close to similar lazy computation representations designed for a different purpose: https://pola-rs.github.io/polars/polars_lazy/prelude/enum.LogicalPlan.html#
//! 
//! Nice ideas about extensibility can be taken from: https://github.com/dcreager/expression-problem-rust
//! 

pub mod builder;
pub mod dot;
pub mod field;
pub mod schema;
pub mod sql;
pub mod transforms;

use std::{cmp, error, fmt, hash, ops::Index, rc::Rc, result};

use colored::Colorize;
use itertools::Itertools;

pub use super::relation::dot::display;
use crate::{
    builder::Ready,
    data_type::{self, function::Function, intervals::Bound, DataType, DataTyped, Integer, Struct, Variant as _},
    expr::{self, Expr, Identifier, Split},
    namer,
    visitor::{self, Acceptor, Dependencies, Visited},
};
pub use builder::{
    JoinBuilder, SetBuilder, MapBuilder, ReduceBuilder, TableBuilder, WithInput, WithSchema, WithoutInput,
    WithoutSchema,
};
pub use field::Field;
pub use schema::Schema;

// Error management

#[derive(Debug)]
pub enum Error {
    InvalidRelation(String),
    InvalidName(String),
    InvalidIndex(String),
    InvalidConversion(String),
    Other(String),
}

impl Error {
    pub fn invalid_relation(relation: impl fmt::Display) -> Error {
        Error::InvalidRelation(format!("{} is invalid", relation))
    }
    pub fn invalid_name(name: impl fmt::Display) -> Error {
        Error::InvalidName(format!("{} is invalid", name))
    }
    pub fn invalid_index(index: impl fmt::Display) -> Error {
        Error::InvalidIndex(format!("{} is invalid", index))
    }
    pub fn invalid_conversion(from: impl fmt::Display, to: impl fmt::Display) -> Error {
        Error::InvalidConversion(format!("Invalid conversion from {} to {}", from, to))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidRelation(desc) => writeln!(f, "InvalidRelation: {}", desc),
            Error::InvalidName(desc) => writeln!(f, "InvalidName: {}", desc),
            Error::InvalidIndex(desc) => writeln!(f, "InvalidIndex: {}", desc),
            Error::InvalidConversion(desc) => writeln!(f, "InvalidConversion: {}", desc),
            Error::Other(err) => writeln!(f, "{}", err),
        }
    }
}

impl error::Error for Error {}

impl From<data_type::function::Error> for Error {
    fn from(err: data_type::function::Error) -> Self {
        Error::Other(err.to_string())
    }
}
impl From<expr::Error> for Error {
    fn from(err: expr::Error) -> Self {
        Error::Other(err.to_string())
    }
}
impl From<crate::io::Error> for Error {
    fn from(err: crate::io::Error) -> Self {
        Error::Other(err.to_string())
    }
}

pub type Result<T> = result::Result<T, Error>;

/// Each expression variant must comply with this trait
pub trait Variant:
    TryFrom<Relation>
    + Into<Relation>
    + Clone
    + fmt::Debug
    + fmt::Display
    + hash::Hash
    + cmp::PartialEq
    + DataTyped
    + for<'a> Index<&'a Identifier>
{
    /// Return the name
    fn name(&self) -> &str;
    /// Return the Schema
    fn schema(&self) -> &Schema;
    /// Return the size bounds
    fn size(&self) -> &Integer;
    /// Return the inputs
    fn inputs(&self) -> Vec<&Relation>;
    /// Access a field of the Relation by index
    fn field_from_index(&self, index: usize) -> Result<&Field> {
        self.schema().field_from_index(index)
    }
    /// Access a field of the Relation by identifier
    fn field_from_identifier(&self, identifier: &Identifier) -> Result<&Field> {
        if identifier.len() == 1 {
            self.schema().field_from_identifier(identifier)
        } else {
            self.inputs()
                .into_iter()
                .find(|&relation| relation.name() == identifier.head().unwrap_or(String::new()))
                .ok_or_else(|| Error::invalid_name(identifier))?
                .schema()
                .field_from_identifier(&identifier.tail()?)
        }
    }
    /// Access a field of the Relation by qualified identifier
    fn field_from_qualified_name(&self, identifier: &Identifier) -> Result<&Field> {
        if let &[ref table, ref field] = identifier.as_slice() {
            if table == self.name() {
                self.schema().field_from_identifier(&field.clone().into())
            } else {
                Err(Error::invalid_name(identifier))
            }
        } else {
            Err(Error::invalid_name(identifier))
        }
    }
}

// Table Relation

/// Produces rows from a table provider by reference or from the context
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Table {
    /// The name of the table
    pub name: String,
    /// The schema description of the output
    pub schema: Schema,
    /// The size of the table
    pub size: Integer,
}

impl Table {
    /// Main constructor
    pub fn new(name: String, schema: Schema, size: Integer) -> Self {
        Table { name, schema, size }
    }

    /// From schema
    pub fn from_schema<S: Into<Schema>>(schema: S) -> Table {
        Table::new(
            namer::new_name("table"),
            schema.into(),
            Integer::from_min(0),
        )
    }

    /// A builder
    pub fn builder() -> TableBuilder<WithoutSchema> {
        TableBuilder::new()
    }
}

impl fmt::Display for Table {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name.to_string().bold().red())
    }
}

impl DataTyped for Table {
    fn data_type(&self) -> DataType {
        self.schema.data_type()
    }
}

impl Variant for Table {
    fn name(&self) -> &str {
        &self.name
    }

    fn schema(&self) -> &Schema {
        &self.schema
    }

    fn size(&self) -> &Integer {
        &self.size
    }

    fn inputs(&self) -> Vec<&Relation> {
        Vec::new()
    }
}

// Map Relation

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct OrderBy {
    pub expr: Expr,
    pub asc: bool,
}

impl OrderBy {
    pub fn new(expr: Expr, asc: bool) -> Self {
        OrderBy { expr, asc }
    }
}

/// Map Relation
/// Maps, project, filter, sort, limit
/// Basically, it can pack many PEP transforms and propagates the range of variables
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Map {
    /// The name of the output
    pub name: String,
    /// The list of expressions (SELECT items)
    pub projection: Vec<Expr>,
    /// The predicate expression, which must have Boolean type (WHERE clause).
    pub filter: Option<Expr>,
    /// The sort expressions (SORT)
    pub order_by: Vec<OrderBy>,
    /// The limit (LIMIT value)
    pub limit: Option<usize>,
    /// The schema description of the output
    pub schema: Schema,
    /// The size of the Map
    pub size: Integer,
    /// The incoming logical plan
    pub input: Rc<Relation>,
}

impl Map {
    /// Important properties such as:
    /// * schema
    /// * size
    /// Are built at construction time, while less important are lazily recomputed
    pub fn new(
        name: String,
        named_exprs: Vec<(String, Expr)>,
        filter: Option<Expr>,
        order_by: Vec<OrderBy>,
        limit: Option<usize>,
        input: Rc<Relation>,
    ) -> Self {
        assert!(Split::from_iter(named_exprs.clone()).len() == 1);
        let (schema, exprs) = Map::schema_exprs(named_exprs, &input);
        let size = Map::size(&input);
        Map {
            name,
            projection: exprs,
            filter,
            order_by,
            schema,
            size,
            limit,
            input,
        }
    }

    /// Compute the schema and exprs of the reduce
    fn schema_exprs(named_exprs: Vec<(String, Expr)>, input: &Relation) -> (Schema, Vec<Expr>) {
        let (fields, exprs) = named_exprs
            .into_iter()
            .map(|(name, expr)| {
                (
                    Field::new(name, expr.super_image(&input.data_type()).unwrap(), None),
                    expr,
                )
            })
            .unzip();
        (Schema::new(fields), exprs)
    }

    /// Compute the size of the map
    /// The size of the map has the same upper bound but no positive lower bound
    fn size(input: &Relation) -> Integer {
        input.size().max().map_or_else(
            || Integer::from_min(0),
            |&max| Integer::from_interval(0, max),
        )
    }

    pub fn builder() -> MapBuilder<WithoutInput> {
        MapBuilder::new()
    }
}

impl fmt::Display for Map {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let named_exprs: Vec<String> = self
            .projection
            .iter()
            .zip(self.schema.fields().iter())
            .map(|(expr, field)| format!("{} AS {}", expr, field.name()))
            .collect();
        let mut query = format!(
            "{} {} {} ( {} ) {} {}",
            "SELECT".to_string().bold().blue(),
            named_exprs.join(", "),
            "FROM".to_string().bold().blue(),
            self.input,
            "AS".to_string().bold().blue(),
            self.name().purple()
        );
        if let Some(cond) = &self.filter {
            query = format!("{} {} {}", query, "WHERE".to_string().bold().blue(), cond)
        }
        if !self.order_by.is_empty() {
            let order_by: Vec<String> = self
                .order_by
                .iter()
                .map(|OrderBy { expr: x, asc: b }| {
                    format!("{} {}", x, if *b { "ASC" } else { "DESC" })
                })
                .collect();
            query = format!(
                "{} {} {}",
                query,
                "ORDER BY".to_string().bold().blue(),
                order_by.join(", ")
            )
        }
        if let Some(limit) = &self.limit {
            query = format!("{} {} {}", query, "LIMIT".to_string().bold().blue(), limit)
        }
        write!(f, "{}", query)
    }
}

impl DataTyped for Map {
    fn data_type(&self) -> DataType {
        self.schema.data_type()
    }
}

impl Variant for Map {
    fn name(&self) -> &str {
        &self.name
    }

    fn schema(&self) -> &Schema {
        &self.schema
    }

    fn size(&self) -> &Integer {
        &self.size
    }

    fn inputs(&self) -> Vec<&Relation> {
        vec![&self.input]
    }
}

// Reduce Relation

/// Aggregates its input based on a set of grouping and aggregate
/// expressions (e.g. SUM).
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Reduce {
    /// The name of the output
    pub name: String,
    /// Aggregate expressions
    pub aggregate: Vec<Expr>,
    /// Grouping expressions
    pub group_by: Vec<Expr>,
    /// The schema description of the output
    pub schema: Schema,
    /// The size of the Reduce
    pub size: Integer,
    /// The incoming relation
    pub input: Rc<Relation>,
}

impl Reduce {
    /// Important properties such as:
    /// * schema
    /// * size
    /// Are built at construction time, while less important are lazily recomputed
    pub fn new(
        name: String,
        named_exprs: Vec<(String, Expr)>,
        group_by: Vec<Expr>,
        input: Rc<Relation>,
    ) -> Self {
        // assert!(Split::from_iter(named_exprs.clone()).len()==1);
        let (schema, exprs) = Reduce::schema_exprs(named_exprs, &input);
        let size = Reduce::size(&input);
        Reduce {
            name,
            aggregate: exprs,
            group_by,
            schema,
            size,
            input,
        }
    }

    /// Compute the schema and exprs of the reduce
    fn schema_exprs(named_exprs: Vec<(String, Expr)>, input: &Relation) -> (Schema, Vec<Expr>) {
        // The input schema HAS to be a Struct
        let input_data_type: Struct = input.data_type().try_into().unwrap();
        let input_columns_data_type: DataType =
            Struct::from_schema_size(input_data_type, input.size()).into();
        let (fields, exprs) = named_exprs
            .into_iter()
            .map(|(name, expr)| {
                (
                    Field::new(
                        name,
                        expr.super_image(&input_columns_data_type).unwrap(),
                        None,
                    ),
                    expr,
                )
            })
            .unzip();
        (Schema::new(fields), exprs)
    }

    /// Compute the size of the reduce
    /// The size of the reduce can be the same as its input and will be at least 0
    fn size(input: &Relation) -> Integer {
        input.size().max().map_or_else(
            || Integer::from_min(0),
            |&max| Integer::from_interval(0, max),
        )
    }

    pub fn builder() -> ReduceBuilder<WithoutInput> {
        ReduceBuilder::new()
    }
}

impl fmt::Display for Reduce {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let named_exprs: Vec<String> = self
            .aggregate
            .iter()
            .zip(self.schema.fields().iter())
            .map(|(expr, field)| {
                format!(
                    "{} {} {}",
                    expr,
                    "AS".to_string().bold().blue(),
                    field.name()
                )
            })
            .collect();
        let mut query = format!(
            "{} {} {} ( {} ) {} {}",
            "SELECT".to_string().bold().blue(),
            named_exprs.join(", "),
            "FROM".to_string().bold().blue(),
            self.input,
            "AS".to_string().bold().blue(),
            self.name().purple()
        );
        if !self.group_by.is_empty() {
            query = format!(
                "{} {} {}",
                query,
                "GROUP BY".to_string().bold().blue(),
                self.group_by.iter().map(|x| format!("{x}")).join(", ")
            )
        }
        write!(f, "{}", query)
    }
}

impl DataTyped for Reduce {
    fn data_type(&self) -> DataType {
        self.schema.data_type()
    }
}

impl Variant for Reduce {
    fn name(&self) -> &str {
        &self.name
    }

    fn schema(&self) -> &Schema {
        &self.schema
    }

    fn size(&self) -> &Integer {
        &self.size
    }

    fn inputs(&self) -> Vec<&Relation> {
        vec![&self.input]
    }
}

// Join Relation

/// Join type
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum JoinOperator {
    Inner(JoinConstraint),
    LeftOuter(JoinConstraint),
    RightOuter(JoinConstraint),
    FullOuter(JoinConstraint),
    Cross,
}

impl fmt::Display for JoinOperator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                JoinOperator::Inner(_) => "INNER",
                JoinOperator::LeftOuter(_) => "LEFT",
                JoinOperator::RightOuter(_) => "RIGHT",
                JoinOperator::FullOuter(_) => "FULL",
                JoinOperator::Cross => "CROSS",
            }
        )
    }
}

/// Join constraint
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum JoinConstraint {
    On(Expr),
    Using(Vec<Identifier>),
    Natural,
    None,
}

/// Join two relations on one or more join columns
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Join {
    /// The name of the output
    pub name: String,
    /// Join constraint
    pub operator: JoinOperator,
    /// The schema description of the output
    pub schema: Schema,
    /// The size of the Join
    pub size: Integer,
    /// Left input
    pub left: Rc<Relation>,
    /// Right input
    pub right: Rc<Relation>,
}

impl Join {
    pub fn new(
        name: String,
        left_names: Vec<String>,
        right_names: Vec<String>,
        operator: JoinOperator,
        left: Rc<Relation>,
        right: Rc<Relation>,
    ) -> Self {
        let schema = Join::schema(left_names, right_names, &left, &right);
        // The size of the join can go from 0 to
        let size = Join::size(&operator, &left, &right);
        Join {
            name,
            operator,
            schema,
            size,
            left,
            right,
        }
    }

    /// Compute the schema and exprs of the reduce
    fn schema(
        left_names: Vec<String>,
        right_names: Vec<String>,
        left: &Relation,
        right: &Relation,
    ) -> Schema {
        let left_fields = left_names
            .into_iter()
            .zip(left.schema().iter())
            .map(|(name, field)| Field::from_name_data_type(name, field.data_type()));
        let right_fields = right_names
            .into_iter()
            .zip(right.schema().iter())
            .map(|(name, field)| Field::from_name_data_type(name, field.data_type()));
        left_fields.chain(right_fields).collect()
    }

    /// Compute the size of the join
    fn size(operator: &JoinOperator, left: &Relation, right: &Relation) -> Integer {
        let left_size_max = left.size().max().cloned().unwrap_or(<i64 as Bound>::max());
        let right_size_max = right.size().max().cloned().unwrap_or(<i64 as Bound>::max());
        // TODO Review this
        match operator {
            JoinOperator::Inner(_) => Integer::from_interval(0, left_size_max.min(right_size_max)),
            JoinOperator::LeftOuter(_) => Integer::from_interval(0, left_size_max),
            JoinOperator::RightOuter(_) => Integer::from_interval(0, right_size_max),
            JoinOperator::FullOuter(_) => {
                Integer::from_interval(0, left_size_max.max(right_size_max))
            }
            JoinOperator::Cross => Integer::from_interval(0, left_size_max * right_size_max),
        }
    }

    /// Iterate over fields and input names
    pub fn field_inputs<'a>(&'a self) -> impl Iterator<Item = (Identifier, Identifier)> + 'a {
        let field_identifiers = self
            .schema()
            .iter()
            .map(|f| Identifier::from_qualified_name(self.name(), f.name()));
        let left_identifiers = self
            .left
            .schema()
            .iter()
            .map(|f| Identifier::from_qualified_name(self.left.name(), f.name()));
        let right_identifiers = self
            .right
            .schema()
            .iter()
            .map(|f| Identifier::from_qualified_name(self.right.name(), f.name()));
        field_identifiers
            .zip(left_identifiers.chain(right_identifiers))
            .map(|(f, i)| (f, i))
    }

    pub fn builder() -> JoinBuilder<WithoutInput, WithoutInput> {
        JoinBuilder::new()
    }
}

impl fmt::Display for Join {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let columns: Vec<Identifier> =
            self.left
                .schema()
                .iter()
                .map(|field| Identifier::from_qualified_name(self.left.name(), field.name()))
                .chain(
                    self.right.schema().iter().map(|field| {
                        Identifier::from_qualified_name(self.right.name(), field.name())
                    }),
                )
                .collect();
        let named_columns: Vec<String> = columns
            .iter()
            .zip(self.schema().iter())
            .map(|(column, field)| {
                format!(
                    "{} {} {}",
                    column,
                    "AS".to_string().bold().blue(),
                    field.name().purple()
                )
            })
            .collect();
        let operator = format!("{} {}", self.operator, "JOIN".to_string().bold().blue());
        let constraint = match &self.operator {
            JoinOperator::Inner(constraint)
            | JoinOperator::LeftOuter(constraint)
            | JoinOperator::RightOuter(constraint)
            | JoinOperator::FullOuter(constraint) => match constraint {
                JoinConstraint::On(expr) => format!("{} {}", "ON".to_string().bold().blue(), expr),
                JoinConstraint::Using(identifiers) => format!(
                    "{} {}",
                    "USING".to_string().bold().blue(),
                    identifiers.iter().join(", ")
                ),
                JoinConstraint::Natural => todo!(),
                JoinConstraint::None => todo!(),
            },
            JoinOperator::Cross => format!(""),
        };
        write!(
            f,
            "{} {} {} ( {} ) {} {} {} ( {} ) {} {} {}",
            "SELECT".to_string().bold().blue(),
            named_columns.join(", "),
            "FROM".to_string().bold().blue(),
            self.left,
            "AS".to_string().bold().blue(),
            self.left.name().purple(),
            operator,
            self.right,
            "AS".to_string().bold().blue(),
            self.right.name().purple(),
            constraint,
        )
    }
}

impl DataTyped for Join {
    fn data_type(&self) -> DataType {
        self.schema.data_type()
    }
}

impl Variant for Join {
    fn name(&self) -> &str {
        &self.name
    }

    fn schema(&self) -> &Schema {
        &self.schema
    }

    fn size(&self) -> &Integer {
        &self.size
    }

    fn inputs(&self) -> Vec<&Relation> {
        vec![&self.left, &self.right]
    }
}

// Set operations

/// Set op
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum SetOperator {
    Union,
    Except,
    Intersect,
}

impl fmt::Display for SetOperator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                SetOperator::Union => "UNION",
                SetOperator::Except => "EXCEPT",
                SetOperator::Intersect => "INTERSECT",
            }
        )
    }
}

/// Set quantifier
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum SetQuantifier {
    All,
    Distinct,
    None,
}

impl fmt::Display for SetQuantifier {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                SetQuantifier::All => "ALL",
                SetQuantifier::Distinct => "DISTINCT",
                SetQuantifier::None => "NONE",
            }
        )
    }
}
/// Apply a Set operation
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Set {
    /// The name of the output
    pub name: String,
    /// Set operator
    pub operator: SetOperator,
    /// Set quantifier
    pub quantifier: SetQuantifier,
    /// The schema description of the output
    pub schema: Schema,
    /// The size of the Set
    pub size: Integer,
    /// Left input
    pub left: Rc<Relation>,
    /// Right input
    pub right: Rc<Relation>,
}


impl Set {
    pub fn new(
        name: String,
        names: Vec<String>,
        operator: SetOperator,
        quantifier: SetQuantifier,
        left: Rc<Relation>,
        right: Rc<Relation>,
    ) -> Self {
        let schema = Set::schema(names, &operator, &quantifier, &left, &right);
        // The size of the join can go from 0 to
        let size = Set::size(&operator, &quantifier, &left, &right);
        Set {
            name,
            operator,
            quantifier,
            schema,
            size,
            left,
            right,
        }
    }

    /// Compute the schema and exprs of the reduce
    fn schema(
        names: Vec<String>,
        operator: &SetOperator,
        quantifier: &SetQuantifier,
        left: &Relation,
        right: &Relation,
    ) -> Schema {
        names
            .into_iter()
            .zip(left.schema().iter().zip(right.schema().iter()))
            .map(|(name, (left_field, right_field))| Field::from_name_data_type(name, match operator {
                SetOperator::Union => left_field.data_type().super_union(&right_field.data_type()).unwrap(),
                SetOperator::Except => left_field.data_type(),
                SetOperator::Intersect => left_field.data_type().super_intersection(&right_field.data_type()).unwrap(),
            }))
            .collect()
    }

    /// Compute the size of the join
    fn size(operator: &SetOperator, quantifier: &SetQuantifier, left: &Relation, right: &Relation) -> Integer {
        let left_size_max = left.size().max().cloned().unwrap_or(<i64 as Bound>::max());
        let right_size_max = right.size().max().cloned().unwrap_or(<i64 as Bound>::max());
        // TODO Improve this
        match operator {
            SetOperator::Union => Integer::from_interval(left_size_max.min(right_size_max), left_size_max + right_size_max),
            SetOperator::Except => Integer::from_interval(0, left_size_max),
            SetOperator::Intersect => Integer::from_interval(0, left_size_max.min(right_size_max)),
        }
    }

    pub fn builder() -> SetBuilder<WithoutInput, WithoutInput> {
        SetBuilder::new()
    }
}

impl fmt::Display for Set {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let operator = match self.quantifier {
            SetQuantifier::All | SetQuantifier::Distinct => format!("{} {}", self.operator, self.quantifier),
            SetQuantifier::None => format!("{}", self.operator),
        };
        write!(
            f,
            "{}\n{}\n{}",
            self.left,
            operator.bold().blue(),
            self.right,
        )
    }
}

impl DataTyped for Set {
    fn data_type(&self) -> DataType {
        self.schema.data_type()
    }
}

impl Variant for Set {
    fn name(&self) -> &str {
        &self.name
    }

    fn schema(&self) -> &Schema {
        &self.schema
    }

    fn size(&self) -> &Integer {
        &self.size
    }

    fn inputs(&self) -> Vec<&Relation> {
        vec![&self.left, &self.right]
    }
}

// The Relation

/// A Relation enum
/// Inspired by: https://calcite.apache.org/
/// similar to: https://docs.rs/sqlparser/latest/sqlparser/ast/enum.TableFactor.html
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Relation {
    Table(Table),
    Map(Map),
    Reduce(Reduce),
    Join(Join),
    Set(Set),
}

impl Relation {
    pub fn inputs(&self) -> Vec<&Relation> {
        match self {
            Relation::Map(map) => vec![map.input.as_ref()],
            Relation::Table(_) => Vec::new(),
            Relation::Reduce(reduce) => vec![reduce.input.as_ref()],
            Relation::Join(join) => vec![join.left.as_ref(), join.right.as_ref()],
            Relation::Set(set) => vec![set.left.as_ref(), set.right.as_ref()],
        }
    }

    pub fn input_schemas(&self) -> Vec<&Schema> {
        self.inputs().into_iter().map(|r| r.schema()).collect()
    }

    pub fn input_fields(&self) -> Vec<&Field> {
        self.inputs()
            .into_iter()
            .flat_map(|r| r.schema().fields())
            .collect()
    }

    // Builders

    /// Build a table
    pub fn table() -> TableBuilder<WithoutSchema> {
        Builder::table()
    }

    /// Build a map
    pub fn map() -> MapBuilder<WithoutInput> {
        Builder::map()
    }

    /// Build a reduce
    pub fn reduce() -> ReduceBuilder<WithoutInput> {
        Builder::reduce()
    }

    /// Build a reduce
    pub fn join() -> JoinBuilder<WithoutInput, WithoutInput> {
        Builder::join()
    }

    /// Build a reduce
    pub fn set() -> SetBuilder<WithoutInput, WithoutInput> {
        Builder::set()
    }
}

// Implements Acceptor, Visitor and derive an iterator and a few other Visitor driven functions

/// Implement the Acceptor trait
impl<'a> Acceptor<'a> for Relation {
    fn dependencies(&'a self) -> Dependencies<'a, Self> {
        // A relation only depends on its inputs
        self.inputs().into_iter().collect()
    }
}

impl<'a> IntoIterator for &'a Relation {
    type Item = &'a Relation;
    type IntoIter = visitor::Iter<'a, Relation>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// Visitors

/// A Visitor for the type Expr
pub trait Visitor<'a, T: Clone> {
    fn table(&self, table: &'a Table) -> T;
    fn map(&self, map: &'a Map, input: T) -> T;
    fn reduce(&self, reduce: &'a Reduce, input: T) -> T;
    fn join(&self, join: &'a Join, left: T, right: T) -> T;
    fn set(&self, set: &'a Set, left: T, right: T) -> T;
}

/// Implement a specific visitor to dispatch the dependencies more easily
impl<'a, T: Clone, V: Visitor<'a, T>> visitor::Visitor<'a, Relation, T> for V {
    fn visit(&self, acceptor: &'a Relation, dependencies: Visited<'a, Relation, T>) -> T {
        match acceptor {
            Relation::Table(table) => self.table(table),
            Relation::Map(map) => self.map(map, dependencies.get(&map.input).clone()),
            Relation::Reduce(reduce) => {
                self.reduce(reduce, dependencies.get(&reduce.input).clone())
            }
            Relation::Join(join) => self.join(
                join,
                dependencies.get(&join.left).clone(),
                dependencies.get(&join.right).clone(),
            ),
            Relation::Set(set) => self.set(
                set,
                dependencies.get(&set.left).clone(),
                dependencies.get(&set.right).clone(),
            ),
        }
    }
}

/// Implement basic Variant conversions
macro_rules! impl_conversions {
    ( $Variant:ident ) => {
        impl From<$Variant> for Relation {
            fn from(v: $Variant) -> Self {
                Relation::$Variant(v)
            }
        }

        impl TryFrom<Relation> for $Variant {
            type Error = Error;

            fn try_from(relation: Relation) -> Result<Self> {
                if let Relation::$Variant(v) = relation {
                    Ok(v)
                } else {
                    Err(Error::invalid_conversion(relation, stringify!($Variant)))
                }
            }
        }
    };
}

/// Implement Relation traits
macro_rules! impl_traits {
    ( $( $Variant:ident ),* ) => {
        // Accessors
        impl Variant for Relation {
            fn name(&self) -> &str {
                match self {
                    $(Relation::$Variant(variant) => variant.name(),)*
                }
            }

            fn schema(&self) -> &Schema {
                match self {
                    $(Relation::$Variant(variant) => variant.schema(),)*
                }
            }

            fn size(&self) -> &Integer {
                match self {
                    $(Relation::$Variant(variant) => variant.size(),)*
                }
            }

            fn inputs(&self) -> Vec<&Relation> {
                match self {
                    $(Relation::$Variant(variant) => variant.inputs(),)*
                }
            }
        }

        impl fmt::Display for Relation {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self {
                    $(Relation::$Variant(variant) => variant.fmt(f),)*
                }
            }
        }

        impl DataTyped for Relation {
            fn data_type(&self) -> DataType {
                match self {
                    $(Relation::$Variant(variant) => variant.data_type(),)*
                }
            }
        }

        impl Index<usize> for Relation {
            type Output = Field;

            fn index(&self, index: usize) -> &Self::Output {
                self.field_from_index(index).unwrap()
            }
        }

        $(
            impl Index<usize> for $Variant {
                type Output = Field;

                fn index(&self, index: usize) -> &Self::Output {
                    self.field_from_index(index).unwrap()
                }
            }
        )*

        impl Index<&Identifier> for Relation {
            type Output = Field;

            fn index(&self, identifier: &Identifier) -> &Self::Output {
                self.field_from_identifier(identifier).unwrap()
            }
        }

        $(
            impl Index<&Identifier> for $Variant {
                type Output = Field;

                fn index(&self, identifier: &Identifier) -> &Self::Output {
                    self.field_from_identifier(identifier).unwrap()
                }
            }
        )*

        $(impl_conversions!($Variant);)*
    }
}

impl_traits!(Table, Map, Reduce, Join, Set);

// A Relation builder

pub struct Builder;

impl Builder {
    pub fn table() -> TableBuilder<WithoutSchema> {
        Table::builder()
    }

    pub fn map() -> MapBuilder<WithoutInput> {
        Map::builder()
    }

    pub fn reduce() -> ReduceBuilder<WithoutInput> {
        Reduce::builder()
    }

    pub fn join() -> JoinBuilder<WithoutInput, WithoutInput> {
        Join::builder()
    }

    pub fn set() -> SetBuilder<WithoutInput, WithoutInput> {
        Set::builder()
    }
}

impl Ready<Relation> for TableBuilder<WithSchema> {
    type Error = Error;

    fn try_build(self) -> Result<Relation> {
        Ok(Ready::<Table>::try_build(self)?.into())
    }
}

impl Ready<Relation> for MapBuilder<WithInput> {
    type Error = Error;

    fn try_build(self) -> Result<Relation> {
        Ok(Ready::<Map>::try_build(self)?.into())
    }
}

impl Ready<Relation> for ReduceBuilder<WithInput> {
    type Error = Error;

    fn try_build(self) -> Result<Relation> {
        Ok(Ready::<Reduce>::try_build(self)?.into())
    }
}

impl Ready<Relation> for JoinBuilder<WithInput, WithInput> {
    type Error = Error;

    fn try_build(self) -> Result<Relation> {
        Ok(Ready::<Join>::try_build(self)?.into())
    }
}

impl Ready<Relation> for SetBuilder<WithInput, WithInput> {
    type Error = Error;

    fn try_build(self) -> Result<Relation> {
        Ok(Ready::<Set>::try_build(self)?.into())
    }
}

#[cfg(test)]
mod tests {
    use super::{schema::Schema, *};
    use crate::{builder::With, data_type::DataType};

    #[test]
    fn test_table() {
        let schema: Schema = vec![
            ("a", DataType::float()),
            ("b", DataType::float_interval(-2., 2.)),
            ("c", DataType::float()),
            ("d", DataType::float_interval(0., 1.)),
        ]
        .into_iter()
        .collect();
        let table = Table::from_schema(schema);
        println!("{}: {}", table, table.data_type());
    }

    #[test]
    fn test_index() {
        let schema: Schema = vec![
            ("a", DataType::float()),
            ("b", DataType::float_interval(-2., 2.)),
            ("c", DataType::float()),
            ("d", DataType::float_interval(0., 1.)),
            ("e", DataType::structured([("f", DataType::integer())])),
        ]
        .into_iter()
        .collect();
        let table = Table::from_schema(schema);
        println!("{}: {}", table, table.data_type());
        let field = table.field_from_identifier(&["c"].into()).unwrap();
        println!("{}: {}", field, field.data_type());
        let field = table.field_from_identifier(&["e"].into()).unwrap();
        println!("{}: {}", field, field.data_type());
    }

    #[test]
    fn test_table_builder() {
        let schema: Schema = vec![
            ("a", DataType::float()),
            ("b", DataType::float_interval(-2., 2.)),
            ("c", DataType::float()),
            ("d", DataType::float_interval(0., 1.)),
        ]
        .into_iter()
        .collect();
        let table: Relation = Relation::table().schema(schema.clone()).build();
        println!("{}: {}", table, table.data_type());
        // The builder requires the schema to be present
        let table: Relation = Relation::table().name("Name").schema(schema).build();
        println!("{}: {}", table, table.data_type());
        assert_eq!(table.name(), "Name");
    }

    #[test]
    fn test_map_builder() {
        let schema: Schema = vec![
            ("a", DataType::float()),
            ("b", DataType::float_interval(-2., 2.)),
            ("c", DataType::float()),
            ("d", DataType::float_interval(0., 1.)),
        ]
        .into_iter()
        .collect();
        let table: Relation = Relation::table().schema(schema).build();
        let map: Relation = Relation::map()
            .with(Expr::exp(Expr::col("a")))
            .input(table)
            .with(Expr::col("b") + Expr::col("d"))
            .build();
        println!("map = {}", map);
        println!("map.data_type() = {}", map.data_type());
        println!("map.schema() = {}", map.schema());
    }

    #[test]
    fn test_reduce_builder() {
        let schema: Schema = vec![
            ("a", DataType::float()),
            ("b", DataType::float_interval(-2., 2.)),
            ("c", DataType::float()),
            ("d", DataType::float_interval(0., 1.)),
        ]
        .into_iter()
        .collect();
        let table: Relation = Relation::table().schema(schema).build();
        let reduce: Relation = Relation::reduce()
            .with(Expr::sum(Expr::col("a")))
            .group_by(Expr::col("a"))
            .input(table)
            // .with(Expr::count(Expr::col("b")))
            .build();
        println!("reduce = {}", reduce);
        println!("reduce.data_type() = {}", reduce.data_type());
        println!("reduce.schema() = {}", reduce.schema());
    }

    #[test]
    fn test_join_builder() {
        let left_schema: Schema = vec![
            ("a", DataType::float()),
            ("b", DataType::float_interval(-2., 2.)),
            ("c", DataType::float_interval(0., 1.)),
            ("id", DataType::integer()),
        ]
        .into_iter()
        .collect();
        let right_schema: Schema = vec![
            ("x", DataType::float()),
            ("y", DataType::float_interval(-2., 2.)),
            ("c", DataType::float_interval(0., 1.)),
            ("id", DataType::integer()),
        ]
        .into_iter()
        .collect();
        let left: Relation = Relation::table().name("left").schema(left_schema).build();
        let right: Relation = Relation::table().name("right").schema(right_schema).build();
        let join: Relation = Relation::join()
            .left(left)
            .right(right)
            .on(Expr::eq(
                Expr::qcol("left", "id"),
                Expr::qcol("right", "id"),
            ))
            .build();
        println!("join = {}", join);
        println!("join.data_type() = {}", join.data_type());
        println!("join.schema() = {}", join.schema());
    }

    #[test]
    fn test_relation_builder() {
        namer::reset();
        let schema: Schema = vec![
            ("a", DataType::float()),
            ("b", DataType::float_interval(-2., 2.)),
            ("c", DataType::float()),
            ("d", DataType::float_interval(0., 1.)),
        ]
        .into_iter()
        .collect();
        let table: Relation = Relation::table()
            .name("table")
            .schema(schema.clone())
            .size(100)
            .build();
        println!("table = {}", table);
        println!("table[a] = {}", table[&"a".into()]);
        let map: Relation = Relation::map()
            .with(Expr::exp(Expr::col("a")))
            .input(table.clone())
            .with(Expr::col("b") + Expr::col("d"))
            .build();
        println!("map = {}", map);
        println!("map[0] = {}", map[0]);
        println!("map[table.a] = {}", map[&["table", "a"].into()]);
        let join: Relation = Relation::join()
            .cross()
            .left(table.clone())
            .right(map)
            .build();
        println!("join = {}", join);
    }

    fn build_complex_relation() -> Rc<Relation> {
        namer::reset();
        let schema: Schema = vec![
            ("a", DataType::float()),
            ("b", DataType::float_interval(-2., 2.)),
            ("c", DataType::float()),
            ("d", DataType::float_interval(0., 1.)),
        ]
        .into_iter()
        .collect();
        let table: Rc<Relation> = Rc::new(
            Relation::table()
                .name("table")
                .schema(schema.clone())
                .size(1000)
                .build(),
        );
        let map: Rc<Relation> = Rc::new(
            Relation::map()
                .name("map_1")
                .with(Expr::exp(Expr::col("a")))
                .input(table.clone())
                .with(Expr::col("b") + Expr::col("d"))
                .build(),
        );
        let join: Rc<Relation> = Rc::new(
            Relation::join()
                .name("join")
                .cross()
                .left(table.clone())
                .right(map.clone())
                .build(),
        );
        let map_2: Rc<Relation> = Rc::new(
            Relation::map()
                .name("map_2")
                .with(Expr::exp(Expr::col(join[4].name())))
                .input(join.clone())
                .with(Expr::col(join[0].name()) + Expr::col(join[1].name()))
                .build(),
        );
        let join_2: Rc<Relation> = Rc::new(
            Relation::join()
                .name("join_2")
                .cross()
                .left(join.clone())
                .right(map_2.clone())
                .build(),
        );
        join_2
    }

    #[test]
    fn test_iter() {
        let relation = build_complex_relation();
        println!("{relation}");
        for rel in relation.iter() {
            println!("relation name = {}", rel.name());
        }
    }

    #[test]
    fn test_field_inputs() {
        let relation = build_complex_relation();
        println!("{relation}");
        if let Relation::Join(join) = &(*relation) {
            for (f, i) in join.field_inputs() {
                println!("field = {f}, input = {i}");
            }
        }
    }

    #[test]
    fn test_map() {
        namer::reset();
        let schema: Schema = vec![
            ("a", DataType::float()),
            ("b", DataType::float_interval(-2., 2.)),
            ("c", DataType::float()),
            ("d", DataType::float_interval(0., 1.)),
        ]
        .into_iter()
        .collect();
        println!("Schema: {}", schema);
        let table: Rc<Relation> = Rc::new(
            Relation::table()
                .name("table")
                .schema(schema.clone())
                .build(),
        );
        println!("Table: {}", table);
        let map: Rc<Relation> = Rc::new(
            Relation::map()
                .name("map_1")
                .with(Expr::exp(Expr::col("a")))
                .input(table.clone())
                .with(Expr::col("b") + Expr::col("d"))
                .build(),
        );
        println!("MAP: {}", map);
    }

    #[test]
    fn test_map_with_reduce() {
        namer::reset();
        let schema: Schema = vec![
            ("a", DataType::float()),
            ("b", DataType::float_interval(-2., 2.)),
            ("c", DataType::float()),
            ("d", DataType::float_interval(0., 1.)),
        ]
        .into_iter()
        .collect();
        println!("Schema: {}", schema);
        let table: Rc<Relation> = Rc::new(
            Relation::table()
                .name("table")
                .schema(schema.clone())
                .size(100)
                .build(),
        );
        println!("Table: {}", table);
        let map: Rc<Relation> = Rc::new(
            Relation::map()
                .name("map_1")
                .with(("a", expr!(cos(count(d) + 1))))
                .with(("b", expr!(sin(count(d) + 1) - sum(a))))
                .input(table.clone())
                .build(),
        );
        println!("MAP: {}", map);
    }
}
