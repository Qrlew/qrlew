//! # `Relation` definition and manipulation
//!
//! This module defines the `Relation` struct
//! A `Relation` is the lazy representation of a computation that can be compiled into DP
//!

pub mod builder;
pub mod dot;
pub mod field;
pub mod rewriting;
pub mod schema;
pub mod sql;

use std::{
    cmp, error, fmt, hash,
    ops::{Deref, Index},
    result,
    sync::Arc,
};

use colored::Colorize;
use itertools::Itertools;

use crate::{
    builder::Ready,
    data_type::{
        self, function::Function, intervals::Bound, DataType, DataTyped, Integer, Struct, Value,
        Variant as _,
    },
    expr::{
        self, aggregate::Aggregate, function, AggregateColumn, Column, Expr,
        Function as ExprFunction, Identifier, Split,
    },
    hierarchy::Hierarchy,
    namer,
    visitor::{self, Acceptor, Dependencies, Visited},
};
pub use builder::{
    JoinBuilder, MapBuilder, ReduceBuilder, SetBuilder, TableBuilder, ValuesBuilder, WithInput,
    WithSchema, WithoutInput, WithoutSchema,
};
pub use field::{Constraint, Field};
pub use schema::Schema;

pub const LEFT_INPUT_NAME: &str = "_LEFT_";
pub const RIGHT_INPUT_NAME: &str = "_RIGHT_";

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
    pub fn other(value: impl fmt::Display) -> Error {
        Error::Other(format!("Error with {}", value))
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
{
    /// Return the name
    fn name(&self) -> &str;
    /// Return the Schema
    fn schema(&self) -> &Schema;
    /// Return the size bounds
    fn size(&self) -> &Integer;
    /// Return the inputs
    fn inputs(&self) -> Vec<&Relation>;
    /// Return the input hierarchy
    fn input_hierarchy(&self) -> Hierarchy<&Relation> {
        //TODO Add inputs recursively
        self.inputs().into_iter().map(|r| ([r.name()], r)).collect()
    }
    /// Return the fields
    fn fields(&self) -> Vec<&Field> {
        self.schema().iter().collect()
    }
    /// Access a field of the Relation by index
    fn field_from_index(&self, index: usize) -> Result<&Field> {
        self.schema().field_from_index(index)
    }
}

// Table Relation

/// Produces rows from a table provider by reference or from the context
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Table {
    /// The name of the table
    name: String,
    /// The path to the actual table
    path: Identifier,
    /// The schema description of the output
    schema: Schema,
    /// The size of the table
    size: Integer,
}

impl Table {
    /// Main constructor
    pub fn new(name: String, path: Identifier, schema: Schema, size: Integer) -> Self {
        Table {
            name,
            path,
            schema,
            size,
        }
    }

    /// From schema
    pub fn from_schema<S: Into<Schema>>(schema: S) -> Table {
        let path: String = namer::new_name("table");
        Table::new(
            path.clone(),
            path.into(),
            schema.into(),
            Integer::from_min(0),
        )
    }

    /// Return the path
    pub fn path(&self) -> &Identifier {
        &self.path
    }

    /// A builder
    pub fn builder() -> TableBuilder<WithoutSchema> {
        TableBuilder::new()
    }
}

impl fmt::Display for Table {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.path)
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
        vec![]
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
/// Basically, it can pack many PUP transforms and propagates the range of variables
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Map {
    /// The name of the output
    name: String,
    /// The list of expressions (SELECT items)
    projection: Vec<Expr>,
    /// The predicate expression, which must have Boolean type (WHERE clause). It is applied on the input columns.
    filter: Option<Expr>,
    /// The sort expressions (SORT)
    order_by: Vec<OrderBy>,
    /// The limit (LIMIT value)
    limit: Option<usize>,
    /// The schema description of the output
    schema: Schema,
    /// The size of the Map
    size: Integer,
    /// The incoming logical plan
    input: Arc<Relation>,
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
        input: Arc<Relation>,
    ) -> Self {
        assert!(Split::from_iter(named_exprs.clone()).len() == 1);
        let (schema, exprs) = Map::schema_exprs(named_exprs, &filter, &input);
        let size = Map::size(&input, limit);
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

    /// Compute the schema and exprs of the map
    fn schema_exprs(
        named_exprs: Vec<(String, Expr)>,
        filter: &Option<Expr>,
        input: &Relation,
    ) -> (Schema, Vec<Expr>) {
        let mut input_data_type = input.data_type();
        if let Some(f) = filter {
            input_data_type = input_data_type.filter(f)
        };
        let (fields, exprs) = named_exprs
            .into_iter()
            .map(|(name, expr)| {
                (
                    Field::new(
                        name,
                        expr.super_image(&input_data_type).unwrap(),
                        if let Expr::Column(c) = expr.clone() {
                            input.schema()[c.last().unwrap()].constraint()
                        } else {
                            None
                        },
                    ),
                    expr,
                )
            })
            .unzip();
        (Schema::new(fields), exprs)
    }

    /// Compute the size of the map
    /// The size of the map has the same upper bound but no positive lower bound
    /// In the case of a `limit` the uuper bound, the minimum between
    /// the limit and the input max size is taken.
    fn size(input: &Relation, limit: Option<usize>) -> Integer {
        input.size().max().map_or_else(
            || Integer::from_min(0),
            |&max| {
                Integer::from_interval(
                    0,
                    match limit {
                        Some(limit_val) => std::cmp::min(limit_val as i64, max),
                        None => max,
                    },
                )
            },
        )
    }
    /// Get projections
    pub fn projection(&self) -> &[Expr] {
        &self.projection
    }
    /// Get filter
    pub fn filter(&self) -> &Option<Expr> {
        &self.filter
    }
    /// Get order_by
    pub fn order_by(&self) -> &[OrderBy] {
        &self.order_by
    }
    /// Get limit
    pub fn limit(&self) -> &Option<usize> {
        &self.limit
    }
    /// Get the input
    pub fn input(&self) -> &Relation {
        &self.input
    }
    /// Get names and expressions
    pub fn field_exprs(&self) -> Vec<(&Field, &Expr)> {
        self.schema.iter().zip(self.projection.iter()).collect()
    }
    /// Get names and expressions
    pub fn named_exprs(&self) -> Vec<(&str, &Expr)> {
        self.schema
            .iter()
            .map(|f| f.name())
            .zip(self.projection.iter())
            .collect()
    }
    /// Return a new builder
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
            "{} {} {} ( {} )",
            "SELECT".to_string().bold().blue(),
            named_exprs.join(", "),
            "FROM".to_string().bold().blue(),
            self.input,
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
    name: String,
    /// Aggregate expressions
    aggregate: Vec<AggregateColumn>,
    /// Grouping expressions
    group_by: Vec<Column>,
    /// The schema description of the output
    schema: Schema,
    /// The size of the Reduce
    size: Integer,
    /// The incoming relation
    input: Arc<Relation>,
}

impl Reduce {
    /// Important properties such as:
    /// * schema
    /// * size
    /// Are built at construction time, while less important are lazily recomputed
    pub fn new(
        name: String,
        named_aggregate: Vec<(String, AggregateColumn)>,
        group_by: Vec<Column>,
        input: Arc<Relation>,
    ) -> Self {
        // assert!(Split::from_iter(named_exprs.clone()).len()==1);
        let (schema, aggregate) = Reduce::schema_aggregate(named_aggregate, &input);
        let size = Reduce::size(&input);
        Reduce {
            name,
            aggregate,
            group_by,
            schema,
            size,
            input,
        }
    }

    /// Compute the schema and exprs of the reduce
    fn schema_aggregate(
        named_aggregate_columns: Vec<(String, AggregateColumn)>,
        input: &Relation,
    ) -> (Schema, Vec<AggregateColumn>) {
        // The input schema HAS to be a Struct
        let input_data_type: Struct = input.data_type().try_into().unwrap();
        let input_columns_data_type: DataType =
            Struct::from_schema_size(input_data_type, input.size()).into();
        let has_one_group = named_aggregate_columns
            .iter()
            .filter(|(_, agg)| matches!(agg.aggregate(), &Aggregate::First))
            .count()
            == 1;
        let (fields, aggregates) = named_aggregate_columns
            .into_iter()
            .map(|(name, aggregate_column)| {
                (
                    Field::new(
                        name,
                        aggregate_column
                            .super_image(&input_columns_data_type)
                            .unwrap(),
                        if has_one_group && aggregate_column.aggregate() == &Aggregate::First {
                            Some(Constraint::Unique)
                        } else {
                            None
                        },
                    ),
                    aggregate_column,
                )
            })
            .unzip();
        (Schema::new(fields), aggregates)
    }
    /// Compute the size of the reduce
    /// The size of the reduce can be the same as its input and will be at least 0
    fn size(input: &Relation) -> Integer {
        input.size().max().map_or_else(
            || Integer::from_min(0),
            |&max| Integer::from_interval(0, max),
        )
    }
    /// Get aggregate exprs
    pub fn aggregate(&self) -> &[AggregateColumn] {
        &self.aggregate
    }
    /// Get group_by
    pub fn group_by(&self) -> &[Column] {
        &self.group_by
    }
    /// Get the input
    pub fn input(&self) -> &Relation {
        &self.input
    }
    /// Get names and expressions
    pub fn field_aggregates(&self) -> Vec<(&Field, &AggregateColumn)> {
        self.schema.iter().zip(self.aggregate.iter()).collect()
    }
    /// Get names and expressions
    pub fn named_aggregates(&self) -> Vec<(&str, &AggregateColumn)> {
        self.schema
            .iter()
            .map(|f| f.name())
            .zip(self.aggregate.iter())
            .collect()
    }
    /// Return a new builder
    pub fn builder() -> ReduceBuilder<WithoutInput> {
        ReduceBuilder::new()
    }
    /// Get group_by_names
    pub fn group_by_names(&self) -> Vec<&str> {
        self.group_by
            .iter()
            .filter_map(|col| col.last().ok())
            .collect()
    }
}

impl fmt::Display for Reduce {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let named_exprs: Vec<String> = self
            .aggregate
            .iter()
            .zip(self.schema.fields().iter())
            .map(|(aggregate, field)| {
                format!(
                    "{} {} {}",
                    aggregate.deref(),
                    "AS".to_string().bold().blue(),
                    field.name()
                )
            })
            .collect();
        let mut query = format!(
            "{} {} {} ( {} )",
            "SELECT".to_string().bold().blue(),
            named_exprs.join(", "),
            "FROM".to_string().bold().blue(),
            self.input,
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

impl JoinOperator {
    /// Rename all exprs in the operator
    pub fn rename<'a>(&'a self, columns: &'a Hierarchy<Identifier>) -> Self {
        match self {
            JoinOperator::Inner(c) => JoinOperator::Inner(c.rename(columns)),
            JoinOperator::LeftOuter(c) => JoinOperator::LeftOuter(c.rename(columns)),
            JoinOperator::RightOuter(c) => JoinOperator::RightOuter(c.rename(columns)),
            JoinOperator::FullOuter(c) => JoinOperator::FullOuter(c.rename(columns)),
            JoinOperator::Cross => JoinOperator::Cross,
        }
    }

    /// Returns the schema of `left` and `right` relations filtered by
    /// the expression equivalent to the Join operator
    fn filtered_schemas(&self, left: &Relation, right: &Relation) -> (Schema, Schema) {
        let dt = DataType::structured([
            (Join::left_name(), left.schema().data_type()),
            (Join::right_name(), right.schema().data_type()),
        ]);

        let dt = dt.filter_by_join_operator(self);
        let left_schema: Schema = dt[Join::left_name()].clone().into();
        let left_schema: Schema = left_schema
            .iter()
            .map(|f| {
                Field::new(
                    f.name().to_string(),
                    f.data_type(),
                    left.schema()[f.name()].constraint(),
                )
            })
            .collect();
        let right_schema: Schema = dt[Join::right_name()].clone().into();
        let right_schema: Schema = right_schema
            .iter()
            .map(|f| {
                Field::new(
                    f.name().to_string(),
                    f.data_type(),
                    right.schema()[f.name()].constraint(),
                )
            })
            .collect();
        (left_schema, right_schema)
    }

    fn has_unique_constraint(&self, left_schema: &Schema, right_schema: &Schema) -> (bool, bool) {
        match self {
            JoinOperator::Inner(c)
            | JoinOperator::LeftOuter(c)
            | JoinOperator::RightOuter(c)
            | JoinOperator::FullOuter(c) => c.has_unique_constraint(left_schema, right_schema),
            JoinOperator::Cross => (false, false),
        }
    }
}

impl DataType {
    /// Returns a new `DataType` clone of the current `DataType`
    /// filtered by the `Expr` equivalent to the `JoinOperator`
    fn filter_by_join_operator(&self, join_op: &JoinOperator) -> DataType {
        match join_op {
            JoinOperator::Inner(c) => {
                let x = Expr::from((c, self));
                self.filter(&x)
            }
            JoinOperator::LeftOuter(c) => {
                let x = Expr::from((c, self));
                let filtered_data_type = self.filter(&x);
                DataType::structured([
                    (Join::left_name(), self[Join::left_name()].clone()),
                    (
                        Join::right_name(),
                        filtered_data_type[Join::right_name()].clone(),
                    ),
                ])
            }
            JoinOperator::RightOuter(c) => {
                let x = Expr::from((c, self));
                let filtered_data_type = self.filter(&x);
                DataType::structured([
                    (
                        Join::left_name(),
                        filtered_data_type[Join::left_name()].clone(),
                    ),
                    (Join::right_name(), self[Join::right_name()].clone()),
                ])
            }
            JoinOperator::FullOuter(_) | JoinOperator::Cross => self.clone(),
        }
    }
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

impl JoinConstraint {
    /// Rename all exprs in the constraint
    pub fn rename<'a>(&'a self, columns: &'a Hierarchy<Identifier>) -> Self {
        match self {
            JoinConstraint::On(expr) => JoinConstraint::On(expr.rename(columns)),
            JoinConstraint::Using(identifiers) => JoinConstraint::Using(
                identifiers
                    .iter()
                    .map(|i| columns.get(i).unwrap().clone())
                    .collect(),
            ),
            JoinConstraint::Natural => JoinConstraint::Natural,
            JoinConstraint::None => JoinConstraint::None,
        }
    }

    /// Returns a tuple of bool where
    /// the first (resp. second) item is `true` if
    /// - the current `JoinConstraint` is an `On`
    ///     - the wrapped expression if of type `(Column(_) = Column(_))` AND
    ///     - the field of column belonging to the left (resp. right) relation has a `Unique` or `PrimaryKey` constraint
    /// - the current `JoinConstraint`` is a `Using`: the field of column belonging to the left (resp. right) relation has a `Unique` or `PrimaryKey` constraint
    pub fn has_unique_constraint(
        &self,
        left_schema: &Schema,
        right_schema: &Schema,
    ) -> (bool, bool) {
        match self {
            JoinConstraint::On(x) => match x {
                Expr::Function(f) if f.function() == function::Function::Eq => {
                    let fields_with_unique_or_primary_key_constraint = Hierarchy::from_iter(
                        left_schema
                            .iter()
                            .map(|f| {
                                (
                                    vec![Join::left_name(), f.name()],
                                    f.has_unique_or_primary_key_constraint(),
                                )
                            })
                            .chain(right_schema.iter().map(|f| {
                                (
                                    vec![Join::right_name(), f.name()],
                                    f.has_unique_or_primary_key_constraint(),
                                )
                            })),
                    );
                    let mut left = false;
                    let mut right = false;
                    if let Expr::Column(c) = &f.arguments()[0] {
                        if fields_with_unique_or_primary_key_constraint
                            .get_key_value(c)
                            .unwrap()
                            .0[0]
                            == Join::left_name()
                        {
                            left = fields_with_unique_or_primary_key_constraint[c.as_slice()]
                        } else {
                            right = fields_with_unique_or_primary_key_constraint[c.as_slice()]
                        }
                    }
                    if let Expr::Column(c) = &f.arguments()[1] {
                        if fields_with_unique_or_primary_key_constraint
                            .get_key_value(c)
                            .unwrap()
                            .0[0]
                            == Join::left_name()
                        {
                            left = fields_with_unique_or_primary_key_constraint[c.as_slice()]
                        } else {
                            right = fields_with_unique_or_primary_key_constraint[c.as_slice()]
                        }
                    }
                    (left, right)
                }
                _ => (false, false),
            },
            JoinConstraint::Using(v) if v.len() == 1 => {
                let left = left_schema
                    .field(v[0].last().unwrap())
                    .map(|f| f.has_unique_or_primary_key_constraint())
                    .unwrap_or(false);
                let right = right_schema
                    .field(v[0].last().unwrap())
                    .map(|f| f.has_unique_or_primary_key_constraint())
                    .unwrap_or(false);
                (left, right)
            }
            _ => (false, false),
        }
    }
}

impl From<(&JoinConstraint, &DataType)> for Expr {
    fn from(value: (&JoinConstraint, &DataType)) -> Self {
        let (constraint, dt) = value;
        match constraint {
            JoinConstraint::On(x) => x.clone(),
            JoinConstraint::Using(x) => x.iter().fold(Expr::val(true), |f, v| {
                Expr::and(
                    f,
                    Expr::eq(
                        Expr::qcol(Join::left_name(), v.head().unwrap()),
                        Expr::qcol(Join::right_name(), v.head().unwrap()),
                    ),
                )
            }),
            JoinConstraint::Natural => {
                let h = dt[Join::right_name()].hierarchy();
                let v = dt[Join::left_name()]
                    .hierarchy()
                    .into_iter()
                    .filter_map(|(s, _)| h.get(&s).map(|_| Identifier::from(s)))
                    .collect::<Vec<_>>();
                (&JoinConstraint::Using(v), dt).into()
            }
            JoinConstraint::None => Expr::val(true),
        }
    }
}

/// Join two relations on one or more join columns
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Join {
    /// The name of the output
    name: String,
    /// Join constraint
    operator: JoinOperator,
    /// The schema description of the output
    schema: Schema,
    /// The size of the Join
    size: Integer,
    /// Left input
    left: Arc<Relation>,
    /// Right input
    right: Arc<Relation>,
}

impl Join {
    pub fn new(
        name: String,
        left_names: Vec<String>,
        right_names: Vec<String>,
        operator: JoinOperator,
        left: Arc<Relation>,
        right: Arc<Relation>,
    ) -> Self {
        let schema = Join::schema(left_names, right_names, &left, &right, &operator);
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
    pub fn left_name() -> &'static str {
        LEFT_INPUT_NAME
    }
    pub fn right_name() -> &'static str {
        RIGHT_INPUT_NAME
    }
    /// Compute the schema and exprs of the join
    fn schema(
        left_names: Vec<String>,
        right_names: Vec<String>,
        left: &Relation,
        right: &Relation,
        operator: &JoinOperator,
    ) -> Schema {
        let (left_schema, right_schema) = operator.filtered_schemas(left, right);
        let (left_is_unique, right_is_unique) =
            operator.has_unique_constraint(left.schema(), right.schema());
        let transform_datatype_in_optional_left: bool = match operator {
            JoinOperator::LeftOuter(_) | JoinOperator::Inner(_) | JoinOperator::Cross => false,
            _ => true,
        };
        let transform_datatype_in_optional_right = match operator {
            JoinOperator::RightOuter(_) | JoinOperator::Inner(_) | JoinOperator::Cross => false,
            _ => true,
        };
        let left_fields = left_names
            .into_iter()
            .zip(left_schema.iter())
            .map(|(name, field)| {
                Field::new(
                    name,
                    if transform_datatype_in_optional_left {
                        DataType::optional(field.data_type())
                    } else {
                        field.data_type()
                    },
                    if right_is_unique {
                        field.constraint()
                    } else {
                        None
                    },
                )
            });
        let right_fields = right_names
            .into_iter()
            .zip(right_schema.iter())
            .map(|(name, field)| {
                Field::new(
                    name,
                    if transform_datatype_in_optional_right {
                        DataType::optional(field.data_type())
                    } else {
                        field.data_type()
                    },
                    if left_is_unique {
                        field.constraint()
                    } else {
                        None
                    },
                )
            });
        left_fields.chain(right_fields).collect()
    }

    /// Compute the size of the join
    fn size(operator: &JoinOperator, left: &Relation, right: &Relation) -> Integer {
        let left_size_max = left.size().max().cloned().unwrap_or(<i64 as Bound>::max());
        let right_size_max = right.size().max().cloned().unwrap_or(<i64 as Bound>::max());
        let (left, right) = operator.has_unique_constraint(left.schema(), right.schema());
        let max = if left || right {
            left_size_max.max(right_size_max)
        } else {
            left_size_max.saturating_mul(right_size_max)
        };
        Integer::from_interval(0, max)
    }

    /// Iterate over fields and input names
    pub fn field_inputs<'a>(&'a self) -> impl Iterator<Item = (String, Identifier)> + 'a {
        let field_identifiers = self.schema().iter().map(|f| f.name().to_string());
        let left_identifiers = self
            .left
            .schema()
            .iter()
            .map(|f| Identifier::from_qualified_name(Self::left_name(), f.name()));
        let right_identifiers = self
            .right
            .schema()
            .iter()
            .map(|f| Identifier::from_qualified_name(Self::right_name(), f.name()));
        field_identifiers
            .zip(left_identifiers.chain(right_identifiers))
            .map(|(f, i)| (f, i))
    }
    /// Get the hierarchy of names
    pub fn names(&self) -> Hierarchy<String> {
        Hierarchy::from_iter(self.field_inputs().map(|(n, i)| (i, n)))
    }
    /// Get join operator
    pub fn operator(&self) -> &JoinOperator {
        &self.operator
    }
    /// Get left input
    pub fn left(&self) -> &Relation {
        &self.left
    }
    /// Get right input
    pub fn right(&self) -> &Relation {
        &self.right
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
            Join::left_name().purple(),
            operator,
            self.right,
            "AS".to_string().bold().blue(),
            Join::right_name().purple(),
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
    ByName,
    AllByName,
    DistinctByName,
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
                SetQuantifier::ByName => "BY NAME",
                SetQuantifier::AllByName => "ALL BY NAME",
                SetQuantifier::DistinctByName => "DISTINCT BY NAME",
            }
        )
    }
}
/// Apply a Set operation
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Set {
    /// The name of the output
    name: String,
    /// Set operator
    operator: SetOperator,
    /// Set quantifier
    quantifier: SetQuantifier,
    /// The schema description of the output
    schema: Schema,
    /// The size of the Set
    size: Integer,
    /// Left input
    left: Arc<Relation>,
    /// Right input
    right: Arc<Relation>,
}

impl Set {
    pub fn new(
        name: String,
        names: Vec<String>,
        operator: SetOperator,
        quantifier: SetQuantifier,
        left: Arc<Relation>,
        right: Arc<Relation>,
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

    /// Compute the schema and exprs of the set
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
            .map(|(name, (left_field, right_field))| {
                Field::from_name_data_type(
                    name,
                    match operator {
                        SetOperator::Union => left_field
                            .data_type()
                            .super_union(&right_field.data_type())
                            .unwrap(),
                        SetOperator::Except => left_field.data_type(),
                        SetOperator::Intersect => left_field
                            .data_type()
                            .super_intersection(&right_field.data_type())
                            .unwrap(),
                    },
                )
            })
            .collect()
    }

    /// Compute the size of the join
    fn size(
        operator: &SetOperator,
        quantifier: &SetQuantifier,
        left: &Relation,
        right: &Relation,
    ) -> Integer {
        let left_size_max = left.size().max().cloned().unwrap_or(<i64 as Bound>::max());
        let right_size_max = right.size().max().cloned().unwrap_or(<i64 as Bound>::max());
        // TODO Improve this
        match operator {
            SetOperator::Union => Integer::from_interval(
                left_size_max.min(right_size_max),
                left_size_max + right_size_max,
            ),
            SetOperator::Except => Integer::from_interval(0, left_size_max),
            SetOperator::Intersect => Integer::from_interval(0, left_size_max.min(right_size_max)),
        }
    }
    /// Get set operator
    pub fn operator(&self) -> &SetOperator {
        &self.operator
    }
    /// Get set quantifier
    pub fn quantifier(&self) -> &SetQuantifier {
        &self.quantifier
    }
    /// Get left input
    pub fn left(&self) -> &Relation {
        &self.left
    }
    /// Get right input
    pub fn right(&self) -> &Relation {
        &self.right
    }
    pub fn builder() -> SetBuilder<WithoutInput, WithoutInput> {
        SetBuilder::new()
    }
}

impl fmt::Display for Set {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let operator = match self.quantifier {
            SetQuantifier::All
            | SetQuantifier::Distinct
            | SetQuantifier::ByName
            | SetQuantifier::AllByName
            | SetQuantifier::DistinctByName => {
                format!("{} {}", self.operator, self.quantifier)
            }
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
/// Values
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Values {
    /// The name of the output
    name: String,
    /// The values
    values: Vec<Value>,
    /// The schema description of the output
    schema: Schema,
    /// The size of the Set
    size: Integer,
}

impl Values {
    pub fn new(name: String, values: Vec<Value>) -> Self {
        let schema = Values::schema(&name, &values);
        let size = Integer::from(values.len() as i64);
        Values {
            name,
            values,
            schema,
            size: size.into(),
        }
    }

    /// Compute the schema of the Values
    fn schema(name: &str, values: &Vec<Value>) -> Schema {
        let list: data_type::List = Value::list(values.iter().cloned())
            .data_type()
            .try_into()
            .unwrap();
        Schema::from_field((name.to_string(), list.data_type().clone()))
    }

    pub fn builder() -> ValuesBuilder {
        ValuesBuilder::new()
    }
}

impl fmt::Display for Values {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "[ {} ]",
            self.values.iter().map(|v| v.to_string()).join(", ")
        )
    }
}

impl DataTyped for Values {
    fn data_type(&self) -> DataType {
        self.schema.data_type()
    }
}

impl Variant for Values {
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
        vec![]
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
    Values(Values),
}

impl Relation {
    pub fn inputs(&self) -> Vec<&Relation> {
        match self {
            Relation::Map(map) => vec![map.input.as_ref()],
            Relation::Table(_) => vec![],
            Relation::Reduce(reduce) => vec![reduce.input.as_ref()],
            Relation::Join(join) => vec![join.left.as_ref(), join.right.as_ref()],
            Relation::Set(set) => vec![set.left.as_ref(), set.right.as_ref()],
            Relation::Values(_) => vec![],
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

    /// Build a values
    pub fn values() -> ValuesBuilder {
        Builder::values()
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
    fn values(&self, values: &'a Values) -> T;
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
            Relation::Values(values) => self.values(values),
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

        $(impl_conversions!($Variant);)*
    }
}

impl_traits!(Table, Map, Reduce, Join, Set, Values);

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

    pub fn values() -> ValuesBuilder {
        Values::builder()
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

impl Ready<Relation> for ValuesBuilder {
    type Error = Error;

    fn try_build(self) -> Result<Relation> {
        Ok(Ready::<Values>::try_build(self)?.into())
    }
}

#[cfg(test)]
mod tests {
    use super::{schema::Schema, *};
    use crate::{builder::With, data_type::DataType, display::Dot};

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
    fn test_values() {
        let values = Values::new(
            "values".to_string(),
            vec![Value::from(1.0), Value::from(2.0), Value::from(10)],
        );
        assert_eq!(
            values.data_type(),
            DataType::structured(vec![(
                "values",
                DataType::from(data_type::Float::from_values(vec![1., 2., 10.]))
            )])
        );
        assert_eq!(values.size(), &Integer::from(3 as i64));
        println!("{}", values);
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
        let table: Relation = Relation::table().schema(schema).name("my_table").build();
        let map: Relation = Relation::map()
            .with(("my_a", Expr::exp(Expr::col("a"))))
            .input(table)
            .with(Expr::col("b") + Expr::col("d"))
            .filter(Expr::gt(Expr::col("a"), expr!(0)))
            .build();
        println!("map = {}", map);
        println!("map.data_type() = {:?}", map.data_type());
        map.display_dot().unwrap();
        assert_eq!(
            map.schema().field_from_index(0).unwrap().data_type(),
            DataType::float_min(1.)
        );
        assert_eq!(
            map.schema().field_from_index(1).unwrap().data_type(),
            DataType::float_interval(-2., 3.)
        );
    }

    #[test]
    fn test_reduce_builder() {
        let schema: Schema = vec![
            ("a", DataType::integer_interval(0, 10)),
            ("b", DataType::float_interval(0., 1.)),
            ("c", DataType::float()),
            ("d", DataType::float_interval(0., 1.)),
        ]
        .into_iter()
        .collect();
        let table: Relation = Relation::table().schema(schema).size(100).build();

        let reduce: Relation = Relation::reduce()
            .with(("my_sum", Expr::sum(Expr::col("b"))))
            .with(("a", Expr::first(Expr::col("a"))))
            .group_by(Expr::col("a"))
            .input(table.clone())
            .build();
        assert_eq!(
            reduce.data_type(),
            DataType::structured([
                ("my_sum", DataType::float_interval(0., 100.)),
                ("a", DataType::integer_interval(0, 10)),
            ])
        );

        let reduce: Relation = Relation::reduce()
            .with(("my_sum", Expr::sum(Expr::col("b"))))
            .with(("my_a", Expr::first(expr!(3 * a))))
            .group_by(expr!(3 * a))
            .input(table)
            .build();
        assert_eq!(
            reduce.data_type(),
            DataType::structured([
                ("my_sum", DataType::float_interval(0., 100.)),
                ("my_a", DataType::integer_interval(0, 30)),
            ])
        );
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
        let join: Join = Relation::join()
            .inner()
            .on(Expr::eq(
                Expr::qcol(LEFT_INPUT_NAME, "id"),
                Expr::qcol(RIGHT_INPUT_NAME, "id"),
            ))
            //.left_names(vec!["l_a", "l_b", "l_c", "l_id"])
            .left(left)
            .right(right)
            .build();
        println!("join.names() = {}", join.names());
        let relation = Relation::from(join);
        println!("relation = {}", relation);
        println!("relation.data_type() = {}", relation.data_type());
        println!("relation.schema() = {}", relation.schema());
    }

    #[test]
    fn test_auto_join_builder() {
        let schema: Schema = vec![
            ("a", DataType::float()),
            ("b", DataType::float_interval(-2., 2.)),
            ("c", DataType::float_interval(0., 1.)),
            ("id", DataType::integer()),
        ]
        .into_iter()
        .collect();
        let table: Relation = Relation::table().name("my_table").schema(schema).build();
        let join: Join = Relation::join()
            .left(table.clone())
            .right(table.clone())
            .left_outer()
            .on(Expr::eq(
                Expr::qcol(LEFT_INPUT_NAME, "id"),
                Expr::qcol(RIGHT_INPUT_NAME, "id"),
            ))
            .build();
        println!("join.names() = {}", join.names());
        let relation = Relation::from(join);
        println!("relation = {}", relation);
        println!("relation.data_type() = {}", relation.data_type());
        println!("relation.schema() = {}", relation.schema());
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
        let map: Relation = Relation::map()
            .with(Expr::exp(Expr::col("a")))
            .input(table.clone())
            .with(Expr::col("b") + Expr::col("d"))
            .build();
        println!("map = {}", map);
        println!("map[0] = {}", map[0]);
        let join: Relation = Relation::join()
            .cross()
            .left(table.clone())
            .right(map)
            .build();
        println!("join = {}", join);
    }

    fn build_complex_relation() -> Arc<Relation> {
        namer::reset();
        let schema: Schema = vec![
            ("a", DataType::float()),
            ("b", DataType::float_interval(-2., 2.)),
            ("c", DataType::float()),
            ("d", DataType::float_interval(0., 1.)),
        ]
        .into_iter()
        .collect();
        let table: Arc<Relation> = Arc::new(
            Relation::table()
                .name("table")
                .schema(schema.clone())
                .size(1000)
                .build(),
        );
        let map: Arc<Relation> = Arc::new(
            Relation::map()
                .name("map_1")
                .with(Expr::exp(Expr::col("a")))
                .input(table.clone())
                .with(Expr::col("b") + Expr::col("d"))
                .build(),
        );
        let join: Arc<Relation> = Arc::new(
            Relation::join()
                .name("join")
                .cross()
                .left(table.clone())
                .right(map.clone())
                .build(),
        );
        let map_2: Arc<Relation> = Arc::new(
            Relation::map()
                .name("map_2")
                .with(Expr::exp(Expr::col(join[4].name())))
                .input(join.clone())
                .with(Expr::col(join[0].name()) + Expr::col(join[1].name()))
                .build(),
        );
        let join_2: Arc<Relation> = Arc::new(
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
            ("c", DataType::float_values([1., 3.4])),
            ("d", DataType::float_interval(0., 1.)),
        ]
        .into_iter()
        .collect();
        println!("Schema: {}", schema);
        let table: Arc<Relation> = Arc::new(
            Relation::table()
                .name("table")
                .schema(schema.clone())
                .build(),
        );
        println!("Table: {}", table);
        let map: Arc<Relation> = Arc::new(
            Relation::map()
                .name("map_1")
                .with(("A", Expr::exp(Expr::col("a"))))
                .input(table.clone())
                .with(("B", Expr::col("b") + Expr::col("d")))
                .with(("C", expr!(gt(c, 5))))
                .build(),
        );
        println!("MAP: {}", map);
        assert_eq!(
            map.data_type(),
            DataType::structured(vec![
                ("A", DataType::float_min(0.)),
                ("B", DataType::float_interval(-2., 3.)),
                ("C", DataType::boolean_value(false)),
            ])
        );
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
        let table: Arc<Relation> = Arc::new(
            Relation::table()
                .name("table")
                .schema(schema.clone())
                .size(100)
                .build(),
        );
        println!("Table: {}", table);
        let map: Arc<Relation> = Arc::new(
            Relation::map()
                .name("map_1")
                .with(("a", expr!(cos(count(d) + 1))))
                .with(("b", expr!(sin(count(d) + 1) - sum(a))))
                .input(table.clone())
                .build(),
        );
        println!("MAP: {}", map);
    }

    #[test]
    fn test_from_join_constraint() {
        let table1 = DataType::structured([
            ("a", DataType::float_interval(-10., 10.)),
            ("b", DataType::integer_interval(-8, 34)),
            ("c", DataType::float_interval(0., 50.)),
        ]);
        let table2 = DataType::structured([
            ("a", DataType::float_interval(0., 20.)),
            ("b", DataType::integer_interval(-1, 14)),
            ("d", DataType::integer_interval(-10, 20)),
        ]);
        let data_type =
            DataType::structured([(Join::left_name(), table1), (Join::right_name(), table2)]);

        // ON
        let x = Expr::eq(Expr::qcol(Join::left_name(), "a"), Expr::col("d"));
        let jc_x = Expr::from((&JoinConstraint::On(x.clone()), &data_type));
        assert_eq!(jc_x, x);

        // USING
        let v = vec![Identifier::from_name("a"), Identifier::from_name("b")];
        let true_x = Expr::and(
            Expr::and(
                Expr::val(true),
                Expr::eq(
                    Expr::qcol(Join::left_name(), "a"),
                    Expr::qcol(Join::right_name(), "a"),
                ),
            ),
            Expr::eq(
                Expr::qcol(Join::left_name(), "b"),
                Expr::qcol(Join::right_name(), "b"),
            ),
        );
        let jc_x = Expr::from((&JoinConstraint::Using(v), &data_type));
        assert_eq!(jc_x, true_x);

        // NATURAL
        let true_x = Expr::and(
            Expr::and(
                Expr::val(true),
                Expr::eq(
                    Expr::qcol(Join::left_name(), "a"),
                    Expr::qcol(Join::right_name(), "a"),
                ),
            ),
            Expr::eq(
                Expr::qcol(Join::left_name(), "b"),
                Expr::qcol(Join::right_name(), "b"),
            ),
        );
        let jc_x = Expr::from((&JoinConstraint::Natural, &data_type));
        assert_eq!(jc_x, true_x);

        // NONE
        let jc_x = Expr::from((&JoinConstraint::None, &data_type));
        assert_eq!(jc_x, Expr::val(true));
    }

    #[test]
    fn test_filter_data_type_inner_join() {
        let table1 = DataType::structured([
            ("a", DataType::float_interval(-3., 3.)),
            ("b", DataType::integer_interval(-1, 3)),
            ("c", DataType::float_interval(0., 5.)),
        ]);
        let table2 = DataType::structured([
            ("a", DataType::float_interval(0., 20.)),
            ("b", DataType::integer_interval(-2, 2)),
            ("d", DataType::integer_interval(-2, 1)),
        ]);
        let data_type =
            DataType::structured([(Join::left_name(), table1), (Join::right_name(), table2)]);

        // ON
        let x = Expr::eq(
            Expr::qcol(Join::left_name(), "a"),
            Expr::qcol(Join::right_name(), "d"),
        );
        let join_op = JoinOperator::Inner(JoinConstraint::On(x.clone()));
        let filtered_table1 = DataType::structured([
            ("a", DataType::integer_interval(-2, 1)),
            ("b", DataType::integer_interval(-1, 3)),
            ("c", DataType::float_interval(0., 5.)),
        ]);
        let filtered_table2 = DataType::structured([
            ("a", DataType::float_interval(0., 20.)),
            ("b", DataType::integer_interval(-2, 2)),
            ("d", DataType::integer_interval(-2, 1)),
        ]);
        let filtered_data_type = DataType::structured([
            (Join::left_name(), filtered_table1),
            (Join::right_name(), filtered_table2),
        ]);
        assert_eq!(
            data_type.filter_by_join_operator(&join_op),
            filtered_data_type
        );

        // USING
        let v = vec![Identifier::from_name("a")];
        let join_op = JoinOperator::Inner(JoinConstraint::Using(v));
        let filtered_table1 = DataType::structured([
            ("a", DataType::float_interval(0., 3.)),
            ("b", DataType::integer_interval(-1, 3)),
            ("c", DataType::float_interval(0., 5.)),
        ]);
        let filtered_table2 = DataType::structured([
            ("a", DataType::float_interval(0., 3.)),
            ("b", DataType::integer_interval(-2, 2)),
            ("d", DataType::integer_interval(-2, 1)),
        ]);
        let filtered_data_type = DataType::structured([
            (Join::left_name(), filtered_table1),
            (Join::right_name(), filtered_table2),
        ]);
        assert_eq!(
            data_type.filter_by_join_operator(&join_op),
            filtered_data_type
        );

        // NATURAL
        let join_op = JoinOperator::Inner(JoinConstraint::Natural);
        let filtered_table1 = DataType::structured([
            ("a", DataType::float_interval(0., 3.)),
            ("b", DataType::integer_interval(-1, 2)),
            ("c", DataType::float_interval(0., 5.)),
        ]);
        let filtered_table2 = DataType::structured([
            ("a", DataType::float_interval(0., 3.)),
            ("b", DataType::integer_interval(-1, 2)),
            ("d", DataType::integer_interval(-2, 1)),
        ]);
        let filtered_data_type = DataType::structured([
            (Join::left_name(), filtered_table1),
            (Join::right_name(), filtered_table2),
        ]);
        assert_eq!(
            data_type.filter_by_join_operator(&join_op),
            filtered_data_type
        );

        // NONE
        let join_op = JoinOperator::Inner(JoinConstraint::None);
        assert_eq!(data_type.filter_by_join_operator(&join_op), data_type);
    }

    #[test]
    fn test_filter_data_type_left_join() {
        let table1 = DataType::structured([
            ("a", DataType::float_interval(-3., 3.)),
            ("b", DataType::integer_interval(-1, 3)),
            ("c", DataType::float_interval(0., 5.)),
        ]);
        let table2 = DataType::structured([
            ("a", DataType::float_interval(0., 20.)),
            ("b", DataType::integer_interval(-2, 2)),
            ("d", DataType::integer_interval(-2, 1)),
        ]);
        let data_type = DataType::structured([
            (Join::left_name(), table1.clone()),
            (Join::right_name(), table2),
        ]);

        // ON
        let x = Expr::eq(
            Expr::qcol(Join::left_name(), "a"),
            Expr::qcol(Join::right_name(), "d"),
        );
        let join_op = JoinOperator::LeftOuter(JoinConstraint::On(x.clone()));
        let filtered_table2 = DataType::structured([
            ("a", DataType::float_interval(0., 20.)),
            ("b", DataType::integer_interval(-2, 2)),
            ("d", DataType::integer_interval(-2, 1)),
        ]);
        let filtered_data_type = DataType::structured([
            (Join::left_name(), table1.clone()),
            (Join::right_name(), filtered_table2),
        ]);
        assert_eq!(
            data_type.filter_by_join_operator(&join_op),
            filtered_data_type
        );

        // USING
        let v = vec![Identifier::from_name("a")];
        let join_op = JoinOperator::LeftOuter(JoinConstraint::Using(v));
        let filtered_table2 = DataType::structured([
            ("a", DataType::float_interval(0., 3.)),
            ("b", DataType::integer_interval(-2, 2)),
            ("d", DataType::integer_interval(-2, 1)),
        ]);
        let filtered_data_type = DataType::structured([
            (Join::left_name(), table1.clone()),
            (Join::right_name(), filtered_table2),
        ]);
        assert_eq!(
            data_type.filter_by_join_operator(&join_op),
            filtered_data_type
        );

        // NATURAL
        let join_op = JoinOperator::LeftOuter(JoinConstraint::Natural);
        let filtered_table2 = DataType::structured([
            ("a", DataType::float_interval(0., 3.)),
            ("b", DataType::integer_interval(-1, 2)),
            ("d", DataType::integer_interval(-2, 1)),
        ]);
        let filtered_data_type = DataType::structured([
            (Join::left_name(), table1.clone()),
            (Join::right_name(), filtered_table2),
        ]);
        assert_eq!(
            data_type.filter_by_join_operator(&join_op),
            filtered_data_type
        );

        // NONE
        let join_op = JoinOperator::LeftOuter(JoinConstraint::None);
        assert_eq!(data_type.filter_by_join_operator(&join_op), data_type);
    }

    #[test]
    fn test_filter_data_type_right_join() {
        let table1 = DataType::structured([
            ("a", DataType::float_interval(-3., 3.)),
            ("b", DataType::integer_interval(-1, 3)),
            ("c", DataType::float_interval(0., 5.)),
        ]);
        let table2 = DataType::structured([
            ("a", DataType::float_interval(0., 20.)),
            ("b", DataType::integer_interval(-2, 2)),
            ("d", DataType::integer_interval(-2, 1)),
        ]);
        let data_type = DataType::structured([
            (Join::left_name(), table1),
            (Join::right_name(), table2.clone()),
        ]);

        // ON
        let x = Expr::eq(
            Expr::qcol(Join::left_name(), "a"),
            Expr::qcol(Join::right_name(), "d"),
        );
        let join_op = JoinOperator::RightOuter(JoinConstraint::On(x.clone()));
        let filtered_table1 = DataType::structured([
            ("a", DataType::integer_interval(-2, 1)),
            ("b", DataType::integer_interval(-1, 3)),
            ("c", DataType::float_interval(0., 5.)),
        ]);
        let filtered_data_type = DataType::structured([
            (Join::left_name(), filtered_table1),
            (Join::right_name(), table2.clone()),
        ]);
        assert_eq!(
            data_type.filter_by_join_operator(&join_op),
            filtered_data_type
        );

        // USING
        let v = vec![Identifier::from_name("a")];
        let join_op = JoinOperator::RightOuter(JoinConstraint::Using(v));
        let filtered_table1 = DataType::structured([
            ("a", DataType::float_interval(0., 3.)),
            ("b", DataType::integer_interval(-1, 3)),
            ("c", DataType::float_interval(0., 5.)),
        ]);
        let filtered_data_type = DataType::structured([
            (Join::left_name(), filtered_table1),
            (Join::right_name(), table2.clone()),
        ]);
        assert_eq!(
            data_type.filter_by_join_operator(&join_op),
            filtered_data_type
        );

        // NATURAL
        let join_op = JoinOperator::RightOuter(JoinConstraint::Natural);
        let filtered_table1 = DataType::structured([
            ("a", DataType::float_interval(0., 3.)),
            ("b", DataType::integer_interval(-1, 2)),
            ("c", DataType::float_interval(0., 5.)),
        ]);
        let filtered_data_type = DataType::structured([
            (Join::left_name(), filtered_table1),
            (Join::right_name(), table2.clone()),
        ]);
        assert_eq!(
            data_type.filter_by_join_operator(&join_op),
            filtered_data_type
        );

        // NONE
        let join_op = JoinOperator::RightOuter(JoinConstraint::None);
        assert_eq!(data_type.filter_by_join_operator(&join_op), data_type);
    }

    #[test]
    fn test_build_join_with_unique_constraint() {
        namer::reset();
        let schema1: Schema = vec![
            ("a", DataType::integer(), Some(Constraint::Unique)),
            ("b", DataType::float_interval(-2., 2.), None),
        ]
        .into_iter()
        .collect();
        let table1: Arc<Relation> = Arc::new(
            Relation::table()
                .name("table1")
                .schema(schema1.clone())
                .size(1000)
                .build(),
        );

        let schema2: Schema = vec![
            ("a", DataType::integer()),
            ("c", DataType::float_interval(-2., 2.)),
        ]
        .into_iter()
        .collect();
        let table2: Arc<Relation> = Arc::new(
            Relation::table()
                .name("table2")
                .schema(schema2.clone())
                .size(20)
                .build(),
        );

        let join: Join = Relation::join()
            .name("join")
            .inner()
            .on_eq("a", "a")
            .left(table1.clone())
            .right(table2.clone())
            .build();
        assert_eq!(join.size(), &Integer::from_interval(0, 1000));

        let join: Join = Relation::join()
            .name("join")
            .inner()
            .on_eq("a", "a")
            .left(table2.clone())
            .right(table1.clone())
            .build();
        assert_eq!(join.size(), &Integer::from_interval(0, 1000));

        let join: Join = Relation::join()
            .name("join")
            .left_outer()
            .on_eq("a", "a")
            .left(table2.clone())
            .right(table1.clone())
            .build();
        assert_eq!(join.size(), &Integer::from_interval(0, 1000));

        let join: Join = Relation::join()
            .name("join")
            .right_outer()
            .on_eq("a", "a")
            .left(table2.clone())
            .right(table1.clone())
            .build();
        assert_eq!(join.size(), &Integer::from_interval(0, 1000));

        let join: Join = Relation::join()
            .name("join")
            .full_outer()
            .on_eq("a", "a")
            .left(table2.clone())
            .right(table1.clone())
            .build();
        assert_eq!(join.size(), &Integer::from_interval(0, 1000));

        let join: Join = Relation::join()
            .name("join")
            .full_outer()
            .on_eq("a", "b")
            .left(table2.clone())
            .right(table1.clone())
            .build();
        assert_eq!(join.size(), &Integer::from_interval(0, 20000));
    }

    #[test]
    fn test_build_unique_constraint() {
        let schema1: Schema = vec![
            ("a", DataType::integer(), Some(Constraint::Unique)),
            ("b", DataType::float_interval(-2., 2.), None),
        ]
        .into_iter()
        .collect();
        let table1: Relation = Relation::table()
            .name("table1")
            .schema(schema1.clone())
            .size(1000)
            .build();

        let reduce: Relation = Relation::reduce()
            .name("reduce1")
            .with(("sum_a", expr!(sum(a))))
            .with_group_by_column("b")
            .input(table1.clone())
            .build();
        let correct_schema: Schema = vec![
            ("sum_a", DataType::integer(), None),
            (
                "b",
                DataType::float_interval(-2., 2.),
                Some(Constraint::Unique),
            ),
        ]
        .into_iter()
        .collect();
        assert_eq!(reduce.schema(), &correct_schema);

        let map: Relation = Relation::map()
            .name("map")
            .with(("my_sum", expr!(sum_a)))
            .with(("my_b", expr!(b)))
            .with(("my_b3", expr!(3 * b)))
            .input(reduce)
            .build();
        let correct_schema: Schema = vec![
            ("my_sum", DataType::integer(), None),
            (
                "my_b",
                DataType::float_interval(-2., 2.),
                Some(Constraint::Unique),
            ),
            ("my_b3", DataType::float_interval(-6., 6.), None),
        ]
        .into_iter()
        .collect();
        assert_eq!(map.schema(), &correct_schema);
        map.display_dot().unwrap();

        let join: Relation = Relation::join()
            .inner()
            .left(table1)
            .right(map)
            .on_eq("b", "my_b")
            .build();
        join.display_dot().unwrap();
        println!("{}", join.schema());
    }

    #[test]
    fn test_constraint_fields_in_joins() {
        namer::reset();
        let schema1: Schema = vec![
            ("a", DataType::float(), Some(Constraint::Unique)),
            ("b", DataType::float(), None),
            ("c", DataType::float(), Some(Constraint::Unique)),
        ]
        .into_iter()
        .collect();
        let table1: Arc<Relation> = Arc::new(
            Relation::table()
                .name("table1")
                .schema(schema1.clone())
                .size(1000)
                .build(),
        );

        let schema2: Schema = vec![
            ("a", DataType::float(), None),
            ("d", DataType::float(), Some(Constraint::Unique)),
            ("e", DataType::float(), Some(Constraint::Unique)),
        ]
        .into_iter()
        .collect();
        let table2: Arc<Relation> = Arc::new(
            Relation::table()
                .name("table2")
                .schema(schema2.clone())
                .size(20)
                .build(),
        );

        // the joining columns are not unique
        let join: Join = Relation::join()
            .name("join")
            .inner()
            .on_eq("b", "a")
            .left(table1.clone())
            .right(table2.clone())
            .left_names(vec!["a1", "b", "c"])
            .right_names(vec!["a2", "d", "e"])
            .build();
        let correct_schema: Schema = vec![
            ("a1", DataType::float(), None),
            ("b", DataType::float(), None),
            ("c", DataType::float(), None),
            ("a2", DataType::float(), None),
            ("d", DataType::float(), None),
            ("e", DataType::float(), None),
        ]
        .into_iter()
        .collect();
        assert_eq!(join.schema(), &correct_schema);

        // the left joining column is unique
        let join: Join = Relation::join()
            .name("join")
            .inner()
            .on_eq("a", "a")
            .left(table1.clone())
            .right(table2.clone())
            .left_names(vec!["a1", "b", "c"])
            .right_names(vec!["a2", "d", "e"])
            .build();
        let correct_schema: Schema = vec![
            ("a1", DataType::float(), None),
            ("b", DataType::float(), None),
            ("c", DataType::float(), None),
            ("a2", DataType::float(), None),
            ("d", DataType::float(), Some(Constraint::Unique)),
            ("e", DataType::float(), Some(Constraint::Unique)),
        ]
        .into_iter()
        .collect();
        assert_eq!(join.schema(), &correct_schema);

        // the right joining column is unique
        let join: Join = Relation::join()
            .name("join")
            .inner()
            .on_eq("b", "d")
            .left(table1.clone())
            .right(table2.clone())
            .left_names(vec!["a1", "b", "c"])
            .right_names(vec!["a2", "d", "e"])
            .build();
        let correct_schema: Schema = vec![
            ("a1", DataType::float(), Some(Constraint::Unique)),
            ("b", DataType::float(), None),
            ("c", DataType::float(), Some(Constraint::Unique)),
            ("a2", DataType::float(), None),
            ("d", DataType::float(), None),
            ("e", DataType::float(), None),
        ]
        .into_iter()
        .collect();
        assert_eq!(join.schema(), &correct_schema);

        // the joining columns are  unique
        let join: Join = Relation::join()
            .name("join")
            .inner()
            .on_eq("a", "d")
            .left(table1.clone())
            .right(table2.clone())
            .left_names(vec!["a1", "b", "c"])
            .right_names(vec!["a2", "d", "e"])
            .build();
        let correct_schema: Schema = vec![
            ("a1", DataType::float(), Some(Constraint::Unique)),
            ("b", DataType::float(), None),
            ("c", DataType::float(), Some(Constraint::Unique)),
            ("a2", DataType::float(), None),
            ("d", DataType::float(), Some(Constraint::Unique)),
            ("e", DataType::float(), Some(Constraint::Unique)),
        ]
        .into_iter()
        .collect();
        assert_eq!(join.schema(), &correct_schema);
    }
}
