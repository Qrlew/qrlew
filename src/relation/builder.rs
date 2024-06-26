use std::{hash::Hash, sync::Arc};

use itertools::Itertools;

use super::{
    Error, Join, JoinOperator, Map, OrderBy, Reduce, Relation, Result, Schema, Set, SetOperator,
    SetQuantifier, Table, Values, Variant,
};
use crate::{
    builder::{Ready, With, WithIterator},
    data_type::{Integer, Value},
    expr::{self, AggregateColumn, Expr, Identifier, Split},
    hierarchy::Hierarchy,
    namer::{self, FIELD, JOIN, MAP, REDUCE, SET},
    And,
};

// A Table builder
#[derive(Debug, Default)]
pub struct WithoutSchema;
pub struct WithSchema(Schema);

/*
Table builder
 */

/// A table builder
#[derive(Debug, Default)]
pub struct TableBuilder<RequireSchema> {
    /// The name of the table (may be derived from the path)
    name: Option<String>,
    /// The path of the table (may be derived from the name)
    path: Option<Identifier>,
    /// The schema description of the output
    schema: RequireSchema,
    /// The size of the dataset
    size: Option<i64>,
}

impl TableBuilder<WithoutSchema> {
    pub fn new() -> Self {
        TableBuilder::default()
    }
}

impl<RequireSchema> TableBuilder<RequireSchema> {
    pub fn name<S: Into<String>>(mut self, name: S) -> Self {
        let name: String = name.into();
        self.name = Some(name.clone());
        self.path = self.path.or_else(|| Some(name.into()));
        self
    }

    pub fn path<I: Into<Identifier>>(mut self, path: I) -> Self {
        let path: Identifier = path.into();
        self.path = Some(path.clone());
        self.name = self.name.or_else(|| Some(path.iter().join("_")));
        self
    }

    pub fn size(mut self, size: i64) -> Self {
        self.size = Some(size);
        self
    }

    pub fn schema<S: Into<Schema>>(self, schema: S) -> TableBuilder<WithSchema> {
        TableBuilder {
            name: self.name,
            path: self.path,
            schema: WithSchema(schema.into()),
            size: self.size,
        }
    }
}

impl Ready<Table> for TableBuilder<WithSchema> {
    type Error = Error;

    fn try_build(self) -> Result<Table> {
        let name = self.name.unwrap_or_else(|| namer::new_name("table"));
        let path = self.path.unwrap_or_else(|| name.clone().into());
        let size = self
            .size
            .map_or_else(|| Integer::from_min(0), |size| Integer::from_value(size));
        Ok(Table::new(name, path, self.schema.0, size))
    }
}

/*
Map Builder
 */

// A Map builder
#[derive(Debug, Default, Hash)]
pub struct WithoutInput;
#[derive(Debug, Hash)]
pub struct WithInput(Arc<Relation>);

/// A Builder for Map relations
#[derive(Clone, Debug, Default, Hash)]
pub struct MapBuilder<RequireInput> {
    name: Option<String>,
    split: Split,
    limit: Option<usize>,
    offset: Option<usize>,
    // The ultimate input
    input: RequireInput,
}

impl MapBuilder<WithoutInput> {
    pub fn new() -> Self {
        MapBuilder::default()
    }
}

impl<RequireInput> MapBuilder<RequireInput> {
    pub fn name<S: Into<String>>(mut self, name: S) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn split<S: Into<Split>>(mut self, split: S) -> Self {
        self.split = split.into();
        self
    }

    pub fn filter(mut self, filter: Expr) -> Self {
        self.split = self.split.map_last(|split| match split {
            Split::Map(map) => Split::from(map).and(Split::filter(filter).into()),
            Split::Reduce(reduce) => Split::Reduce(expr::Reduce::new(
                reduce.named_aggregates,
                reduce.group_by,
                reduce.map.as_deref().map(|m| {
                    Split::from(m.clone())
                        .and(Split::filter(filter).into())
                        .into_map()
                }),
            )),
        });
        self
    }

    pub fn filter_iter(self, iter: Vec<Expr>) -> Self {
        let filter = iter
            .into_iter()
            .fold(Expr::val(true), |f, x| Expr::and(f, x));
        self.filter(filter)
    }

    pub fn order_by(mut self, expr: Expr, asc: bool) -> Self {
        self.split = self.split.and(Split::order_by(expr, asc).into());
        self
    }

    pub fn order_by_iter(self, iter: Vec<(Expr, bool)>) -> Self {
        iter.into_iter().fold(self, |w, (x, b)| w.order_by(x, b))
    }

    /// Add a group by
    pub fn group_by(mut self, expr: Expr) -> Self {
        let s = Split::group_by(expr.into());
        self.split = self.split.and(s.into());
        self
    }

    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit.into());
        self
    }

    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset.into());
        self
    }

    /// Initialize a builder with filtered existing map
    pub fn filter_fields_with<P: Fn(&str) -> bool>(
        self,
        map: Map,
        predicate: P,
    ) -> MapBuilder<WithInput> {
        let Map {
            name,
            projection,
            filter,
            order_by,
            limit,
            offset,
            schema,
            input,
            ..
        } = map;
        let builder = self
            .name(name)
            .with_iter(
                schema
                    .into_iter()
                    .zip(projection)
                    .filter_map(|(field, expr)| {
                        predicate(field.name()).then_some((field.name().to_string(), expr))
                    }),
            )
            .input(input);
        // Filter
        let builder = filter.into_iter().fold(builder, |b, f| b.filter(f));
        // Order by
        let builder = order_by
            .into_iter()
            .fold(builder, |b, o| b.order_by(o.expr, o.asc));
        // Limit
        let builder = limit.into_iter().fold(builder, |b, l| b.limit(l));
        // Offset
        let builder = offset.into_iter().fold(builder, |b, l| b.offset(l));
        builder
    }

    /// Initialize a builder with filtered existing map
    pub fn map_with<F: Fn(&str, Expr) -> Expr>(self, map: Map, f: F) -> MapBuilder<WithInput> {
        let Map {
            name,
            projection,
            filter,
            order_by,
            limit,
            offset,
            schema,
            input,
            ..
        } = map;
        let builder = self
            .name(name)
            .with_iter(
                schema
                    .into_iter()
                    .zip(projection)
                    .map(|(field, expr)| (field.name().to_string(), f(field.name(), expr))),
            )
            .input(input);
        // Filter
        let builder = filter.into_iter().fold(builder, |b, f| b.filter(f));
        // Order by
        let builder = order_by
            .into_iter()
            .fold(builder, |b, o| b.order_by(o.expr, o.asc));
        // Limit
        let builder = limit.into_iter().fold(builder, |b, l| b.limit(l));
        // Offset
        let builder = offset.into_iter().fold(builder, |b, l| b.offset(l));
        builder
    }

    /// Initialize a builder with filtered existing map
    pub fn rename_with<F: Fn(&str, &Expr) -> String>(
        self,
        map: Map,
        f: F,
    ) -> MapBuilder<WithInput> {
        let Map {
            name,
            projection,
            filter,
            order_by,
            limit,
            offset,
            schema,
            input,
            ..
        } = map;
        let builder = self
            .name(name)
            .with_iter(
                schema
                    .into_iter()
                    .zip(projection)
                    .map(|(field, expr)| (f(field.name(), &expr), expr)),
            )
            .input(input);
        // Filter
        let builder = filter.into_iter().fold(builder, |b, f| b.filter(f));
        // Order by
        let builder = order_by
            .into_iter()
            .fold(builder, |b, o| b.order_by(o.expr, o.asc));
        // Limit
        let builder = limit.into_iter().fold(builder, |b, l| b.limit(l));
        // Offset
        let builder = offset.into_iter().fold(builder, |b, l| b.offset(l));
        builder
    }

    /// Initialize a builder with an existing map and filter by an `Expr` that depends on the input columns
    pub fn filter_with(self, map: Map, predicate: Expr) -> MapBuilder<WithInput> {
        let Map {
            name,
            projection,
            filter,
            order_by,
            limit,
            offset,
            schema,
            input,
            ..
        } = map;
        let builder = self
            .name(name)
            .with_iter(
                schema
                    .iter()
                    .zip(projection.clone())
                    .map(|(field, expr)| (field.name().to_string(), expr)),
            )
            .input(input);
        // Filter
        let filter = if let Some(x) = filter {
            Expr::and(x, predicate)
        } else {
            predicate
        };
        let builder = builder.filter(filter);
        // Order by
        let builder = order_by
            .into_iter()
            .fold(builder, |b, o| b.order_by(o.expr, o.asc));
        // Limit
        let builder = limit.into_iter().fold(builder, |b, l| b.limit(l));
        // Offset
        let builder = offset.into_iter().fold(builder, |b, l| b.offset(l));
        builder
    }

    pub fn input<R: Into<Arc<Relation>>>(self, input: R) -> MapBuilder<WithInput> {
        MapBuilder {
            name: self.name,
            split: self.split,
            limit: self.limit,
            offset: self.offset,
            input: WithInput(input.into()),
        }
    }
}

impl<RequireInput> With<Expr> for MapBuilder<RequireInput> {
    fn with(self, expr: Expr) -> Self {
        let name = namer::name_from_content(FIELD, &expr);
        self.with((name, expr))
    }
}

impl<RequireInput, S: Into<String>> With<(S, Expr)> for MapBuilder<RequireInput> {
    fn with(mut self, (name, expr): (S, Expr)) -> Self {
        self.split = self.split.and(Split::from((name.into(), expr)));
        self
    }
}

impl<RequireInput> With<Map, MapBuilder<WithInput>> for MapBuilder<RequireInput> {
    fn with(self, map: Map) -> MapBuilder<WithInput> {
        let Map {
            name,
            projection,
            filter,
            order_by,
            limit,
            offset,
            schema,
            input,
            ..
        } = map;
        let builder = self
            .name(name)
            .with_iter(
                schema
                    .into_iter()
                    .zip(projection)
                    .map(|(field, expr)| (field.name().to_string(), expr)),
            )
            .input(input);
        // Filter
        let builder = filter.into_iter().fold(builder, |b, f| b.filter(f));
        // Order by
        let builder = order_by
            .into_iter()
            .fold(builder, |b, o| b.order_by(o.expr, o.asc));
        // Limit
        let builder = limit.into_iter().fold(builder, |b, l| b.limit(l));
        // Offset
        let builder = offset.into_iter().fold(builder, |b, l| b.offset(l));
        builder
    }
}

/// Methods to cal on the finalized builder
impl MapBuilder<WithInput> {
    fn build_name(&self) -> String {
        self.name
            .clone()
            .unwrap_or_else(|| namer::name_from_content(MAP, &self))
    }
}

impl Ready<Map> for MapBuilder<WithInput> {
    type Error = Error;

    fn try_build(self) -> Result<Map> {
        // Build the name
        let name = self.build_name();
        if let Split::Map(map) = self.split {
            // Build the input
            let input = match map.reduce {
                Some(reduce) => Arc::new(
                    ReduceBuilder::new()
                        .split(*reduce)
                        .input(self.input.0)
                        .try_build()?,
                ),
                None => self.input.0,
            };
            // Build the Relation
            Ok(Map::new(
                name,
                map.named_exprs,
                map.filter,
                map.order_by
                    .into_iter()
                    .map(|(e, a)| OrderBy::new(e, a))
                    .collect(),
                self.limit,
                self.offset,
                input,
            ))
        } else {
            Err(Error::invalid_relation(self.split))
        }
    }
}

/*
Reduce Builder
 */

/// A Reduce builder
#[derive(Debug, Default, Hash)]
pub struct ReduceBuilder<RequireInput> {
    name: Option<String>,
    split: Split,
    // The ultimate input
    input: RequireInput,
}

impl ReduceBuilder<WithoutInput> {
    pub fn new() -> Self {
        ReduceBuilder::default()
    }
}

impl<RequireInput> ReduceBuilder<RequireInput> {
    pub fn name<S: Into<String>>(mut self, name: S) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn split<S: Into<Split>>(mut self, split: S) -> Self {
        self.split = split.into();
        self
    }

    pub fn group_by<E: Into<Expr>>(mut self, expr: E) -> Self {
        let s = Split::group_by(expr.into());
        self.split = self.split.and(s.into());
        self
    }

    pub fn group_by_iter<E: Into<Expr>, I: IntoIterator<Item = E>>(self, iter: I) -> Self {
        iter.into_iter().fold(self, |w, i| w.group_by(i))
    }

    pub fn filter(mut self, filter: Expr) -> Self {
        self.split = self.split.map_last(|split| match split {
            Split::Map(map) => Split::from(map).and(Split::filter(filter).into()),
            Split::Reduce(reduce) => Split::Reduce(expr::Reduce::new(
                reduce.named_aggregates,
                reduce.group_by,
                reduce.map.as_deref().map(|m| {
                    Split::from(m.clone())
                        .and(Split::filter(filter).into())
                        .into_map()
                }),
            )),
        });
        self
    }

    pub fn input<R: Into<Arc<Relation>>>(self, input: R) -> ReduceBuilder<WithInput> {
        ReduceBuilder {
            name: self.name,
            split: self.split,
            input: WithInput(input.into()),
        }
    }

    /// Initialize a builder with filtered existing reduce
    pub fn filter_fields_with<P: Fn(&str) -> bool>(
        self,
        reduce: Reduce,
        predicate: P,
    ) -> ReduceBuilder<WithInput> {
        let Reduce {
            name,
            aggregate,
            group_by,
            schema,
            input,
            ..
        } = reduce;
        let builder = self
            .name(name)
            .with_iter(
                schema
                    .into_iter()
                    .zip(aggregate)
                    .filter_map(|(field, aggregate)| {
                        predicate(field.name()).then_some((field.name().to_string(), aggregate))
                    }),
            )
            .input(input);
        // Group by
        let builder = group_by.into_iter().fold(builder, |b, g| b.group_by(g));
        builder
    }

    /// Add a group by column
    pub fn with_group_by_column<S: Into<String>>(mut self, column: S) -> Self {
        let name = column.into();
        self = self.group_by(Expr::col(name.clone()));
        self = self.with((name.clone(), AggregateColumn::from(name)));
        self
    }

    /// Rename fields in the reduce
    pub fn rename_with<F: Fn(&str, &Expr) -> String>(
        self,
        reduce: Reduce,
        f: F,
    ) -> ReduceBuilder<WithInput> {
        let Reduce {
            name,
            aggregate,
            group_by,
            schema,
            size: _,
            input,
        } = reduce;
        let builder = self
            .name(name)
            .with_iter(
                schema
                    .into_iter()
                    .zip(aggregate)
                    .map(|(field, aggregate)| (f(field.name(), &aggregate), aggregate)),
            )
            .group_by_iter(group_by.into_iter())
            .input(input);
        builder
    }
}

impl<RequireInput> With<Expr> for ReduceBuilder<RequireInput> {
    fn with(self, expr: Expr) -> Self {
        let name = namer::name_from_content(FIELD, &expr);
        self.with((name, expr))
    }
}

impl<RequireInput, S: Into<String>> With<(S, Expr)> for ReduceBuilder<RequireInput> {
    fn with(mut self, (name, expr): (S, Expr)) -> Self {
        self.split = self.split.and(Split::from((name, expr)));
        self
    }
}

impl<RequireInput> With<AggregateColumn> for ReduceBuilder<RequireInput> {
    fn with(self, aggregate: AggregateColumn) -> Self {
        let name = namer::name_from_content(FIELD, &aggregate);
        self.with((name, aggregate))
    }
}

impl<RequireInput, S: Into<String>> With<(S, AggregateColumn)> for ReduceBuilder<RequireInput> {
    fn with(mut self, (name, aggregate): (S, AggregateColumn)) -> Self {
        self.split = self.split.and(Split::reduce(name, aggregate).into());
        self
    }
}

impl<RequireInput> With<Reduce, ReduceBuilder<WithInput>> for ReduceBuilder<RequireInput> {
    fn with(self, reduce: Reduce) -> ReduceBuilder<WithInput> {
        let Reduce {
            name,
            aggregate,
            group_by,
            schema,
            input,
            ..
        } = reduce;
        let builder = self
            .name(name)
            .with_iter(
                schema
                    .into_iter()
                    .zip(aggregate)
                    .map(|(field, expr)| (field.name().to_string(), expr)),
            )
            .input(input);
        // Group by
        let builder = group_by.into_iter().fold(builder, |b, g| b.group_by(g));
        builder
    }
}

/// Methods to cal on the finalized builder
impl ReduceBuilder<WithInput> {
    fn build_name(&self) -> String {
        self.name
            .clone()
            .unwrap_or_else(|| namer::name_from_content(REDUCE, &self))
    }
}

impl Ready<Reduce> for ReduceBuilder<WithInput> {
    type Error = Error;

    fn try_build(self) -> Result<Reduce> {
        // Build the name
        let name = self.build_name();
        if let Split::Reduce(reduce) = self.split {
            // Build the input
            let input = match reduce.map {
                Some(map) => Arc::new(
                    MapBuilder::new()
                        .split(*map)
                        .input(self.input.0)
                        .try_build()?,
                ),
                None => self.input.0,
            };
            // Build the Relation
            Ok(Reduce::new(
                name,
                reduce.named_aggregates,
                reduce.group_by,
                input,
            ))
        } else {
            Err(Error::invalid_relation(self.split))
        }
    }
}

/*
Join Builder
 */

/// A Join builder
#[derive(Debug, Default, Hash)]
pub struct JoinBuilder<RequireLeftInput, RequireRightInput> {
    name: Option<String>,
    /// Input names indexed by qualified names
    names: Hierarchy<String>,
    /// Names by position as a fallback
    left_names: Vec<String>,
    /// Names by position as a fallback
    right_names: Vec<String>,
    operator: Option<JoinOperator>,
    left: RequireLeftInput,
    right: RequireRightInput,
}

impl JoinBuilder<WithoutInput, WithoutInput> {
    pub fn new() -> Self {
        JoinBuilder::default()
    }
}

impl<RequireLeftInput, RequireRightInput> JoinBuilder<RequireLeftInput, RequireRightInput> {
    pub fn name<S: Into<String>>(mut self, name: S) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn left_names<S: Into<String>>(mut self, names: Vec<S>) -> Self {
        self.left_names = names.into_iter().map(S::into).collect();
        self
    }

    pub fn right_names<S: Into<String>>(mut self, names: Vec<S>) -> Self {
        self.right_names = names.into_iter().map(S::into).collect();
        self
    }

    pub fn names(mut self, names: Hierarchy<String>) -> Self {
        self.names = names;
        self
    }

    pub fn inner(mut self, expr: Expr) -> Self {
        self.operator = Some(JoinOperator::Inner(expr));
        self
    }

    pub fn left_outer(mut self, expr: Expr) -> Self {
        self.operator = Some(JoinOperator::LeftOuter(expr));
        self
    }

    pub fn right_outer(mut self, expr: Expr) -> Self {
        self.operator = Some(JoinOperator::RightOuter(expr));
        self
    }

    pub fn full_outer(mut self, expr: Expr) -> Self {
        self.operator = Some(JoinOperator::FullOuter(expr));
        self
    }

    pub fn cross(mut self) -> Self {
        self.operator = Some(JoinOperator::Cross);
        self
    }
    /// Add an on condition
    pub fn on(mut self, expr: Expr) -> Self {
        self.operator = match self.operator {
            Some(JoinOperator::Inner(_)) => Some(JoinOperator::Inner(expr)),
            Some(JoinOperator::LeftOuter(_)) => Some(JoinOperator::LeftOuter(expr)),
            Some(JoinOperator::RightOuter(_)) => Some(JoinOperator::RightOuter(expr)),
            Some(JoinOperator::FullOuter(_)) => Some(JoinOperator::FullOuter(expr)),
            Some(JoinOperator::Cross) => Some(JoinOperator::Cross),
            None => Some(JoinOperator::Inner(expr)),
        };
        self
    }

    pub fn on_eq(self, left: &str, right: &str) -> Self {
        let x = Expr::eq(
            Expr::qcol(Join::left_name(), left),
            Expr::qcol(Join::right_name(), right),
        );
        self.on(x)
    }

    pub fn on_iter<I: IntoIterator<Item = Expr>>(mut self, exprs: I) -> Self {
        self = self.on(Expr::and_iter(exprs));
        self
    }

    /// Add a condition to the ON
    pub fn and(mut self, expr: Expr) -> Self {
        self.operator = match self.operator {
            Some(JoinOperator::Inner(x)) => Some(JoinOperator::Inner(Expr::and(expr, x))),
            Some(JoinOperator::LeftOuter(x)) => Some(JoinOperator::LeftOuter(Expr::and(expr, x))),
            Some(JoinOperator::RightOuter(x)) => Some(JoinOperator::RightOuter(Expr::and(expr, x))),
            Some(JoinOperator::FullOuter(x)) => Some(JoinOperator::FullOuter(Expr::and(expr, x))),
            op => op,
        };
        self
    }

    /// Set directly the full JOIN operator
    pub fn operator(mut self, operator: JoinOperator) -> Self {
        self.operator = Some(operator);
        self
    }

    pub fn left<R: Into<Arc<Relation>>>(
        self,
        input: R,
    ) -> JoinBuilder<WithInput, RequireRightInput> {
        JoinBuilder {
            name: self.name,
            left_names: self.left_names,
            right_names: self.right_names,
            operator: self.operator,
            left: WithInput(input.into()),
            right: self.right,
            names: self.names,
        }
    }

    pub fn right<R: Into<Arc<Relation>>>(
        self,
        input: R,
    ) -> JoinBuilder<RequireLeftInput, WithInput> {
        JoinBuilder {
            name: self.name,
            left_names: self.left_names,
            right_names: self.right_names,
            operator: self.operator,
            left: self.left,
            right: WithInput(input.into()),
            names: self.names,
        }
    }
}

impl<RequireLeftInput, RequireRightInput> With<Join, JoinBuilder<WithInput, WithInput>>
    for JoinBuilder<RequireLeftInput, RequireRightInput>
{
    fn with(self, join: Join) -> JoinBuilder<WithInput, WithInput> {
        let Join {
            name,
            operator,
            schema: _,
            size: _,
            left,
            right,
        } = join;
        let builder = self.name(name).operator(operator).left(left).right(right);
        builder
    }
}

impl Ready<Join> for JoinBuilder<WithInput, WithInput> {
    type Error = Error;

    fn try_build(self) -> Result<Join> {
        let name = self
            .name
            .clone()
            .unwrap_or(namer::name_from_content(JOIN, &self));
        let operator = self.operator.unwrap_or(JoinOperator::Cross);
        let left_names = self
            .left
            .0
            .schema()
            .iter()
            .enumerate()
            .map(|(i, field)| {
                self.names
                    .get(&[Join::left_name().to_string(), field.name().to_string()])
                    .unwrap_or(self.left_names.get(i).unwrap_or(&namer::name_from_content(
                        FIELD,
                        &(Join::left_name(), &field),
                    )))
                    .to_string()
            })
            .collect();
        let right_names = self
            .right
            .0
            .schema()
            .iter()
            .enumerate()
            .map(|(i, field)| {
                self.names
                    .get(&[Join::right_name().to_string(), field.name().to_string()])
                    .unwrap_or(self.right_names.get(i).unwrap_or(&namer::name_from_content(
                        FIELD,
                        &(Join::right_name(), &field),
                    )))
                    .to_string()
            })
            .collect();
        Ok(Join::new(
            name,
            left_names,
            right_names,
            operator,
            self.left.0,
            self.right.0,
        ))
    }
}

/*
Set Builder
 */

/// A Set builder
#[derive(Debug, Default, Hash)]
pub struct SetBuilder<RequireLeftInput, RequireRightInput> {
    name: Option<String>,
    operator: Option<SetOperator>,
    quantifier: Option<SetQuantifier>,
    left: RequireLeftInput,
    right: RequireRightInput,
}

impl SetBuilder<WithoutInput, WithoutInput> {
    pub fn new() -> Self {
        SetBuilder::default()
    }
}

impl<RequireLeftInput, RequireRightInput> SetBuilder<RequireLeftInput, RequireRightInput> {
    pub fn name<S: Into<String>>(mut self, name: S) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn union(mut self) -> Self {
        self.operator = Some(SetOperator::Union);
        self
    }

    pub fn except(mut self) -> Self {
        self.operator = Some(SetOperator::Except);
        self
    }

    pub fn intersect(mut self) -> Self {
        self.operator = Some(SetOperator::Intersect);
        self
    }

    pub fn all(mut self) -> Self {
        self.quantifier = Some(SetQuantifier::All);
        self
    }

    pub fn distinct(mut self) -> Self {
        self.quantifier = Some(SetQuantifier::Distinct);
        self
    }

    /// Set directly the SetOperator
    pub fn operator(mut self, operator: SetOperator) -> Self {
        self.operator = Some(operator);
        self
    }

    /// Set directly the SetQuantifier
    pub fn quantifier(mut self, quantifier: SetQuantifier) -> Self {
        self.quantifier = Some(quantifier);
        self
    }

    pub fn left<R: Into<Arc<Relation>>>(
        self,
        input: R,
    ) -> SetBuilder<WithInput, RequireRightInput> {
        SetBuilder {
            name: self.name,
            operator: self.operator,
            quantifier: self.quantifier,
            left: WithInput(input.into()),
            right: self.right,
        }
    }

    pub fn right<R: Into<Arc<Relation>>>(
        self,
        input: R,
    ) -> SetBuilder<RequireLeftInput, WithInput> {
        SetBuilder {
            name: self.name,
            operator: self.operator,
            quantifier: self.quantifier,
            left: self.left,
            right: WithInput(input.into()),
        }
    }
}

impl<RequireLeftInput, RequireRightInput> With<Set, SetBuilder<WithInput, WithInput>>
    for SetBuilder<RequireLeftInput, RequireRightInput>
{
    fn with(self, set: Set) -> SetBuilder<WithInput, WithInput> {
        let Set {
            name,
            operator,
            quantifier,
            schema: _,
            size: _,
            left,
            right,
        } = set;
        let builder = self
            .name(name)
            .operator(operator)
            .quantifier(quantifier)
            .left(left)
            .right(right);
        builder
    }
}

impl Ready<Set> for SetBuilder<WithInput, WithInput> {
    type Error = Error;

    fn try_build(self) -> Result<Set> {
        let name = self
            .name
            .clone()
            .unwrap_or(namer::name_from_content(SET, &self));
        let names = self
            .left
            .0
            .schema()
            .iter()
            .zip(self.right.0.schema().iter())
            .map(|(left_field, right_field)| {
                if left_field.name() == right_field.name() {
                    left_field.name().to_string()
                } else {
                    namer::name_from_content(
                        FIELD,
                        &(&self.left.0, &self.right.0, left_field, right_field),
                    )
                }
            })
            .collect();
        let operator = self.operator.unwrap_or(SetOperator::Union);
        let quantifier = self.quantifier.unwrap_or(SetQuantifier::None);
        Ok(Set::new(
            name,
            names,
            operator,
            quantifier,
            self.left.0,
            self.right.0,
        ))
    }
}

/*
Values builder
 */

/// A values builder
#[derive(Debug, Default)]
pub struct ValuesBuilder {
    /// The name
    name: Option<String>,
    /// The Value
    values: Vec<Value>,
}

impl ValuesBuilder {
    pub fn new() -> Self {
        ValuesBuilder::default()
    }

    pub fn name<S: Into<String>>(mut self, name: S) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn values<L: IntoIterator<Item = V>, V: Into<Value>>(mut self, values: L) -> Self {
        self.values = values.into_iter().map(|v| v.into()).collect();
        self
    }
}

impl Ready<Values> for ValuesBuilder {
    type Error = Error;

    fn try_build(self) -> Result<Values> {
        let name = self.name.unwrap_or_else(|| namer::new_name("values"));
        let values = self.values;
        Ok(Values::new(name, values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{data_type::DataTyped, display::Dot, DataType};

    #[test]
    fn test_map_building() {
        let table: Relation = Relation::table()
            .path(["db", "schema", "table"])
            .schema(
                Schema::builder()
                    .with(("a", DataType::float_range(1.0..=1.1)))
                    .with(("b", DataType::float_values([0.1, 1.0, 5.0, -1.0, -5.0])))
                    .with(("c", DataType::float_range(0.0..=5.0)))
                    .with(("d", DataType::float_values([0.0, 1.0, 2.0, -1.0])))
                    .with(("x", DataType::float_range(0.0..=2.0)))
                    .with(("y", DataType::float_range(0.0..=5.0)))
                    .with(("z", DataType::float_range(9.0..=11.)))
                    .with(("t", DataType::float_range(0.9..=1.1)))
                    .build(),
            )
            .build();
        println!("Table = {table}");
        let map: Relation = Relation::map()
            .with(("A", Expr::col("a")))
            .with(("B", Expr::col("b")))
            .input(table)
            .build();
        println!("Map = {map}");
        let reduce: Relation = Relation::reduce()
            .with(("S", Expr::sum(Expr::col("A"))))
            .with_group_by_column("B")
            .input(map)
            .build();
        println!("Reduce = {reduce}");
    }

    #[test]
    fn test_reduce_building() {
        let table: Relation = Relation::table()
            .path(["db", "schema", "table"])
            .schema(
                Schema::builder()
                    .with(("a", DataType::float_range(1.0..=1.1)))
                    .with(("b", DataType::float_values([0.1, 1.0, 5.0, -1.0, -5.0])))
                    .with(("c", DataType::float_range(0.0..=5.0)))
                    .with(("d", DataType::float_values([0.0, 1.0, 2.0, -1.0])))
                    .with(("x", DataType::float_range(0.0..=2.0)))
                    .with(("y", DataType::float_range(0.0..=5.0)))
                    .with(("z", DataType::float_range(9.0..=11.)))
                    .with(("t", DataType::float_range(0.9..=1.1)))
                    .build(),
            )
            .build();
        println!("Table = {table}");
        let reduce: Relation = Relation::reduce()
            .with(("S", AggregateColumn::sum("a")))
            // .with_group_by_column("b")
            .group_by(Expr::col("b"))
            .input(table)
            .build();
        println!("Reduce = {reduce}");
        reduce.display_dot().unwrap();
    }

    #[test]
    fn test_map_reduce() {
        let table: Relation = Relation::table()
            .path(["db", "schema", "table"])
            .schema(
                Schema::builder()
                    .with(("a", DataType::float_range(1.0..=1.1)))
                    .with(("b", DataType::float_values([0.1, 1.0, 5.0, -1.0, -5.0])))
                    .with(("c", DataType::float_range(0.0..=5.0)))
                    .with(("d", DataType::float_values([0.0, 1.0, 2.0, -1.0])))
                    .with(("x", DataType::float_range(0.0..=2.0)))
                    .with(("y", DataType::float_range(0.0..=5.0)))
                    .with(("z", DataType::float_range(9.0..=11.)))
                    .with(("t", DataType::float_range(0.9..=1.1)))
                    .build(),
            )
            .build();
        let map: Relation = Map::builder()
            .with(("a", Expr::col("a")))
            .with(("b", Expr::col("b")))
            .input(table)
            .build();
        let reduce: Relation = Relation::reduce()
            .with(("S", AggregateColumn::sum("a")))
            .with(("count", Expr::sum(Expr::val(1.))))
            .group_by(Expr::col("b"))
            .input(map)
            .build();
        println!("Reduce = {reduce}");
        reduce.display_dot().unwrap();
    }

    #[test]
    fn test_join_building() {
        use crate::{
            ast,
            display::Dot,
            hierarchy::Path,
            io::{postgresql, Database},
        };
        use itertools::Itertools;
        let mut database = postgresql::test_database();
        let join: Relation = Relation::join()
            .left(database.relations().get(&"table_1".path()).unwrap().clone())
            .right(
                database
                    .relations()
                    .get(&["table_2".into()])
                    .unwrap()
                    .clone(),
            )
            .on_eq("d", "x")
            .and(Expr::lt(Expr::col("a"), Expr::col("x")))
            .build();
        join.display_dot().unwrap();
        println!("Join = {join}");
        let query = &ast::Query::from(&join).to_string();
        println!(
            "{}\n{}",
            format!("{query}"),
            database
                .query(query)
                .unwrap()
                .iter()
                .map(ToString::to_string)
                .join("\n")
        );
    }

    #[test]
    fn test_join() {
        let table1: Relation = Relation::table()
            .name("table")
            .schema(
                Schema::builder()
                    .with(("a", DataType::integer_range(1..=10)))
                    .with(("b", DataType::integer_values([1, 2, 5, 6, 7, 8])))
                    .build(),
            )
            .build();

        let table2: Relation = Relation::table()
            .name("table")
            .schema(
                Schema::builder()
                    .with(("c", DataType::integer_range(5..=20)))
                    .with(("d", DataType::integer_range(1..=100)))
                    .build(),
            )
            .build();

        let join: Relation = Relation::join()
            .left(table1)
            .right(table2)
            .left_outer(Expr::eq(Expr::col("a"), Expr::col("c")))
            .left_names(vec!["a1", "b1"])
            //.on_iter(vec![Expr::eq(Expr::col("a"), Expr::col("c")), Expr::eq(Expr::col("b"), Expr::col("d"))])
            .build();
        join.display_dot().unwrap();
    }

    #[test]
    fn test_map_filter() {
        let table: Relation = Relation::table()
            .name("table")
            .schema(
                Schema::builder()
                    .with(("a", DataType::float_range(1.0..=1.1)))
                    .with(("b", DataType::float_values([0.1, 1.0, 5.0, -1.0, -5.0])))
                    .with(("c", DataType::float_range(0.0..=5.0)))
                    .build(),
            )
            .build();
        let map: Relation = Relation::map()
            .with(("A", Expr::col("a")))
            .with(("B", Expr::col("b")))
            .filter(Expr::gt(Expr::col("a"), Expr::val(0.5)))
            .filter(Expr::eq(Expr::col("b"), Expr::val(0.5)))
            .input(table.clone())
            .build();
        if let Relation::Map(m) = map {
            assert_eq!(m.filter.unwrap(), expr!(eq(b, 0.5)))
        }

        let map: Relation = Relation::map()
            .with(("A", Expr::col("a")))
            .with(("B", Expr::col("b")))
            .filter_iter(vec![
                Expr::gt(Expr::col("a"), Expr::val(0.5)),
                Expr::eq(Expr::col("b"), Expr::val(0.6)),
            ])
            .input(table)
            .build();
        if let Relation::Map(m) = map {
            assert_eq!(
                m.filter.unwrap(),
                Expr::and(
                    Expr::and(Expr::val(true), Expr::gt(Expr::col("a"), Expr::val(0.5))),
                    Expr::eq(Expr::col("b"), Expr::val(0.6))
                )
            )
        }
    }

    #[test]
    fn test_reduce_filter() {
        let table: Relation = Relation::table()
            .path(["db", "schema", "table"])
            .schema(
                Schema::builder()
                    .with(("a", DataType::float_range(1.0..=1.1)))
                    .with(("b", DataType::float_values([0.1, 1.0, 5.0, -1.0, -5.0])))
                    .with(("c", DataType::float_range(0.0..=5.0)))
                    .build(),
            )
            .build();

        // Reduce inputing a Map
        let relation: Relation = Relation::reduce()
            .with(("S", expr!(sum(3 * a))))
            .group_by(Expr::col("b"))
            .input(table.clone())
            .filter(expr!(gt(b, 0.5)))
            .build();
        //relation.display_dot().unwrap();
        assert_eq!(
            relation.inputs()[0].schema()[1].data_type(),
            DataType::float_values([1.0, 5.0])
        );

        // Simple Reduce
        let relation: Relation = Relation::reduce()
            .with(("S", expr!(sum(a))))
            .with_group_by_column("b")
            .filter(expr!(gt(b, 0.5)))
            .input(table.clone())
            .build();
        relation.display_dot().unwrap();
        assert_eq!(
            relation.inputs()[0].schema()[1].data_type(),
            DataType::float_values([1.0, 5.0])
        );
        assert_eq!(
            relation.data_type()["b"],
            DataType::float_values([1.0, 5.0])
        );

        let reduce: Relation = Relation::reduce()
            .with(("S", AggregateColumn::sum("a")))
            .group_by(Expr::col("b"))
            .filter(expr!(gt(a, 1.05)))
            .input(table)
            .build();
        assert_eq!(
            relation.inputs()[0].schema()[0].data_type(),
            DataType::float_range(1.0..=1.1)
        );
        reduce.display_dot().unwrap();
    }

    #[test]
    fn test_map_filter_with() {
        let table: Relation = Relation::table()
            .name("table")
            .schema(
                Schema::builder()
                    .with(("a", DataType::float_range(1.0..=1.1)))
                    .with(("b", DataType::float_values([0.1, 1.0, 5.0, -1.0, -5.0])))
                    .with(("c", DataType::float_range(0.0..=5.0)))
                    .build(),
            )
            .build();

        let map: Relation = Relation::map()
            .with(("A", Expr::col("a")))
            .with(("B", Expr::col("b")))
            .filter(Expr::gt(Expr::col("a"), Expr::val(0.5)))
            .input(table.clone())
            .build();
        if let Relation::Map(m) = map {
            println!("Map = {}", m);
            let filtered_map: Map = Relation::map()
                .filter_with(m, Expr::lt(Expr::col("a"), Expr::val(0.9)))
                .build();
            assert_eq!(
                filtered_map.filter.unwrap(),
                Expr::and(
                    Expr::gt(Expr::col("a"), Expr::val(0.5)),
                    Expr::lt(Expr::col("a"), Expr::val(0.9))
                )
            )
        }

        let map: Relation = Relation::map()
            .with(("A", Expr::col("a")))
            .with(("B", Expr::col("b")))
            .input(table.clone())
            .build();
        if let Relation::Map(m) = map {
            println!("Map = {}", m);
            let filtered_map: Map = Relation::map()
                .filter_with(m, Expr::lt(Expr::col("a"), Expr::val(0.9)))
                .build();
            assert_eq!(
                filtered_map.filter.unwrap(),
                Expr::lt(Expr::col("a"), Expr::val(0.9))
            )
        }

        let map: Relation = Relation::map()
            .with(("a", Expr::col("a")))
            .with(("b", Expr::col("b")))
            .input(table.clone())
            .build();
        if let Relation::Map(m) = map {
            println!("Map = {}", m);
            let filtered_map: Map = Relation::map()
                .filter_with(m, Expr::lt(Expr::col("a"), Expr::val(0.9)))
                .build();
            assert_eq!(
                filtered_map.filter.unwrap(),
                Expr::lt(Expr::col("a"), Expr::val(0.9))
            )
        }
    }

    #[test]
    fn test_values() {
        // empty
        let values = Relation::values().build();
        assert_eq!(Values::new("values_0".to_string(), vec![]), values);

        // float
        let values = Relation::values().name("MyValues").values(vec![5.]).build();
        assert_eq!(
            Values::new("MyValues".to_string(), vec![Value::float(5.)]),
            values
        );

        // list of float
        let values = Relation::values()
            .name("MyValues")
            .values([1., 3., 5.])
            .build();
        assert_eq!(
            Values::new(
                "MyValues".to_string(),
                vec![1.0.into(), 3.0.into(), 5.0.into()]
            ),
            values
        );

        // list of float
        let values: Relation = Relation::values()
            .name("MyValues")
            .values([
                Value::from(1.),
                Value::from(6),
                Value::from("a".to_string()),
            ])
            .build();
        println!("{}", values);
        println!("{}", values.data_type());
    }

    #[test]
    fn test_auto_join_building() {
        use crate::{
            ast,
            display::Dot,
            hierarchy::Path,
            io::{postgresql, Database},
        };
        let mut database = postgresql::test_database();
        let table1 = database
            .relations()
            .get(&"table_1".path())
            .unwrap()
            .as_ref()
            .clone();
        let join: Relation = Relation::join()
            .left(table1.clone())
            .right(table1.clone())
            .inner(Expr::val(true))
            .on_eq("d", "d")
            .names(Hierarchy::<String>::from_iter(vec![
                ([Join::left_name(), "a"], "a1".to_string()),
                ([Join::left_name(), "d"], "d1".to_string()),
            ]))
            .left_names(vec!["l_a", "l_b", "l_c", "l_d"])
            .right_names(vec!["a", "b", "c", "d"])
            .build();
        join.display_dot().unwrap();
        println!("Join = {join}");
        let query = &ast::Query::from(&join).to_string();
        println!("query = {}", query);
        assert_eq!(
            join.data_type(),
            DataType::structured(vec![
                ("a1", DataType::float_interval(0., 10.)),
                ("l_b", DataType::optional(DataType::float_interval(-1., 1.))),
                (
                    "l_c",
                    DataType::date_interval(
                        chrono::NaiveDate::from_ymd_opt(1980, 12, 06).unwrap(),
                        chrono::NaiveDate::from_ymd_opt(2023, 12, 06).unwrap(),
                    )
                ),
                ("d1", DataType::integer_interval(0, 10)),
                ("a", DataType::float_interval(0., 10.)),
                ("b", DataType::optional(DataType::float_interval(-1., 1.))),
                (
                    "c",
                    DataType::date_interval(
                        chrono::NaiveDate::from_ymd_opt(1980, 12, 06).unwrap(),
                        chrono::NaiveDate::from_ymd_opt(2023, 12, 06).unwrap(),
                    )
                ),
                ("d", DataType::integer_interval(0, 10)),
            ])
        );
        _ = database
            .query(query)
            .unwrap()
            .iter()
            .map(ToString::to_string)
            .join("\n");
    }
}
