use std::{hash::Hash, rc::Rc};

use super::{
    Error, Join, JoinConstraint, JoinOperator,
    Set, SetOperator, SetQuantifier,
    Map, OrderBy,
    Reduce, Relation, Result, Schema,
    Table, Variant,
};
use crate::{
    builder::{Ready, With, WithIterator},
    data_type::Integer,
    expr::{self, Expr, Identifier, Split, split},
    namer::{self, FIELD, JOIN, SET, MAP, REDUCE},
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
    /// The name of the table
    name: Option<String>,
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
        self.name = Some(name.into());
        self
    }

    pub fn size(mut self, size: i64) -> Self {
        self.size = Some(size);
        self
    }

    pub fn schema<S: Into<Schema>>(self, schema: S) -> TableBuilder<WithSchema> {
        TableBuilder {
            name: self.name,
            schema: WithSchema(schema.into()),
            size: self.size,
        }
    }
}

impl Ready<Table> for TableBuilder<WithSchema> {
    type Error = Error;

    fn try_build(self) -> Result<Table> {
        let name = self.name.unwrap_or_else(|| namer::new_name("table"));
        let size = self
            .size
            .map_or_else(|| Integer::from_min(0), |size| Integer::from_value(size));
        Ok(Table::new(name, self.schema.0, size))
    }
}

/*
Map Builder
 */

// A Map builder
#[derive(Debug, Default, Hash)]
pub struct WithoutInput;
#[derive(Debug, Hash)]
pub struct WithInput(Rc<Relation>);

/// A Builder for Map relations
#[derive(Clone, Debug, Default, Hash)]
pub struct MapBuilder<RequireInput> {
    name: Option<String>,
    split: Split,
    limit: Option<usize>,
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

    // TODO filter should maybe be possible on aggregates
    pub fn filter(mut self, filter: Expr) -> Self {
        self.split = self.split.map_last(|split| match split {
            Split::Map(map) => Split::from(map).and(Split::filter(filter).into()),
            Split::Reduce(reduce) => Split::Reduce(expr::Reduce::new(
                reduce.named_exprs,
                reduce.group_by,
                Some(Split::filter(filter.into())),
            )),
        });
        self
    }

    // TODO Does order by applies to the top split?
    pub fn order_by(mut self, expr: Expr, asc: bool) -> Self {
        self.split = self.split.and(Split::order_by(expr, asc).into());
        self
    }

    /// Add a group by
    pub fn group_by(mut self, expr: Expr) -> Self {
        self.split = self
            .split
            .map_last_reduce(|reduce| reduce.and(Split::group_by(expr)));
        self
    }

    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit.into());
        self
    }

    /// Initialize a builder with filtered existing map
    pub fn filter_with<P: Fn(&str) -> bool>(self, map: Map, predicate: P) -> MapBuilder<WithInput> {
        let Map { name, projection, filter, order_by, limit, schema, input, .. } = map;
        let builder = self.name(name)
            .with_iter(schema.into_iter().zip(projection).filter_map(|(field, expr)| predicate(field.name()).then_some((field.name().to_string(), expr))))
            .input(input);
        // Filter
        let builder = filter.into_iter().fold(builder, |b, f|b.filter(f));
        // Order by
        let builder = order_by.into_iter().fold(builder, |b, o|b.order_by(o.expr, o.asc));
        // Limit
        let builder = limit.into_iter().fold(builder, |b, l|b.limit(l));
        builder
    }

    /// Initialize a builder with filtered existing map
    pub fn map_with<F: Fn(&str, Expr) -> Expr>(self, map: Map, f: F) -> MapBuilder<WithInput> {
        let Map { name, projection, filter, order_by, limit, schema, input, .. } = map;
        let builder = self.name(name)
            .with_iter(schema.into_iter().zip(projection).map(|(field, expr)| (field.name().to_string(), f(field.name(), expr))))
            .input(input);
        // Filter
        let builder = filter.into_iter().fold(builder, |b, f|b.filter(f));
        // Order by
        let builder = order_by.into_iter().fold(builder, |b, o|b.order_by(o.expr, o.asc));
        // Limit
        let builder = limit.into_iter().fold(builder, |b, l|b.limit(l));
        builder
    }

    pub fn input<R: Into<Rc<Relation>>>(self, input: R) -> MapBuilder<WithInput> {
        MapBuilder {
            name: self.name,
            split: self.split,
            limit: self.limit,
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
        let Map { name, projection, filter, order_by, limit, schema, input, .. } = map;
        let builder = self.name(name)
            .with_iter(schema.into_iter().zip(projection).map(|(field, expr)| (field.name().to_string(), expr)))
            .input(input);
        // Filter
        let builder = filter.into_iter().fold(builder, |b, f|b.filter(f));
        // Order by
        let builder = order_by.into_iter().fold(builder, |b, o|b.order_by(o.expr, o.asc));
        // Limit
        let builder = limit.into_iter().fold(builder, |b, l|b.limit(l));
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
                Some(reduce) => Rc::new(
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
        self.split = self.split.and(Split::group_by(expr.into()).into());
        self
    }

    pub fn input<R: Into<Rc<Relation>>>(self, input: R) -> ReduceBuilder<WithInput> {
        ReduceBuilder {
            name: self.name,
            split: self.split,
            input: WithInput(input.into()),
        }
    }

    /// Initialize a builder with filtered existing reduce
    pub fn filter_with<P: Fn(&str) -> bool>(self, reduce: Reduce, predicate: P) -> ReduceBuilder<WithInput> {
        let Reduce { name, aggregate, group_by, schema, input, .. } = reduce;
        let builder = self.name(name)
            .with_iter(schema.into_iter().zip(aggregate).filter_map(|(field, expr)| predicate(field.name()).then_some((field.name().to_string(), expr))))
            .input(input);
        // Group by
        let builder = group_by.into_iter().fold(builder, |b, g|b.group_by(g));
        builder
    }

    /// Add a group by column
    pub fn with_group_by_column<S: Into<String>>(mut self, column: S) -> Self {
        let name = column.into();
        let expr = Expr::col(name.clone());
        self.split = self.split.and(Split::group_by(expr.clone()).into());
        self.split = self.split.and(split::Reduce::new(vec![(name, expr)], vec![], None).into());
        self
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
        self.split = self.split.and(Split::from((name.into(), expr)));
        self
    }
}

impl<RequireInput> With<Reduce, ReduceBuilder<WithInput>> for ReduceBuilder<RequireInput> {
    fn with(self, reduce: Reduce) -> ReduceBuilder<WithInput> {
        let Reduce { name, aggregate, group_by, schema, input, .. } = reduce;
        let builder = self.name(name)
            .with_iter(schema.into_iter().zip(aggregate).map(|(field, expr)| (field.name().to_string(), expr)))
            .input(input);
        // Group by
        let builder = group_by.into_iter().fold(builder, |b, g|b.group_by(g));
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
                Some(map) => Rc::new(
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
                reduce.named_exprs,
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
    pub fn name<S: Into<String>>(
        mut self,
        name: S,
    ) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn inner(mut self) -> Self {
        self.operator = Some(JoinOperator::Inner(JoinConstraint::Natural));
        self
    }

    pub fn left_outer(mut self) -> Self {
        self.operator = Some(JoinOperator::LeftOuter(JoinConstraint::Natural));
        self
    }

    pub fn right_outer(mut self) -> Self {
        self.operator = Some(JoinOperator::RightOuter(JoinConstraint::Natural));
        self
    }

    pub fn full_outer(mut self) -> Self{
        self.operator = Some(JoinOperator::FullOuter(JoinConstraint::Natural));
        self
    }

    pub fn cross(mut self) -> Self {
        self.operator = Some(JoinOperator::Cross);
        self
    }

    pub fn on(mut self, expr: Expr) -> Self {
        self.operator = match self.operator {
            Some(JoinOperator::Inner(_)) => Some(JoinOperator::Inner(JoinConstraint::On(expr))),
            Some(JoinOperator::LeftOuter(_)) => {
                Some(JoinOperator::LeftOuter(JoinConstraint::On(expr)))
            }
            Some(JoinOperator::RightOuter(_)) => {
                Some(JoinOperator::RightOuter(JoinConstraint::On(expr)))
            }
            Some(JoinOperator::FullOuter(_)) => {
                Some(JoinOperator::FullOuter(JoinConstraint::On(expr)))
            }
            Some(JoinOperator::Cross) => Some(JoinOperator::Cross),
            None => Some(JoinOperator::Inner(JoinConstraint::On(expr))),
        };
        self
    }

    pub fn using<I: Into<Identifier>>(
        mut self,
        using: I,
    ) -> Self {
        let using: Identifier = using.into();
        self.operator = match self.operator {
            Some(JoinOperator::Inner(JoinConstraint::Using(mut identifiers))) => {
                identifiers.push(using);
                Some(JoinOperator::Inner(JoinConstraint::Using(identifiers)))
            }
            Some(JoinOperator::LeftOuter(JoinConstraint::Using(mut identifiers))) => {
                identifiers.push(using);
                Some(JoinOperator::LeftOuter(JoinConstraint::Using(identifiers)))
            }
            Some(JoinOperator::RightOuter(JoinConstraint::Using(mut identifiers))) => {
                identifiers.push(using);
                Some(JoinOperator::RightOuter(JoinConstraint::Using(identifiers)))
            }
            Some(JoinOperator::FullOuter(JoinConstraint::Using(mut identifiers))) => {
                identifiers.push(using);
                Some(JoinOperator::FullOuter(JoinConstraint::Using(identifiers)))
            }
            Some(JoinOperator::Inner(_)) => {
                Some(JoinOperator::Inner(JoinConstraint::Using(vec![using])))
            }
            Some(JoinOperator::LeftOuter(_)) => {
                Some(JoinOperator::LeftOuter(JoinConstraint::Using(vec![using])))
            }
            Some(JoinOperator::RightOuter(_)) => {
                Some(JoinOperator::RightOuter(JoinConstraint::Using(vec![using])))
            }
            Some(JoinOperator::FullOuter(_)) => {
                Some(JoinOperator::FullOuter(JoinConstraint::Using(vec![using])))
            }
            Some(JoinOperator::Cross) => Some(JoinOperator::Cross),
            None => Some(JoinOperator::Inner(JoinConstraint::Using(vec![using]))),
        };
        self
    }

    /// Set directly the full JOIN operator
    pub fn operator(
        mut self,
        operator: JoinOperator,
    ) -> Self {
        self.operator = Some(operator);
        self
    }

    pub fn left<R: Into<Rc<Relation>>>(
        self,
        input: R,
    ) -> JoinBuilder<WithInput, RequireRightInput> {
        JoinBuilder {
            name: self.name,
            operator: self.operator,
            left: WithInput(input.into()),
            right: self.right,
        }
    }

    pub fn right<R: Into<Rc<Relation>>>(
        self,
        input: R,
    ) -> JoinBuilder<RequireLeftInput, WithInput> {
        JoinBuilder {
            name: self.name,
            operator: self.operator,
            left: self.left,
            right: WithInput(input.into()),
        }
    }
}

impl Ready<Join> for JoinBuilder<WithInput, WithInput> {
    type Error = Error;

    fn try_build(self) -> Result<Join> {
        let name = self
            .name
            .clone()
            .unwrap_or(namer::name_from_content(JOIN, &self));
        let left_names = self
            .left
            .0
            .schema()
            .iter()
            .map(|field| namer::name_from_content(FIELD, &(&self.left.0, &field)))
            .collect();
        let right_names = self
            .right
            .0
            .schema()
            .iter()
            .map(|field| namer::name_from_content(FIELD, &(&self.right.0, &field)))
            .collect();
        let operator = self
            .operator
            .unwrap_or(JoinOperator::Inner(JoinConstraint::Natural));
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
    pub fn name<S: Into<String>>(
        mut self,
        name: S,
    ) -> Self {
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
    pub fn operator(
        mut self,
        operator: SetOperator,
    ) -> Self {
        self.operator = Some(operator);
        self
    }

    /// Set directly the SetQuantifier
    pub fn quantifier(
        mut self,
        quantifier: SetQuantifier,
    ) -> Self {
        self.quantifier = Some(quantifier);
        self
    }

    pub fn left<R: Into<Rc<Relation>>>(
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

    pub fn right<R: Into<Rc<Relation>>>(
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

impl Ready<Set> for SetBuilder<WithInput, WithInput> {
    type Error = Error;

    fn try_build(self) -> Result<Set> {
        let name = self
            .name
            .clone()
            .unwrap_or(namer::name_from_content(SET, &self));
        let names = self.left.0.schema().iter().zip(self.right.0.schema().iter())
            .map(|(left_field, right_field)| if left_field.name()==right_field.name() {
                left_field.name().to_string()
            } else {
                namer::name_from_content(FIELD, &(&self.left.0, &self.right.0, left_field, right_field))
            })
            .collect();
        let operator = self
            .operator
            .unwrap_or(SetOperator::Union);
        let quantifier = self
            .quantifier
            .unwrap_or(SetQuantifier::None);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DataType;

    #[test]
    fn test_map_building() {
        let table: Relation = Relation::table().name("table")
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
            ).build();
        println!("Table = {table}");
        let map: Relation =  Relation::map()
            .with(("A", Expr::col("a")))
            .with(("B", Expr::col("b")))
            .input(table).build();
        println!("Map = {map}");
        let reduce: Relation =  Relation::reduce()
            .with(("S", Expr::sum(Expr::col("A"))))
            .with_group_by_column("B")
            .input(map).build();
        println!("Reduce = {reduce}");
    }
}