use std::{hash::Hash, rc::Rc};

use super::{
    Error, Join, JoinConstraint, JoinOperator, Map, OrderBy, Reduce, Relation, Result, Schema,
    Table, Variant,
};
use crate::{
    builder::{Ready, With},
    data_type::Integer,
    expr::{self, Expr, Identifier, Split},
    namer::{self, FIELD, JOIN, MAP, REDUCE},
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
    pub fn new() -> TableBuilder<WithoutSchema> {
        TableBuilder::default()
    }
}

impl<RequireSchema> TableBuilder<RequireSchema> {
    pub fn name<S: Into<String>>(mut self, name: S) -> TableBuilder<RequireSchema> {
        self.name = Some(name.into());
        self
    }

    pub fn size(mut self, size: i64) -> TableBuilder<RequireSchema> {
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
    pub fn new() -> MapBuilder<WithoutInput> {
        MapBuilder::default()
    }
}

impl<RequireInput> MapBuilder<RequireInput> {
    pub fn name<S: Into<String>>(mut self, name: S) -> MapBuilder<RequireInput> {
        self.name = Some(name.into());
        self
    }

    pub fn split<S: Into<Split>>(mut self, split: S) -> MapBuilder<RequireInput> {
        self.split = split.into();
        self
    }

    // TODO filter should maybe be possible on aggregates
    pub fn filter(mut self, filter: Expr) -> MapBuilder<RequireInput> {
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
    pub fn order_by(mut self, expr: Expr, asc: bool) -> MapBuilder<RequireInput> {
        self.split = self.split.and(Split::order_by(expr, asc).into());
        self
    }

    /// Add a group by
    pub fn group_by(mut self, expr: Expr) -> MapBuilder<RequireInput> {
        self.split = self
            .split
            .map_last_reduce(|reduce| reduce.and(Split::group_by(expr)));
        self
    }

    pub fn limit(mut self, limit: usize) -> MapBuilder<RequireInput> {
        self.limit = Some(limit.into());
        self
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
    pub fn new() -> ReduceBuilder<WithoutInput> {
        ReduceBuilder::default()
    }
}

impl<RequireInput> ReduceBuilder<RequireInput> {
    pub fn name<S: Into<String>>(mut self, name: S) -> ReduceBuilder<RequireInput> {
        self.name = Some(name.into());
        self
    }

    pub fn split<S: Into<Split>>(mut self, split: S) -> ReduceBuilder<RequireInput> {
        self.split = split.into();
        self
    }

    pub fn group_by<E: Into<Expr>>(mut self, expr: E) -> ReduceBuilder<RequireInput> {
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
    pub fn new() -> JoinBuilder<WithoutInput, WithoutInput> {
        JoinBuilder::default()
    }
}

impl<RequireLeftInput, RequireRightInput> JoinBuilder<RequireLeftInput, RequireRightInput> {
    pub fn name<S: Into<String>>(
        mut self,
        name: S,
    ) -> JoinBuilder<RequireLeftInput, RequireRightInput> {
        self.name = Some(name.into());
        self
    }

    pub fn inner(mut self) -> JoinBuilder<RequireLeftInput, RequireRightInput> {
        self.operator = Some(JoinOperator::Inner(JoinConstraint::Natural));
        self
    }

    pub fn left_outer(mut self) -> JoinBuilder<RequireLeftInput, RequireRightInput> {
        self.operator = Some(JoinOperator::LeftOuter(JoinConstraint::Natural));
        self
    }

    pub fn right_outer(mut self) -> JoinBuilder<RequireLeftInput, RequireRightInput> {
        self.operator = Some(JoinOperator::RightOuter(JoinConstraint::Natural));
        self
    }

    pub fn full_outer(mut self) -> JoinBuilder<RequireLeftInput, RequireRightInput> {
        self.operator = Some(JoinOperator::FullOuter(JoinConstraint::Natural));
        self
    }

    pub fn cross(mut self) -> JoinBuilder<RequireLeftInput, RequireRightInput> {
        self.operator = Some(JoinOperator::Cross);
        self
    }

    pub fn on(mut self, expr: Expr) -> JoinBuilder<RequireLeftInput, RequireRightInput> {
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
    ) -> JoinBuilder<RequireLeftInput, RequireRightInput> {
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
    ) -> JoinBuilder<RequireLeftInput, RequireRightInput> {
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
