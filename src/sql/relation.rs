//! This file provides tools for converting a ast::Statement
//! into the corresponding Qrlew Relation.
//! Example: `Expr::try_from(sql_parser_statement)`

use super::{
    query_names::{IntoQueryNamesVisitor, QueryNames},
    visitor::Visitor,
    Error, Result,
};
use crate::{
    ast,
    builder::{Ready, With, WithIterator, WithoutContext},
    dialect::{Dialect, GenericDialect},
    expr::{Expr, Identifier, Split, Reduce},
    hierarchy::{Hierarchy, Path},
    namer::{self, FIELD},
    parser::Parser,
    relation::{
        Join, JoinOperator, MapBuilder, Relation, SetOperator, SetQuantifier,
        Variant as _, WithInput,
        LEFT_INPUT_NAME, RIGHT_INPUT_NAME
    },
    tokenizer::Tokenizer,
    visitor::{Acceptor, Dependencies, Visited},
    types::And, display::Dot
};
use itertools::Itertools;
use std::{
    convert::TryFrom,
    iter::{once, Iterator},
    result,
    str::FromStr,
    sync::Arc,
    ops::Deref
};

/*
Before we visit queries to build Relations we must collect the namespaces of all subqueries to map the right names to the right Relations.
This is done in the query_names module.
 */

/// A visitor for AST queries conversion into relations
/// The Hierarchy of Relations is the context in which the query is converted, typically the list of tables with their Path
/// The QueryNames is the map of sub-query referrenced by their names, so that links can be unfolded
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct TryIntoRelationVisitor<'a>(&'a Hierarchy<Arc<Relation>>, QueryNames<'a>);

impl<'a> TryIntoRelationVisitor<'a> {
    fn new(relations: &'a Hierarchy<Arc<Relation>>, query_names: QueryNames<'a>) -> Self {
        TryIntoRelationVisitor(relations, query_names)
    }
}

//TODO Add columns as (alias.col -> field, name.col -> field) for all fields

// A few useful conversions

impl From<ast::SetOperator> for SetOperator {
    fn from(value: ast::SetOperator) -> Self {
        match value {
            ast::SetOperator::Union => SetOperator::Union,
            ast::SetOperator::Except => SetOperator::Except,
            ast::SetOperator::Intersect => SetOperator::Intersect,
        }
    }
}

impl From<ast::SetQuantifier> for SetQuantifier {
    fn from(value: ast::SetQuantifier) -> Self {
        match value {
            ast::SetQuantifier::All => SetQuantifier::All,
            ast::SetQuantifier::Distinct => SetQuantifier::Distinct,
            ast::SetQuantifier::None => SetQuantifier::None,
            ast::SetQuantifier::ByName => SetQuantifier::ByName,
            ast::SetQuantifier::AllByName => SetQuantifier::AllByName,
            ast::SetQuantifier::DistinctByName => SetQuantifier::DistinctByName,
        }
    }
}

// This is RelationWithColumns from_xxx method

/// A struct to hold Relations with column mapping in the FROM
struct RelationWithColumns(Arc<Relation>, Hierarchy<Identifier>);

impl RelationWithColumns {
    fn new(relation: Arc<Relation>, columns: Hierarchy<Identifier>) -> Self {
        RelationWithColumns(relation, columns)
    }
}

/// A struct to hold the query being visited and its Relations
struct VisitedQueryRelations<'a>(
    Hierarchy<Arc<Relation>>,
    Visited<'a, ast::Query, Result<Arc<Relation>>>,
);

impl<'a> VisitedQueryRelations<'a> {
    fn new(
        try_into_relation_visitor: &TryIntoRelationVisitor<'a>,
        query: &'a ast::Query,
        visited: Visited<'a, ast::Query, Result<Arc<Relation>>>,
    ) -> Self {
        let TryIntoRelationVisitor(relations, query_names) = try_into_relation_visitor;
        let mut relations: Hierarchy<Arc<Relation>> = (*relations).clone();
        relations.extend(
            query_names
                .name_referred(query)
                .map(|(name, referred)| (name.clone(), visited.get(referred).clone().unwrap())),
        );
        VisitedQueryRelations(relations, visited)
    }

    /// Convert a TableFactor into a RelationWithColumns
    fn try_from_table_factor(
        &self,
        table_factor: &'a ast::TableFactor,
    ) -> Result<RelationWithColumns> {
        let VisitedQueryRelations(relations, visited) = self;
        // Process the table_factor

        match &table_factor {
            ast::TableFactor::Table { name, alias, .. } => {
                let relation = relations
                    .get(&name.cloned())
                    .cloned()
                    .ok_or(Error::parsing_error(format!("Unknown table: {name}")))?;
                let name = alias
                    .clone()
                    .map(|a| a.name.cloned())
                    .unwrap_or(name.cloned());
                let columns: Hierarchy<Identifier> = relation
                    .schema()
                    .iter()
                    .map(|f| {
                        (
                            name.cloned()
                                .into_iter()
                                .chain(once(f.name().to_string()))
                                .collect_vec(),
                            [f.name()].into(),
                        )
                    })
                    .collect();
                Ok(RelationWithColumns::new(relation, columns))
            }
            ast::TableFactor::Derived {
                subquery, alias, ..
            } => {
                let relation = visited.get(subquery).clone()?;
                let name = alias
                    .clone()
                    .map(|a| a.name.cloned())
                    .unwrap_or(relation.name().cloned());
                let columns: Hierarchy<Identifier> = relation
                    .schema()
                    .iter()
                    .map(|f| {
                        (
                            name.cloned()
                                .into_iter()
                                .chain(once(f.name().to_string()))
                                .collect_vec(),
                            [f.name()].into(),
                        )
                    })
                    .collect();
                Ok(RelationWithColumns::new(relation, columns))
            }
            _ => todo!(),
        }
    }

    fn try_from_join_constraint_with_columns(
        &self,
        join_constraint: &ast::JoinConstraint,
        columns: &'a Hierarchy<Identifier>,
    ) -> Result<Expr> {
        Ok(match join_constraint {
            ast::JoinConstraint::On(expr) => expr.with(columns).try_into()?,
            ast::JoinConstraint::Using(idents) => { // the "Using (id)" condition is equivalent to "ON _LEFT_.id = _RIGHT_.id"
                Expr::and_iter(
                    idents.into_iter()
                    .map(|id| Expr::eq(
                        Expr::Column(Identifier::from(vec![LEFT_INPUT_NAME.to_string(), id.value.to_string()])),
                        Expr::Column(Identifier::from(vec![RIGHT_INPUT_NAME.to_string(), id.value.to_string()])),
                    ))
                )
            },
            ast::JoinConstraint::Natural => { // the NATURAL condition is equivalent to a "ON _LEFT_.col1 = _RIGHT_.col1 AND _LEFT_.col2 = _RIGHT_.col2" where col1, col2... are the columns present in both tables
                let tables = columns.iter()
                .map(|(k, _)| k.iter().take(k.len() - 1).map(|s| s.to_string()).collect::<Vec<_>>())
                .dedup()
                .collect::<Vec<_>>();
                assert_eq!(tables.len(), 2);
                let columns_1 = columns.filter(tables[0].as_slice());
                let columns_2 = columns.filter(tables[1].as_slice());
                let columns_1 = columns_1
                .iter()
                .map(|(k, _)| k.last().unwrap())
                .collect::<Vec<_>>();
                let columns_2 = columns_2
                .iter()
                .map(|(k, _)| k.last().unwrap())
                .collect::<Vec<_>>();

                Expr::and_iter(
                    columns_1
                    .iter()
                    .filter_map(|col| columns_2.contains(&col).then_some(col))
                    .map(|id| Expr::eq(
                        Expr::Column(Identifier::from(vec![LEFT_INPUT_NAME.to_string(), id.to_string()])),
                        Expr::Column(Identifier::from(vec![RIGHT_INPUT_NAME.to_string(), id.to_string()]))
                    ))
                )
            },
            ast::JoinConstraint::None => todo!(),
        })
    }

    fn try_from_join_operator_with_columns(
        &self,
        join_operator: &ast::JoinOperator,
        columns: &'a Hierarchy<Identifier>,
    ) -> Result<JoinOperator> {
        match join_operator {
            ast::JoinOperator::Inner(join_constraint) => Ok(JoinOperator::Inner(
                self.try_from_join_constraint_with_columns(join_constraint, columns)?
            )),
            ast::JoinOperator::LeftOuter(join_constraint) => Ok(JoinOperator::LeftOuter(
                self.try_from_join_constraint_with_columns(join_constraint, columns)?,
            )),
            ast::JoinOperator::RightOuter(join_constraint) => Ok(JoinOperator::RightOuter(
                self.try_from_join_constraint_with_columns(join_constraint, columns)?,
            )),
            ast::JoinOperator::FullOuter(join_constraint) => Ok(JoinOperator::FullOuter(
                self.try_from_join_constraint_with_columns(join_constraint, columns)?,
            )),
            ast::JoinOperator::CrossJoin => Ok(JoinOperator::Cross),
            _ => todo!(), //TODO implement other JOIN later
        }
    }

    /// Convert a TableWithJoins into a RelationWithColumns
    fn try_from_table_with_joins(
        &self,
        table_with_joins: &'a ast::TableWithJoins,
    ) -> Result<RelationWithColumns> {
        // Process the relation
        // Then the JOIN if needed
        let result = table_with_joins.joins.iter().fold(
            self.try_from_table_factor(&table_with_joins.relation),
            |left, ast_join| {
                let RelationWithColumns(left_relation, left_columns) = left?;
                let RelationWithColumns(right_relation, right_columns) =
                    self.try_from_table_factor(&ast_join.relation)?;
                let left_columns = left_columns.map(|i| {
                    let mut v = vec![Join::left_name().to_string()];
                    v.extend(i.to_vec());
                    v.into()
                });
                let right_columns = right_columns.map(|i| {
                    let mut v = vec![Join::right_name().to_string()];
                    v.extend(i.to_vec());
                    v.into()
                });
                let all_columns = left_columns.with(right_columns);
                let operator = self.try_from_join_operator_with_columns(
                    &ast_join.join_operator,
                    // &all_columns.filter_map(|i| Some(i.split_last().ok()?.0)),//TODO remove this
                    &all_columns,
                )?;
                // We build a Join
                let join: Join = Relation::join()
                    .operator(operator)
                    .left(left_relation)
                    .right(right_relation)
                    .build();

                // We collect column mapping inputs should map to new names (hence the inversion)
                let new_columns: Hierarchy<Identifier> =
                    join.field_inputs().map(|(f, i)| (i, f.into())).collect();
                let composed_columns = all_columns.and_then(new_columns.clone());

                // If the join contraint is of type "USING" or "NATURAL", add a map to coalesce the duplicate columns
                let relation = match &ast_join.join_operator {
                    ast::JoinOperator::Inner(ast::JoinConstraint::Using(v))
                    | ast::JoinOperator::LeftOuter(ast::JoinConstraint::Using(v))
                    | ast::JoinOperator::RightOuter(ast::JoinConstraint::Using(v))
                    | ast::JoinOperator::FullOuter(ast::JoinConstraint::Using(v)) => {
                        join.remove_duplicates_and_coalesce(
                            v.into_iter().map(|id| id.value.to_string()).collect(),
                            &new_columns
                        )
                    },
                    ast::JoinOperator::Inner(ast::JoinConstraint::Natural)
                    | ast::JoinOperator::LeftOuter(ast::JoinConstraint::Natural)
                    | ast::JoinOperator::RightOuter(ast::JoinConstraint::Natural)
                    | ast::JoinOperator::FullOuter(ast::JoinConstraint::Natural) => {
                        let v = join.left().fields()
                            .into_iter()
                            .filter_map(|f| join.right().schema().field(f.name()).is_ok().then_some(f.name().to_string()))
                            .collect();
                        join.remove_duplicates_and_coalesce(v,&new_columns)
                    },
                    ast::JoinOperator::LeftSemi(_) => todo!(),
                    ast::JoinOperator::RightSemi(_) => todo!(),
                    ast::JoinOperator::LeftAnti(_) => todo!(),
                    ast::JoinOperator::RightAnti(_) => todo!(),
                    _ => Relation::from(join),
                };

                // We should compose hierarchies
                Ok(RelationWithColumns::new(Arc::new(relation), composed_columns))
            },
        );
        result
    }

    /// Convert a Vec<TableWithJoins> into a Relation
    fn try_from_tables_with_joins(
        &self,
        tables_with_joins: &'a Vec<ast::TableWithJoins>,
    ) -> Result<RelationWithColumns> {
        // TODO consider more tables
        // For now, only consider the first element
        // It should eventually be cross joined as described in: https://www.postgresql.org/docs/current/queries-table-expressions.html
        self.try_from_table_with_joins(&tables_with_joins[0])
    }

    /// Build a relation from the
    fn try_from_select_items_selection_and_group_by(
        &self,
        names: &'a Hierarchy<String>,
        select_items: &'a [ast::SelectItem],
        selection: &'a Option<ast::Expr>,
        group_by: &'a ast::GroupByExpr,
        from: Arc<Relation>,
        having: &'a Option<ast::Expr>,
        distinct: &'a Option<ast::Distinct>,
    ) -> Result<Arc<Relation>> {
        // Collect all expressions with their aliases
        let mut named_exprs: Vec<(String, Expr)> = vec![];
        // Columns from names
        let columns = &names.map(|s| s.clone().into());
        for select_item in select_items {
            match select_item {
                ast::SelectItem::UnnamedExpr(expr) => named_exprs.push((
                    match expr {
                        // Pull the original name for implicit aliasing
                        ast::Expr::Identifier(ident) => ident.value.clone(),
                        ast::Expr::CompoundIdentifier(idents) => {
                            idents.last().unwrap().value.clone()
                        }
                        expr => namer::name_from_content(FIELD, &expr),
                    },
                    Expr::try_from(expr.with(columns))?,
                )),
                ast::SelectItem::ExprWithAlias { expr, alias } => {
                    named_exprs.push((alias.clone().value, Expr::try_from(expr.with(columns))?))
                }
                ast::SelectItem::QualifiedWildcard(_, _) => todo!(),
                ast::SelectItem::Wildcard(_) => {
                    for field in from.schema().iter() {
                        named_exprs.push((field.name().to_string(), Expr::col(field.name())))
                    }
                }
            }
        }
        // Prepare the GROUP BY
        let group_by  = match group_by {
            ast::GroupByExpr::All => todo!(),
            ast::GroupByExpr::Expressions(group_by_exprs) => group_by_exprs
                .iter()
                .map(|e| e.with(columns).try_into())
                .collect::<Result<Vec<Expr>>>()?,
        };
        // If the GROUP BY contains aliases, then replace them by the corresponding expression in `named_exprs`.
        // Note that we mimic postgres behaviour and support only GROUP BY alias column (no other expressions containing aliases are allowed)
        // The aliases cannot be used in HAVING
        let group_by = group_by.into_iter()
            .map(|x| match &x {
                Expr::Column(c) if columns.get_key_value(&c).is_none() && c.len() == 1 => {
                    named_exprs
                        .iter()
                        .find(|&(name, _)| name == &c[0])
                        .map(|(_, expr)| expr.clone())
                        .unwrap_or(x)
                },
                _ => x
            })
            .collect::<Vec<_>>();
        // Add the having in named_exprs
        let having = if let Some(expr) = having {
            let having_name = namer::name_from_content(FIELD, &expr);
            let mut expr = Expr::try_from(expr.with(columns))?;
            let columns = named_exprs
                .iter()
                .map(|(s, x)| (Expr::col(s.to_string()), x.clone()))
                .collect();
            expr = expr.replace(columns).0;
            let columns = group_by
                .iter()
                .filter_map(|x| {
                    matches!(x, Expr::Column(_)).then_some((x.clone(), Expr::first(x.clone())))
                })
                .collect();
            expr = expr.replace(columns).0;
            named_exprs.push((having_name.clone(), expr));
            Some(having_name)
        } else {
            None
        };
        // Build the Map or Reduce based on the type of split
        // If group_by is non-empty, start with them so that aggregations can take them into account
        let split = if group_by.is_empty() {
            Split::from_iter(named_exprs)
        } else {
            let group_by = group_by.clone().into_iter()
            .fold(Split::Reduce(Reduce::default()),
            |s, expr| s.and(Split::Reduce(Split::group_by(expr)))
            );
            named_exprs.into_iter()
            .fold(group_by,
                |s, named_expr| s.and(named_expr.into())
            )
        };
        // Prepare the WHERE
        let filter: Option<Expr> = selection
            .as_ref()
            .map(|e| e.with(columns).try_into())
            .map_or(Ok(None), |r| r.map(Some))?;
        // Build a Relation
        let mut relation: Relation = match split {
            Split::Map(map) => {
                let builder = Relation::map().split(map);
                let builder = filter.into_iter().fold(builder, |b, e| b.filter(e));
                let builder = group_by.into_iter().fold(builder, |b, e| b.group_by(e));
                builder.input(from).build()
            }
            Split::Reduce(reduce) => {
                let builder = Relation::reduce().split(reduce);
                let builder = filter.into_iter().fold(builder, |b, e| b.filter(e));
                let builder = group_by.into_iter().fold(builder, |b, e| b.group_by(e));
                builder.input(from).build()
            }
        };
        if let Some(h) = having {
            relation = Relation::map()
                .with_iter(
                    relation
                        .fields()
                        .iter()
                        .filter_map(|f| (f.name() != h).then_some((f.name(), Expr::col(f.name()))))
                        .collect::<Vec<_>>(),
                )
                .filter(Expr::col(h))
                .input(relation)
                .build();
        }
        if let Some(distinct) = distinct {
            if matches!(distinct, ast::Distinct::On(_)) {
                return Err(Error::other("DISTINCT IN is not supported"));
            }
            relation = relation.distinct()
        }
        Ok(Arc::new(relation))
    }

    /// Convert a Select into a Relation
    fn try_from_select(&self, select: &'a ast::Select) -> Result<RelationWithColumns> {
        let ast::Select {
            projection,
            from,
            selection,
            group_by,
            distinct,
            top,
            into,
            lateral_views,
            cluster_by,
            distribute_by,
            sort_by,
            having,
            named_window,
            qualify,
        } = select;
        if top.is_some() {
            return Err(Error::other("TOP is not supported"));
        }
        if into.is_some() {
            return Err(Error::other("INTO is not supported"));
        }
        if !lateral_views.is_empty() {
            return Err(Error::other("LATERAL VIEWS are not supported"));
        }
        if !cluster_by.is_empty() {
            return Err(Error::other("CLUSTER BY is not supported"));
        }
        if !distribute_by.is_empty() {
            return Err(Error::other("DISTRIBUTE BY is not supported"));
        }
        if !sort_by.is_empty() {
            return Err(Error::other("SORT BY is not supported"));
        }
        if !named_window.is_empty() {
            return Err(Error::other("NAMED WINDOW is not supported"));
        }
        if qualify.is_some() {
            return Err(Error::other("QUALIFY is not supported"));
        }
        let RelationWithColumns(from, columns) = self.try_from_tables_with_joins(from)?;
        let relation = self.try_from_select_items_selection_and_group_by(
            &columns.filter_map(|i| Some(i.split_last().ok()?.0)),
            projection,
            selection,
            group_by,
            from,
            having,
            distinct
        )?;
        Ok(RelationWithColumns::new(relation, columns))
    }

    fn try_from_limit(&self, limit: &'a ast::Expr) -> Result<usize> {
        if let ast::Expr::Value(ast::Value::Number(number, false)) = limit {
            Ok(usize::from_str(number)?)
        } else {
            Err(Error::parsing_error(limit))
        }
    }

    fn try_from_offset(&self, offset: &'a ast::Offset) -> Result<usize> {
        if let ast::Expr::Value(ast::Value::Number(number, false)) = &offset.value {
            Ok(usize::from_str(&number)?)
        } else {
            Err(Error::parsing_error(offset))
        }
    }

    /// Convert a Query into a Relation
    fn try_from_query(&self, query: &'a ast::Query) -> Result<Arc<Relation>> {
        let ast::Query {
            body,
            order_by,
            limit,
            offset,
            ..
        } = query;
        match body.as_ref() {
            ast::SetExpr::Select(select) => {
                let RelationWithColumns(relation, columns) =
                    self.try_from_select(select.as_ref())?;
                if order_by.is_empty() && limit.is_none() && offset.is_none() {
                    Ok(relation)
                } else {
                    // Build a relation with ORDER BY and LIMIT if needed
                    let relation_builder = Relation::map();
                    // We add all the columns
                    let relation_builder = relation
                        .schema()
                        .iter()
                        .fold(relation_builder, |builder, field| {
                            builder.with((field.name(), Expr::col(field.name())))
                        });
                    // Add input
                    let relation_builder = relation_builder.input(relation);
                    // Add ORDER BYs
                    let relation_builder: Result<MapBuilder<WithInput>> = order_by.iter().fold(
                        Ok(relation_builder),
                        |builder, ast::OrderByExpr { expr, asc, .. }| {
                            Ok(builder?
                                .order_by(expr.with(&columns).try_into()?, asc.unwrap_or(true)))
                        },
                    );
                    // Add LIMITs
                    let relation_builder: Result<MapBuilder<WithInput>> =
                        limit.iter().fold(relation_builder, |builder, limit| {
                            Ok(builder?.limit(self.try_from_limit(limit)?))
                        });
                    // Add OFFSET
                    let relation_builder: Result<MapBuilder<WithInput>> =
                    offset.iter().fold(relation_builder, |builder, offset| {
                        Ok(builder?.offset(self.try_from_offset(offset)?))
                    });
                    // Build a relation with ORDER BY and LIMIT
                    Ok(Arc::new(relation_builder?.try_build()?))
                }
            }
            ast::SetExpr::SetOperation {
                op,
                set_quantifier,
                left,
                right,
            } => match (left.as_ref(), right.as_ref()) {
                (ast::SetExpr::Select(left_select), ast::SetExpr::Select(right_select)) => {
                    let RelationWithColumns(left_relation, _left_columns) =
                        self.try_from_select(left_select.as_ref())?;
                    let RelationWithColumns(right_relation, _right_columns) =
                        self.try_from_select(right_select.as_ref())?;
                    let relation_builder = Relation::set()
                        .operator(op.clone().into())
                        .quantifier(set_quantifier.clone().into())
                        .left(left_relation)
                        .right(right_relation);
                    // Build a Relation from set operation
                    Ok(Arc::new(relation_builder.try_build()?))
                }
                _ => panic!("We only support set operations over SELECTs"),
            },
            _ => todo!(),
        }
    }
}

impl<'a> Visitor<'a, Result<Arc<Relation>>> for TryIntoRelationVisitor<'a> {
    fn dependencies(&self, acceptor: &'a ast::Query) -> Dependencies<'a, ast::Query> {
        let TryIntoRelationVisitor(_relations, query_names) = self;
        let mut dependencies = acceptor.dependencies();
        // Add subqueries from the body
        dependencies.extend(
            query_names
                .iter()
                .filter_map(|((query, _name), dependency)| {
                    if (*query) == acceptor {
                        dependency.clone()
                    } else {
                        None
                    }
                }),
        );
        dependencies
    }
    fn query(
        &self,
        query: &'a ast::Query,
        visited: Visited<'a, ast::Query, Result<Arc<Relation>>>,
    ) -> Result<Arc<Relation>> {
        // Pull relations accessible from this query
        let visited_query_relations = VisitedQueryRelations::new(self, query, visited);
        // Retrieve a relation before ORDER BY and LIMIT
        let relation = visited_query_relations.try_from_query(query)?;
        Ok(relation)
    }
}

/*
To convert a Query into a Relation, one need to pack it with a few named Relations to refer to
 */

/// A struct holding a query and a context for conversion to Relation
/// This is the main entrypoint of this module
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct QueryWithRelations<'a>(&'a ast::Query, &'a Hierarchy<Arc<Relation>>);

impl<'a> QueryWithRelations<'a> {
    pub fn new(query: &'a ast::Query, relations: &'a Hierarchy<Arc<Relation>>) -> Self {
        QueryWithRelations(query, relations)
    }

    pub fn query(&self) -> &ast::Query {
        self.0
    }

    pub fn relations(&self) -> &Hierarchy<Arc<Relation>> {
        self.1
    }
}

impl<'a> With<&'a Hierarchy<Arc<Relation>>, QueryWithRelations<'a>> for &'a ast::Query {
    fn with(self, input: &'a Hierarchy<Arc<Relation>>) -> QueryWithRelations<'a> {
        QueryWithRelations::new(self, input)
    }
}

impl<'a> TryFrom<QueryWithRelations<'a>> for Relation {
    type Error = Error;

    fn try_from(value: QueryWithRelations<'a>) -> result::Result<Self, Self::Error> {
        // Pull values from the object
        let QueryWithRelations(query, relations) = value;
        // Visit the query to get query names
        let query_names = query.accept(IntoQueryNamesVisitor);
        // Visit for conversion
        query
            .accept(TryIntoRelationVisitor::new(relations, query_names))
            .map(|r| r.as_ref().clone())
    }
}

/// A simple SQL query parser with dialect
pub fn parse_with_dialect<D: Dialect>(query: &str, dialect: D) -> Result<ast::Query> {
    let mut tokenizer = Tokenizer::new(&dialect, query);
    let tokens = tokenizer.tokenize()?;
    let mut parser = Parser::new(&dialect).with_tokens(tokens);
    let expr = parser.parse_query()?;
    Ok(expr)
}

/// A simple SQL query parser to test the code
pub fn parse(query: &str) -> Result<ast::Query> {
    parse_with_dialect(query, GenericDialect)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use colored::Colorize;

    use super::*;
    use crate::{
        builder::Ready,
        data_type::{DataType, DataTyped, Variant},
        display::Dot,
        relation::schema::Schema,
        io::{Database, postgresql}
    };

    #[test]
    fn test_map_from_query() {
        let query = parse("SELECT exp(table.a) FROM schema.table").unwrap();
        let schema: Schema = vec![
            ("a", DataType::float()),
            ("b", DataType::float_interval(-2., 2.)),
            ("c", DataType::float()),
            ("d", DataType::float_interval(0., 1.)),
        ]
        .into_iter()
        .collect();
        let table = Relation::table()
            .name("tab")
            .schema(schema.clone())
            .size(100)
            .build();
        let map = Relation::try_from(QueryWithRelations::new(
            &query,
            &Hierarchy::from([(["schema", "table"], Arc::new(table))]),
        ))
        .unwrap();
        print!("{}", map);
    }

    #[test]
    fn test_parse() {
        let query = parse("SELECT 2 * my_table.price FROM schema.table AS my_table").unwrap();
        //println!("\nquery: {:?}", query);
        println!("\n{}", query.to_string());
        let schema: Schema = vec![("price", DataType::float_interval(1., 4.))]
            .into_iter()
            .collect();
        let table = Relation::table()
            .name("tab")
            .schema(schema.clone())
            .size(100)
            .build();
        let map = Relation::try_from(QueryWithRelations::new(
            &query,
            &Hierarchy::from([(["schema", "table"], Arc::new(table))]),
        ))
        .unwrap();
        let query2 = &ast::Query::from(&map);
        //println!("\nquery2: {:?}", query2);
        println!("\n{}", query2.to_string());
        map.display_dot().unwrap();
    }

    #[test]
    fn test_parse_auto_join() {
        let query = parse("SELECT 2 * my_table.price + table2.price FROM schema.table AS my_table JOIN schema.table AS table2 ON my_table.id = table2.id").unwrap();
        //println!("\nquery: {:?}", query);
        println!("\n{}", query.to_string());
        let schema: Schema = vec![
            ("price", DataType::float_interval(1., 4.)),
            ("id", DataType::text()),
        ]
        .into_iter()
        .collect();
        let table = Relation::table()
            .name("tab")
            .schema(schema.clone())
            .size(100)
            .build();
        println!("Table = {:?}", table);
        let relation = Relation::try_from(QueryWithRelations::new(
            &query,
            &Hierarchy::from([(["schema", "table"], Arc::new(table))]),
        ))
        .unwrap();
        let query2 = &ast::Query::from(&relation);
        //println!("\nquery2: {:?}", query2);
        println!("\n{}", query2.to_string());
        relation.display_dot().unwrap();
    }

    fn complex_query() -> ast::Query {
        parse(r#"
            with view_1 as (select * from schema.table_1),
            view_2 as (select * from view_1)
            select 2*a, b+1 from (select view_2.c as a, right.d as b from (select * from view_2) LEFT OUTER JOIN schema.table_2 as right ON view_1.id = table_2.id);
            select * from table_1;
        "#).unwrap()
    }

    #[test]
    fn test_query_names() {
        let query_1 = parse("select * from table_1").unwrap();
        let query_2 = parse("select * from table_2").unwrap();
        let query_3 = parse("select * from table_3").unwrap();
        let query_4 = parse("select * from table_4").unwrap();
        let name_1 = ast::ObjectName(vec!["name_1".into()]);
        let name_2 = ast::ObjectName(vec!["name_2".into()]);
        let name_3 = ast::ObjectName(vec!["name_3".into()]);
        println!("query_1 = {}", query_1.to_string().blue());
        let mut query_names_1 = QueryNames::new();
        let mut query_names_2 = QueryNames::new();
        let mut query_names_3 = QueryNames::new();
        query_names_2.insert((&query_1, name_1), None);
        query_names_2.insert((&query_1, name_2), Some(&query_2));
        query_names_3.insert((&query_1, name_3.clone()), Some(&query_3));
        query_names_3.insert((&query_2, name_3.clone()), None);
        query_names_3.insert((&query_3, name_3.clone()), None);
        query_names_1.extend(query_names_2);
        query_names_1.extend(query_names_3);
        query_names_1.set(name_3.clone(), &query_4);
    }

    #[test]
    fn test_query_names_visitor() {
        let query = complex_query();
        println!("Query = {}", query.to_string().blue());
        let visitor = IntoQueryNamesVisitor;
        let query_names = query.accept(visitor);
        println!("{}", query_names);
    }

    const QUERIES: &[&str] = &["
        with view_1 as (select a, b from schema.table_1),
        view_2 as (select a, b from view_1)
        select 2*a, b+1 from (select view_2.c as a, right.d as b from (select a, b from view_2) LEFT OUTER JOIN schema.table_2 as right ON view_1.id = table_2.id);",
        "
        with view_1 as (select a,b,c,d from schema.table_1)
        select 2*a, b+1 from (select c as a, d as b from view_1) join table_2 on a=b;",
        "
        with view_1 as (select a,b,c,d from schema.table_1)
        select a from (select c as a, d as b from view_1) join table_2 on a=b;",
        "
        with view_1 as (select a,b,c,d from schema.table_1)
        select cos(0.1*view_1.b) as k, table_2.u as l, table_2.v, tab.a from (select c as a, d as b from view_1) as tab join table_2 on a=b join view_1 on a=b;",
        "
        with view_1 as (select table_1.a,b,c as s,d from schema.table_1),
        view_2 as (select 2*sum(tu.a) as a, count(tu.b) as b, sum(table_2.u) as u, 10+sum(table_2.v) as v from (select s as a, d as b from view_1) as tu join table_2 on a=b)
        select cos(0.1*ta.b) as cs, tb.b as l, tb.a from view_2 as ta left outer join table_1 as tb on ta.a=tb.b;",
        ];

    #[test]
    fn test_try_from_complex_query() {
        let query = parse(QUERIES[4]).unwrap();
        let schema_1: Schema = vec![
            ("a", DataType::float()),
            ("b", DataType::float_interval(-2., 2.)),
            ("c", DataType::float()),
            ("d", DataType::float_interval(0., 1.)),
        ]
        .into_iter()
        .collect();
        let schema_2: Schema = vec![
            ("u", DataType::float()),
            ("v", DataType::float_interval(-2., 2.)),
            ("w", DataType::float()),
            ("x", DataType::float_interval(0., 1.)),
        ]
        .into_iter()
        .collect();
        let table_1 = Relation::table()
            .name("tab_1")
            .schema(schema_1.clone())
            .size(100)
            .build();
        let table_2 = Relation::table()
            .name("tab_2")
            .schema(schema_2.clone())
            .size(100)
            .build();
        let relation = Relation::try_from(QueryWithRelations::new(
            &query,
            &Hierarchy::from([
                (["schema", "table_1"], Arc::new(table_1)),
                (["schema", "table_2"], Arc::new(table_2)),
            ]),
        ))
        .unwrap();
        println!("relation = {relation}");
        let q = ast::Query::from(&relation);
        println!("query = {q}");
        relation.display_dot();
    }

    #[test]
    fn test_try_from_simple_query() {
        let query = parse(
            "
            WITH view_1 as (select 1+sum(2*a), count(b) from table_1)
            SELECT * FROM view_1;
        ",
        )
        .unwrap();
        let schema_1: Schema = vec![
            ("a", DataType::float_interval(-1., 3.)),
            ("b", DataType::float_interval(-2., 2.)),
            ("c", DataType::float()),
            ("d", DataType::float_interval(0., 1.)),
        ]
        .into_iter()
        .collect();
        let schema_2: Schema = vec![
            ("u", DataType::float()),
            ("v", DataType::float_interval(-2., 2.)),
            ("w", DataType::float()),
            ("x", DataType::float_interval(0., 1.)),
        ]
        .into_iter()
        .collect();
        let table_1 = Relation::table()
            .name("tab_1")
            .schema(schema_1.clone())
            .size(100)
            .build();
        let table_2 = Relation::table()
            .name("tab_2")
            .schema(schema_2.clone())
            .size(100)
            .build();
        let relation = Relation::try_from(QueryWithRelations::new(
            &query,
            &Hierarchy::from([
                (["schema", "table_1"], Arc::new(table_1)),
                (["schema", "table_2"], Arc::new(table_2)),
            ]),
        ))
        .unwrap();
        println!("relation = {relation}");
        let q = ast::Query::from(&relation);
        println!("query = {q}");
        relation.display_dot().unwrap();
    }

    #[test]
    fn test_try_from_aggregate_query() {
        let query = parse(
            "
            SELECT 1+count(a) as c, sum(b+1) as s FROM table_1;
        ",
        )
        .unwrap();
        let schema_1: Schema = vec![
            ("a", DataType::float_interval(-1., 3.)),
            ("b", DataType::float_interval(-2., 2.)),
            ("c", DataType::float()),
            ("d", DataType::float_interval(0., 1.)),
        ]
        .into_iter()
        .collect();
        let table_1 = Relation::table()
            .name("tab_1")
            .schema(schema_1.clone())
            .size(100)
            .build();
        let relation = Relation::try_from(QueryWithRelations::new(
            &query,
            &Hierarchy::from([(["schema", "table_1"], Arc::new(table_1))]),
        ))
        .unwrap();
        println!("relation = {relation}");
        let q = ast::Query::from(&relation);
        println!("query = {q}");
        relation.display_dot().unwrap();
    }

    #[test]
    fn test_where() {
        let query = parse(
            "
            --SELECT 1+SUM(a), count(b) FROM table_1 WHERE a>4;
            --SELECT SUM(a), count(b) FROM table_1 WHERE a IN (1, 2, 13);
            SELECT a, SUM(b) FROM table_1 WHERE a IN (1, -0.5, 2, 13) GROUP BY a;
            --SELECT a, count(b) FROM table_1 GROUP BY a WHERE a IN (1, 2, 13);
        ",
        )
        .unwrap();
        let schema_1: Schema = vec![
            ("a", DataType::float_interval(-1., 3.)),
            ("b", DataType::float_interval(-2., 2.)),
            ("c", DataType::float()),
            ("d", DataType::float_interval(0., 1.)),
        ]
        .into_iter()
        .collect();
        let table_1 = Relation::table()
            .name("tab_1")
            .schema(schema_1.clone())
            .size(100)
            .build();
        let relation = Relation::try_from(QueryWithRelations::new(
            &query,
            &Hierarchy::from([(["schema", "table_1"], Arc::new(table_1))]),
        ))
        .unwrap();
        println!("relation = {relation:#?}");
        let q = ast::Query::from(&relation);
        println!("query = {q}");
        relation.display_dot().unwrap();
    }

    #[test]
    fn test_reduce_where() {
        let query = parse(
            "
            SELECT SUM(a), count(b) FROM table_1 WHERE a>4;
        ",
        )
        .unwrap();
        let schema_1: Schema = vec![
            ("a", DataType::float_interval(-1., 3.)),
            ("b", DataType::float_interval(-2., 2.)),
            ("c", DataType::float()),
            ("d", DataType::float_interval(0., 1.)),
        ]
        .into_iter()
        .collect();
        let table_1 = Relation::table()
            .name("tab_1")
            .schema(schema_1.clone())
            .size(100)
            .build();
        let relation = Relation::try_from(QueryWithRelations::new(
            &query,
            &Hierarchy::from([(["schema", "table_1"], Arc::new(table_1))]),
        ))
        .unwrap();
        println!("relation = {relation:#?}");
        let q = ast::Query::from(&relation);
        println!("query = {q}");
        relation.display_dot().unwrap();
    }

    #[test]
    fn test_group_by_alias() {
        let query = parse(
            "
            SELECT a AS my_a, SUM(b) AS sum_b FROM table_1 GROUP BY my_a;
        ",
        )
        .unwrap();
        let schema_1: Schema = vec![
            ("a", DataType::integer()),
            ("b", DataType::float_interval(-10., 10.)),
        ]
        .into_iter()
        .collect();
        let table_1 = Relation::table()
            .name("tab_1")
            .schema(schema_1.clone())
            .size(100)
            .build();
        let relation = Relation::try_from(QueryWithRelations::new(
            &query,
            &Hierarchy::from([(["schema", "table_1"], Arc::new(table_1))]),
        ))
        .unwrap();
        let q = ast::Query::from(&relation);
        println!("query = {q}");
        relation.display_dot().unwrap();
        assert_eq!(
            relation.data_type(),
            DataType::structured(vec![
                ("my_a", DataType::integer()),
                ("sum_b", DataType::float_interval(-1000., 1000.)),
            ])
        );
    }

    #[test]
    fn test_case() {
        let query =
            parse("SELECT CASE WHEN SUM(a) = 5 THEN 5 ELSE 4 * AVG(a) END FROM table_1").unwrap();
        let schema_1: Schema = vec![("a", DataType::float_interval(0., 10.))]
            .into_iter()
            .collect();
        let table_1 = Relation::table()
            .name("tab_1")
            .schema(schema_1.clone())
            .size(100)
            .build();
        let relation = Relation::try_from(QueryWithRelations::new(
            &query,
            &Hierarchy::from([(["schema", "table_1"], Arc::new(table_1))]),
        ))
        .unwrap();
        println!("relation = {relation}");
        relation.display_dot().unwrap();
        let q = ast::Query::from(&relation);
        println!("query = {q}");
    }

    #[test]
    fn test_group_by_columns() {
        let query = parse("SELECT a, sum(b) as s FROM table_1 GROUP BY a").unwrap();
        let schema_1: Schema = vec![
            ("a", DataType::integer_interval(0, 10)),
            ("b", DataType::float_interval(0., 10.)),
        ]
        .into_iter()
        .collect();
        let table_1: Relation = Relation::table()
            .name("tab_1")
            .path(["schema", "table_1"])
            .schema(schema_1.clone())
            .size(100)
            .build();
        let relation = Relation::try_from(QueryWithRelations::new(
            &query,
            &Hierarchy::from([(["schema", "table_1"], Arc::new(table_1))]),
        ));
        // .unwrap();
        // println!("relation = {relation}");
        // relation.display_dot().unwrap();
        // let q = ast::Query::from(&relation);
        // println!("query = {q}");
    }

    #[test]
    fn test_count_all() {
        let query = parse("SELECT count(*) FROM table_1 GROUP BY a").unwrap();
        let schema_1: Schema = vec![
            ("a", DataType::integer_interval(0, 10)),
            ("b", DataType::float_interval(0., 10.)),
        ]
        .into_iter()
        .collect();
        let table_1: Relation = Relation::table()
            .name("tab_1")
            .path(["schema", "table_1"])
            .schema(schema_1.clone())
            .size(100)
            .build();
        let relation = Relation::try_from(QueryWithRelations::new(
            &query,
            &Hierarchy::from([(["schema", "table_1"], Arc::new(table_1))]),
        ))
        .unwrap();
        println!("relation = {relation}");
        relation.display_dot().unwrap();
        let q = ast::Query::from(&relation);
        println!("query = {q}");
    }

    #[test]
    fn test_reduce_with_only_group_by_columns() {
        let query = parse("SELECT a AS a FROM table_1 GROUP BY a").unwrap();
        let schema_1: Schema = vec![("a", DataType::integer_interval(0, 10))]
            .into_iter()
            .collect();
        let table_1 = Relation::table()
            .name("tab_1")
            .schema(schema_1.clone())
            .size(100)
            .build();
        let relation = Relation::try_from(QueryWithRelations::new(
            &query,
            &Hierarchy::from([(["schema", "table_1"], Arc::new(table_1))]),
        ))
        .unwrap();
        println!("relation = {relation}");
        relation.display_dot().unwrap();
        let q = ast::Query::from(&relation);
        println!("query = {q}");
    }

    #[test]
    fn test_reduce_with_only_group_by_columns_multiple_map_reduce() {
        let schema_1: Schema = vec![("a", DataType::integer_interval(0, 10))]
            .into_iter()
            .collect();
        let table_1 = Relation::table()
            .name("tab_1")
            .schema(schema_1.clone())
            .size(100)
            .build();

        let query = parse(
            "
        WITH tab_a AS (SELECT a FROM table_1),
        tab_b AS (SELECT a FROM tab_a GROUP BY a),
        tab_c AS (SELECT LOG(a) AS log_a FROM tab_b),
        tab_d AS (SELECT SUM(log_a) AS aaa FROM tab_c)
        SELECT aaa FROM tab_d",
        )
        .unwrap();
        let relation = Relation::try_from(QueryWithRelations::new(
            &query,
            &Hierarchy::from([(["schema", "table_1"], Arc::new(table_1))]),
        ))
        .unwrap();
        println!("relation = {relation}");
        relation.display_dot().unwrap();
    }

    #[test]
    #[ignore]
    fn test_values() {
        let query = parse("SELECT a FROM (VALUES (1), (2), (3)) AS t1 (a) ;").unwrap();
        let schema_1: Schema = vec![
            ("a", DataType::integer_interval(0, 10)),
            ("b", DataType::float_interval(0., 10.)),
        ]
        .into_iter()
        .collect();
        let table_1 = Relation::table()
            .name("tab_1")
            .schema(schema_1.clone())
            .size(100)
            .build();
        let relation = Relation::try_from(QueryWithRelations::new(
            &query,
            &Hierarchy::from([(["schema", "table_1"], Arc::new(table_1))]),
        ))
        .unwrap();
        println!("relation = {relation}");
        relation.display_dot().unwrap();
        let q = ast::Query::from(&relation);
        println!("query = {q}");
    }

    #[test]
    fn test_having() {
        let query = parse("SELECT SUM(b) As my_sum FROM table_1 HAVING COUNT(b) > 4;").unwrap();
        let schema_1: Schema = vec![
            ("a", DataType::integer_interval(0, 10)),
            ("b", DataType::float_interval(0., 10.)),
        ]
        .into_iter()
        .collect();
        let table_1 = Relation::table()
            .name("table_1")
            .schema(schema_1.clone())
            .size(100)
            .build();
        let relation = Relation::try_from(QueryWithRelations::new(
            &query,
            &Hierarchy::from([(["schema", "table_1"], Arc::new(table_1))]),
        ))
        .unwrap();
        relation.display_dot().unwrap();
        println!("relation = {relation}");
        assert_eq!(
            relation.data_type(),
            DataType::structured(vec![("my_sum", DataType::float_interval(0., 1000.))])
        );

        let q = ast::Query::from(&relation);
        println!("query = {q}");

        let query = parse("SELECT SUM(b) AS my_sum FROM table_1 GROUP BY a HAVING COUNT(b) > 4 and a > 4 AND my_sum > 6;").unwrap();
        let schema_1: Schema = vec![
            ("a", DataType::integer_values([0, 2, 4, 10])),
            ("b", DataType::float_interval(0., 10.)),
        ]
        .into_iter()
        .collect();
        let table_1 = Relation::table()
            .name("table_1")
            .schema(schema_1.clone())
            .size(100)
            .build();
        let relation = Relation::try_from(QueryWithRelations::new(
            &query,
            &Hierarchy::from([(["schema", "table_1"], Arc::new(table_1))]),
        ))
        .unwrap();
        println!("relation = {relation}");
        relation.display_dot().unwrap();
        let q = ast::Query::from(&relation);
        println!("query = {q}");
        assert_eq!(
            relation.data_type(),
            DataType::structured(vec![("my_sum", DataType::float_interval(0., 1000.))])
        );

        let query =
            parse("SELECT SUM(b) AS my_sum FROM table_1 GROUP BY a HAVING a > 40;").unwrap();
        let schema_1: Schema = vec![
            ("a", DataType::integer_values([0, 2, 4, 10])),
            ("b", DataType::float_interval(0., 10.)),
        ]
        .into_iter()
        .collect();
        let table_1 = Relation::table()
            .name("table_1")
            .schema(schema_1.clone())
            .size(100)
            .build();
        let relation = Relation::try_from(QueryWithRelations::new(
            &query,
            &Hierarchy::from([(["schema", "table_1"], Arc::new(table_1))]),
        ))
        .unwrap();
        println!("relation = {relation}");
        relation.display_dot().unwrap();
        let q = ast::Query::from(&relation);
        println!("query = {q}");
        assert_eq!(
            relation.data_type(),
            DataType::structured(vec![("my_sum", DataType::float().try_empty().unwrap())])
        );

        let query = parse("SELECT SUM(b) AS my_sum FROM table_1 GROUP BY a HAVING COUNT(b) > 4 and a > 40 AND my_sum > 6;").unwrap();
        let schema_1: Schema = vec![
            ("a", DataType::integer_values([0, 2, 4, 10])),
            ("b", DataType::float_interval(0., 10.)),
        ]
        .into_iter()
        .collect();
        let table_1 = Relation::table()
            .name("table_1")
            .schema(schema_1.clone())
            .size(100)
            .build();
        let relation = Relation::try_from(QueryWithRelations::new(
            &query,
            &Hierarchy::from([(["schema", "table_1"], Arc::new(table_1))]),
        ))
        .unwrap();
        println!("relation = {relation}");
        relation.display_dot().unwrap();
        let q = ast::Query::from(&relation);
        println!("query = {q}");
        assert_eq!(
            relation.data_type(),
            DataType::structured(vec![("my_sum", DataType::float().try_empty().unwrap())])
        );
    }

    #[test]
    fn test_group_by_exprs() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        let query_str = "SELECT 3*d, COUNT(*) AS my_count FROM table_1 GROUP BY 3*d;";
        let query = parse(query_str).unwrap();
        let relation = Relation::try_from(QueryWithRelations::new(
            &query,
            &relations
        ))
        .unwrap();
        relation.display_dot().unwrap();
        println!("relation = {relation}");
        assert_eq!(
            relation.data_type(),
            DataType::structured(vec![
                ("field_fp0x", DataType::integer_interval(0, 30)),
                ("my_count", DataType::integer_interval(0, 10)),
            ])
        );
        let query: &str = &ast::Query::from(&relation).to_string();
        println!("{query}");
        _ = database
            .query(query)
            .unwrap()
            .iter()
            .map(ToString::to_string);
    }

    #[test]
    fn test_distinct_in_select() {
        let query = parse("SELECT DISTINCT a, b FROM table_1;").unwrap();
        let schema_1: Schema = vec![
            ("a", DataType::integer_interval(0, 10)),
            ("b", DataType::float_interval(0., 10.)),
        ]
        .into_iter()
        .collect();
        let table_1 = Relation::table()
            .name("table_1")
            .schema(schema_1.clone())
            .size(100)
            .build();
        let relation = Relation::try_from(QueryWithRelations::new(
            &query,
            &Hierarchy::from([(["schema", "table_1"], Arc::new(table_1))]),
            ))
        .unwrap();
        relation.display_dot().unwrap();
        println!("relation = {relation}");
        assert_eq!(
            relation.data_type(),
            DataType::structured(vec![
                ("a", DataType::integer_interval(0, 10)),
                ("b", DataType::float_interval(0., 10.)),
            ])
        );
    }

    #[test]
    fn test_join_with_using() {
        namer::reset();
        let table_1: Relation = Relation::table()
            .name("table_1")
            .schema(
                vec![
                    ("a", DataType::integer_interval(0, 10)),
                    ("b", DataType::float_interval(20., 50.)),
                ].into_iter()
                .collect::<Schema>()
            )
            .size(100)
            .build();
        let table_2: Relation  = Relation::table()
            .name("table_2")
            .schema(
                vec![
                    ("a", DataType::integer_interval(-5, 5)),
                    ("c", DataType::float()),
                ].into_iter()
                .collect::<Schema>()
            )
            .size(100)
            .build();
        let relations = Hierarchy::from([
            (["schema", "table_1"], Arc::new(table_1)),
            (["schema", "table_2"], Arc::new(table_2)),
        ]);

        // INNER JOIN
        let query = parse("SELECT * FROM table_1 INNER JOIN table_2 USING (a)").unwrap();
        let relation = Relation::try_from(QueryWithRelations::new(
            &query,
            &relations,
            ))
        .unwrap();
        relation.display_dot().unwrap();
        assert!(matches!(relation.data_type(), DataType::Struct(_)));
        if let DataType::Struct(s) = relation.data_type() {
            assert_eq!(s[0], Arc::new(DataType::integer_interval(0, 5)));
            assert_eq!(s[1], Arc::new(DataType::float_interval(20., 50.)));
            assert_eq!(s[2], Arc::new(DataType::float()));
        }

        // LEFT JOIN
        let query = parse("SELECT * FROM table_1 LEFT JOIN table_2 USING (a)").unwrap();
        let relation = Relation::try_from(QueryWithRelations::new(
            &query,
            &relations,
            ))
        .unwrap();
        relation.display_dot().unwrap();
        assert!(matches!(relation.data_type(), DataType::Struct(_)));
        if let DataType::Struct(s) = relation.data_type() {
            assert_eq!(s[0], Arc::new(DataType::integer_interval(0, 10)));
            assert_eq!(s[1], Arc::new(DataType::float_interval(20., 50.)));
            assert_eq!(s[2], Arc::new(DataType::optional(DataType::float())));
        }

        // RIGHT JOIN
        let query = parse("SELECT * FROM table_1 RIGHT JOIN table_2 USING (a)").unwrap();
        let relation = Relation::try_from(QueryWithRelations::new(
            &query,
            &relations,
            ))
        .unwrap();
        relation.display_dot().unwrap();
        assert!(matches!(relation.data_type(), DataType::Struct(_)));
        if let DataType::Struct(s) = relation.data_type() {
            assert_eq!(s[0], Arc::new(DataType::integer_interval(-5, 5)));
            assert_eq!(s[1], Arc::new(DataType::optional(DataType::float_interval(20., 50.))));
            assert_eq!(s[2], Arc::new(DataType::float()));
        }

        // FULL JOIN
        let query = parse("SELECT * FROM table_1 FULL JOIN table_2 USING (a)").unwrap();
        let relation = Relation::try_from(QueryWithRelations::new(
            &query,
            &relations,
            ))
        .unwrap();
        relation.display_dot().unwrap();
        assert!(matches!(relation.data_type(), DataType::Struct(_)));
        if let DataType::Struct(s) = relation.data_type() {
            assert_eq!(s[0], Arc::new(DataType::optional(DataType::integer_interval(-5, 10))));
            assert_eq!(s[1], Arc::new(DataType::optional(DataType::float_interval(20., 50.))));
            assert_eq!(s[2], Arc::new(DataType::optional(DataType::float())));
        }
    }

    #[test]
    fn test_join_with_natural() {
        namer::reset();
        let table_1: Relation = Relation::table()
            .name("table_1")
            .schema(
                vec![
                    ("a", DataType::integer_interval(0, 10)),
                    ("b", DataType::float_interval(20., 50.)),
                    ("d", DataType::float_interval(-10., 50.)),
                ].into_iter()
                .collect::<Schema>()
            )
            .size(100)
            .build();
        let table_2: Relation  = Relation::table()
            .name("table_2")
            .schema(
                vec![
                    ("a", DataType::integer_interval(-5, 5)),
                    ("c", DataType::float()),
                    ("d", DataType::float_interval(10., 100.)),
                ].into_iter()
                .collect::<Schema>()
            )
            .size(100)
            .build();
        let relations = Hierarchy::from([
            (["schema", "table_1"], Arc::new(table_1)),
            (["schema", "table_2"], Arc::new(table_2)),
        ]);

        // INNER JOIN
        let query = parse("SELECT * FROM table_1 NATURAL INNER JOIN table_2").unwrap();
        let relation = Relation::try_from(QueryWithRelations::new(
            &query,
            &relations,
            ))
        .unwrap();
        relation.display_dot().unwrap();
        assert!(matches!(relation.data_type(), DataType::Struct(_)));
        if let DataType::Struct(s) = relation.data_type() {
            assert_eq!(s[0], Arc::new(DataType::integer_interval(0, 5)));
            assert_eq!(s[1], Arc::new(DataType::float_interval(10., 50.)));
            assert_eq!(s[2], Arc::new(DataType::float_interval(20., 50.)));
            assert_eq!(s[3], Arc::new(DataType::float()));
        }

        // LEFT JOIN
        let query = parse("SELECT * FROM table_1 NATURAL LEFT JOIN table_2").unwrap();
        let relation = Relation::try_from(QueryWithRelations::new(
            &query,
            &relations,
            ))
        .unwrap();
        relation.display_dot().unwrap();
        assert!(matches!(relation.data_type(), DataType::Struct(_)));
        if let DataType::Struct(s) = relation.data_type() {
            assert_eq!(s[0], Arc::new(DataType::integer_interval(0, 10)));
            assert_eq!(s[1], Arc::new(DataType::float_interval(-10., 50.)));
            assert_eq!(s[2], Arc::new(DataType::float_interval(20., 50.)));
            assert_eq!(s[3], Arc::new(DataType::optional(DataType::float())));
        }

        // RIGHT JOIN
        let query = parse("SELECT * FROM table_1 NATURAL RIGHT JOIN table_2").unwrap();
        let relation = Relation::try_from(QueryWithRelations::new(
            &query,
            &relations,
            ))
        .unwrap();
        relation.display_dot().unwrap();
        assert!(matches!(relation.data_type(), DataType::Struct(_)));
        if let DataType::Struct(s) = relation.data_type() {
            assert_eq!(s[0], Arc::new(DataType::integer_interval(-5, 5)));
            assert_eq!(s[1], Arc::new(DataType::float_interval(10., 100.)));
            assert_eq!(s[2], Arc::new(DataType::optional(DataType::float_interval(20., 50.))));
            assert_eq!(s[3], Arc::new(DataType::float()));

        }

        // FULL JOIN
        let query = parse("SELECT * FROM table_1 NATURAL FULL JOIN table_2").unwrap();
        let relation = Relation::try_from(QueryWithRelations::new(
            &query,
            &relations,
            ))
        .unwrap();
        relation.display_dot().unwrap();
        assert!(matches!(relation.data_type(), DataType::Struct(_)));
        if let DataType::Struct(s) = relation.data_type() {
            assert_eq!(s[0], Arc::new(DataType::optional(DataType::integer_interval(-5, 10))));
            assert_eq!(s[1], Arc::new(DataType::optional(DataType::float_interval(-10., 100.))));
            assert_eq!(s[2], Arc::new(DataType::optional(DataType::float_interval(20., 50.))));
            assert_eq!(s[3], Arc::new(DataType::optional(DataType::float())));
        }
    }
}
