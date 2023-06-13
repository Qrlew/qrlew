//! This file provides tools for converting a sqlparser::ast::Statement
//! into the corresponding Qrlew Relation.
//! Example: `Expr::try_from(sql_parser_statement)`

use super::{
    query_names::{IntoQueryNamesVisitor, QueryNames},
    visitor::Visitor,
    Error, Result,
};
use crate::{
    builder::{Ready, With, WithoutContext},
    expr::{Expr, Identifier, Split},
    hierarchy::{Hierarchy, Path},
    namer::{self, FIELD},
    relation::{
        Join, JoinConstraint, JoinOperator, MapBuilder, Relation, Set, SetOperator, SetQuantifier,
        Variant as _, WithInput,
    },
    visitor::{Acceptor, Dependencies, Visited},
};
use itertools::Itertools;
use sqlparser::{
    ast,
    dialect::{Dialect, GenericDialect},
    parser::Parser,
    tokenizer::Tokenizer,
};
use std::{
    convert::TryFrom,
    iter::{once, Iterator},
    rc::Rc,
    result,
    str::FromStr,
};

/*
Before we visit queries to build Relations we must collect the namespaces of all subqueries to map the right names to the right Relations.
This is done in the query_names module.
 */

/// A visitor for AST queries conversion into relations
/// The Hierarchy of Relations is the context in which the query is converted, typically the list of tables with their Path
/// The QueryNames is the map of sub-query referrenced by their names, so that links can be unfolded
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct TryIntoRelationVisitor<'a>(&'a Hierarchy<Rc<Relation>>, QueryNames<'a>);

impl<'a> TryIntoRelationVisitor<'a> {
    fn new(relations: &'a Hierarchy<Rc<Relation>>, query_names: QueryNames<'a>) -> Self {
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
        }
    }
}

// This is RelationWithColumns from_xxx method

/// A struct to hold Relations with column mapping in the FROM
struct RelationWithColumns(Rc<Relation>, Hierarchy<Identifier>);

impl RelationWithColumns {
    fn new(relation: Rc<Relation>, columns: Hierarchy<Identifier>) -> Self {
        RelationWithColumns(relation, columns)
    }
}

/// A struct to hold the query being visited and its Relations
struct VisitedQueryRelations<'a>(
    Hierarchy<Rc<Relation>>,
    Visited<'a, ast::Query, Result<Rc<Relation>>>,
);

impl<'a> VisitedQueryRelations<'a> {
    fn new(
        try_into_relation_visitor: &TryIntoRelationVisitor<'a>,
        query: &'a ast::Query,
        visited: Visited<'a, ast::Query, Result<Rc<Relation>>>,
    ) -> Self {
        let TryIntoRelationVisitor(relations, query_names) = try_into_relation_visitor;
        let mut relations: Hierarchy<Rc<Relation>> = (*relations).clone();
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
            ast::TableFactor::Table {
                name, alias: None, ..
            } => {
                let relation = relations
                    .get(&name.cloned())
                    .cloned()
                    .ok_or(Error::parsing_error(format!("Unknown table: {name}")))?;
                let columns: Hierarchy<Identifier> = relation
                    .schema()
                    .iter()
                    .map(|f| {
                        (
                            name.cloned()
                                .into_iter()
                                .chain(once(f.name().to_string()))
                                .collect_vec(),
                            [relation.name(), f.name()].into(),
                        )
                    })
                    .collect();
                Ok(RelationWithColumns::new(relation, columns))
            }
            ast::TableFactor::Table {
                name,
                alias: Some(alias),
                ..
            } => {
                // TODO Only the table can be aliased for now -> Fix this
                let relation = relations
                    .get(&name.cloned())
                    .cloned()
                    .ok_or(Error::parsing_error(format!("Unknown table: {name}")))?;
                let columns: Hierarchy<Identifier> = relation
                    .schema()
                    .iter()
                    .map(|f| {
                        (
                            alias
                                .name
                                .cloned()
                                .into_iter()
                                .chain(once(f.name().to_string()))
                                .collect_vec(),
                            [relation.name(), f.name()].into(),
                        )
                    })
                    .collect();
                Ok(RelationWithColumns::new(relation, columns))
            }
            ast::TableFactor::Derived {
                subquery,
                alias: None,
                ..
            } => {
                let relation = visited.get(subquery).clone()?;
                let columns: Hierarchy<Identifier> = relation
                    .schema()
                    .iter()
                    .map(|f| {
                        (
                            relation
                                .name()
                                .cloned()
                                .into_iter()
                                .chain(once(f.name().to_string()))
                                .collect_vec(),
                            [relation.name(), f.name()].into(),
                        )
                    })
                    .collect();
                Ok(RelationWithColumns::new(relation, columns))
            }
            ast::TableFactor::Derived {
                subquery,
                alias: Some(alias),
                ..
            } => {
                // TODO Only the table can be aliased for now -> Fix this
                let relation = visited.get(subquery).clone()?;
                let columns: Hierarchy<Identifier> = relation
                    .schema()
                    .iter()
                    .map(|f| {
                        (
                            alias
                                .name
                                .cloned()
                                .into_iter()
                                .chain(once(f.name().to_string()))
                                .collect_vec(),
                            [relation.name(), f.name()].into(),
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
    ) -> Result<JoinConstraint> {
        match join_constraint {
            ast::JoinConstraint::On(expr) => Ok(JoinConstraint::On(expr.with(columns).try_into()?)),
            ast::JoinConstraint::Using(idents) => Ok(JoinConstraint::Using(
                idents
                    .into_iter()
                    .map(|ident| Identifier::from(ident.value.clone()))
                    .collect(),
            )),
            ast::JoinConstraint::Natural => Ok(JoinConstraint::Natural),
            ast::JoinConstraint::None => Ok(JoinConstraint::None),
        }
    }

    fn try_from_join_operator_with_columns(
        &self,
        join_operator: &ast::JoinOperator,
        columns: &'a Hierarchy<Identifier>,
    ) -> Result<JoinOperator> {
        match join_operator {
            ast::JoinOperator::Inner(join_constraint) => Ok(JoinOperator::Inner(
                self.try_from_join_constraint_with_columns(join_constraint, columns)?,
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
            |left, join| {
                let RelationWithColumns(left_relation, left_columns) = left?;
                let RelationWithColumns(right_relation, right_columns) =
                    self.try_from_table_factor(&join.relation)?;
                let all_columns = left_columns.with(right_columns);
                let operator = self.try_from_join_operator_with_columns(
                    &join.join_operator,
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
                let composed_columns = all_columns.and_then(new_columns);
                let relation = Rc::new(Relation::from(join));
                // We should compose hierarchies
                Ok(RelationWithColumns::new(relation, composed_columns))
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
        group_by: &'a Vec<ast::Expr>,
        from: Rc<Relation>,
    ) -> Result<Rc<Relation>> {
        // Collect all expressions with their aliases
        let mut named_exprs: Vec<(String, Expr)> = Vec::new();
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
        // Build the Map or Reduce based on the type of split
        let split = Split::from_iter(named_exprs);
        // Prepare the WHERE
        let filter: Option<Expr> = selection
            .as_ref()
            .map(|e| e.with(columns).try_into())
            .map_or(Ok(None), |r| r.map(Some))?;
        // Prepare the GROUP BY
        let group_by: Result<Vec<Expr>> = group_by
            .iter()
            .map(|e| e.with(columns).try_into())
            .collect();
        // Build a Relation
        let relation = match split {
            Split::Map(map) => {
                let builder = Relation::map().split(map);
                let builder = filter.into_iter().fold(builder, |b, e| b.filter(e));
                let builder = group_by?.into_iter().fold(builder, |b, e| b.group_by(e));
                builder.input(from).build()
            }
            Split::Reduce(reduce) => {
                let builder = Relation::reduce().split(reduce);
                let builder = group_by?.into_iter().fold(builder, |b, e| b.group_by(e));
                builder.input(from).build()
            }
        };
        Ok(Rc::new(relation))
    }

    /// Convert a Select into a Relation
    fn try_from_select(&self, select: &'a ast::Select) -> Result<RelationWithColumns> {
        let ast::Select {
            projection,
            from,
            selection,
            group_by,
            ..
        } = select;
        let RelationWithColumns(from, columns) = self.try_from_tables_with_joins(from)?;
        let relation = self.try_from_select_items_selection_and_group_by(
            &columns.filter_map(|i| Some(i.split_last().ok()?.0)),
            projection,
            selection,
            group_by,
            from,
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

    /// Convert a Query into a Relation
    fn try_from_query(&self, query: &'a ast::Query) -> Result<Rc<Relation>> {
        let ast::Query {
            body,
            order_by,
            limit,
            ..
        } = query;
        match body.as_ref() {
            ast::SetExpr::Select(select) => {
                let RelationWithColumns(relation, columns) =
                    self.try_from_select(select.as_ref())?;
                // let names = &columns.filter_map(|i| Some(i.split_last().ok()?.0));//TODO remove this
                if order_by.is_empty() && limit.is_none() {
                    Ok(relation)
                } else {
                    // Build a relation with ORDER BY and LIMIT if needed
                    let relation_bulider = Relation::map();
                    // We add all the columns
                    let relation_builder = relation
                        .schema()
                        .iter()
                        .fold(relation_bulider, |builder, field| {
                            builder.with((field.name(), Expr::col(field.name())))
                        });
                    // Add input
                    let relation_bulider = relation_builder.input(relation);
                    // Add ORDER BYs
                    let relation_bulider: Result<MapBuilder<WithInput>> = order_by.iter().fold(
                        Ok(relation_bulider),
                        |builder, ast::OrderByExpr { expr, asc, .. }| {
                            Ok(builder?
                                .order_by(expr.with(&columns).try_into()?, asc.unwrap_or(true)))
                        },
                    );
                    // Add LIMITs
                    let relation_bulider: Result<MapBuilder<WithInput>> =
                        limit.iter().fold(relation_bulider, |builder, limit| {
                            Ok(builder?.limit(self.try_from_limit(limit)?))
                        });
                    // Build a relation with ORDER BY and LIMIT
                    Ok(Rc::new(relation_bulider?.try_build()?))
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
                    let relation_bulider = Relation::set()
                        .operator(op.clone().into())
                        .quantifier(set_quantifier.clone().into())
                        .left(left_relation)
                        .right(right_relation);
                    // Build a Relation from set operation
                    Ok(Rc::new(relation_bulider.try_build()?))
                }
                _ => panic!("We only support set operations over SELECTs"),
            },
            _ => todo!(),
        }
    }
}

impl<'a> Visitor<'a, Result<Rc<Relation>>> for TryIntoRelationVisitor<'a> {
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
        visited: Visited<'a, ast::Query, Result<Rc<Relation>>>,
    ) -> Result<Rc<Relation>> {
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
pub struct QueryWithRelations<'a>(&'a ast::Query, &'a Hierarchy<Rc<Relation>>);

impl<'a> QueryWithRelations<'a> {
    pub fn new(query: &'a ast::Query, relations: &'a Hierarchy<Rc<Relation>>) -> Self {
        QueryWithRelations(query, relations)
    }
}

impl<'a> With<&'a Hierarchy<Rc<Relation>>, QueryWithRelations<'a>> for &'a ast::Query {
    fn with(self, input: &'a Hierarchy<Rc<Relation>>) -> QueryWithRelations<'a> {
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
            .map(|r| (*r).clone())
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
    use std::rc::Rc;

    use colored::Colorize;

    use super::*;
    use crate::{builder::Ready, data_type::DataType, display::Dot, relation::schema::Schema};

    #[test]
    fn test_map_from_query() {
        let query = parse("SELECT exp(a) FROM shema.table").unwrap();
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
            &Hierarchy::from([(["shema", "table"], Rc::new(table))]),
        ))
        .unwrap();
        print!("{}", map)
    }

    #[ignore]
    #[test]
    fn test_parse() {
        let query = parse("SELECT 2 * (price - 1) FROM schema.table").unwrap();
        let query = parse("SELECT 2 * price FROM schema.table").unwrap();
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
            &Hierarchy::from([(["schema", "table"], Rc::new(table))]),
        ))
        .unwrap();
        let query2 = &ast::Query::from(&map);
        //println!("\nquery2: {:?}", query2);
        println!("\n{}", query2.to_string());
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
        println!("BEFORE: {query_names_1}");
        query_names_1.set(name_3.clone(), &query_4);
        println!("AFTER: {query_names_1}");
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
        select cos(0.1*ta.b) as cs, tb.b as l, tb.a from view_2 as ta left outer join table_1 as tb on a=b;",
        ];

    #[ignore]
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
                (["schema", "table_1"], Rc::new(table_1)),
                (["schema", "table_2"], Rc::new(table_2)),
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
                (["schema", "table_1"], Rc::new(table_1)),
                (["schema", "table_2"], Rc::new(table_2)),
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
            &Hierarchy::from([(["schema", "table_1"], Rc::new(table_1))]),
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
            SELECT 1+SUM(a), count(b) FROM table_1 WHERE a>4;
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
            &Hierarchy::from([(["schema", "table_1"], Rc::new(table_1))]),
        ))
        .unwrap();
        println!("relation = {relation:#?}");
        let q = ast::Query::from(&relation);
        println!("query = {q}");
        relation.display_dot().unwrap();
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
            &Hierarchy::from([(["schema", "table_1"], Rc::new(table_1))]),
        ))
        .unwrap();
        println!("relation = {relation}");
        relation.display_dot().unwrap();
        let q = ast::Query::from(&relation);
        println!("query = {q}");
    }

    #[test]
    fn test_group_by_columns() {
        let query =
            parse("SELECT a, sum(b) as s FROM table_1 GROUP BY a").unwrap();
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
            &Hierarchy::from([(["schema", "table_1"], Rc::new(table_1))]),
        ))
        .unwrap();
        println!("relation = {relation}");
        relation.display_dot().unwrap();
        let q = ast::Query::from(&relation);
        println!("query = {q}");
    }
}
