//! Methods to convert Relations to ast::Query
use serde::de::value;

use super::{
    Error, Join, JoinOperator, Map, OrderBy, Reduce, Relation, Result, Set, SetOperator,
    SetQuantifier, Table, Values, Variant as _, Visitor,
};
use crate::{
    ast,
    data_type::{DataType, DataTyped},
    dialect_translation::{postgresql::PostgreSqlTranslator, RelationToQueryTranslator},
    expr::{identifier::Identifier, Expr},
    visitor::Acceptor,
};
use std::{collections::HashSet, convert::TryFrom, iter::Iterator, ops::Deref};

/// A simple Relation -> ast::Query conversion Visitor using CTE
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct FromRelationVisitor<T: RelationToQueryTranslator> {
    translator: T,
}

impl<T: RelationToQueryTranslator> FromRelationVisitor<T> {
    pub fn new(translator: T) -> Self {
        FromRelationVisitor { translator }
    }
}

impl From<Identifier> for ast::ObjectName {
    fn from(value: Identifier) -> Self {
        ast::ObjectName(value.into_iter().map(|s| ast::Ident::new(s)).collect())
    }
}

impl From<SetOperator> for ast::SetOperator {
    fn from(value: SetOperator) -> Self {
        match value {
            SetOperator::Union => ast::SetOperator::Union,
            SetOperator::Except => ast::SetOperator::Except,
            SetOperator::Intersect => ast::SetOperator::Intersect,
        }
    }
}

impl From<SetQuantifier> for ast::SetQuantifier {
    fn from(value: SetQuantifier) -> Self {
        match value {
            SetQuantifier::All => ast::SetQuantifier::All,
            SetQuantifier::Distinct => ast::SetQuantifier::Distinct,
            SetQuantifier::None => ast::SetQuantifier::None,
            SetQuantifier::ByName => ast::SetQuantifier::ByName,
            SetQuantifier::AllByName => ast::SetQuantifier::AllByName,
            SetQuantifier::DistinctByName => ast::SetQuantifier::DistinctByName,
        }
    }
}

fn values_query(rows: Vec<Vec<ast::Expr>>) -> ast::Query {
    ast::Query {
        with: None,
        body: Box::new(ast::SetExpr::Values(ast::Values {
            explicit_row: false,
            rows,
        })),
        order_by: vec![],
        limit: None,
        offset: None,
        fetch: None,
        locks: vec![],
        limit_by: vec![],
        for_clause: None,
    }
}

fn table_with_joins(relation: ast::TableFactor, joins: Vec<ast::Join>) -> ast::TableWithJoins {
    ast::TableWithJoins { relation, joins }
}

fn ctes_from_query(query: ast::Query) -> Vec<ast::Cte> {
    query.with.map(|with| with.cte_tables).unwrap_or_default()
}

fn all() -> Vec<ast::SelectItem> {
    vec![ast::SelectItem::Wildcard(
        ast::WildcardAdditionalOptions::default(),
    )]
}

fn select_from_query(query: ast::Query) -> ast::Select {
    match query.body.as_ref() {
        ast::SetExpr::Select(select) => select.as_ref().clone(),
        _ => panic!("Non select query"), // It is okay to panic as this should not happen in our context and is a private function
    }
}

/// Build a set operation
fn set_operation(
    with: Vec<ast::Cte>,
    operator: ast::SetOperator,
    quantifier: ast::SetQuantifier,
    left: ast::Select,
    right: ast::Select,
) -> ast::Query {
    ast::Query {
        with: (!with.is_empty()).then_some(ast::With {
            recursive: false,
            cte_tables: with,
        }),
        body: Box::new(ast::SetExpr::SetOperation {
            op: operator,
            set_quantifier: quantifier,
            left: Box::new(ast::SetExpr::Select(Box::new(left))),
            right: Box::new(ast::SetExpr::Select(Box::new(right))),
        }),
        order_by: vec![],
        limit: None,
        offset: None,
        fetch: None,
        locks: vec![],
        limit_by: vec![],
        for_clause: None,
    }
}

impl<'a, T: RelationToQueryTranslator> Visitor<'a, ast::Query> for FromRelationVisitor<T> {
    fn table(&self, table: &'a Table) -> ast::Query {
        self.translator.query(
            vec![],
            vec![ast::SelectItem::Wildcard(
                ast::WildcardAdditionalOptions::default(),
            )],
            table_with_joins(
                self.translator.table_factor(&table.clone().into(), None),
                vec![],
            ),
            None,
            ast::GroupByExpr::Expressions(vec![]),
            vec![],
            None,
            None,
        )
    }

    fn map(&self, map: &'a Map, input: ast::Query) -> ast::Query {
        // Pull the existing CTEs
        let mut input_ctes = ctes_from_query(input);
        // Add input query to CTEs
        input_ctes.push(
            self.translator.cte(
                self.translator.identifier( &(map.name().into()) )[0].clone(),
                map.schema()
                    .iter()
                    .map(|field| self.translator.identifier( &(field.name().into()) )[0].clone())
                    .collect(),
                self.translator.query(
                    vec![],
                    map.projection
                        .clone()
                        .into_iter()
                        .zip(map.schema.clone())
                        .map(|(expr, field)| ast::SelectItem::ExprWithAlias {
                            expr: self.translator.expr(&expr),
                            alias: self.translator.identifier( &(field.name().into()) )[0].clone(),
                        })
                        .collect(),
                    table_with_joins(
                        self.translator
                            .table_factor(map.input.as_ref().into(), None),
                        vec![],
                    ),
                    map.filter.as_ref().map(|expr| self.translator.expr(expr)),
                    ast::GroupByExpr::Expressions(vec![]),
                    map.order_by
                        .iter()
                        .map(|OrderBy { expr, asc }| ast::OrderByExpr {
                            expr: self.translator.expr(expr),
                            asc: Some(*asc),
                            nulls_first: None,
                        })
                        .collect(),
                    map.limit.map(|limit| {
                        ast::Expr::Value(ast::Value::Number(limit.to_string(), false))
                    }),
                    map.offset.map(|offset| ast::Offset {
                        value: ast::Expr::Value(ast::Value::Number(offset.to_string(), false)),
                        rows: ast::OffsetRows::None,
                    }),
                ),
            ),
        );
        self.translator.query(
            input_ctes,
            all(),
            table_with_joins(
                self.translator.table_factor(&map.clone().into(), None),
                vec![],
            ),
            None,
            ast::GroupByExpr::Expressions(vec![]),
            vec![],
            map.limit
                .map(|limit| ast::Expr::Value(ast::Value::Number(limit.to_string(), false))),
            None,
        )
    }

    fn reduce(&self, reduce: &'a Reduce, input: ast::Query) -> ast::Query {
        // Pull the existing CTEs
        let mut input_ctes = ctes_from_query(input);
        // Add input query to CTEs
        input_ctes.push(
            self.translator.cte(
                self.translator.identifier( &(reduce.name().into()) )[0].clone(),
                reduce
                    .schema()
                    .iter()
                    .map(|field| self.translator.identifier( &(field.name().into()) )[0].clone())
                    .collect(),
                self.translator.query(
                    vec![],
                    reduce
                        .aggregate
                        .clone()
                        .into_iter()
                        .zip(reduce.schema.clone())
                        .map(|(aggregate, field)| ast::SelectItem::ExprWithAlias {
                            expr: self.translator.expr(aggregate.deref()),
                            alias: self.translator.identifier( &(field.name().into()) )[0].clone(),
                        })
                        .collect(),
                    table_with_joins(
                        self.translator
                            .table_factor(reduce.input.as_ref().into(), None),
                        vec![],
                    ),
                    None,
                    ast::GroupByExpr::Expressions(
                        reduce
                            .group_by
                            .iter()
                            .map(|col| self.translator.expr(&Expr::Column(col.clone())))
                            .collect(),
                    ),
                    vec![],
                    None,
                    None,
                ),
            ),
        );
        self.translator.query(
            input_ctes,
            all(),
            table_with_joins(
                self.translator.table_factor(&reduce.clone().into(), None),
                vec![],
            ),
            None,
            ast::GroupByExpr::Expressions(vec![]),
            vec![],
            None,
            None,
        )
    }

    fn join(&self, join: &'a Join, left: ast::Query, right: ast::Query) -> ast::Query {
        // Pull the existing CTEs
        let mut exist: HashSet<ast::Cte> = HashSet::new();
        let mut input_ctes: Vec<ast::Cte> = vec![];
        ctes_from_query(left).into_iter().for_each(|cte| {
            if exist.insert(cte.clone()) {
                input_ctes.push(cte)
            }
        });
        ctes_from_query(right).into_iter().for_each(|cte| {
            if exist.insert(cte.clone()) {
                input_ctes.push(cte)
            }
        });

        // Add input query to CTEs
        input_ctes.push(
            self.translator.cte(
                self.translator.identifier( &(join.name().into()) )[0].clone(),
                join.schema()
                    .iter()
                    .map(|field| self.translator.identifier( &(field.name().into()) )[0].clone())
                    .collect(),
                self.translator.query(
                    vec![],
                    self.translator.join_projection(join), //self.translator.join_projection(),
                    table_with_joins(
                        self.translator
                            .table_factor(join.left.as_ref().into(), Some(Join::left_name())),
                        vec![ast::Join {
                            relation: self
                                .translator
                                .table_factor(join.right.as_ref().into(), Some(Join::right_name())),
                            join_operator: self.translator.join_operator(&join.operator),
                        }],
                    ),
                    None,
                    ast::GroupByExpr::Expressions(vec![]),
                    vec![],
                    None,
                    None,
                ),
            ),
        );
        self.translator.query(
            input_ctes,
            all(),
            table_with_joins(
                self.translator.table_factor(&join.clone().into(), None),
                vec![],
            ),
            None,
            ast::GroupByExpr::Expressions(vec![]),
            vec![],
            None,
            None,
        )
    }

    fn set(&self, set: &'a Set, left: ast::Query, right: ast::Query) -> ast::Query {
        // Pull the existing CTEs
        let mut exist: HashSet<ast::Cte> = HashSet::new();
        let mut input_ctes: Vec<ast::Cte> = vec![];
        ctes_from_query(left.clone()).into_iter().for_each(|cte| {
            if exist.insert(cte.clone()) {
                input_ctes.push(cte)
            }
        });
        ctes_from_query(right.clone()).into_iter().for_each(|cte| {
            if exist.insert(cte.clone()) {
                input_ctes.push(cte)
            }
        });
        // Add input query to CTEs
        input_ctes.push(
            self.translator.cte(
                set.name().into(),
                set.schema()
                    .iter()
                    .map(|field| self.translator.identifier( &(field.name().into()) )[0].clone())
                    .collect(),
                set_operation(
                    vec![],
                    set.operator.clone().into(),
                    set.quantifier.clone().into(),
                    select_from_query(left),
                    select_from_query(right),
                ),
            ),
        );
        self.translator.query(
            input_ctes,
            all(),
            table_with_joins(
                self.translator.table_factor(&set.clone().into(), None),
                vec![],
            ),
            None,
            ast::GroupByExpr::Expressions(vec![]),
            vec![],
            None,
            None,
        )
    }

    fn values(&self, values: &'a Values) -> ast::Query {
        let rows = values
            .values
            .iter()
            .cloned()
            .map(|v| vec![ast::Expr::from(&Expr::Value(v))])
            .collect();

        let value_name = self.translator.identifier(&(values.name.as_str().into()))[0].clone();
        let from = ast::TableWithJoins {
            relation: ast::TableFactor::Derived {
                lateral: false,
                subquery: Box::new(values_query(rows)),
                alias: Some(ast::TableAlias {
                    name: value_name.clone(),
                    columns: vec![value_name],
                }),
            },
            joins: vec![],
        };
        let cte_query = self.translator.query(
            vec![],
            all(),
            from,
            None,
            ast::GroupByExpr::Expressions(vec![]),
            vec![],
            None,
            None,
        );
        let value_name = self.translator.identifier( &(values.name().into()) )[0].clone();
        let input_ctes =
            vec![self
                .translator
                .cte(value_name.clone(), vec![value_name], cte_query)];
        self.translator.query(
            input_ctes,
            all(),
            table_with_joins(
                self.translator.table_factor(&values.clone().into(), None),
                vec![],
            ),
            None,
            ast::GroupByExpr::Expressions(vec![]),
            vec![],
            None,
            None,
        )
    }
}

/// Based on the FromRelationVisitor implement the From trait
impl From<&Relation> for ast::Query {
    fn from(value: &Relation) -> Self {
        let dialect_translator = PostgreSqlTranslator;
        value.accept(FromRelationVisitor::new(dialect_translator))
    }
}

impl Table {
    /// Build the CREATE TABLE statement
    pub fn create<T: RelationToQueryTranslator>(&self, translator: T) -> ast::Statement {
        translator.create(self)
    }

    pub fn insert<T: RelationToQueryTranslator>(
        &self,
        prefix: &str,
        translator: T,
    ) -> ast::Statement {
        translator.insert(prefix, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        builder::{Ready, With},
        data_type::{DataType, Value},
        display::Dot,
        namer,
        relation::schema::Schema,
    };
    use std::sync::Arc;

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
                .size(100)
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
                .limit(100)
                .offset(20)
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
    fn test_from_table_relation() {
        // let relation = build_complex_relation();
        let schema: Schema = vec![
            ("a", DataType::float()),
            ("b", DataType::float_interval(-2., 2.)),
            ("c", DataType::float()),
            ("d", DataType::float_interval(0., 1.)),
        ]
        .into_iter()
        .collect();
        let table: Relation = Relation::table()
            .name("Name")
            .schema(schema.clone())
            .build();
        let query = ast::Query::from(&table);
        println!("query = {query}");
    }

    #[test]
    fn test_from_complex_relation() {
        let relation = build_complex_relation();
        let relation = relation.as_ref();
        relation.display_dot().unwrap();
        let query = ast::Query::from(relation);
        println!("query = {query}");
    }

    #[test]
    fn test_display_join() {
        namer::reset();
        let schema: Schema = vec![("b", DataType::float_interval(-2., 2.))]
            .into_iter()
            .collect();
        let left: Relation = Relation::table()
            .name("left")
            .schema(schema.clone())
            .size(1000)
            .build();
        let right: Relation = Relation::table()
            .name("right")
            .schema(schema.clone())
            .size(1000)
            .build();

        let join: Relation = Relation::join()
            .name("join")
            .left_outer(Expr::val(true))
            .on_eq("b", "b")
            .left(left)
            .right(right)
            .build();

        let query = ast::Query::from(&join);
        println!("query = {}", query.to_string());
    }

    #[test]
    fn test_display_values() {
        namer::reset();
        let values: Relation = Relation::values()
            .name("my_values")
            .values([Value::from(3.), Value::from(4)])
            .build();

        let query = ast::Query::from(&values);
        assert_eq!(
            query.to_string(),
            r#"WITH "my_values" ("my_values") AS (SELECT * FROM (VALUES (3), (4)) AS "my_values" ("my_values")) SELECT * FROM "my_values""#.to_string()
        );

        let schema: Schema = vec![("b", DataType::float_interval(-2., 2.))]
            .into_iter()
            .collect();
        let table: Relation = Relation::table()
            .name("table")
            .schema(schema.clone())
            .size(1000)
            .build();

        let join: Relation = Relation::join().left(values).right(table).cross().build();
        let query = ast::Query::from(&join);
        assert_eq!(
            query.to_string(),
            r#"WITH "my_values" ("my_values") AS (SELECT * FROM (VALUES (3), (4)) AS "my_values" ("my_values")), "join_zs1x" ("field_gu2a", "field_b8x4") AS (SELECT * FROM "my_values" AS "_LEFT_" CROSS JOIN "table" AS "_RIGHT_") SELECT * FROM "join_zs1x""#.to_string()
        );
    }
}
