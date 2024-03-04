use std::sync::Arc;

use crate::{
    expr,
    hierarchy::Hierarchy,
    relation::sql::FromRelationVisitor,
    sql::{parse_with_dialect, query_names::IntoQueryNamesVisitor},
    visitor::Acceptor,
    Relation,
};

use super::{function_builder, RelationWithTranslator, QueryToRelationTranslator, RelationToQueryTranslator};
use sqlparser::{ast, dialect::PostgreSqlDialect};

use crate::sql::{Error, Result};
#[derive(Clone, Copy)]
pub struct PostgreSqlTranslator;

impl RelationToQueryTranslator for PostgreSqlTranslator {
    fn first(&self, expr: &expr::Expr) -> ast::Expr {
        ast::Expr::from(expr)
    }

    fn mean(&self, expr: &expr::Expr) -> ast::Expr {
        let arg = self.expr(expr);
        function_builder("AVG", vec![arg], false)
    }

    fn var(&self, expr: &expr::Expr) -> ast::Expr {
        let arg = self.expr(expr);
        function_builder("VARIANCE", vec![arg], false)
    }

    fn std(&self, expr: &expr::Expr) -> ast::Expr {
        let arg = self.expr(expr);
        function_builder("STDDEV", vec![arg], false)
    }

    fn trunc(&self, exprs: Vec<&expr::Expr>) -> ast::Expr {
        // TRUNC in postgres has a problem:
        // In TRUNC(double_precision_number, precision) if precision is specified it fails
        // If it is not specified it passes considering precision = 0.
        // SELECT TRUNC(CAST (0.12 AS DOUBLE PRECISION), 0) fails
        // SELECT TRUNC(CAST (0.12 AS DOUBLE PRECISION)) passes.
        // Here we check precision, if it is 0 we remove it (such that the precision is implicit).
        let ast_exprs: Vec<ast::Expr> = exprs.into_iter().map(|expr| self.expr(expr)).collect();
        ast::Expr::Function(ast::Function {
            name: ast::ObjectName(vec![ast::Ident::from("TRUNC")]),
            args: ast_exprs
                .into_iter()
                .filter_map(|e| {
                    (e != ast::Expr::Value(ast::Value::Number("0".to_string(), false)))
                        .then_some(ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(e)))
                })
                .collect(),
            over: None,
            distinct: false,
            special: false,
            order_by: vec![],
            filter: None,
            null_treatment: None,
        })
    }

    fn round(&self, exprs: Vec<&expr::Expr>) -> ast::Expr {
        // Same as TRUNC
        // what if I wanted to do round(0, 0)
        let ast_exprs: Vec<ast::Expr> = exprs.into_iter().map(|expr| self.expr(expr)).collect();
        ast::Expr::Function(ast::Function {
            name: ast::ObjectName(vec![ast::Ident::from("ROUND")]),
            args: ast_exprs
                .into_iter()
                .filter_map(|e| {
                    (e != ast::Expr::Value(ast::Value::Number("0".to_string(), false)))
                        .then_some(ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(e)))
                })
                .collect(),
            over: None,
            distinct: false,
            special: false,
            order_by: vec![],
            filter: None,
            null_treatment: None,
        })
    }
}

impl QueryToRelationTranslator for PostgreSqlTranslator {
    type D = PostgreSqlDialect;

    fn dialect(&self) -> Self::D {
        PostgreSqlDialect {}
    }
}

#[cfg(test)]
mod tests {
    use sqlparser::dialect;

    use super::*;
    use crate::{
        builder::{Ready, With},
        data_type::{DataType, Value as _},
        display::Dot,
        expr::Expr,
        namer,
        relation::{schema::Schema, Relation, TableBuilder},
        sql::{parse, relation::QueryWithRelations},
    };
    use std::sync::Arc;

    fn assert_same_query_str(query_1: &str, query_2: &str) {
        let a_no_whitespace: String = query_1.chars().filter(|c| !c.is_whitespace()).collect();
        let b_no_whitespace: String = query_2.chars().filter(|c| !c.is_whitespace()).collect();
        assert_eq!(a_no_whitespace, b_no_whitespace);
    }

    #[test]
    fn test_query() -> Result<()> {
        let translator = PostgreSqlTranslator;
        let query_str = "SELECT POSITION('o' IN z) AS col FROM table_2";
        let query = parse_with_dialect(query_str, translator.dialect())?;
        println!("{:?}", query);
        Ok(())
    }

    #[test]
    fn test_map() -> Result<()> {
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
        let relations = Hierarchy::from([(["schema", "table"], Arc::new(table))]);

        let query_str = "SELECT log(table.d + 1) FROM schema.table";
        let translator = PostgreSqlTranslator;
        let query = parse_with_dialect(query_str, translator.dialect())?;
        let query_with_relation = QueryWithRelations::new(&query, &relations);
        let relation = Relation::try_from((query_with_relation, translator))?;

        // let retranslated: ast::Query::from()
        print!("{}", relation);
        Ok(())
    }

    #[test]
    fn test_table_special() -> Result<()> {
        let table: Relation = TableBuilder::new()
            .path(["MY SPECIAL TABLE"])
            .name("my_table")
            .size(100)
            .schema(
                Schema::empty()
                    .with(("Id", DataType::integer_interval(0, 1000)))
                    .with(("Na.Me", DataType::text()))
                    .with(("inc&ome", DataType::float_interval(100.0, 200000.0)))
                    .with(("normal_col",  DataType::text())),
            )
            .build();
        let relations = Hierarchy::from([(["schema", "MY SPECIAL TABLE"], Arc::new(table))]);
        let query_str = r#"SELECT "Id", NORMAL_COL, "Na.Me" FROM "MY SPECIAL TABLE" ORDER BY "Id" "#;
        let translator = PostgreSqlTranslator;
        let query = parse_with_dialect(query_str, translator.dialect())?;
        let query_with_relation = QueryWithRelations::new(&query, &relations);
        let relation = Relation::try_from((query_with_relation, translator))?;
        println!("\n {} \n", relation);
        let rel_with_traslator = RelationWithTranslator(&relation, translator);
        let retranslated = ast::Query::from(rel_with_traslator);
        print!("{}", retranslated);
        let translated = r#"
        WITH "map_mou5" ("Id","normal_col","Na.Me") AS (
            SELECT "Id" AS "Id", "normal_col" AS "normal_col", "Na.Me" AS "Na.Me" FROM "MY SPECIAL TABLE"
        ), "map_0swv"("Id","normal_col","Na.Me") AS (
            SELECT "Id" AS "Id", "normal_col" AS "normal_col", "Na.Me" AS "Na.Me" FROM "map_mou5" ORDER BY "Id" ASC
        ) SELECT * FROM "map_0swv"
        "#;
        // assert_same_query_str(&retranslated.to_string(), translated);
        Ok(())
    }
}
