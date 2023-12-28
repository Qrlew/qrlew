use std::sync::Arc;

use crate::{
    expr,
    hierarchy::Hierarchy,
    relation::sql::FromRelationVisitor,
    sql::{parse_with_dialect, query_names::IntoQueryNamesVisitor},
    visitor::Acceptor,
    Relation,
};

use super::{function_builder, QueryToRelationTranslator, RelationToQueryTranslator};
use sqlparser::{ast, dialect::PostgreSqlDialect};

use crate::sql::{Error, Result};
#[derive(Clone, Copy)]
pub struct PostgresTranslator;

impl RelationToQueryTranslator for PostgresTranslator {
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

    fn position(&self, exprs: Vec<&expr::Expr>) -> ast::Expr {
        assert!(exprs.len() == 2);
        let ast_exprs: Vec<ast::Expr> = exprs.into_iter().map(|expr| self.expr(expr)).collect();
        ast::Expr::Position {
            expr: Box::new(ast_exprs[0].clone()),
            r#in: Box::new(ast_exprs[1].clone()),
        }
    }

    fn substr_with_size(&self, exprs: Vec<&expr::Expr>) -> ast::Expr {
        assert!(exprs.len() == 3);
        let ast_exprs: Vec<ast::Expr> = exprs.into_iter().map(|expr| self.expr(expr)).collect();
        ast::Expr::Substring {
            expr: Box::new(ast_exprs[0].clone()),
            substring_from: Some(Box::new(ast_exprs[1].clone())),
            substring_for: Some(Box::new(ast_exprs[2].clone())),
            special: false,
        }
    }

    fn is_null(&self, expr: &expr::Expr) -> ast::Expr {
        let ast_expr: ast::Expr = self.expr(expr);
        ast::Expr::IsNull(Box::new(ast_expr))
    }

    fn ilike(&self, exprs: Vec<&expr::Expr>) -> ast::Expr {
        assert!(exprs.len() == 2);
        let ast_exprs: Vec<ast::Expr> = exprs.into_iter().map(|expr| self.expr(expr)).collect();
        ast::Expr::ILike {
            negated: false,
            expr: Box::new(ast_exprs[0].clone()),
            pattern: Box::new(ast_exprs[1].clone()),
            escape_char: None,
        }
    }

    fn like(&self, exprs: Vec<&expr::Expr>) -> ast::Expr {
        assert!(exprs.len() == 2);
        let ast_exprs: Vec<ast::Expr> = exprs.into_iter().map(|expr| self.expr(expr)).collect();
        ast::Expr::Like {
            negated: false,
            expr: Box::new(ast_exprs[0].clone()),
            pattern: Box::new(ast_exprs[1].clone()),
            escape_char: None,
        }
    }
}

impl QueryToRelationTranslator for PostgresTranslator {
    type D = PostgreSqlDialect;

    fn dialect(&self) -> Self::D {
        PostgreSqlDialect {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        builder::{Ready, With},
        data_type::{DataType, Value as _},
        display::Dot,
        expr::Expr,
        namer,
        relation::{schema::Schema, Relation},
        sql::{parse, relation::QueryWithRelations},
    };
    use std::sync::Arc;

    #[test]
    fn test_query() -> Result<()> {
        let translator = PostgresTranslator;
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
        let translator = PostgresTranslator;
        let query = parse_with_dialect(query_str, translator.dialect())?;
        let query_with_relation = QueryWithRelations::new(&query, &relations);
        let relation = Relation::try_from((query_with_relation, translator))?;

        // let retranslated: ast::Query::from()
        print!("{}", relation);
        Ok(())
    }
}
