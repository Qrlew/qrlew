use super::{function_builder, QueryToRelationTranslator, RelationToQueryTranslator};
use sqlparser::{ast, dialect::RedshiftSqlDialect};

#[derive(Clone, Copy)]
pub struct RedshiftSqlTranslator;

// Copied from postgres since it is very similar.
impl RelationToQueryTranslator for RedshiftSqlTranslator {
    fn first(&self, expr: ast::Expr) -> ast::Expr {
        expr
    }

    fn mean(&self, expr: ast::Expr) -> ast::Expr {
        function_builder("AVG", vec![expr], false)
    }

    fn var(&self, expr: ast::Expr) -> ast::Expr {
        function_builder("VARIANCE", vec![expr], false)
    }

    fn std(&self, expr: ast::Expr) -> ast::Expr {
        function_builder("STDDEV", vec![expr], false)
    }

    fn trunc(&self, exprs: Vec<ast::Expr>) -> ast::Expr {
        // TRUNC in postgres has a problem:
        // In TRUNC(double_precision_number, precision) if precision is specified it fails
        // If it is not specified it passes considering precision = 0.
        // SELECT TRUNC(CAST (0.12 AS DOUBLE PRECISION), 0) fails
        // SELECT TRUNC(CAST (0.12 AS DOUBLE PRECISION)) passes.
        // Here we check precision, if it is 0 we remove it (such that the precision is implicit).
        let func_args_list = ast::FunctionArgumentList {
            duplicate_treatment: None,
            args: exprs
                .into_iter()
                .filter_map(|e| {
                    (e != ast::Expr::Value(ast::Value::Number("0".to_string(), false)))
                        .then_some(ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(e)))
                })
                .collect(),
            clauses: vec![],
        };
        ast::Expr::Function(ast::Function {
            name: ast::ObjectName(vec![ast::Ident::from("TRUNC")]),
            args: ast::FunctionArguments::List(func_args_list),
            over: None,
            filter: None,
            null_treatment: None,
            within_group: vec![],
        })
    }

    fn round(&self, exprs: Vec<ast::Expr>) -> ast::Expr {
        // Same as TRUNC
        // what if I wanted to do round(0, 0)
        let func_args_list = ast::FunctionArgumentList {
            duplicate_treatment: None,
            args: exprs
                .into_iter()
                .filter_map(|e| {
                    (e != ast::Expr::Value(ast::Value::Number("0".to_string(), false)))
                        .then_some(ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(e)))
                })
                .collect(),
            clauses: vec![],
        };
        ast::Expr::Function(ast::Function {
            name: ast::ObjectName(vec![ast::Ident::from("ROUND")]),
            args: ast::FunctionArguments::List(func_args_list),
            over: None,
            filter: None,
            null_treatment: None,
            within_group: vec![],
        })
    }
}

impl QueryToRelationTranslator for RedshiftSqlTranslator {
    type D = RedshiftSqlDialect;

    fn dialect(&self) -> Self::D {
        RedshiftSqlDialect {}
    }
}