use super::{function_builder, QueryToRelationTranslator, RelationToQueryTranslator};
use sqlparser::{ast, dialect::MySqlDialect};

#[derive(Clone, Copy)]
pub struct MySqlTranslator;

impl RelationToQueryTranslator for MySqlTranslator {
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
}


impl QueryToRelationTranslator for MySqlTranslator {
    type D = MySqlDialect;

    fn dialect(&self) -> Self::D {
        MySqlDialect {}
    }
}

