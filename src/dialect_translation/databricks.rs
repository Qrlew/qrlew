use super::{function_builder, QueryToRelationTranslator, RelationToQueryTranslator};
use sqlparser::{ast, dialect::DatabricksDialect};

use crate::{
    expr::{self},
};


#[derive(Clone, Copy)]
pub struct DatabricksTranslator;

impl RelationToQueryTranslator for DatabricksTranslator {
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

    fn identifier(&self, value: &expr::Identifier) -> Vec<ast::Ident> {
        value
            .iter()
            .map(|r| ast::Ident::with_quote('`', r))
            .collect()
    }

    fn cte(&self, name: ast::Ident, _columns: Vec<ast::Ident>, query: ast::Query) -> ast::Cte {
        ast::Cte {
            alias: ast::TableAlias {
                name,
                columns: vec![],
            },
            query: Box::new(query),
            from: None,
            materialized: None,
        }
    }
}


impl QueryToRelationTranslator for DatabricksTranslator {
    type D = DatabricksDialect;

    fn dialect(&self) -> Self::D {
        DatabricksDialect {}
    }
}
