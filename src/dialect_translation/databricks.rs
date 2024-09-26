use super::{function_builder, QueryToRelationTranslator, RelationToQueryTranslator};
use sqlparser::{ast, dialect::DatabricksDialect};

use crate::expr::{self};

#[derive(Clone, Copy)]
pub struct DatabricksTranslator;

impl RelationToQueryTranslator for DatabricksTranslator {
    fn identifier(&self, value: &expr::Identifier) -> Vec<ast::Ident> {
        value
            .iter()
            .map(|r| ast::Ident::with_quote('`', r))
            .collect()
    }

    fn first(&self, expr: ast::Expr) -> ast::Expr {
        expr
    }

    fn var(&self, expr: ast::Expr) -> ast::Expr {
        function_builder("VARIANCE", vec![expr], false)
    }

    fn cast_as_text(&self, expr: ast::Expr) -> ast::Expr {
        ast::Expr::Cast {
            expr: Box::new(expr),
            data_type: ast::DataType::String(None),
            format: None,
            kind: ast::CastKind::Cast,
        }
    }
    fn cast_as_float(&self, expr: ast::Expr) -> ast::Expr {
        function_builder("FLOAT", vec![expr], false)
    }
    /// It converts EXTRACT(epoch FROM column) into
    /// UNIX_TIMESTAMP(col)
    fn extract_epoch(&self, expr: ast::Expr) -> ast::Expr {
        function_builder("UNIX_TIMESTAMP", vec![expr], false)
    }

    fn format_float_value(&self, value: f64) -> ast::Expr {
        let max_precision = 37;
        let formatted = if value.abs() < 1e-10 || value.abs() > 1e10 {
            // If the value is too small or too large, switch to scientific notation
            format!("{:.precision$e}", value, precision = max_precision)
        } else {
            // Otherwise, use the default float formatting with the specified precision
            format!("{}", value)
        };
        ast::Expr::Value(ast::Value::Number(formatted, false))
    }
}

impl QueryToRelationTranslator for DatabricksTranslator {
    type D = DatabricksDialect;

    fn dialect(&self) -> Self::D {
        DatabricksDialect {}
    }
}
