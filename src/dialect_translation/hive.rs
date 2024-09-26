use super::{function_builder, QueryToRelationTranslator, RelationToQueryTranslator};
use crate::expr::{self};
use sqlparser::{ast, dialect::HiveDialect};

#[derive(Clone, Copy)]
pub struct HiveTranslator;

//Is the same as MySql for now.
impl RelationToQueryTranslator for HiveTranslator {
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

    fn random(&self) -> ast::Expr {
        function_builder("RAND", vec![], false)
    }
    /// Converting LOG to LOG10
    fn log(&self, expr: ast::Expr) -> ast::Expr {
        function_builder("LOG10", vec![expr], false)
    }
    fn cast_as_text(&self, expr: ast::Expr) -> ast::Expr {
        ast::Expr::Cast {
            expr: Box::new(expr),
            data_type: ast::DataType::Char(None),
            format: None,
            kind: ast::CastKind::Cast,
        }
    }
    fn extract_epoch(&self, expr: ast::Expr) -> ast::Expr {
        function_builder("UNIX_TIMESTAMP", vec![expr], false)
    }
    /// For mysql CAST(expr AS INTEGER) should be converted to
    /// CAST(expr AS SIGNED [INTEGER]) which produces a BigInt value.
    /// CONVERT can be also used as CONVERT(expr, SIGNED)
    /// however st::DataType doesn't support SIGNED [INTEGER].
    /// We fix it by creating a function CONVERT(expr, SIGNED).
    fn cast_as_integer(&self, expr: ast::Expr) -> ast::Expr {
        let signed = ast::Expr::Identifier(ast::Ident {
            value: "SIGNED".to_string(),
            quote_style: None,
        });
        function_builder("CONVERT", vec![expr, signed], false)
    }
}

impl QueryToRelationTranslator for HiveTranslator {
    type D = HiveDialect;

    fn dialect(&self) -> Self::D {
        HiveDialect {}
    }
}
