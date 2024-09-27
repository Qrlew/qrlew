use super::{function_builder, QueryToRelationTranslator, RelationToQueryTranslator};
use crate::{
    expr::{self},
    relation::{Join, Variant as _},
};
use sqlparser::{ast, dialect::HiveDialect};

#[derive(Clone, Copy)]
pub struct HiveTranslator;

// Using the same translations as in bigquery since it should be similar.
// HiveTranslator is not tested at the moment.
impl RelationToQueryTranslator for HiveTranslator {
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
    /// Converting LOG to LOG10
    fn log(&self, expr: ast::Expr) -> ast::Expr {
        function_builder("LOG10", vec![expr], false)
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
        ast::Expr::Cast {
            expr: Box::new(expr),
            data_type: ast::DataType::Float64,
            format: None,
            kind: ast::CastKind::Cast,
        }
    }
    fn substr(&self, exprs: Vec<ast::Expr>) -> ast::Expr {
        assert!(exprs.len() == 2);
        function_builder("SUBSTR", exprs, false)
    }
    fn substr_with_size(&self, exprs: Vec<ast::Expr>) -> ast::Expr {
        assert!(exprs.len() == 3);
        function_builder("SUBSTR", exprs, false)
    }
    /// Converting MD5(X) to TO_HEX(MD5(X))
    fn md5(&self, expr: ast::Expr) -> ast::Expr {
        let md5_function = function_builder("MD5", vec![expr], false);
        function_builder("TO_HEX", vec![md5_function], false)
    }
    fn random(&self) -> ast::Expr {
        function_builder("RAND", vec![], false)
    }
    fn join_projection(&self, join: &Join) -> Vec<ast::SelectItem> {
        join.left()
            .schema()
            .iter()
            .map(|f| self.expr(&expr::Expr::qcol(Join::left_name(), f.name())))
            .chain(
                join.right()
                    .schema()
                    .iter()
                    .map(|f| self.expr(&expr::Expr::qcol(Join::right_name(), f.name()))),
            )
            .zip(join.schema().iter())
            .map(|(expr, field)| ast::SelectItem::ExprWithAlias {
                expr,
                alias: field.name().into(),
            })
            .collect()
    }
    /// It converts EXTRACT(epoch FROM column) into
    /// UNIX_SECONDS(CAST(col AS TIMESTAMP))
    fn extract_epoch(&self, expr: ast::Expr) -> ast::Expr {
        let cast = ast::Expr::Cast {
            expr: Box::new(expr),
            data_type: ast::DataType::Timestamp(None, ast::TimezoneInfo::None),
            format: None,
            kind: ast::CastKind::Cast,
        };
        function_builder("UNIX_SECONDS", vec![cast], false)
    }
}

impl QueryToRelationTranslator for HiveTranslator {
    type D = HiveDialect;

    fn dialect(&self) -> Self::D {
        HiveDialect {}
    }
}
