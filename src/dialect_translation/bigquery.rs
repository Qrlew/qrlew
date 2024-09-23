use crate::{
    expr::{self},
    relation::{Join, Variant as _},
};

use super::{function_builder, QueryToRelationTranslator, RelationToQueryTranslator};
use sqlparser::{ast, dialect::BigQueryDialect};

#[derive(Clone, Copy)]
pub struct BigQueryTranslator;

impl RelationToQueryTranslator for BigQueryTranslator {
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
    fn cast_as_float(&self,expr:ast::Expr) -> ast::Expr {
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
    fn extract_epoch(&self, expr:ast::Expr) -> ast::Expr {
        let cast = ast::Expr::Cast {
            expr: Box::new(expr),
            data_type: ast::DataType::Timestamp(None, ast::TimezoneInfo::None),
            format: None,
            kind: ast::CastKind::Cast,
        };
        function_builder("UNIX_SECONDS", vec![cast], false)
    }
}

impl QueryToRelationTranslator for BigQueryTranslator {
    type D = BigQueryDialect;

    fn dialect(&self) -> Self::D {
        BigQueryDialect {}
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::{
        builder::{Ready, With},
        data_type::DataType,
        dialect_translation::RelationWithTranslator,
        expr::Expr,
        namer,
        relation::{schema::Schema, Relation},
    };
    use std::sync::Arc;

    fn assert_same_query_str(query_1: &str, query_2: &str) {
        let a_no_whitespace: String = query_1.chars().filter(|c| !c.is_whitespace()).collect();
        let b_no_whitespace: String = query_2.chars().filter(|c| !c.is_whitespace()).collect();
        assert_eq!(a_no_whitespace, b_no_whitespace);
    }

    #[test]
    fn test_rel_to_query() {
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
                .with(Expr::col("a"))
                .input(table.clone())
                .build(),
        );
        let rel_with_traslator = RelationWithTranslator(map.as_ref(), BigQueryTranslator);
        let query = ast::Query::from(rel_with_traslator);
        let translated = r#"
            WITH `map_1` AS (SELECT `a` AS `field_s7n2` FROM `table`) SELECT * FROM `map_1`
        "#;
        assert_same_query_str(&query.to_string(), translated);
    }
}
