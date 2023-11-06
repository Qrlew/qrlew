use crate::{expr, relation::sql::FromRelationVisitor, visitor::Acceptor, Relation};

use super::IntoDialectTranslator;
use sqlparser::{ast, dialect::MsSqlDialect};

pub struct MSSQLTranslator;

impl IntoDialectTranslator for MSSQLTranslator {
    type D = MsSqlDialect;

    fn dialect(&self) -> Self::D {
        MsSqlDialect {}
    }

    /// Converting LN to LOG
    fn from_ln(&self, expr: &expr::Expr) -> ast::Expr {
        let ast_expr = self.expr(expr);
        let function_arg_expr = ast::FunctionArgExpr::Expr(ast_expr);
        let function_args = ast::FunctionArg::Unnamed(function_arg_expr);
        let name = ast::ObjectName(vec![ast::Ident::from("LOG")]);
        let funtion = ast::Function {
            name,
            args: vec![function_args],
            over: None,
            distinct: false,
            special: false,
            order_by: vec![],
        };
        ast::Expr::Function(funtion)
    }

    /// Converting RANDOM to RAND
    fn from_random(&self) -> ast::Expr {
        ast::Expr::Function(ast::Function {
            name: ast::ObjectName(vec![ast::Ident::from("RAND")]),
            args: vec![],
            over: None,
            distinct: false,
            special: false,
            order_by: vec![],
        })
    }

    /// Converting MD5(X) to CONVERT(VARCHAR(MAX), HASHBYTES('MD5', X), 2)
    fn from_md5(&self, expr: &expr::Expr) -> ast::Expr {
        let ast_expr = self.expr(expr);
        let ast_expr_as_function_arg =
            ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(ast_expr));
        let md5_literal = ast::Expr::Value(ast::Value::SingleQuotedString("MD5".to_string()));
        let md5_literal_as_function_arg =
            ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(md5_literal));
        let hash_byte = ast::Expr::Function(ast::Function {
            name: ast::ObjectName(vec![ast::Ident::from("HASHBYTES")]),
            args: vec![md5_literal_as_function_arg, ast_expr_as_function_arg],
            over: None,
            distinct: false,
            special: false,
            order_by: vec![],
        });

        let hash_byte_as_function_arg =
            ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(hash_byte));

        // VARCHAR(MAX) is treated as a function with MAX argument as identifier.
        let varchartype_expr = ast::Expr::Function(ast::Function {
            name: ast::ObjectName(vec![ast::Ident::from("VARCHAR")]),
            args: vec![ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(
                ast::Expr::Identifier(ast::Ident::from("MAX")),
            ))],
            over: None,
            distinct: false,
            special: false,
            order_by: vec![],
        });

        let varchartype_as_function_arg =
            ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(varchartype_expr));

        ast::Expr::Function(ast::Function {
            name: ast::ObjectName(vec![ast::Ident::from("CONVERT")]),
            args: vec![
                varchartype_as_function_arg,
                hash_byte_as_function_arg,
                ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(ast::Expr::Value(
                    ast::Value::Number("2".to_string(), false),
                ))),
            ],
            over: None,
            distinct: false,
            special: false,
            order_by: vec![],
        })
    }
}

pub struct RelationWithMSSQLTranslator<'a>(pub &'a Relation, pub MSSQLTranslator);

impl<'a> From<RelationWithMSSQLTranslator<'a>> for ast::Query {
    fn from(value: RelationWithMSSQLTranslator) -> Self {
        let RelationWithMSSQLTranslator(relation, translator) = value;
        relation.accept(FromRelationVisitor::new(translator))
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
        sql::parse,
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
    fn test_translation() {
        let input_sql = r#"
        SELECT 3 AS tre, True AS bool, 'sasa' AS str1, CONCAT(1, 2, 3) AS conc, ABS(col) AS abs_col
        "#;
        //let query = parse(input_sql).unwrap();
        let query = parse(input_sql).unwrap();
        println!("{:?}", query)
    }

    #[test]
    fn test_translation_from_relation() {
        let binding = build_complex_relation();
        let rel = binding.as_ref();
        let query = ast::Query::from(rel);
        println!("{}", query)
    }

    #[test]
    fn test_map() {
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
                .with(Expr::ln(Expr::col("a")))
                .input(table.clone())
                .build(),
        );
        map.display_dot().unwrap();

        let query = ast::Query::from(map.as_ref());
        print!("NOT TRANSLATED: \n{}\n", query);

        let rel_with_traslator = RelationWithMSSQLTranslator(map.as_ref(), MSSQLTranslator);
        let query = ast::Query::from(rel_with_traslator);
        print!("TRANSLATED: \n{}\n", query);
    }

    //
    #[test]
    fn test_md5() {
        let input_sql = r#"
        SELECT CONVERT( VARCHAR(MAX), HASHBYTES('MD5', X), 2) FROM table_x
        "#;
        //let query = parse(input_sql).unwrap();
        let query = parse(input_sql).unwrap();
        println!("{:?}", query);

        let schema: Schema = vec![("a", DataType::float())].into_iter().collect();
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
                .with(Expr::md5(Expr::col("a")))
                .input(table.clone())
                .build(),
        );
        map.display_dot().unwrap();

        let query = ast::Query::from(map.as_ref());
        print!("NOT TRANSLATED: \n{}\n", query);

        let rel_with_traslator = RelationWithMSSQLTranslator(map.as_ref(), MSSQLTranslator);
        let query = ast::Query::from(rel_with_traslator);
        print!("TRANSLATED: \n{}\n", query);
    }
}
