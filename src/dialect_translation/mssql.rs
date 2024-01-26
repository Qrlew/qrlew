use crate::{
    data_type::{DataType, DataTyped as _},
    expr::{self, Function as _},
    hierarchy::Hierarchy,
    relation::{sql::FromRelationVisitor, Relation, Table, Variant as _},
    visitor::Acceptor,
    WithoutContext,
};

use super::{function_builder, QueryToRelationTranslator, RelationToQueryTranslator, Result};
use sqlparser::{
    ast::{self, CharacterLength},
    dialect::MsSqlDialect,
};
#[derive(Clone, Copy)]
pub struct MsSqlTranslator;

impl RelationToQueryTranslator for MsSqlTranslator {
    /// Identifiers are back quoted
    fn identifier(&self, value: &expr::Identifier) -> Vec<ast::Ident> {
        let quoting_char: char = '"';
        value
            .iter()
            .map(|i| ast::Ident::with_quote(quoting_char, i))
            .collect()
    }

    fn first(&self, expr: &expr::Expr) -> ast::Expr {
        ast::Expr::from(expr)
    }

    fn mean(&self, expr: &expr::Expr) -> ast::Expr {
        let arg = self.expr(expr);
        function_builder("AVG", vec![arg], false)
    }

    fn std(&self, expr: &expr::Expr) -> ast::Expr {
        let arg = self.expr(expr);
        function_builder("STDEV", vec![arg], false)
    }

    /// Converting LN to LOG
    fn ln(&self, expr: &expr::Expr) -> ast::Expr {
        let arg = self.expr(expr);
        function_builder("LOG", vec![arg], false)
    }

    /// Converting RANDOM to RAND(CHECKSUM(NEWID()))
    fn random(&self) -> ast::Expr {
        let new_id = function_builder("NEWID", vec![], false);
        let check_sum = function_builder("CHECKSUM", vec![new_id], false);
        function_builder("RAND", vec![check_sum], false)
    }

    /// Converting MD5(X) to CONVERT(VARCHAR(MAX), HASHBYTES('MD5', X), 2)
    fn md5(&self, expr: &expr::Expr) -> ast::Expr {
        // In sql parser 0.4 it has been introduced CONVERT as an expression
        // but if doesn't allow for a style argument (see the doc here:
        // https://learn.microsoft.com/fr-fr/sql/t-sql/functions/cast-and-convert-transact-sql?view=sql-server-ver16)
        // which is needed for the convertion.
        // So we can't parse the fllowing:
        // let input_sql = r#"
        // SELECT CONVERT(X, VARCHAR(MAX)) FROM table_x
        // "#;
        // TODO: If we need to parse it, maybe we can use something like CONVERT_SARUS such that it can be
        // seen as a function from the parser. But maybe we don't need to parse it.
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
            filter: None,
            null_treatment: None,
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
            filter: None,
            null_treatment: None,
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
            filter: None,
            null_treatment: None,
        })
    }

    fn cast_as_boolean(&self, expr: &expr::Expr) -> ast::Expr {
        // It should be CAST(expr AS BIT) but BIT is not a valid ast::DataType
        // So we cast it to INT
        let casted_to_integer = expr::Expr::cast_as_integer(expr.clone());
        ast::Expr::from(&casted_to_integer)
    }

    fn cast_as_text(&self, expr: &expr::Expr) -> ast::Expr {
        let ast_expr = self.expr(expr);
        ast::Expr::Cast {
            expr: Box::new(ast_expr),
            data_type: ast::DataType::Nvarchar(Some(255)),
            format: None,
        }
    }
    fn substr(&self, exprs: Vec<&expr::Expr>) -> ast::Expr {
        assert!(exprs.len() == 3);
        let ast_exprs: Vec<ast::Expr> = exprs.into_iter().map(|expr| self.expr(expr)).collect();
        ast::Expr::Substring {
            expr: Box::new(ast_exprs[0].clone()),
            substring_from: Some(Box::new(ast_exprs[1].clone())),
            substring_for: Some(Box::new(ast_exprs[2].clone())),
            special: true,
        }
    }
    fn substr_with_size(&self, exprs: Vec<&expr::Expr>) -> ast::Expr {
        assert!(exprs.len() == 3);
        let ast_exprs: Vec<ast::Expr> = exprs.into_iter().map(|expr| self.expr(expr)).collect();
        ast::Expr::Substring {
            expr: Box::new(ast_exprs[0].clone()),
            substring_from: Some(Box::new(ast_exprs[1].clone())),
            substring_for: Some(Box::new(ast_exprs[2].clone())),
            special: true,
        }
    }
    fn ceil(&self, expr: &expr::Expr) -> ast::Expr {
        let arg = self.expr(expr);
        function_builder("CEILING", vec![arg], false)
    }

    // fn from_extract_epoch(&self, expr: &expr::Expr) -> ast::Expr {
    //     //EXTRACT(EPOCH FROM col1) is not supported yet
    //     todo!()
    // }

    // used during onboarding in order to have datetime with a proper format.
    // This is not needed when we will remove the cast in string of the datetime
    // during the onboarding
    // CAST(col AS VARCHAR/TEXT) -> CONVERT(VARCHAR, col, 126)

    // TODO: some functions are not supported yet.
    // EXTRACT(epoch FROM column) -> DATEDIFF(SECOND, '19700101', column)
    // Concat(a, b) has to take at least 2 args, it can take empty string as well.
    // onboarding, charset query: SELECT DISTINCT REGEXP_SPLIT_TO_TABLE(anon_2.name ,'') AS "regexp_split" ...
    // onboarding, sampling, remove WHERE RAND().
    // onboarding CAST(col AS Boolean) -> CAST(col AS BIT)
    // onboarding Literal True/Fale -> 1/0.

    /// MSSQL queries don't support LIMIT but TOP in the SELECT statement instated
    fn query(
        &self,
        with: Vec<ast::Cte>,
        projection: Vec<ast::SelectItem>,
        from: ast::TableWithJoins,
        selection: Option<ast::Expr>,
        group_by: ast::GroupByExpr,
        order_by: Vec<ast::OrderByExpr>,
        limit: Option<ast::Expr>,
        offset: Option<ast::Offset>,
    ) -> ast::Query {
        let top = limit.map(|e| ast::Top {
            with_ties: false,
            percent: false,
            quantity: Some(ast::TopQuantity::Expr(e)),
        });
        ast::Query {
            with: (!with.is_empty()).then_some(ast::With {
                recursive: false,
                cte_tables: with,
            }),
            body: Box::new(ast::SetExpr::Select(Box::new(ast::Select {
                distinct: None,
                top,
                projection,
                into: None,
                from: vec![from],
                lateral_views: vec![],
                selection,
                group_by,
                cluster_by: vec![],
                distribute_by: vec![],
                sort_by: vec![],
                having: None,
                qualify: None,
                named_window: vec![],
            }))),
            order_by,
            limit: None,
            offset: offset,
            fetch: None,
            locks: vec![],
            limit_by: vec![],
            for_clause: None,
        }
    }

    fn create(&self, table: &Table) -> ast::Statement {
        ast::Statement::CreateTable {
            or_replace: false,
            temporary: false,
            external: false,
            global: None,
            if_not_exists: false,
            transient: false,
            name: table.path().clone().into(),
            columns: table
                .schema()
                .iter()
                .map(|f| ast::ColumnDef {
                    name: f.name().into(),
                    // Need to override some convertions
                    data_type: { translate_data_type(f.data_type()) },
                    collation: None,
                    options: if let DataType::Optional(_) = f.data_type() {
                        vec![]
                    } else {
                        vec![ast::ColumnOptionDef {
                            name: None,
                            option: ast::ColumnOption::NotNull,
                        }]
                    },
                })
                .collect(),
            constraints: vec![],
            hive_distribution: ast::HiveDistributionStyle::NONE,
            hive_formats: None,
            table_properties: vec![],
            with_options: vec![],
            file_format: None,
            location: None,
            query: None,
            without_rowid: false,
            like: None,
            clone: None,
            engine: None,
            default_charset: None,
            collation: None,
            on_commit: None,
            on_cluster: None,
            order_by: None,
            strict: false,
            comment: None,
            auto_increment_offset: None,
            partition_by: None,
            cluster_by: None,
            options: None,
        }
    }
}

impl QueryToRelationTranslator for MsSqlTranslator {
    type D = MsSqlDialect;

    fn dialect(&self) -> Self::D {
        MsSqlDialect {}
    }

    fn try_function(
        &self,
        func: &ast::Function,
        context: &Hierarchy<expr::Identifier>,
    ) -> Result<expr::Expr> {
        let function_name: &str = &func.name.0.iter().next().unwrap().value.to_lowercase()[..];

        match function_name {
            "log" => self.try_ln(func, context),
            "convert" => self.try_md5(func, context),
            // "rand" => self.try_random(func, context),
            _ => {
                // I can't call IntoRelationTranslator::try_function since it is overriden. I can still use expr::Expr::try_from.
                let expr = ast::Expr::Function(func.clone());
                expr::Expr::try_from(expr.with(context))
            }
        }
    }
    /// CONVERT(VARCHAR(MAX), HASHBYTES('MD5', X), 2) to Converting MD5(X)
    fn try_md5(
        &self,
        func: &ast::Function,
        context: &Hierarchy<expr::Identifier>,
    ) -> Result<expr::Expr> {
        // need to check func.args:
        let args = &func.args;
        // We expect 2 args
        if args.len() != 3 {
            let expr = ast::Expr::Function(func.clone());
            expr::Expr::try_from(expr.with(context))
        } else {
            let is_first_arg_valid = is_varchar_valid(&args[0]);
            let is_last_arg_valid = is_literal_two_arg(&args[2]);
            let extract_x_arg = extract_hashbyte_expression_if_valid(&args[1]);
            if is_first_arg_valid && is_last_arg_valid && extract_x_arg.is_some() {
                // Code to execute when both booleans are true and the option is Some
                let converted_x_arg =
                    self.try_function_args(vec![extract_x_arg.unwrap()], context)?;
                Ok(expr::Expr::md5(converted_x_arg[0].clone()))
            } else {
                let expr = ast::Expr::Function(func.clone());
                expr::Expr::try_from(expr.with(context))
            }
        }
    }
}

fn is_literal_two_arg(func_arg: &ast::FunctionArg) -> bool {
    let expected_literal_func_arg = ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(
        ast::Expr::Value(ast::Value::Number("2".to_string(), false)),
    ));
    if func_arg == &expected_literal_func_arg {
        true
    } else {
        false
    }
}

fn is_varchar_valid(func_arg: &ast::FunctionArg) -> bool {
    match func_arg {
        ast::FunctionArg::Unnamed(e) => match e {
            ast::FunctionArgExpr::Expr(e) => match e {
                ast::Expr::Function(f) => {
                    if f.name == ast::ObjectName(vec!["VARCHAR".into()]) {
                        true
                    } else {
                        false
                    }
                }
                _ => false,
            },
            _ => false,
        },
        _ => false,
    }
}

fn extract_hashbyte_expression_if_valid(func_arg: &ast::FunctionArg) -> Option<ast::FunctionArg> {
    let expected_f_name = ast::ObjectName(vec![ast::Ident::from("HASHBYTES")]);
    let md5_literal = ast::Expr::Value(ast::Value::SingleQuotedString("MD5".to_string()));
    let expected_first_arg = ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(md5_literal));
    match func_arg {
        ast::FunctionArg::Named { .. } => None,
        ast::FunctionArg::Unnamed(fargexpr) => match fargexpr {
            ast::FunctionArgExpr::Expr(e) => match e {
                ast::Expr::Function(f) => {
                    if (f.name == expected_f_name) && (f.args[0] == expected_first_arg) {
                        Some(f.args[1].clone())
                    } else {
                        None
                    }
                }
                _ => None,
            },
            _ => None,
        },
    }
}

// method to override DataType -> ast::DataType
fn translate_data_type(dtype: DataType) -> ast::DataType {
    match dtype {
        DataType::Text(_) => ast::DataType::Nvarchar(Some(255)),
        //DataType::Boolean(_) => Boolean should be displayed as BIT for MSSQL,
        // SQLParser doesn't support the BIT DataType (mssql equivalent of bool)
        DataType::Optional(o) => translate_data_type(o.data_type().clone()),
        _ => dtype.into(),
    }
}

#[cfg(test)]
#[cfg(feature = "mssql")]
mod tests {
    use sqlparser::dialect::GenericDialect;

    use super::*;
    use crate::{
        builder::{Ready, With},
        data_type::{DataType, Value as _},
        dialect_translation::RelationWithTranslator,
        display::Dot,
        expr::Expr,
        io::{mssql, Database as _},
        namer,
        relation::{schema::Schema, Relation, Variant as _},
        sql::{parse, parse_expr, parse_with_dialect, relation::QueryWithRelations},
    };
    use std::sync::Arc;

    #[test]
    fn test_limit() {
        let mut database = mssql::test_database();
        let relations = database.relations();

        let query = "SELECT * FROM table_1 LIMIT 30";

        let relation = Relation::try_from(With::with(&parse(query).unwrap(), &relations)).unwrap();

        let rel_with_traslator = RelationWithTranslator(&relation, MsSqlTranslator);
        let translated_query = &ast::Query::from(rel_with_traslator).to_string()[..];
        println!("{}", translated_query);

        let _ = database.query(translated_query).unwrap();
    }

    #[test]
    fn test_cast() {
        let mut database = mssql::test_database();
        let relations = database.relations();

        let query = "SELECT CAST(1 AS boolean) FROM table_2";

        let relation = Relation::try_from(With::with(&parse(query).unwrap(), &relations)).unwrap();

        let rel_with_traslator = RelationWithTranslator(&relation, MsSqlTranslator);
        let translated_query = &ast::Query::from(rel_with_traslator).to_string()[..];
        println!("{}", translated_query);

        let _ = database.query(translated_query).unwrap();
    }

    #[test]
    fn test_cast_bis() {
        let mut database = mssql::test_database();
        let relations = database.relations();
        let query = parse(
            r#"
        SELECT
            CAST(CASE WHEN a > 1 THEN 1 ELSE 0 END AS BOOLEAN) as col
        FROM table_1
        "#,
        )
        .unwrap();

        let relation = Relation::try_from(With::with(&query, &relations)).unwrap();

        let rel_with_traslator = RelationWithTranslator(&relation, MsSqlTranslator);
        let translated_query = &ast::Query::from(rel_with_traslator).to_string()[..];
        println!("{}", translated_query);

        let _ = database.query(translated_query).unwrap();
    }

    fn assert_same_query_str(query_1: &str, query_2: &str) {
        let a_no_whitespace: String = query_1.chars().filter(|c| !c.is_whitespace()).collect();
        let b_no_whitespace: String = query_2.chars().filter(|c| !c.is_whitespace()).collect();
        assert_eq!(a_no_whitespace, b_no_whitespace);
    }

    #[test]
    fn test_ln_rel_to_query() {
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
                .with(Expr::ln(Expr::col("a")))
                .input(table.clone())
                .build(),
        );
        let rel_with_traslator = RelationWithTranslator(map.as_ref(), MsSqlTranslator);
        let query = ast::Query::from(rel_with_traslator);
        let translated = r#"
            WITH map_1 (field_li80) AS (SELECT LOG("a") AS field_li80 FROM "table") SELECT * FROM "map_1"
        "#;
        assert_same_query_str(&query.to_string(), translated);
    }

    #[test]
    fn test_md5() -> Result<()> {
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

        let query = ast::Query::from(map.as_ref());
        print!("NOT TRANSLATED: \n{}\n", query);

        let translator = MsSqlTranslator;
        let rel = map.as_ref();
        let rel_with_traslator = RelationWithTranslator(rel, translator);
        let query = ast::Query::from(rel_with_traslator);
        print!("TRANSLATED: \n{}\n", query);
        let translated = r#"
            WITH map_1(field_cg_6) AS (
                SELECT CONVERT(VARCHAR(MAX), HASHBYTES('MD5',"a"), 2) AS field_cg_6
                FROM "table"
            )
            SELECT * FROM "map_1"
        "#;
        assert_same_query_str(&query.to_string(), translated);
        Ok(())
    }
}
