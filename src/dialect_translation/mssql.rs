use crate::{
    data_type::{DataType, DataTyped as _},
    expr::{self},
    hierarchy::Hierarchy,
    relation::{Table, Variant as _},
    WithoutContext,
};

use super::{function_builder, QueryToRelationTranslator, RelationToQueryTranslator, Result};
use sqlparser::{
    ast::{self},
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

    fn first(&self, expr: ast::Expr) -> ast::Expr {
        expr
    }

    fn mean(&self, expr: ast::Expr) -> ast::Expr {
        function_builder("AVG", vec![expr], false)
    }

    fn std(&self, expr: ast::Expr) -> ast::Expr {
        function_builder("STDEV", vec![expr], false)
    }

    /// Converting LN to LOG
    fn ln(&self, expr: ast::Expr) -> ast::Expr {
        function_builder("LOG", vec![expr], false)
    }
    /// Converting LOG to LOG10
    fn log(&self, expr: ast::Expr) -> ast::Expr {
        function_builder("LOG10", vec![expr], false)
    }

    /// Converting RANDOM to RAND(CHECKSUM(NEWID()))
    fn random(&self) -> ast::Expr {
        let new_id = function_builder("NEWID", vec![], false);
        let check_sum = function_builder("CHECKSUM", vec![new_id], false);
        function_builder("RAND", vec![check_sum], false)
    }

    fn char_length(&self, expr:ast::Expr) -> ast::Expr {
        function_builder("LEN", vec![expr], false)
    }

    /// Converting MD5(X) to CONVERT(VARCHAR(MAX), HASHBYTES('MD5', X), 2)
    fn md5(&self, expr: ast::Expr) -> ast::Expr {
        // Construct HASHBYTES('MD5', X)
        let md5_literal = ast::Expr::Value(ast::Value::SingleQuotedString("MD5".to_string()));
        let md5_literal_as_function_arg =
            ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(md5_literal));
        let ast_expr_as_function_arg = ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(expr));

        let func_args_list = ast::FunctionArgumentList {
            duplicate_treatment: None,
            args: vec![md5_literal_as_function_arg, ast_expr_as_function_arg],
            clauses: vec![],
        };
        let hash_byte_expr = ast::Expr::Function(ast::Function {
            name: ast::ObjectName(vec![ast::Ident::from("HASHBYTES")]),
            args: ast::FunctionArguments::List(func_args_list),
            over: None,
            filter: None,
            null_treatment: None,
            within_group: vec![],
        });

        // Construct the CONVERT expr
        ast::Expr::Convert {
            expr: Box::new(hash_byte_expr),
            data_type: Some(ast::DataType::Varchar(Some(ast::CharacterLength::Max))),
            charset: None,
            target_before_value: true,
            styles: vec![ast::Expr::Value(ast::Value::Number("2".to_string(), false))],
        }
    }

    fn cast_as_boolean(&self, expr: ast::Expr) -> ast::Expr {
        // It should be CAST(expr AS BIT) but BIT is not a valid ast::DataType
        // So we cast it to INT
        self.cast_as_integer(expr)
    }

    fn cast_as_text(&self, expr: ast::Expr) -> ast::Expr {
        ast::Expr::Cast {
            expr: Box::new(expr),
            data_type: ast::DataType::Nvarchar(Some(ast::CharacterLength::IntegerLength {
                length: 255,
                unit: None,
            })),
            format: None,
            kind: ast::CastKind::Cast,
        }
    }
    fn substr(&self, exprs: Vec<ast::Expr>) -> ast::Expr {
        assert!(exprs.len() == 3);
        ast::Expr::Substring {
            expr: Box::new(exprs[0].clone()),
            substring_from: Some(Box::new(exprs[1].clone())),
            substring_for: Some(Box::new(exprs[2].clone())),
            special: true,
        }
    }
    fn substr_with_size(&self, exprs: Vec<ast::Expr>) -> ast::Expr {
        assert!(exprs.len() == 3);
        ast::Expr::Substring {
            expr: Box::new(exprs[0].clone()),
            substring_from: Some(Box::new(exprs[1].clone())),
            substring_for: Some(Box::new(exprs[2].clone())),
            special: true,
        }
    }
    fn ceil(&self, expr: ast::Expr) -> ast::Expr {
        function_builder("CEILING", vec![expr], false)
    }
    fn extract_epoch(&self, expr: ast::Expr) -> ast::Expr {
        let second = ast::Expr::Identifier(ast::Ident {
            value: "SECOND".to_string(),
            quote_style: None,
        });
        let unix = ast::Expr::Value(ast::Value::SingleQuotedString("19700101".to_string()));
        function_builder("DATEDIFF", vec![second, unix, expr], false)
    }

    fn concat(&self, exprs:Vec<ast::Expr>) -> ast::Expr {
        let literal = ast::Expr::Value(ast::Value::SingleQuotedString("".to_string()));
        let expanded_exprs: Vec<_> = exprs
            .iter()
            .cloned()  
            .chain(std::iter::once(literal))
            .collect();
        function_builder("CONCAT", expanded_exprs, false)
    }

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

        let translated_projection: Vec<ast::SelectItem> = projection.iter().map(case_from_not).collect();
        let translated_selection: Option<ast::Expr> = selection.and_then(none_from_where_random).and_then(boolean_expr_from_identifier);

        ast::Query {
            with: (!with.is_empty()).then_some(ast::With {
                recursive: false,
                cte_tables: with,
            }),
            body: Box::new(ast::SetExpr::Select(Box::new(ast::Select {
                distinct: None,
                top,
                projection: translated_projection,
                into: None,
                from: vec![from],
                lateral_views: vec![],
                selection: translated_selection,
                group_by,
                cluster_by: vec![],
                distribute_by: vec![],
                sort_by: vec![],
                having: None,
                qualify: None,
                named_window: vec![],
                window_before_qualify: false,
                value_table_mode: None,
                connect_by: None,
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
            name: ast::ObjectName(self.identifier(&(table.path().clone().into()))),
            columns: table
                .schema()
                .iter()
                .map(|f| ast::ColumnDef {
                    name: self.identifier(&(f.name().into()))[0].clone(),
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
        println!("{}", function_name);
        match function_name {
            "log" => self.try_ln(func, context),
            "log10" => self.try_log(func, context),
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
        let args = match &func.args {
            ast::FunctionArguments::None | ast::FunctionArguments::Subquery(_) => vec![],
            ast::FunctionArguments::List(l) => l.args.iter().collect(),
        };
        // We expect 2 args
        if args.len() != 3 {
            let expr = ast::Expr::Function(func.clone());
            expr::Expr::try_from(expr.with(context))
        } else {
            let is_first_arg_valid = is_varchar_valid(&args[0]);
            let is_last_arg_valid = is_literal_two_arg(&args[2]);
            let extract_x_arg = extract_hashbyte_expression_if_valid(&args[1]);
            if is_first_arg_valid && is_last_arg_valid && extract_x_arg.is_some() {
                let function_args = ast::FunctionArgumentList {
                    duplicate_treatment: None,
                    args: vec![extract_x_arg.unwrap()],
                    clauses: vec![],
                };
                let converted_x_arg =
                    self.try_function_args(ast::FunctionArguments::List(function_args), context)?;
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
                    let arg_vec = match &f.args {
                        ast::FunctionArguments::None | ast::FunctionArguments::Subquery(_) => {
                            vec![]
                        }
                        ast::FunctionArguments::List(func_args) => func_args.args.iter().collect(),
                    };
                    if (f.name == expected_f_name) && (arg_vec[0] == &expected_first_arg) {
                        Some(arg_vec[1].clone())
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

// It converts a `NOT (col IS NULL)` in the SELECT items into a CASE expression
fn case_from_not(select_item: &ast::SelectItem) -> ast::SelectItem {
    match select_item {
        ast::SelectItem::ExprWithAlias { expr, alias } => ast::SelectItem::ExprWithAlias { expr: case_from_not_expr(expr), alias: alias.clone() },
        ast::SelectItem::UnnamedExpr(expr) => ast::SelectItem::UnnamedExpr(case_from_not_expr(expr)),
        _ =>  select_item.clone()
    }
}

fn case_from_not_expr(expr: &ast::Expr) -> ast::Expr {
    match expr {
        ast::Expr::UnaryOp { op, expr } => case_from_not_unary_op(op, expr),
        _ => expr.clone()
    }
}

fn case_from_not_unary_op(op: &ast::UnaryOperator, expr: &Box<ast::Expr>) -> ast::Expr {
    match op {
        ast::UnaryOperator::Not => {
            // NOT( some_bool_expr ) -> CASE WHEN some_bool_expr THEN 0 ELSE 1
            let when_expr = vec![expr.as_ref().clone()];
            let then_expr  = vec![ast::Expr::Value(ast::Value::Number("0".to_string(), false))];
            let else_expr = Box::new(ast::Expr::Value(ast::Value::Number("1".to_string(), false)));
            ast::Expr::Case {
                operand: None,
                conditions: when_expr,
                results: then_expr,
                else_result: Some(else_expr),
            }
        },
        _ => ast::Expr::UnaryOp {op: op.clone(), expr: expr.clone()}
    }
}

// WHERE expretion modifications:

/// Often sampling queries uses WHERE RAND(CHECKSUM(NEWID())) < x but in mssql
/// this doesn't associate a random value for each row.
/// Use ruther this approach to sample:
/// https://www.sqlservercentral.com/forums/topic/whats-the-best-way-to-get-a-sample-set-of-a-big-table-without-primary-key#post-1948778 
/// Careful!! If RAND function is found the WHERE will be set to None.
fn none_from_where_random(expr: ast::Expr) -> Option<ast::Expr> {
    if has_rand_func(&expr) {
        None
    } else {
        Some(expr)
    }
}

// It checks recursively if the Expr is RAND function.
fn has_rand_func(expr: &ast::Expr) -> bool {
    match expr {
        ast::Expr::Function(func) => {
            let ast::Function {name, ..} = func;
            let rand_func_name = ast::ObjectName(vec![ast::Ident::from("RAND")]);
            &rand_func_name == name
        },
        ast::Expr::BinaryOp { left , .. } => has_rand_func(left.as_ref()),
        ast::Expr::Nested(expr) => has_rand_func(expr.as_ref()),
        _ => false
    }
}

// In Mssql WHERE col is not accepted.
// This function converts WHERE col -> WHERE col=1
fn boolean_expr_from_identifier(expr: ast::Expr) -> Option<ast::Expr> {
    match expr {
        ast::Expr::Identifier(_) => Some(ast::Expr::BinaryOp { left: Box::new(expr), op: ast::BinaryOperator::Eq, right: Box::new(ast::Expr::Value(ast::Value::Number("1".to_string(), false))) }),
        _ => Some(expr)
    }
}

// method to override DataType -> ast::DataType
fn translate_data_type(dtype: DataType) -> ast::DataType {
    match dtype {
        DataType::Text(_) => ast::DataType::Nvarchar(Some(ast::CharacterLength::IntegerLength {
            length: 255,
            unit: None,
        })),
        //DataType::Boolean(_) => Boolean should be displayed as BIT for MSSQL,
        // SQLParser doesn't support the BIT DataType (mssql equivalent of bool)
        DataType::Optional(o) => translate_data_type(o.data_type().clone()),
        _ => dtype.into(),
    }
}

#[cfg(test)]
#[cfg(feature = "mssql")]
mod tests {
    use super::*;
    use crate::{
        builder::{Ready, With}, data_type::DataType, dialect_translation::RelationWithTranslator, display::Dot, expr::Expr, io::{mssql, Database as _}, namer, relation::{schema::Schema, Relation}, sql::parse
    };
    use std::sync::Arc;

    #[test]
    fn test_coalesce() {
        let mut database = mssql::test_database();
        let relations = database.relations();

        let query = "SELECT COALESCE(a, NULL) FROM table_1 LIMIT 30";

        let relation = Relation::try_from(With::with(&parse(query).unwrap(), &relations)).unwrap();
        relation.display_dot().unwrap();
        let rel_with_traslator = RelationWithTranslator(&relation, MsSqlTranslator);
        let translated_query = &ast::Query::from(rel_with_traslator).to_string()[..];
        println!("{}", translated_query);
        let _ = database.query(translated_query).unwrap();
    }

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
    fn test_not() {
        // let mut database = mssql::test_database();
        let schema: Schema = vec![("a", DataType::optional(DataType::float()))].into_iter().collect();
        let table: Arc<Relation> = Arc::new(
            Relation::table()
                .name("table_2")
                .schema(schema.clone())
                .size(100)
                .build(),
        );

        let relations = Hierarchy::from([(vec!["table_2"], table)]);

        let query = "WITH new_tab AS (SELECT NOT (a IS NULL) AS col FROM table_2) SELECT * FROM new_tab WHERE RANDOM()) < (0.9) ";
        // let query = "SELECT DISTINCT a FROM table_2 LIMIT 10";

        let relation = Relation::try_from(With::with(&parse(query).unwrap(), &relations)).unwrap();
        relation.display_dot().unwrap();

        let rel_with_traslator = RelationWithTranslator(&relation, MsSqlTranslator);
        let translated_query = ast::Query::from(rel_with_traslator);
        println!("{:?}", translated_query);
        println!("\n{}\n", translated_query);

        // let _ = database.query(translated_query).unwrap();
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
            WITH "map_1" ("field_li80") AS (SELECT LOG("a") AS "field_li80" FROM "table") SELECT * FROM "map_1"
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
            WITH "map_1"("field_cg_6") AS (
                SELECT CONVERT(VARCHAR(MAX), HASHBYTES('MD5',"a"), 2) AS "field_cg_6"
                FROM "table"
            )
            SELECT * FROM "map_1"
        "#;
        assert_same_query_str(&query.to_string(), translated);
        Ok(())
    }
}
