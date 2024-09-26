use super::{function_builder, QueryToRelationTranslator, RelationToQueryTranslator, Result};
use sqlparser::{ast, dialect::MySqlDialect};
use crate::{data_type::DataTyped as _, expr::{self}, hierarchy::Hierarchy, relation::{Table, Variant as _}, DataType, WithoutContext as _};


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

    fn identifier(&self, value: &expr::Identifier) -> Vec<ast::Ident> {
        value
            .iter()
            .map(|r| ast::Ident::with_quote('`', r))
            .collect()
    }
    fn insert(&self, prefix: &str, table: &Table) -> ast::Statement {
        ast::Statement::Insert(ast::Insert {
            or: None,
            into: true,
            table_name: ast::ObjectName(self.identifier(&(table.path().clone().into()))),
            table_alias: None,
            columns: table
                .schema()
                .iter()
                .map(|f| self.identifier(&(f.name().into()))[0].clone())
                .collect(),
            overwrite: false,
            source: Some(Box::new(ast::Query {
                with: None,
                body: Box::new(ast::SetExpr::Values(ast::Values {
                    explicit_row: false,
                    rows: vec![(1..=table.schema().len())
                        .map(|_| {
                            ast::Expr::Value(ast::Value::Placeholder(format!(
                                "{prefix}"
                            )))
                        })
                        .collect()],
                })),
                order_by: vec![],
                limit: None,
                limit_by: vec![],
                offset: None,
                fetch: None,
                locks: vec![],
                for_clause: None,
            })),
            partitioned: None,
            after_columns: vec![],
            table: false,
            on: None,
            returning: None,
            ignore: false,
            replace_into: false,
            priority: None,
            insert_alias: None,
        })
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
    fn extract_epoch(&self,expr:ast::Expr) -> ast::Expr {
        function_builder("UNIX_TIMESTAMP", vec![expr], false)
    }
    /// For mysql CAST(expr AS INTEGER) should be converted to
    /// CAST(expr AS SIGNED [INTEGER]) which produces a BigInt value.
    /// CONVERT can be also used as CONVERT(expr, SIGNED)
    /// however st::DataType doesn't support SIGNED [INTEGER].
    /// We fix it by creating a function CONVERT(expr, SIGNED).
    fn cast_as_integer(&self,expr:ast::Expr) -> ast::Expr {
        let signed = ast::Expr::Identifier(ast::Ident{value: "SIGNED".to_string(), quote_style: None});
        function_builder("CONVERT", vec![expr, signed], false)
    }

    // encode(source, 'escape') -> source
    // encode(source, 'hex') -> hex(source)
    // encode(source, 'base64') -> to_base64(source)
    fn encode(&self,exprs:Vec<ast::Expr>) -> ast::Expr {
        assert_eq!(exprs.len(), 2);
        let source = exprs[0].clone();
        match &exprs[1] {
            ast::Expr::Value(ast::Value::SingleQuotedString( s)) if s == &"hex".to_string() => function_builder("HEX", vec![source], false),
            ast::Expr::Value(ast::Value::SingleQuotedString(s)) if s == &"base64".to_string()=> function_builder("TO_BASE64", vec![source], false),
            _ => source
        }
    }

    // decode(source, 'hex') -> CONVERT(unhex(source) USING utf8mb4)
    // decode(source, 'escape') -> CONVERT(source USING utf8mb4)
    // decode(source, 'base64') -> CONVERT(from_base64(source) USING utf8mb4)
    fn decode(&self,exprs:Vec<ast::Expr>) -> ast::Expr {
        assert_eq!(exprs.len(), 2);
        let source = exprs[0].clone();
        let binary_expr = match &exprs[1] {
            ast::Expr::Value(ast::Value::SingleQuotedString( s)) if s == &"hex".to_string() => function_builder("UNHEX", vec![source], false),
            ast::Expr::Value(ast::Value::SingleQuotedString(s)) if s == &"base64".to_string()=> function_builder("FROM_BASE64", vec![source], false),
            _ => source
        };
        let char_enc = ast::ObjectName(vec![ast::Ident{value: "utf8mb4".to_string(), quote_style: None}]);
        ast::Expr::Convert { expr: Box::new(binary_expr), data_type: None, charset: Some(char_enc), target_before_value: false, styles: vec![] }
    }


}


impl QueryToRelationTranslator for MySqlTranslator {
    type D = MySqlDialect;

    fn dialect(&self) -> Self::D {
        MySqlDialect {}
    }

    fn try_function(
        &self,
        func: &ast::Function,
        context: &Hierarchy<expr::Identifier>,
    ) -> Result<expr::Expr> {
        let function_name: &str = &func.name.0.iter().next().unwrap().value.to_lowercase()[..];
        let converted = self.try_function_args(func.args.clone(), context)?;

        match function_name {
            "log" => self.try_ln(func, context),
            "log10" => self.try_log(func, context),
            "convert" => self.try_md5(func, context),
            "unhex" => try_encode_decode(converted, EncodeDecodeFormat::Hex),
            "from_base64" => try_encode_decode(converted, EncodeDecodeFormat::Base64),
            "hex" => try_encode(converted, EncodeDecodeFormat::Hex),
            "to_base64" => try_encode(converted, EncodeDecodeFormat::Base64),
            "rand" => Ok(expr::Expr::random(0)),
            _ => {
                let expr = ast::Expr::Function(func.clone());
                expr::Expr::try_from(expr.with(context))
            }
        }
    }
}


// method to override DataType -> ast::DataType
fn translate_data_type(dtype: DataType) -> ast::DataType {
    match dtype {
        DataType::Text(_) => ast::DataType::Varchar(Some(ast::CharacterLength::IntegerLength {
            length: 255,
            unit: None,
        })),
        //DataType::Boolean(_) => Boolean should be displayed as BIT for MSSQL,
        // SQLParser doesn't support the BIT DataType (mssql equivalent of bool)
        DataType::Optional(o) => translate_data_type(o.data_type().clone()),
        _ => dtype.into(),
    }
}
enum EncodeDecodeFormat {
    Hex,
    Base64
}

// unhex(source) -> encode(decode(source, 'hex'), 'escape')
// from_base64(source) -> encode(decode(source, 'base_64'), 'escape')
fn try_encode_decode(exprs: Vec<expr::Expr>, format: EncodeDecodeFormat) -> Result<expr::Expr> {
    assert_eq!(exprs.len(), 1);
    let format = match format {
        EncodeDecodeFormat::Hex => expr::Expr::val("hex".to_string()),
        EncodeDecodeFormat::Base64 => expr::Expr::val("base64".to_string()),
    };
    let decode = expr::Expr::decode(exprs[0].clone(), format);
    let escape = expr::Expr::val("escape".to_string());
    Ok(expr::Expr::encode(decode, escape))
}

// hex(source) -> encode(source, 'hex')
// to_base64(source) -> encode(source, 'base64')
fn try_encode(exprs: Vec<expr::Expr>, format: EncodeDecodeFormat) -> Result<expr::Expr> {
    assert_eq!(exprs.len(), 1);
    let format = match format {
        EncodeDecodeFormat::Hex => expr::Expr::val("hex".to_string()),
        EncodeDecodeFormat::Base64 => expr::Expr::val("base64".to_string()),
    };
    Ok(expr::Expr::encode(exprs[0].clone(), format))
}

#[cfg(test)]
#[cfg(feature = "mysql")]
mod tests {
    use base64::display;
    use itertools::Itertools as _;

    use super::*;
    use crate::{
        builder::{Ready, With}, dialect_translation::{postgresql::PostgreSqlTranslator, RelationWithTranslator}, display::Dot, io::{mysql, postgresql, Database as _}, namer, relation::{schema::Schema, Relation}, sql::{self, parse, relation::QueryWithRelations}
    };


    fn try_from_mssql_query(mysql_query: &str, relations: Hierarchy<std::sync::Arc<Relation>>) -> Relation {
        let parsed_query = sql::relation::parse_with_dialect(mysql_query, MySqlTranslator.dialect()).unwrap();
        // let parsed_query = parse(mysql_query).unwrap();
        let query_with_translator = (QueryWithRelations::new(&parsed_query, &relations), MySqlTranslator);
        Relation::try_from(query_with_translator).unwrap()
    }


    #[test]
    fn test_unhex() {
        let mut mysql_database = mysql::test_database();
        let mut psql_database = postgresql::test_database();
        let relations = mysql_database.relations();

        let initial_mysql_query = "SELECT unhex('50726976617465') FROM table_2 LIMIT 1";
        let rel = try_from_mssql_query(initial_mysql_query, relations);

        let rel_with_traslator = RelationWithTranslator(&rel, PostgreSqlTranslator);
        let psql_query = &ast::Query::from(rel_with_traslator).to_string()[..];

        let rel_with_traslator = RelationWithTranslator(&rel, MySqlTranslator);
        let mysql_query = &ast::Query::from(rel_with_traslator).to_string()[..];

        println!("{}", initial_mysql_query);
        println!("{}", mysql_query);
        println!("{}", psql_query);
        let res_initial_mysql = mysql_database
            .query(initial_mysql_query)
            .unwrap()
            .iter()
            .map(ToString::to_string)
            .join("\n");
        
        let res_mysql = mysql_database
            .query(mysql_query)
            .unwrap()
            .iter()
            .map(ToString::to_string)
            .join("\n");
        let res_psql = psql_database
            .query(psql_query)
            .unwrap()
            .iter()
            .map(ToString::to_string)
            .join("\n");
        assert_eq!(res_mysql, "(Private)".to_string());
        assert_eq!(res_mysql, res_psql);
        assert_eq!(res_mysql, res_initial_mysql)
    }

    #[test]
    fn test_hex() {
        let mut mysql_database = mysql::test_database();
        let mut psql_database = postgresql::test_database();
        let relations = mysql_database.relations();
        
        let initial_mysql_query = "SELECT hex('Private') FROM table_2 LIMIT 1";
        let rel = try_from_mssql_query(initial_mysql_query, relations);

        let rel_with_traslator = RelationWithTranslator(&rel, PostgreSqlTranslator);
        let psql_query = &ast::Query::from(rel_with_traslator).to_string()[..];

        let rel_with_traslator = RelationWithTranslator(&rel, MySqlTranslator);
        let mysql_query = &ast::Query::from(rel_with_traslator).to_string()[..];

        println!("{}", initial_mysql_query);
        println!("{}", mysql_query);
        println!("{}", psql_query);
        let res_initial_mysql = mysql_database
            .query(initial_mysql_query)
            .unwrap()
            .iter()
            .map(ToString::to_string)
            .join("\n");
        
        let res_mysql = mysql_database
            .query(mysql_query)
            .unwrap()
            .iter()
            .map(ToString::to_string)
            .join("\n");
        let res_psql = psql_database
            .query(psql_query)
            .unwrap()
            .iter()
            .map(ToString::to_string)
            .join("\n");
        assert_eq!(res_mysql, "(50726976617465)".to_string());
        assert_eq!(res_mysql, res_psql);
        assert_eq!(res_mysql, res_initial_mysql)
    }

    #[test]
    fn test_from_base64() {
        let mut mysql_database = mysql::test_database();
        let mut psql_database = postgresql::test_database();
        let relations = mysql_database.relations();
        
        let initial_mysql_query = "SELECT from_base64('YWJj') FROM table_2 LIMIT 1";
        let rel = try_from_mssql_query(initial_mysql_query, relations);

        let rel_with_traslator = RelationWithTranslator(&rel, PostgreSqlTranslator);
        let psql_query = &ast::Query::from(rel_with_traslator).to_string()[..];

        let rel_with_traslator = RelationWithTranslator(&rel, MySqlTranslator);
        let mysql_query = &ast::Query::from(rel_with_traslator).to_string()[..];

        println!("{}", initial_mysql_query);
        println!("{}", mysql_query);
        println!("{}", psql_query);
        let res_initial_mysql = mysql_database
            .query(initial_mysql_query)
            .unwrap()
            .iter()
            .map(ToString::to_string)
            .join("\n");
        
        let res_mysql = mysql_database
            .query(mysql_query)
            .unwrap()
            .iter()
            .map(ToString::to_string)
            .join("\n");
        let res_psql = psql_database
            .query(psql_query)
            .unwrap()
            .iter()
            .map(ToString::to_string)
            .join("\n");
        assert_eq!(res_mysql, res_psql);
        assert_eq!(res_mysql, res_initial_mysql);
        assert_eq!(res_mysql, "(abc)".to_string());
    }

    #[test]
    fn test_to_base64() {
        let mut mysql_database = mysql::test_database();
        let mut psql_database = postgresql::test_database();
        let relations = mysql_database.relations();
        
        let initial_mysql_query = "SELECT TO_BASE64('abc') FROM table_2 LIMIT 1";
        let rel = try_from_mssql_query(initial_mysql_query, relations);

        let rel_with_traslator = RelationWithTranslator(&rel, PostgreSqlTranslator);
        let psql_query = &ast::Query::from(rel_with_traslator).to_string()[..];

        let rel_with_traslator = RelationWithTranslator(&rel, MySqlTranslator);
        let mysql_query = &ast::Query::from(rel_with_traslator).to_string()[..];

        println!("{}", initial_mysql_query);
        println!("{}", mysql_query);
        println!("{}", psql_query);
        let res_initial_mysql = mysql_database
            .query(initial_mysql_query)
            .unwrap()
            .iter()
            .map(ToString::to_string)
            .join("\n");
        
        let res_mysql = mysql_database
            .query(mysql_query)
            .unwrap()
            .iter()
            .map(ToString::to_string)
            .join("\n");
        let res_psql = psql_database
            .query(psql_query)
            .unwrap()
            .iter()
            .map(ToString::to_string)
            .join("\n");
        assert_eq!(res_mysql, res_psql);
        assert_eq!(res_mysql, res_initial_mysql);
        assert_eq!(res_mysql, "(YWJj)".to_string());
    }
}