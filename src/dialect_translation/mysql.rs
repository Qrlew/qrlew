use super::{function_builder, QueryToRelationTranslator, RelationToQueryTranslator};
use sqlparser::{ast, dialect::MySqlDialect};
use crate::{data_type::DataTyped as _, expr::{self}, relation::{Table, Variant as _}, DataType};


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
}


impl QueryToRelationTranslator for MySqlTranslator {
    type D = MySqlDialect;

    fn dialect(&self) -> Self::D {
        MySqlDialect {}
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