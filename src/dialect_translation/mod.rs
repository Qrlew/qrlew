//! Tools for queries from one dialect into another
//! A specific Dialect is a struct holding:
//!     - a method to provide a sqlparser::Dialect for the parsing
//!     - methods varying from dialect to dialect regarding the conversion from AST to Expr+Relation and vice-versa

use sqlparser::{ast, dialect::Dialect};

use crate::{
    data_type::DataTyped,
    expr::Identifier,
    hierarchy::Hierarchy,
    relation::{Join, JoinOperator, Table, Variant},
    sql::Result,
    DataType, Relation,
};
use crate::{
    expr::{self},
    relation::sql::FromRelationVisitor,
    visitor::Acceptor,
    WithoutContext,
};

use paste::paste;

pub mod bigquery;
pub mod hive;
pub mod mssql;
pub mod mysql;
pub mod postgresql;
pub mod sqlite;

// TODO: Add translatio errors

/// Constructors for creating trait functions with default implementations for generating AST nullary function expressions
macro_rules! nullary_function_ast_constructor {
    ($( $enum:ident ),*) => {
        paste! {
            $(
                fn [<$enum:snake>](&self) -> ast::Expr {
                    function_builder(stringify!([<$enum:snake:upper>]), vec![], false)
                }
            )*
        }
    }
}

/// Constructors for creating trait functions with default implementations for generating AST unnary function expressions
macro_rules! unary_function_ast_constructor {
    ($( $enum:ident ),*) => {
        paste! {
            $(
                fn [<$enum:snake>](&self, expr: ast::Expr) -> ast::Expr {
                    function_builder(stringify!([<$enum:snake:upper>]), vec![expr], false)
                }
            )*
        }
    }
}

/// Constructors for creating trait functions with default implementations for generating AST extract expressions
macro_rules! extract_ast_expression_constructor {
    ($( $enum:ident ),*) => {
        paste! {
            $(
                fn [<extract_ $enum:snake>](&self, expr: ast::Expr) -> ast::Expr {
                    extract_builder(expr, ast::DateTimeField::$enum)
                }
            )*
        }
    }
}

/// Constructors for creating trait functions with default implementations for generating AST nary function expressions
macro_rules! nary_function_ast_constructor {
    ($( $enum:ident ),*) => {
        paste! {
            $(
                fn [<$enum:snake>](&self, exprs: Vec<ast::Expr>) -> ast::Expr {
                    function_builder(stringify!([<$enum:snake:upper>]), exprs, false)
                }
            )*
        }
    }
}

// Constructor matching all supported qrlew expr::function::Functions and generating AST Expressions.
macro_rules! function_match_constructor {
    (
        $self:expr,
        $args:expr,
        $func:expr,
        ($($binary_op:ident),*),
        ($($nullray:ident),*),
        ($($unary:ident),*),
        ($($nary:ident),*),
        $default:expr
    ) => {
        paste! {
            match $func {
                // expand arms for binary_op
                $(
                    expr::function::Function::$binary_op => binary_op_builder($args[0].clone(), ast::BinaryOperator::$binary_op, $args[1].clone()),
                )*

                // expand arms for nullray
                $(
                    expr::function::Function::$nullray => $self.[<$nullray:snake>](),
                )*

                // expand arms for unary
                $(
                    expr::function::Function::$unary => $self.[<$unary:snake>]($args[0].clone()),
                )*

                // expand arms for nary
                $(
                    expr::function::Function::$nary => $self.[<$nary:snake>]($args.clone()),
                )*
                _ => $default
            }
        }
    }
}

/// Trait constructor for building dialect dependent AST parts with default implementations.
/// We use macros to reduce code repetition in generating functions to convert each Expression variant.
macro_rules! relation_to_query_translator_trait_constructor {
    () => {
        pub trait RelationToQueryTranslator {
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
                ast::Query {
                    with: (!with.is_empty()).then_some(ast::With {
                        recursive: false,
                        cte_tables: with,
                    }),
                    body: Box::new(ast::SetExpr::Select(Box::new(ast::Select {
                        distinct: None,
                        top: None,
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
                        window_before_qualify: false,
                        value_table_mode: None,
                        connect_by: None,
                    }))),
                    order_by,
                    limit,
                    limit_by: vec![],
                    offset: offset,
                    fetch: None,
                    locks: vec![],
                    for_clause: None,
                }
            }

            fn create(&self, table: &Table) -> ast::Statement {
                ast::Statement::CreateTable {
                    or_replace: false,
                    temporary: false,
                    external: false,
                    global: None,
                    if_not_exists: true,
                    transient: false,
                    name: ast::ObjectName(self.identifier(&(table.path().clone().into()))),
                    columns: table
                        .schema()
                        .iter()
                        .map(|f| ast::ColumnDef {
                            name: self.identifier(&(f.name().into()))[0].clone(),
                            data_type: f.data_type().into(),
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
                    comment: None,
                    auto_increment_offset: None,
                    partition_by: None,
                    cluster_by: None,
                    options: None,
                    strict: false,
                }
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
                                .map(|i| {
                                    ast::Expr::Value(ast::Value::Placeholder(format!(
                                        "{prefix}{i}"
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

            fn cte(
                &self,
                name: ast::Ident,
                columns: Vec<ast::Ident>,
                query: ast::Query,
            ) -> ast::Cte {
                ast::Cte {
                    alias: ast::TableAlias { name, columns },
                    query: Box::new(query),
                    from: None,
                    materialized: None,
                }
            }
            fn join_projection(&self, _join: &Join) -> Vec<ast::SelectItem> {
                vec![ast::SelectItem::Wildcard(
                    ast::WildcardAdditionalOptions::default(),
                )]
            }

            fn identifier(&self, value: &expr::Identifier) -> Vec<ast::Ident> {
                value
                    .iter()
                    .map(|r| ast::Ident::with_quote('"', r))
                    .collect()
            }

            fn table_factor(&self, relation: &Relation, alias: Option<&str>) -> ast::TableFactor {
                let alias = alias.map(|s| ast::TableAlias {
                    name: self.identifier(&(s.into()))[0].clone(),
                    columns: vec![],
                });
                match relation {
                    Relation::Table(table) => ast::TableFactor::Table {
                        name: ast::ObjectName(self.identifier(table.path())),
                        alias,
                        args: None,
                        with_hints: vec![],
                        version: None,
                        partitions: vec![],
                    },
                    relation => ast::TableFactor::Table {
                        name: ast::ObjectName(self.identifier(&(relation.name().into()))),
                        alias,
                        args: None,
                        with_hints: vec![],
                        version: None,
                        partitions: vec![],
                    },
                }
            }

            fn join_operator(&self, value: &JoinOperator) -> ast::JoinOperator {
                match value {
                    JoinOperator::Inner(expr) => {
                        ast::JoinOperator::Inner(ast::JoinConstraint::On(self.expr(expr)))
                    }
                    JoinOperator::LeftOuter(expr) => {
                        ast::JoinOperator::LeftOuter(ast::JoinConstraint::On(self.expr(expr)))
                    }
                    JoinOperator::RightOuter(expr) => {
                        ast::JoinOperator::RightOuter(ast::JoinConstraint::On(self.expr(expr)))
                    }
                    JoinOperator::FullOuter(expr) => {
                        ast::JoinOperator::FullOuter(ast::JoinConstraint::On(self.expr(expr)))
                    }
                    JoinOperator::Cross => ast::JoinOperator::CrossJoin,
                }
            }

            fn expr(&self, expr: &expr::Expr) -> ast::Expr {
                expr.accept(ExprToAstVisitor { translator: self })
            }

            fn column(&self, ident: &expr::Column) -> ast::Expr {
                let ast_iden = self.identifier(ident.into());
                if ast_iden.len() > 1 {
                    ast::Expr::CompoundIdentifier(ast_iden)
                } else {
                    ast::Expr::Identifier(ast_iden[0].clone())
                }
            }

            fn value(&self, value: &expr::Value) -> ast::Expr {
                match value {
                    expr::Value::Unit(_) => ast::Expr::Value(ast::Value::Null),
                    expr::Value::Boolean(b) => ast::Expr::Value(ast::Value::Boolean(**b)),
                    expr::Value::Integer(i) => {
                        ast::Expr::Value(ast::Value::Number(format!("{}", **i), false))
                    }
                    expr::Value::Enum(_) => todo!(),
                    expr::Value::Float(f) => {
                        ast::Expr::Value(ast::Value::Number(format!("{}", **f), false))
                    }
                    expr::Value::Text(t) => {
                        ast::Expr::Value(ast::Value::SingleQuotedString(format!("{}", **t)))
                    }
                    expr::Value::Bytes(_) => todo!(),
                    expr::Value::Struct(_) => todo!(),
                    expr::Value::Union(_) => todo!(),
                    expr::Value::Optional(optional_val) => match optional_val.as_deref() {
                        Some(arg) => self.value(arg),
                        None => ast::Expr::Value(ast::Value::Null),
                    },
                    expr::Value::List(l) => ast::Expr::Tuple(
                        l.to_vec()
                            .iter()
                            .map(|v| self.value(v))
                            .collect::<Vec<ast::Expr>>(),
                    ),
                    expr::Value::Set(_) => todo!(),
                    expr::Value::Array(_) => todo!(),
                    expr::Value::Date(_) => todo!(),
                    expr::Value::Time(_) => todo!(),
                    expr::Value::DateTime(_) => todo!(),
                    expr::Value::Duration(_) => todo!(),
                    expr::Value::Id(_) => todo!(),
                    expr::Value::Function(_) => todo!(),
                }
            }

            fn function(
                &self,
                function: &expr::function::Function,
                arguments: Vec<ast::Expr>,
            ) -> ast::Expr {
                function_match_constructor!(
                    self,
                    arguments,
                    function,
                    //binary op functions match
                    (
                        Plus,
                        Minus,
                        Multiply,
                        Divide,
                        Modulo,
                        StringConcat,
                        Gt,
                        Lt,
                        GtEq,
                        LtEq,
                        Eq,
                        NotEq,
                        And,
                        Or,
                        Xor,
                        BitwiseOr,
                        BitwiseAnd,
                        BitwiseXor
                    ),
                    //nullary op functions match
                    (Pi, Newid, CurrentDate, CurrentTime, CurrentTimestamp),
                    //unary functions
                    (
                        Exp,
                        Ln,
                        Log,
                        Abs,
                        Sin,
                        Cos,
                        Sqrt,
                        CharLength,
                        Lower,
                        Upper,
                        Md5,
                        CastAsText,
                        CastAsInteger,
                        CastAsFloat,
                        CastAsBoolean,
                        CastAsDateTime,
                        Ceil,
                        Floor,
                        CastAsDate,
                        CastAsTime,
                        Sign,
                        Unhex,
                        ExtractEpoch,
                        ExtractYear,
                        ExtractMonth,
                        ExtractDay,
                        ExtractHour,
                        ExtractMinute,
                        ExtractSecond,
                        ExtractMicrosecond,
                        ExtractMillisecond,
                        ExtractDow,
                        ExtractWeek,
                        Dayname,
                        UnixTimestamp,
                        Quarter,
                        Date,
                        IsNull
                    ),
                    //nary functions
                    (
                        Pow,
                        Position,
                        Least,
                        Greatest,
                        Coalesce,
                        Rtrim,
                        Ltrim,
                        Substr,
                        Round,
                        Trunc,
                        RegexpContains,
                        Encode,
                        Decode,
                        FromUnixtime,
                        DateFormat,
                        Choose,
                        Like,
                        Ilike,
                        IsBool,
                        Case,
                        SubstrWithSize,
                        RegexpExtract,
                        RegexpReplace,
                        DatetimeDiff
                    ),
                    match function {
                        expr::function::Function::Opposite =>
                            unary_op_builder(ast::UnaryOperator::Minus, arguments[0].clone()),
                        expr::function::Function::Not =>
                            unary_op_builder(ast::UnaryOperator::Not, arguments[0].clone()),
                        expr::function::Function::InList => {
                            if let ast::Expr::Tuple(t) = arguments[1].clone() {
                                ast::Expr::InList {
                                    expr: Box::new(arguments[0].clone()),
                                    list: t.clone(),
                                    negated: false,
                                }
                            } else {
                                todo!()
                            }
                        }
                        expr::function::Function::Random(_) => self.random(),
                        expr::function::Function::Concat(_) => self.concat(arguments),
                        _ => todo!(),
                    }
                )
            }

            fn aggregate(
                &self,
                aggregate: &expr::aggregate::Aggregate,
                argument: ast::Expr,
            ) -> ast::Expr {
                match aggregate {
                    expr::aggregate::Aggregate::Min => self.min(argument),
                    expr::aggregate::Aggregate::Max => self.max(argument),
                    expr::aggregate::Aggregate::Median => self.median(argument),
                    expr::aggregate::Aggregate::NUnique => self.n_unique(argument),
                    expr::aggregate::Aggregate::First => self.first(argument),
                    expr::aggregate::Aggregate::Last => self.last(argument),
                    expr::aggregate::Aggregate::Mean => self.mean(argument),
                    expr::aggregate::Aggregate::List => self.list(argument),
                    expr::aggregate::Aggregate::Count => self.count(argument),
                    expr::aggregate::Aggregate::Quantile(_) => self.quantile(argument),
                    expr::aggregate::Aggregate::Quantiles(_) => self.quantiles(argument),
                    expr::aggregate::Aggregate::Sum => self.sum(argument),
                    expr::aggregate::Aggregate::AggGroups => self.agg_groups(argument),
                    expr::aggregate::Aggregate::Std => self.std(argument),
                    expr::aggregate::Aggregate::Var => self.var(argument),
                    expr::aggregate::Aggregate::MeanDistinct => self.mean_distinct(argument),
                    expr::aggregate::Aggregate::CountDistinct => self.count_distinct(argument),
                    expr::aggregate::Aggregate::SumDistinct => self.sum_distinct(argument),
                    expr::aggregate::Aggregate::StdDistinct => self.std_distinct(argument),
                    expr::aggregate::Aggregate::VarDistinct => self.var_distinct(argument),
                }
            }

            nullary_function_ast_constructor!(
                Random,
                Pi,
                Newid,
                CurrentDate,
                CurrentTime,
                CurrentTimestamp
            );

            unary_function_ast_constructor!(
                Exp,
                Ln,
                Log,
                Abs,
                Sin,
                Cos,
                Sqrt,
                CharLength,
                Lower,
                Upper,
                Md5,
                Ceil,
                Floor,
                CastAsDate,
                CastAsTime,
                Sign,
                Unhex,
                Dayname,
                UnixTimestamp,
                Quarter,
                Date,
                Min,
                Max,
                Median,
                NUnique,
                First,
                Last,
                Mean,
                List,
                Count,
                Quantile,
                Quantiles,
                Sum,
                AggGroups,
                Std,
                Var
            );

            nary_function_ast_constructor!(
                Pow,
                Least,
                Greatest,
                Coalesce,
                Rtrim,
                Ltrim,
                Round,
                Trunc,
                RegexpContains,
                Encode,
                Decode,
                FromUnixtime,
                DateFormat,
                Choose,
                IsBool,
                RegexpExtract,
                RegexpReplace,
                DatetimeDiff,
                Concat
            );

            extract_ast_expression_constructor!(
                Epoch,
                Year,
                Month,
                Day,
                Dow,
                Hour,
                Minute,
                Second,
                Microsecond,
                Millisecond
            );

            fn extract_week(&self, expr: ast::Expr) -> ast::Expr {
                extract_builder(expr, ast::DateTimeField::Week(None))
            }

            fn cast_as_text(&self, expr: ast::Expr) -> ast::Expr {
                cast_builder(expr, ast::DataType::Text)
            }
            fn cast_as_float(&self, expr: ast::Expr) -> ast::Expr {
                cast_builder(expr, ast::DataType::Float(None))
            }
            fn cast_as_integer(&self, expr: ast::Expr) -> ast::Expr {
                cast_builder(expr, ast::DataType::Integer(None))
            }
            fn cast_as_boolean(&self, expr: ast::Expr) -> ast::Expr {
                cast_builder(expr, ast::DataType::Boolean)
            }
            fn cast_as_date_time(&self, expr: ast::Expr) -> ast::Expr {
                cast_builder(expr, ast::DataType::Datetime(None))
            }
            fn case(&self, exprs: Vec<ast::Expr>) -> ast::Expr {
                assert!(exprs.len() == 3);
                let mut when = vec![exprs[0].clone()];
                let mut then = vec![exprs[1].clone()];

                let _else = match &exprs[2] {
                    ast::Expr::Case {
                        conditions,
                        results,
                        else_result,
                        ..
                    } => {
                        when.extend(conditions.clone());
                        then.extend(results.clone());
                        else_result.clone()
                    }
                    s => Some(Box::new(s.clone())),
                };
                case_builder(when, then, _else)
            }
            fn count_distinct(&self, expr: ast::Expr) -> ast::Expr {
                function_builder("COUNT", vec![expr], true)
            }
            fn sum_distinct(&self, expr: ast::Expr) -> ast::Expr {
                function_builder("SUM", vec![expr], true)
            }
            fn mean_distinct(&self, expr: ast::Expr) -> ast::Expr {
                function_builder("AVG", vec![expr], true)
            }
            fn std_distinct(&self, expr: ast::Expr) -> ast::Expr {
                function_builder("STDDEV", vec![expr], true)
            }
            fn var_distinct(&self, expr: ast::Expr) -> ast::Expr {
                function_builder("VARIANCE", vec![expr], true)
            }
            fn position(&self, exprs: Vec<ast::Expr>) -> ast::Expr {
                assert!(exprs.len() == 2);
                ast::Expr::Position {
                    expr: Box::new(exprs[0].clone()),
                    r#in: Box::new(exprs[1].clone()),
                }
            }
            fn substr(&self, exprs: Vec<ast::Expr>) -> ast::Expr {
                assert!(exprs.len() == 2);
                ast::Expr::Substring {
                    expr: Box::new(exprs[0].clone()),
                    substring_from: Some(Box::new(exprs[1].clone())),
                    substring_for: None,
                    special: false,
                }
            }
            fn substr_with_size(&self, exprs: Vec<ast::Expr>) -> ast::Expr {
                assert!(exprs.len() == 3);
                ast::Expr::Substring {
                    expr: Box::new(exprs[0].clone()),
                    substring_from: Some(Box::new(exprs[1].clone())),
                    substring_for: Some(Box::new(exprs[2].clone())),
                    special: false,
                }
            }
            fn is_null(&self, expr: ast::Expr) -> ast::Expr {
                ast::Expr::IsNull(Box::new(expr))
            }
            fn ilike(&self, exprs: Vec<ast::Expr>) -> ast::Expr {
                assert!(exprs.len() == 2);
                ast::Expr::ILike {
                    negated: false,
                    expr: Box::new(exprs[0].clone()),
                    pattern: Box::new(exprs[1].clone()),
                    escape_char: None,
                }
            }
            fn like(&self, exprs: Vec<ast::Expr>) -> ast::Expr {
                assert!(exprs.len() == 2);
                ast::Expr::Like {
                    negated: false,
                    expr: Box::new(exprs[0].clone()),
                    pattern: Box::new(exprs[1].clone()),
                    escape_char: None,
                }
            }
        }
    };
}

relation_to_query_translator_trait_constructor!();

/// Build Sarus Relation from dialect specific AST
pub trait QueryToRelationTranslator {
    type D: Dialect;

    fn dialect(&self) -> Self::D;

    // It converts ast Expressions to sarus expressions
    fn try_expr(&self, expr: &ast::Expr, context: &Hierarchy<Identifier>) -> Result<expr::Expr> {
        match expr {
            ast::Expr::Function(func) => self.try_function(func, context),
            _ => expr::Expr::try_from(expr.with(context)),
        }
    }

    // The construction of qrlew expressions depends dynamically from the function name
    fn try_function(
        &self,
        func: &ast::Function,
        context: &Hierarchy<Identifier>,
    ) -> Result<expr::Expr> {
        let function_name: &str = &func.name.0.iter().next().unwrap().value.to_lowercase()[..];

        match function_name {
            "log" => self.try_log(func, context),
            "ln" => self.try_ln(func, context),
            "md5" => self.try_md5(func, context),
            // "random" => self.try_random(),
            _ => {
                let expr = ast::Expr::Function(func.clone());
                expr::Expr::try_from(expr.with(context))
            }
        }
    }
    fn try_ln(&self, func: &ast::Function, context: &Hierarchy<Identifier>) -> Result<expr::Expr> {
        let converted = self.try_function_args(func.args.clone(), context)?;
        Ok(expr::Expr::ln(converted[0].clone()))
    }

    fn try_log(&self, func: &ast::Function, context: &Hierarchy<Identifier>) -> Result<expr::Expr> {
        let converted = self.try_function_args(func.args.clone(), context)?;
        Ok(expr::Expr::log(converted[0].clone()))
    }

    fn try_md5(&self, func: &ast::Function, context: &Hierarchy<Identifier>) -> Result<expr::Expr> {
        let converted = self.try_function_args(func.args.clone(), context)?;
        Ok(expr::Expr::md5(converted[0].clone()))
    }

    fn try_function_args(
        &self,
        args: ast::FunctionArguments,
        context: &Hierarchy<Identifier>,
    ) -> Result<Vec<expr::Expr>> {
        match args {
            ast::FunctionArguments::None | ast::FunctionArguments::Subquery(_) => Ok(vec![]),
            ast::FunctionArguments::List(arg_list) => arg_list
                .args
                .iter()
                .map(|func_arg| match func_arg {
                    ast::FunctionArg::Named { arg, .. } | ast::FunctionArg::Unnamed(arg) => {
                        self.try_function_arg_expr(arg, context)
                    }
                })
                .collect(),
        }
    }

    fn try_function_arg_expr(
        &self,
        func_arg_expr: &ast::FunctionArgExpr,
        context: &Hierarchy<Identifier>,
    ) -> Result<expr::Expr> {
        match func_arg_expr {
            ast::FunctionArgExpr::Expr(e) => self.try_expr(e, context),
            ast::FunctionArgExpr::QualifiedWildcard(_o) => todo!(),
            ast::FunctionArgExpr::Wildcard => todo!(),
        }
    }
}

// Helpers

// Implement RelationToQueryTranslator for references
impl<T: RelationToQueryTranslator + ?Sized> RelationToQueryTranslator for &T {
    fn column(&self, column: &expr::Column) -> ast::Expr {
        (**self).column(column)
    }

    fn value(&self, value: &expr::Value) -> ast::Expr {
        (**self).value(value)
    }

    fn function(
        &self,
        function: &expr::function::Function,
        arguments: Vec<ast::Expr>,
    ) -> ast::Expr {
        (**self).function(function, arguments)
    }

    fn aggregate(&self, aggregate: &expr::aggregate::Aggregate, argument: ast::Expr) -> ast::Expr {
        (**self).aggregate(aggregate, argument)
    }
}

struct ExprToAstVisitor<T> {
    translator: T,
}

impl<'a, T: RelationToQueryTranslator> expr::Visitor<'a, ast::Expr> for ExprToAstVisitor<T> {
    fn column(&self, column: &'a expr::Column) -> ast::Expr {
        self.translator.column(column)
    }

    fn value(&self, value: &'a expr::Value) -> ast::Expr {
        self.translator.value(value)
    }

    fn function(
        &self,
        function: &'a expr::function::Function,
        arguments: Vec<ast::Expr>,
    ) -> ast::Expr {
        self.translator.function(function, arguments)
    }

    fn aggregate(
        &self,
        aggregate: &'a expr::aggregate::Aggregate,
        argument: ast::Expr,
    ) -> ast::Expr {
        self.translator.aggregate(aggregate, argument)
    }

    fn structured(&self, _fields: Vec<(Identifier, ast::Expr)>) -> ast::Expr {
        todo!()
    }
}

// AST Function expression builder
fn function_builder(name: &str, exprs: Vec<ast::Expr>, distinct: bool) -> ast::Expr {
    let function_args: Vec<ast::FunctionArg> = exprs
        .into_iter()
        .map(|e| {
            let function_arg_expr = ast::FunctionArgExpr::Expr(e);
            ast::FunctionArg::Unnamed(function_arg_expr)
        })
        .collect();
    let function_name = name.to_uppercase();
    let name = ast::ObjectName(vec![ast::Ident::from(&function_name[..])]);
    let ast_distinct = if distinct {
        Some(ast::DuplicateTreatment::Distinct)
    } else {
        None
    };
    let func_args_list = ast::FunctionArgumentList {
        duplicate_treatment: ast_distinct,
        args: function_args,
        clauses: vec![],
    };
    let function = ast::Function {
        name,
        args: ast::FunctionArguments::List(func_args_list),
        over: None,
        filter: None,
        null_treatment: None,
        within_group: vec![],
    };
    ast::Expr::Function(function)
}

// AST CAST expression builder
fn cast_builder(expr: ast::Expr, as_type: ast::DataType) -> ast::Expr {
    ast::Expr::Cast {
        expr: Box::new(expr),
        data_type: as_type,
        format: None,
        kind: ast::CastKind::Cast,
    }
}

// AST CASE expression builder
fn case_builder(
    when: Vec<ast::Expr>,
    then: Vec<ast::Expr>,
    _else: Option<Box<ast::Expr>>,
) -> ast::Expr {
    ast::Expr::Case {
        operand: None,
        conditions: when,
        results: then,
        else_result: _else,
    }
}

// AST Binary oparation expression builder
fn binary_op_builder(left: ast::Expr, op: ast::BinaryOperator, right: ast::Expr) -> ast::Expr {
    ast::Expr::BinaryOp {
        left: Box::new(ast::Expr::Nested(Box::new(left))),
        op,
        right: Box::new(ast::Expr::Nested(Box::new(right))),
    }
}

// AST Unary oparation expression builder
fn unary_op_builder(op: ast::UnaryOperator, expr: ast::Expr) -> ast::Expr {
    ast::Expr::UnaryOp {
        op: op,
        expr: Box::new(ast::Expr::Nested(Box::new(expr))),
    }
}

fn extract_builder(expr: ast::Expr, datetime_field: ast::DateTimeField) -> ast::Expr {
    ast::Expr::Extract {
        field: datetime_field,
        expr: Box::new(expr),
    }
}

pub struct RelationWithTranslator<'a, T: RelationToQueryTranslator>(pub &'a Relation, pub T);

impl<'a, T: RelationToQueryTranslator> From<RelationWithTranslator<'a, T>> for ast::Query {
    fn from(value: RelationWithTranslator<'a, T>) -> Self {
        let RelationWithTranslator(rel, translator) = value;
        rel.accept(FromRelationVisitor::new(translator))
    }
}
