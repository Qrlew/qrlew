//! Tools for queries from one dialect into another
//! A specific Dialect is a struct holding:
//!     - a method to provide a sqlparser::Dialect for the parsing
//!     - methods varying from dialect to dialect regarding the conversion from AST to Expr+Relation and vice-versa
use std::{iter::once, ops::Deref};

use sqlparser::{
    ast,
    dialect::{BigQueryDialect, Dialect, PostgreSqlDialect},
};

use crate::{
    data_type::function::cast,
    expr::{self, Function},
    relation::{self, sql::FromRelationVisitor},
    visitor::Acceptor,
    WithContext, WithoutContext,
};
use crate::{
    data_type::DataTyped,
    expr::Identifier,
    hierarchy::Hierarchy,
    relation::{JoinConstraint, JoinOperator, Table, Variant},
    sql::{
        self, parse, parse_with_dialect,
        relation::{RelationWithColumns, VisitedQueryRelations},
        Error, Result,
    },
    DataType, Relation,
};

use paste::paste;

pub mod bigquery;
pub mod hive;
pub mod mssql;
pub mod mysql;
pub mod postgres;
pub mod sqlite;

// TODO: Add translatio errors

/// Constructors for creating trait functions with default implementations for generating AST nullary function expressions
macro_rules! nullary_function_ast_constructor {
    ($( $enum:ident ),*) => {
        paste! {
            $(
                fn [<from_$enum:snake>](&self) -> ast::Expr {
                    function_builder(stringify!([<$enum:upper>]), vec![])
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
                fn [<from_$enum:snake>](&self, expr: &expr::Expr) -> ast::Expr {
                    let ast_expr = self.expr(expr);
                    function_builder(stringify!([<$enum:upper>]), vec![ast_expr])
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
                fn [<from_$enum:snake>](&self, exprs: Vec<&expr::Expr>) -> ast::Expr {
                    let ast_exprs: Vec<ast::Expr> = exprs.into_iter().map(|expr| self.expr(expr)).collect();
                    function_builder(stringify!([<$enum:upper>]), ast_exprs)
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
        ($($nullay:ident),*),
        ($($unary:ident),*),
        ($($nary:ident),*),
        $default:expr
    ) => {
        paste! {
            match $func {
                // expand arms for binary_op
                $(
                    expr::function::Function::$binary_op => binary_op_builder($self.expr($args[0]), ast::BinaryOperator::$binary_op, $self.expr($args[1])),
                )*

                // expand arms for nullay
                $(
                    expr::function::Function::$nullay => $self.[<from_ $nullay:snake>](),
                )*

                // expand arms for unary
                $(
                    expr::function::Function::$unary => $self.[<from_ $unary:snake>]($args[0]),
                )*

                // expand arms for nary
                $(
                    expr::function::Function::$nary => $self.[<from_ $nary:snake>]($args),
                )*
                _ => $default
            }
        }
    }
}

/// Trait constructor for building dialect dependent AST parts with default implementations.
/// We use macros to reduce code repetition in generating functions to convert each Epression variant.
macro_rules! into_dialect_tranlator_trait_constructor {
    () => {
        pub trait IntoDialectTranslator {
            fn query(
                &self,
                with: Vec<ast::Cte>,
                projection: Vec<ast::SelectItem>,
                from: ast::TableWithJoins,
                selection: Option<ast::Expr>,
                group_by: ast::GroupByExpr,
                order_by: Vec<ast::OrderByExpr>,
                limit: Option<ast::Expr>,
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
                    }))),
                    order_by,
                    limit,
                    limit_by: vec![],
                    offset: None,
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
                    name: table.path().clone().into(),
                    columns: table
                        .schema()
                        .iter()
                        .map(|f| ast::ColumnDef {
                            name: f.name().into(),
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
                    strict: false,
                    comment: None,
                    auto_increment_offset: None,
                }
            }

            fn insert(&self, prefix: &str, table: &Table) -> ast::Statement {
                ast::Statement::Insert {
                    or: None,
                    into: true,
                    table_name: table.path().clone().into(),
                    columns: table.schema().iter().map(|f| f.name().into()).collect(),
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
                }
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
                }
            }

            fn identifier(&self, value: &expr::Identifier) -> Vec<ast::Ident> {
                value.iter().map(ast::Ident::new).collect()
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

            fn join_constraint(&self, value: &JoinConstraint) -> ast::JoinConstraint {
                match value {
                    JoinConstraint::On(expr) => ast::JoinConstraint::On(self.expr(expr)),
                    JoinConstraint::Using(idents) => ast::JoinConstraint::Using(
                        idents
                            .into_iter()
                            .map(|ident| self.identifier(&ident)[0].clone())
                            .collect(),
                    ),
                    JoinConstraint::Natural => ast::JoinConstraint::Natural,
                    JoinConstraint::None => ast::JoinConstraint::None,
                }
            }

            fn join_operator(&self, value: &JoinOperator) -> ast::JoinOperator {
                match value {
                    JoinOperator::Inner(join_constraint) => {
                        ast::JoinOperator::Inner(self.join_constraint(join_constraint))
                    }
                    JoinOperator::LeftOuter(join_constraint) => {
                        ast::JoinOperator::LeftOuter(self.join_constraint(join_constraint))
                    }
                    JoinOperator::RightOuter(join_constraint) => {
                        ast::JoinOperator::RightOuter(self.join_constraint(join_constraint))
                    }
                    JoinOperator::FullOuter(join_constraint) => {
                        ast::JoinOperator::FullOuter(self.join_constraint(join_constraint))
                    }
                    JoinOperator::Cross => ast::JoinOperator::CrossJoin,
                }
            }

            fn expr(&self, expr: &expr::Expr) -> ast::Expr {
                match expr {
                    expr::Expr::Column(ident) => self.from_column(ident),
                    expr::Expr::Value(value) => self.from_value(value),
                    expr::Expr::Function(func) => self.from_function(func),
                    expr::Expr::Aggregate(agg) => self.from_aggregate(agg),
                    expr::Expr::Struct(_) => todo!(),
                }
            }

            fn from_column(&self, ident: &expr::Identifier) -> ast::Expr {
                let ast_iden = self.identifier(ident);
                if ast_iden.len() > 1 {
                    ast::Expr::CompoundIdentifier(ast_iden)
                } else {
                    ast::Expr::Identifier(ast_iden[0].clone())
                }
            }

            fn from_value(&self, value: &expr::Value) -> ast::Expr {
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
                    expr::Value::Optional(_) => todo!(),
                    expr::Value::List(l) => ast::Expr::Tuple(
                        l.to_vec()
                            .iter()
                            .map(|v| self.from_value(v))
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

            fn from_function(&self, func: &Function) -> ast::Expr {
                let binding = func.arguments();
                let args: Vec<&expr::Expr> = binding.iter().collect();

                function_match_constructor!(
                    self,
                    args,
                    func.function(),
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
                    (),
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
                        CastAsFloat,
                        CastAsInteger,
                        CastAsDateTime
                    ),
                    (Pow, Case, Position, Least, Greatest),
                    match func.function() {
                        expr::function::Function::Opposite =>
                            unary_op_builder(ast::UnaryOperator::Minus, self.expr(args[0])),
                        expr::function::Function::Not =>
                            unary_op_builder(ast::UnaryOperator::Not, self.expr(args[0])),
                        expr::function::Function::InList => {
                            if let ast::Expr::Tuple(t) = self.expr(args[1]) {
                                ast::Expr::InList {
                                    expr: Box::new(self.expr(args[0])),
                                    list: t.clone(),
                                    negated: false,
                                }
                            } else {
                                todo!()
                            }
                        }
                        expr::function::Function::Random(_) => self.from_random(),
                        expr::function::Function::Concat(_) => self.from_concat(args),
                        _ => todo!(),
                    }
                )
            }

            fn from_aggregate(&self, agg: &expr::Aggregate) -> ast::Expr {
                let arg = agg.argument();
                match agg.aggregate() {
                    expr::aggregate::Aggregate::Min => self.from_min(arg),
                    expr::aggregate::Aggregate::Max => self.from_max(arg),
                    expr::aggregate::Aggregate::Median => self.from_median(arg),
                    expr::aggregate::Aggregate::NUnique => self.from_n_unique(arg),
                    expr::aggregate::Aggregate::First => self.from_first(arg),
                    expr::aggregate::Aggregate::Last => self.from_last(arg),
                    expr::aggregate::Aggregate::Mean => self.from_mean(arg),
                    expr::aggregate::Aggregate::List => self.from_list(arg),
                    expr::aggregate::Aggregate::Count => self.from_count(arg),
                    expr::aggregate::Aggregate::Quantile(_) => self.from_quantile(arg),
                    expr::aggregate::Aggregate::Quantiles(_) => self.from_quantiles(arg),
                    expr::aggregate::Aggregate::Sum => self.from_sum(arg),
                    expr::aggregate::Aggregate::AggGroups => self.from_agg_groups(arg),
                    expr::aggregate::Aggregate::Std => self.from_std(arg),
                    expr::aggregate::Aggregate::Var => self.from_var(arg),
                    expr::aggregate::Aggregate::MeanDistinct => self.from_mean_distinct(arg),
                    expr::aggregate::Aggregate::CountDistinct => self.from_count_distinct(arg),
                    expr::aggregate::Aggregate::SumDistinct => self.from_sum_distinct(arg),
                    expr::aggregate::Aggregate::StdDistinct => self.from_std_distinct(arg),
                    expr::aggregate::Aggregate::VarDistinct => self.from_var_distinct(arg),
                }
            }

            nullary_function_ast_constructor!(Random);

            unary_function_ast_constructor!(
                Exp, Ln, Log, Abs, Sin, Cos, Sqrt, CharLength, Lower, Upper, Md5, Min, Max, Median,
                NUnique, First, Last, Mean, List, Count, Quantile, Quantiles, Sum, AggGroups, Std,
                Var, MeanDistinct, CountDistinct, SumDistinct, StdDistinct, VarDistinct
            );

            nary_function_ast_constructor!(Pow, Concat, Position, Least, Greatest);

            fn from_cast_as_text(&self, expr: &expr::Expr) -> ast::Expr {
                let ast_expr = self.expr(expr);
                cast_builder(ast_expr, ast::DataType::Text)
            }
            fn from_cast_as_float(&self, expr: &expr::Expr) -> ast::Expr {
                let ast_expr = self.expr(expr);
                cast_builder(ast_expr, ast::DataType::Float(Some(64)))
            }
            fn from_cast_as_integer(&self, expr: &expr::Expr) -> ast::Expr {
                let ast_expr = self.expr(expr);
                cast_builder(ast_expr, ast::DataType::Integer(Some(64)))
            }
            fn from_cast_as_date_time(&self, expr: &expr::Expr) -> ast::Expr {
                let ast_expr = self.expr(expr);
                cast_builder(ast_expr, ast::DataType::Datetime(Some(64)))
            }
            fn from_case(&self, exprs: Vec<&expr::Expr>) -> ast::Expr {
                let ast_exprs: Vec<ast::Expr> =
                    exprs.into_iter().map(|expr| self.expr(expr)).collect();
                case_builder(ast_exprs)
            }
        }
    };
}

into_dialect_tranlator_trait_constructor!();

/// Constructors for creating functions that convert AST functions with
/// a single args to annequivalent sarus functions
macro_rules! try_into_unary_function_constructor {
    ($( $enum:ident ),*) => {
        paste! {
            $(
                fn [<try_into_ $enum:snake>](&self, arg: &ast::Function, context: &Hierarchy<Identifier>) -> Result<expr::Expr> {
                    let converted = self.try_from_function_args(vec![arg.clone()], context)?;
                    Ok(expr::Expr::[<$enum:snake>](converted[0]))
                }
            )*
        }
    }
}

// macro_rules! unary_function_matcher {
//     (
//         $self:expr,
//         $args:expr,
//         $context:expr,
//         $func_name:expr,
//         ($($unary_function:ident),*),
//         $default:expr
//     ) => {
//         paste! {
//             match $func_name {
//                 $(
//                     stringify!([<$unary_function:lower>]) => $self.[<try_from_ $unary_function:snake>]($args[0], $context),
//                 )*
//                 _ => $default
//             }
//         }
//     }
// }

/// Build Sarus Relatioin from dialect speciific AST
// macro_rules! into_ralation_tranlator_trait_constructor {
//     () => {
pub trait IntoRelationTranslator {
    type D: Dialect;

    fn dialect(&self) -> Self::D;

    // It converts ast Expressions to sarus expressions
    fn try_from_expr(
        &self,
        expr: &ast::Expr,
        context: &Hierarchy<Identifier>,
    ) -> Result<expr::Expr> {
        match expr {
            ast::Expr::Function(func) => self.try_from_function(func, context),
            _ => expr::Expr::try_from(expr.with(context)),
        }
    }

    // The construction of qrlew expressions depends dynamically from the function name
    fn try_from_function(
        &self,
        func: &ast::Function,
        context: &Hierarchy<Identifier>,
    ) -> Result<expr::Expr> {
        let function_name: &str = &func.name.0.iter().next().unwrap().value.to_lowercase()[..];

        match function_name {
            "log" => self.try_into_ln(func, context),
            "md5" => self.try_into_md5(func, context),
            // "random" => self.try_into_random(),
            _ => {
                let expr = ast::Expr::Function(func.clone());
                expr::Expr::try_from(expr.with(context))
            }
        }
    }

    fn try_into_ln(
        &self,
        func: &ast::Function,
        context: &Hierarchy<Identifier>,
    ) -> Result<expr::Expr> {
        let converted = self.try_from_function_args(func.args.clone(), context)?;
        Ok(expr::Expr::ln(converted[0].clone()))
    }

    fn try_into_md5(
        &self,
        func: &ast::Function,
        context: &Hierarchy<Identifier>,
    ) -> Result<expr::Expr> {
        let converted = self.try_from_function_args(func.args.clone(), context)?;
        Ok(expr::Expr::md5(converted[0].clone()))
    }

    fn try_from_function_args(
        &self,
        args: Vec<ast::FunctionArg>,
        context: &Hierarchy<Identifier>,
    ) -> Result<Vec<expr::Expr>> {
        args.iter()
            .map(|func_arg| match func_arg {
                ast::FunctionArg::Named { name: _, arg } => {
                    self.try_from_function_arg_expr(arg, context)
                }
                ast::FunctionArg::Unnamed(arg) => self.try_from_function_arg_expr(arg, context),
            })
            .collect()
    }

    fn try_from_function_arg_expr(
        &self,
        func_arg_expr: &ast::FunctionArgExpr,
        context: &Hierarchy<Identifier>,
    ) -> Result<expr::Expr> {
        match func_arg_expr {
            ast::FunctionArgExpr::Expr(e) => self.try_from_expr(e, context),
            ast::FunctionArgExpr::QualifiedWildcard(o) => todo!(),
            ast::FunctionArgExpr::Wildcard => todo!(),
        }
    }
}
//     };
// }

// into_ralation_tranlator_trait_constructor!();

// Helpers

// AST Function expression builder
fn function_builder(name: &str, exprs: Vec<ast::Expr>) -> ast::Expr {
    let function_args: Vec<ast::FunctionArg> = exprs
        .into_iter()
        .map(|e| {
            let function_arg_expr = ast::FunctionArgExpr::Expr(e);
            ast::FunctionArg::Unnamed(function_arg_expr)
        })
        .collect();
    let function_name = name.to_uppercase();
    let name = ast::ObjectName(vec![ast::Ident::from(&function_name[..])]);
    let funtion = ast::Function {
        name,
        args: function_args,
        over: None,
        distinct: false,
        special: false,
        order_by: vec![],
        filter: None,
        null_treatment: None,
    };
    ast::Expr::Function(funtion)
}

// AST CAST expression builder
fn cast_builder(expr: ast::Expr, as_type: ast::DataType) -> ast::Expr {
    ast::Expr::Cast {
        expr: Box::new(expr),
        data_type: as_type,
        format: None,
    }
}

// AST CASE expression builder
fn case_builder(exprs: Vec<ast::Expr>) -> ast::Expr {
    ast::Expr::Case {
        operand: None,
        conditions: vec![exprs[0].clone()],
        results: vec![exprs[1].clone()],
        else_result: exprs.get(2).map(|e| Box::new(e.clone())),
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

pub struct RelationWithTranslator<'a, T: IntoDialectTranslator>(pub &'a Relation, pub T);

impl<'a, T: IntoDialectTranslator> From<RelationWithTranslator<'a, T>> for ast::Query {
    fn from(value: RelationWithTranslator<'a, T>) -> Self {
        let RelationWithTranslator(rel, translator) = value;
        rel.accept(FromRelationVisitor::new(translator))
    }
}
