//! Tools for queries from one dialect into another
//! A specific Dialect is a struct holding:
//!     - a method to provide a sqlparser::Dialect for the parsing
//!     - methods varying from dialect to dialect regarding the conversion from AST to Expr+Relation and vice-versa
use std::ops::Deref;

use sqlparser::{
    ast,
    dialect::{BigQueryDialect, Dialect, PostgreSqlDialect},
};

use crate::sql::{self, parse, parse_with_dialect};
use crate::{
    data_type::function::cast,
    expr::{self, Function},
    relation::{self, sql::FromRelationVisitor},
    visitor::Acceptor,
};

pub mod bigquery;
pub mod hive;
pub mod mssql;
pub mod mysql;
pub mod postgres;

// Should they implement methods for converting from Generic to Dialect and vice-versa.
// Relations are dialect agnostic objects
// We can probably convert them to DialectAwareQueries. Changes for this:
// - FromRelationVisitor needs to have a Dialect
// - In each node of a relation we use methods to construct a query by piece:
// - query -> ast::Query
// - values_query -> ast::Query
// - table_factor -> ast::TableFactor
// - table_with_joins -> ast::TableWithJoins
// - ctes_from_query -> Vec<ast::Cte>
// - cte -> ast::Cte
// - all -> Vec<ast::SelectItem>
// - select_from_query -> ast::Select
// - set_operation -> ast::Query

// What we currently need to translate:
// Quoting identifiers: ast::Ident can be constructed with a quote character.
// Quoting aliases (careful to case-sensitive and case-insensitive Dialects)
// Function mappings:
// Query construction: TOP -> LIMIT, CTE column aliases,

/// Trait for mapping sarus to ast objects
pub trait IntoDialectTranslator {
    type D: Dialect;

    fn dialect(&self) -> Self::D;

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
        let idents_as_vec = ident.deref();
        if idents_as_vec.len() > 1 {
            ast::Expr::CompoundIdentifier(
                idents_as_vec
                    .iter()
                    .map(|name| ast::Ident::from(&name[..]))
                    .collect(),
            )
        } else {
            ast::Expr::Identifier(ast::Ident::from(&idents_as_vec[0][..]))
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

        match func.function() {
            // Unary operators
            expr::function::Function::Opposite => ast::Expr::UnaryOp {
                op: ast::UnaryOperator::Minus,
                expr: Box::new(ast::Expr::Nested(Box::new(self.expr(args[0])))),
            },
            expr::function::Function::Not => ast::Expr::UnaryOp {
                op: ast::UnaryOperator::Not,
                expr: Box::new(ast::Expr::Nested(Box::new(self.expr(args[0])))),
            },
            // Binary operator
            expr::function::Function::Plus => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[0])))),
                op: ast::BinaryOperator::Plus,
                right: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[1])))),
            },
            expr::function::Function::Minus => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[0])))),
                op: ast::BinaryOperator::Minus,
                right: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[1])))),
            },
            expr::function::Function::Multiply => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[0])))),
                op: ast::BinaryOperator::Multiply,
                right: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[1])))),
            },
            expr::function::Function::Divide => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[0])))),
                op: ast::BinaryOperator::Divide,
                right: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[1])))),
            },
            expr::function::Function::Modulo => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[0])))),
                op: ast::BinaryOperator::Modulo,
                right: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[1])))),
            },
            expr::function::Function::StringConcat => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[0])))),
                op: ast::BinaryOperator::StringConcat,
                right: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[1])))),
            },
            expr::function::Function::Gt => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[0])))),
                op: ast::BinaryOperator::Gt,
                right: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[1])))),
            },
            expr::function::Function::Lt => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[0])))),
                op: ast::BinaryOperator::Lt,
                right: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[1])))),
            },
            expr::function::Function::GtEq => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[0])))),
                op: ast::BinaryOperator::GtEq,
                right: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[1])))),
            },
            expr::function::Function::LtEq => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[0])))),
                op: ast::BinaryOperator::LtEq,
                right: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[1])))),
            },
            expr::function::Function::Eq => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[0])))),
                op: ast::BinaryOperator::Eq,
                right: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[1])))),
            },
            expr::function::Function::NotEq => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[0])))),
                op: ast::BinaryOperator::NotEq,
                right: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[1])))),
            },
            expr::function::Function::And => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[0])))),
                op: ast::BinaryOperator::And,
                right: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[1])))),
            },
            expr::function::Function::Or => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[0])))),
                op: ast::BinaryOperator::Or,
                right: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[1])))),
            },
            expr::function::Function::Xor => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[0])))),
                op: ast::BinaryOperator::Xor,
                right: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[1])))),
            },
            expr::function::Function::BitwiseOr => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[0])))),
                op: ast::BinaryOperator::BitwiseOr,
                right: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[1])))),
            },
            expr::function::Function::BitwiseAnd => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[0])))),
                op: ast::BinaryOperator::BitwiseAnd,
                right: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[1])))),
            },
            expr::function::Function::BitwiseXor => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[0])))),
                op: ast::BinaryOperator::BitwiseXor,
                right: Box::new(ast::Expr::Nested(Box::new(self.expr(&args[1])))),
            },
            expr::function::Function::InList => {
                if let ast::Expr::Tuple(t) = self.expr(&args[1]) {
                    ast::Expr::InList {
                        expr: Box::new(self.expr(&args[0])),
                        list: t.clone(),
                        negated: false,
                    }
                } else {
                    todo!()
                }
            }
            // Unary Functions
            expr::function::Function::Exp => self.from_exp(&args[0]),
            expr::function::Function::Ln => self.from_ln(&args[0]),
            expr::function::Function::Log => self.from_log(&args[0]),
            expr::function::Function::Abs => self.from_abs(&args[0]),
            expr::function::Function::Sin => self.from_sin(&args[0]),
            expr::function::Function::Cos => self.from_cos(&args[0]),
            expr::function::Function::Sqrt => self.from_sqrt(&args[0]),
            expr::function::Function::CharLength => self.from_char_length(&args[0]),
            expr::function::Function::Lower => self.from_lower(&args[0]),
            expr::function::Function::Upper => self.from_upper(&args[0]),
            expr::function::Function::Md5 => self.from_md5(&args[0]),

            // nary Functions
            expr::function::Function::Pow => self.from_pow(args),
            expr::function::Function::Case => todo!(),
            expr::function::Function::Concat(_) => self.from_concat(args),
            expr::function::Function::Position => self.from_position(args),

            expr::function::Function::CastAsText => todo!(),
            expr::function::Function::CastAsFloat => todo!(),
            expr::function::Function::CastAsInteger => todo!(),
            expr::function::Function::CastAsDateTime => todo!(),
            expr::function::Function::Least => self.from_least(args),
            expr::function::Function::Greatest => self.from_greatest(args),
            // zero arg
            expr::function::Function::Random(_) => self.from_random(),
        }
    }

    fn from_aggregate(&self, agg: &expr::Aggregate) -> ast::Expr {
        let arg = agg.argument();
        match agg.aggregate() {
            expr::aggregate::Aggregate::Min => self.from_min(arg),
            expr::aggregate::Aggregate::Max => self.from_max(arg),
            expr::aggregate::Aggregate::Median => self.from_median(arg),
            expr::aggregate::Aggregate::NUnique => self.from_nunique(arg),
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
        }
    }
    // Functions with default implementation. Each dialect, when needed, it should override these implementations

    // unary Functions
    fn from_exp(&self, expr: &expr::Expr) -> ast::Expr {
        let ast_expr = self.expr(expr);
        function_builder("EXP", vec![ast_expr])
    }
    fn from_ln(&self, expr: &expr::Expr) -> ast::Expr {
        let ast_expr = self.expr(expr);
        function_builder("LN", vec![ast_expr])
    }
    fn from_log(&self, expr: &expr::Expr) -> ast::Expr {
        let ast_expr = self.expr(expr);
        function_builder("LOG", vec![ast_expr])
    }
    fn from_abs(&self, expr: &expr::Expr) -> ast::Expr {
        let ast_expr = self.expr(expr);
        function_builder("ABS", vec![ast_expr])
    }
    fn from_sin(&self, expr: &expr::Expr) -> ast::Expr {
        let ast_expr = self.expr(expr);
        function_builder("SIN", vec![ast_expr])
    }
    fn from_cos(&self, expr: &expr::Expr) -> ast::Expr {
        let ast_expr = self.expr(expr);
        function_builder("COS", vec![ast_expr])
    }
    fn from_sqrt(&self, expr: &expr::Expr) -> ast::Expr {
        let ast_expr = self.expr(expr);
        function_builder("SQRT", vec![ast_expr])
    }
    fn from_char_length(&self, expr: &expr::Expr) -> ast::Expr {
        let ast_expr = self.expr(expr);
        function_builder("CHAR_LENGTH", vec![ast_expr])
    }
    fn from_lower(&self, expr: &expr::Expr) -> ast::Expr {
        let ast_expr = self.expr(expr);
        function_builder("LOWER", vec![ast_expr])
    }
    fn from_upper(&self, expr: &expr::Expr) -> ast::Expr {
        let ast_expr = self.expr(expr);
        function_builder("UPPER", vec![ast_expr])
    }
    fn from_md5(&self, expr: &expr::Expr) -> ast::Expr {
        let ast_expr = self.expr(expr);
        function_builder("MD5", vec![ast_expr])
    }

    fn from_min(&self, expr: &expr::Expr) -> ast::Expr {
        let ast_expr = self.expr(expr);
        function_builder("MIN", vec![ast_expr])
    }
    fn from_max(&self, expr: &expr::Expr) -> ast::Expr {
        let ast_expr = self.expr(expr);
        function_builder("MAX", vec![ast_expr])
    }
    fn from_median(&self, expr: &expr::Expr) -> ast::Expr {
        let ast_expr = self.expr(expr);
        function_builder("MEDIAN", vec![ast_expr])
    }
    // this should be a count(distinct expr)
    fn from_nunique(&self, expr: &expr::Expr) -> ast::Expr {
        todo!()
    }
    fn from_first(&self, expr: &expr::Expr) -> ast::Expr {
        let ast_expr = self.expr(expr);
        function_builder("FIRST", vec![ast_expr])
    }
    fn from_last(&self, expr: &expr::Expr) -> ast::Expr {
        let ast_expr = self.expr(expr);
        function_builder("LAST", vec![ast_expr])
    }
    fn from_mean(&self, expr: &expr::Expr) -> ast::Expr {
        let ast_expr = self.expr(expr);
        function_builder("AVG", vec![ast_expr])
    }
    //
    fn from_list(&self, expr: &expr::Expr) -> ast::Expr {
        todo!()
    }
    fn from_count(&self, expr: &expr::Expr) -> ast::Expr {
        let ast_expr = self.expr(expr);
        function_builder("COUNT", vec![ast_expr])
    }
    fn from_quantile(&self, expr: &expr::Expr) -> ast::Expr {
        todo!()
    }
    fn from_quantiles(&self, expr: &expr::Expr) -> ast::Expr {
        todo!()
    }
    fn from_sum(&self, expr: &expr::Expr) -> ast::Expr {
        let ast_expr = self.expr(expr);
        function_builder("SUM", vec![ast_expr])
    }
    // ? leaving as todo!() for now
    fn from_agg_groups(&self, expr: &expr::Expr) -> ast::Expr {
        todo!()
    }
    fn from_std(&self, expr: &expr::Expr) -> ast::Expr {
        let ast_expr = self.expr(expr);
        function_builder("STD", vec![ast_expr])
    }
    fn from_var(&self, expr: &expr::Expr) -> ast::Expr {
        let ast_expr = self.expr(expr);
        function_builder("VAR", vec![ast_expr])
    }

    // nary Functions
    fn from_pow(&self, exprs: Vec<&expr::Expr>) -> ast::Expr {
        let ast_exprs: Vec<ast::Expr> = exprs.into_iter().map(|expr| self.expr(expr)).collect();
        function_builder("POW", ast_exprs)
    }
    fn from_concat(&self, exprs: Vec<&expr::Expr>) -> ast::Expr {
        let ast_exprs: Vec<ast::Expr> = exprs.into_iter().map(|expr| self.expr(expr)).collect();
        function_builder("CONCAT", ast_exprs)
    }
    fn from_position(&self, exprs: Vec<&expr::Expr>) -> ast::Expr {
        let ast_exprs: Vec<ast::Expr> = exprs.into_iter().map(|expr| self.expr(expr)).collect();
        function_builder("POSITION", ast_exprs)
    }
    fn from_least(&self, exprs: Vec<&expr::Expr>) -> ast::Expr {
        let ast_exprs: Vec<ast::Expr> = exprs.into_iter().map(|expr| self.expr(expr)).collect();
        function_builder("LEAST", ast_exprs)
    }
    fn from_greatest(&self, exprs: Vec<&expr::Expr>) -> ast::Expr {
        let ast_exprs: Vec<ast::Expr> = exprs.into_iter().map(|expr| self.expr(expr)).collect();
        function_builder("GREATEST", ast_exprs)
    }
    fn from_random(&self) -> ast::Expr {
        function_builder("RANDOM", vec![])
    }

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
        let ast_exprs: Vec<ast::Expr> = exprs.into_iter().map(|expr| self.expr(expr)).collect();
        case_builder(ast_exprs)
    }
}

// Function expression builders
fn function_builder(name: &str, exprs: Vec<ast::Expr>) -> ast::Expr {
    //let function_arg_expr = ast::FunctionArgExpr::Expr(exprs);
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
    };
    ast::Expr::Function(funtion)
}

fn cast_builder(expr: ast::Expr, as_type: ast::DataType) -> ast::Expr {
    ast::Expr::Cast {
        expr: Box::new(expr),
        data_type: as_type,
    }
}

fn case_builder(exprs: Vec<ast::Expr>) -> ast::Expr {
    ast::Expr::Case {
        operand: None,
        conditions: vec![exprs[0].clone()],
        results: vec![exprs[1].clone()],
        else_result: exprs.get(2).map(|e| Box::new(e.clone())),
    }
}
