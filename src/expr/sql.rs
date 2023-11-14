//! Convert Expr into ast::Expr
use crate::{
    ast,
    data_type::DataType,
    expr::{self, Expr},
    visitor::Acceptor,
};
use std::iter::Iterator;

/// A simple Expr -> ast::Expr conversion Visitor
pub struct FromExprVisitor;

impl<'a> expr::Visitor<'a, ast::Expr> for FromExprVisitor {
    fn column(&self, column: &'a expr::Column) -> ast::Expr {
        if column.len() == 1 {
            ast::Expr::Identifier(ast::Ident::new(column.head().unwrap()))
        } else {
            ast::Expr::CompoundIdentifier(column.iter().map(|id| ast::Ident::new(id)).collect())
        }
    }

    fn value(&self, value: &'a expr::Value) -> ast::Expr {
        match value {
            crate::data_type::value::Value::Unit(_) => ast::Expr::Value(ast::Value::Null),
            crate::data_type::value::Value::Boolean(b) => {
                ast::Expr::Value(ast::Value::Boolean(**b))
            }
            crate::data_type::value::Value::Integer(i) => {
                ast::Expr::Value(ast::Value::Number(format!("{}", **i), false))
            }
            crate::data_type::value::Value::Enum(_) => todo!(),
            crate::data_type::value::Value::Float(f) => {
                ast::Expr::Value(ast::Value::Number(format!("{}", **f), false))
            }
            crate::data_type::value::Value::Text(t) => {
                ast::Expr::Value(ast::Value::SingleQuotedString(format!("{}", **t)))
            }
            crate::data_type::value::Value::Bytes(_) => todo!(),
            crate::data_type::value::Value::Struct(_) => todo!(),
            crate::data_type::value::Value::Union(_) => todo!(),
            crate::data_type::value::Value::Optional(_) => todo!(),
            crate::data_type::value::Value::List(l) => ast::Expr::Tuple(
                l.to_vec()
                    .iter()
                    .map(|v| self.value(v))
                    .collect::<Vec<ast::Expr>>(),
            ),
            crate::data_type::value::Value::Set(_) => todo!(),
            crate::data_type::value::Value::Array(_) => todo!(),
            crate::data_type::value::Value::Date(_) => todo!(),
            crate::data_type::value::Value::Time(_) => todo!(),
            crate::data_type::value::Value::DateTime(_) => todo!(),
            crate::data_type::value::Value::Duration(_) => todo!(),
            crate::data_type::value::Value::Id(_) => todo!(),
            crate::data_type::value::Value::Function(_) => todo!(),
        }
    }

    fn function(
        &self,
        function: &'a expr::function::Function,
        arguments: Vec<ast::Expr>,
    ) -> ast::Expr {
        match function {
            expr::function::Function::Opposite => ast::Expr::UnaryOp {
                op: ast::UnaryOperator::Minus,
                expr: Box::new(ast::Expr::Nested(Box::new(arguments[0].clone()))),
            },
            expr::function::Function::Not => ast::Expr::UnaryOp {
                op: ast::UnaryOperator::Not,
                expr: Box::new(ast::Expr::Nested(Box::new(arguments[0].clone()))),
            },
            expr::function::Function::Plus => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(arguments[0].clone()))),
                op: ast::BinaryOperator::Plus,
                right: Box::new(ast::Expr::Nested(Box::new(arguments[1].clone()))),
            },
            expr::function::Function::Minus => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(arguments[0].clone()))),
                op: ast::BinaryOperator::Minus,
                right: Box::new(ast::Expr::Nested(Box::new(arguments[1].clone()))),
            },
            expr::function::Function::Multiply => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(arguments[0].clone()))),
                op: ast::BinaryOperator::Multiply,
                right: Box::new(ast::Expr::Nested(Box::new(ast::Expr::Nested(Box::new(
                    arguments[1].clone(),
                ))))),
            },
            expr::function::Function::Divide => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(arguments[0].clone()))),
                op: ast::BinaryOperator::Divide,
                right: Box::new(ast::Expr::Nested(Box::new(ast::Expr::Nested(Box::new(
                    arguments[1].clone(),
                ))))),
            },
            expr::function::Function::Modulo => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(arguments[0].clone()))),
                op: ast::BinaryOperator::Modulo,
                right: Box::new(ast::Expr::Nested(Box::new(arguments[1].clone()))),
            },
            expr::function::Function::StringConcat => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(arguments[0].clone()))),
                op: ast::BinaryOperator::StringConcat,
                right: Box::new(ast::Expr::Nested(Box::new(arguments[1].clone()))),
            },
            expr::function::Function::Gt => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(arguments[0].clone()))),
                op: ast::BinaryOperator::Gt,
                right: Box::new(ast::Expr::Nested(Box::new(arguments[1].clone()))),
            },
            expr::function::Function::Lt => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(arguments[0].clone()))),
                op: ast::BinaryOperator::Lt,
                right: Box::new(ast::Expr::Nested(Box::new(arguments[1].clone()))),
            },
            expr::function::Function::GtEq => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(arguments[0].clone()))),
                op: ast::BinaryOperator::GtEq,
                right: Box::new(ast::Expr::Nested(Box::new(arguments[1].clone()))),
            },
            expr::function::Function::LtEq => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(arguments[0].clone()))),
                op: ast::BinaryOperator::LtEq,
                right: Box::new(ast::Expr::Nested(Box::new(arguments[1].clone()))),
            },
            expr::function::Function::Eq => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(arguments[0].clone()))),
                op: ast::BinaryOperator::Eq,
                right: Box::new(ast::Expr::Nested(Box::new(arguments[1].clone()))),
            },
            expr::function::Function::NotEq => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(arguments[0].clone()))),
                op: ast::BinaryOperator::NotEq,
                right: Box::new(ast::Expr::Nested(Box::new(arguments[1].clone()))),
            },
            expr::function::Function::And => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(arguments[0].clone()))),
                op: ast::BinaryOperator::And,
                right: Box::new(ast::Expr::Nested(Box::new(arguments[1].clone()))),
            },
            expr::function::Function::Or => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(arguments[0].clone()))),
                op: ast::BinaryOperator::Or,
                right: Box::new(ast::Expr::Nested(Box::new(arguments[1].clone()))),
            },
            expr::function::Function::Xor => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(arguments[0].clone()))),
                op: ast::BinaryOperator::Xor,
                right: Box::new(ast::Expr::Nested(Box::new(arguments[1].clone()))),
            },
            expr::function::Function::BitwiseOr => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(arguments[0].clone()))),
                op: ast::BinaryOperator::BitwiseOr,
                right: Box::new(ast::Expr::Nested(Box::new(arguments[1].clone()))),
            },
            expr::function::Function::BitwiseAnd => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(arguments[0].clone()))),
                op: ast::BinaryOperator::BitwiseAnd,
                right: Box::new(ast::Expr::Nested(Box::new(arguments[1].clone()))),
            },
            expr::function::Function::BitwiseXor => ast::Expr::BinaryOp {
                left: Box::new(ast::Expr::Nested(Box::new(arguments[0].clone()))),
                op: ast::BinaryOperator::BitwiseXor,
                right: Box::new(ast::Expr::Nested(Box::new(arguments[1].clone()))),
            },
            expr::function::Function::Exp
            | expr::function::Function::Ln
            | expr::function::Function::Log
            | expr::function::Function::Abs
            | expr::function::Function::Sin
            | expr::function::Function::Cos
            | expr::function::Function::Sqrt
            | expr::function::Function::Pow
            | expr::function::Function::Md5
            | expr::function::Function::Concat(_)
            | expr::function::Function::CharLength
            | expr::function::Function::Lower
            | expr::function::Function::Upper
            | expr::function::Function::Random(_)
            | expr::function::Function::Least
            | expr::function::Function::Greatest
            | expr::function::Function::Coalesce
            | expr::function::Function::Rtrim
            | expr::function::Function::Ltrim => ast::Expr::Function(ast::Function {
                name: ast::ObjectName(vec![ast::Ident::new(function.to_string())]),
                args: arguments
                    .into_iter()
                    .map(|e| ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(e)))
                    .collect(),
                over: None,
                distinct: false,
                special: false,
                order_by: vec![],
                filter: None,
                null_treatment: None,
            }),
            expr::function::Function::Case => ast::Expr::Case {
                operand: None,
                conditions: vec![arguments[0].clone()],
                results: vec![arguments[1].clone()],
                else_result: Some(Box::new(arguments[2].clone())),
            },
            expr::function::Function::InList => {
                if let ast::Expr::Tuple(t) = &arguments[1] {
                    ast::Expr::InList {
                        expr: arguments[0].clone().into(),
                        list: t.clone(),
                        negated: false,
                    }
                } else {
                    todo!()
                }
            }
            // a,
            expr::function::Function::Position => ast::Expr::Position {
                expr: arguments[0].clone().into(),
                r#in: arguments[1].clone().into(),
            },
            expr::function::Function::CastAsText => ast::Expr::Cast {
                expr: arguments[0].clone().into(),
                data_type: DataType::text().into(),
                format: None,
            },
            expr::function::Function::CastAsFloat => todo!(),
            expr::function::Function::CastAsInteger => todo!(),
            expr::function::Function::CastAsDateTime => todo!(),
        }
    }
    // TODO implement this properly
    fn aggregate(
        &self,
        aggregate: &'a expr::aggregate::Aggregate,
        argument: ast::Expr,
    ) -> ast::Expr {
        match aggregate {
            expr::aggregate::Aggregate::Min => ast::Expr::Function(ast::Function {
                name: ast::ObjectName(vec![ast::Ident {
                    value: String::from("min"),
                    quote_style: None,
                }]),
                args: vec![ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(
                    argument,
                ))],
                over: None,
                distinct: false,
                special: false,
                order_by: vec![],
                filter: None,
                null_treatment: None,
            }),
            expr::aggregate::Aggregate::Max => ast::Expr::Function(ast::Function {
                name: ast::ObjectName(vec![ast::Ident {
                    value: String::from("max"),
                    quote_style: None,
                }]),
                args: vec![ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(
                    argument,
                ))],
                over: None,
                distinct: false,
                special: false,
                order_by: vec![],
                filter: None,
                null_treatment: None,
            }),
            expr::aggregate::Aggregate::Median => todo!(),
            expr::aggregate::Aggregate::NUnique => todo!(),
            // TODO this is a very simple implementation. It will break in non obvious cases. We should fix it.
            expr::aggregate::Aggregate::First => argument,
            // TODO this is a very simple implementation. It will break in non obvious cases. We should fix it.
            expr::aggregate::Aggregate::Last => argument,
            expr::aggregate::Aggregate::Mean => ast::Expr::Function(ast::Function {
                name: ast::ObjectName(vec![ast::Ident {
                    value: String::from("avg"),
                    quote_style: None,
                }]),
                args: vec![ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(
                    argument,
                ))],
                over: None,
                distinct: false,
                special: false,
                order_by: vec![],
                filter: None,
                null_treatment: None,
            }),
            expr::aggregate::Aggregate::List => todo!(),
            expr::aggregate::Aggregate::Count => ast::Expr::Function(ast::Function {
                name: ast::ObjectName(vec![ast::Ident {
                    value: String::from("count"),
                    quote_style: None,
                }]),
                args: vec![ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(
                    argument,
                ))],
                over: None,
                distinct: false,
                special: false,
                order_by: vec![],
                filter: None,
                null_treatment: None,
            }),
            expr::aggregate::Aggregate::Quantile(_) => todo!(),
            expr::aggregate::Aggregate::Quantiles(_) => todo!(),
            expr::aggregate::Aggregate::Sum => ast::Expr::Function(ast::Function {
                name: ast::ObjectName(vec![ast::Ident {
                    value: String::from("sum"),
                    quote_style: None,
                }]),
                args: vec![ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(
                    argument,
                ))],
                over: None,
                distinct: false,
                special: false,
                order_by: vec![],
                filter: None,
                null_treatment: None,
            }),
            expr::aggregate::Aggregate::AggGroups => todo!(),
            expr::aggregate::Aggregate::Std => ast::Expr::Function(ast::Function {
                name: ast::ObjectName(vec![ast::Ident {
                    value: String::from("stddev"),
                    quote_style: None,
                }]),
                args: vec![ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(
                    argument,
                ))],
                over: None,
                distinct: false,
                special: false,
                order_by: vec![],
                filter: None,
                null_treatment: None,
            }),
            expr::aggregate::Aggregate::Var => ast::Expr::Function(ast::Function {
                name: ast::ObjectName(vec![ast::Ident {
                    value: String::from("variance"),
                    quote_style: None,
                }]),
                args: vec![ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(
                    argument,
                ))],
                over: None,
                distinct: false,
                special: false,
                order_by: vec![],
                filter: None,
                null_treatment: None,
            }),
        }
    }

    fn structured(&self, _fields: Vec<(expr::identifier::Identifier, ast::Expr)>) -> ast::Expr {
        todo!()
    }
}

/// Based on the FromExprVisitor implement the From trait
impl From<&Expr> for ast::Expr {
    fn from(value: &Expr) -> Self {
        value.accept(FromExprVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sql::parse_expr;
    use std::convert::TryFrom;

    #[test]
    fn test_from_expr() {
        let ast_expr: ast::Expr = parse_expr("exp(a*cos(sin(x) + 2*a + b))").unwrap();
        println!("ast::expr = {ast_expr}");
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        //assert_eq!(ast_expr, gen_expr)
        assert_eq!(
            gen_expr.to_string(),
            String::from("exp((a) * ((cos(((sin(x)) + ((2) * ((a)))) + (b)))))")
        )
    }

    #[test]
    fn test_from_expr_concat() {
        let ast_expr: ast::Expr = parse_expr("concat(a, b, c)").unwrap();
        println!("ast::expr = {ast_expr}");
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(ast_expr, gen_expr)
    }

    #[test]
    fn test_string_functions() {
        // Lower
        let ast_expr: ast::Expr = parse_expr("lower(my_expr)").unwrap();
        println!("ast::expr = {ast_expr}");
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        assert_eq!(ast_expr, gen_expr);

        // Upper
        let ast_expr: ast::Expr = parse_expr("upper(my_expr)").unwrap();
        println!("ast::expr = {ast_expr}");
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        assert_eq!(ast_expr, gen_expr);

        // CharLength
        let ast_expr: ast::Expr = parse_expr("char_length(my_expr)").unwrap();
        println!("ast::expr = {ast_expr}");
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        assert_eq!(ast_expr, gen_expr);

        // CharLength
        let ast_expr: ast::Expr = parse_expr("position('x' in expr)").unwrap();
        println!("ast::expr = {ast_expr}");
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        assert_eq!(ast_expr, gen_expr);
    }

    #[test]
    fn test_from_expr_with_var() {
        let ast_expr: ast::Expr = parse_expr("variance((sin(x)) + (b))").unwrap();
        println!("ast::expr = {ast_expr}");
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(ast_expr, gen_expr)
    }

    #[test]
    fn test_case() {
        let str_expr = "CASE a WHEN 5 THEN 0 ELSE a END";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        println!("ast::expr = {ast_expr}");
        assert_eq!(ast_expr.to_string(), str_expr.to_string(),);
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        assert_eq!(ast_expr.to_string(), str_expr.to_string(),);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {}", gen_expr.to_string());
        assert_eq!(
            gen_expr.to_string(),
            "CASE WHEN (a) = (5) THEN 0 ELSE a END".to_string(),
        );
    }

    #[test]
    fn test_in() {
        let str_expr = "a IN (4, 5)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        println!("ast::expr = {ast_expr}");
        println!("ast::expr = {:?}", ast_expr);
        assert_eq!(ast_expr.to_string(), str_expr.to_string(),);
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        println!("expr = {:?}", expr);
        assert_eq!(ast_expr.to_string(), str_expr.to_string(),);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {}", gen_expr.to_string());
        assert_eq!(gen_expr.to_string(), "a IN (4, 5)".to_string(),);
    }

    #[test]
    fn test_coalesce() {
        let str_expr = "Coalesce(a, 5)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        println!("ast::expr = {ast_expr}");
        println!("ast::expr = {:?}", ast_expr);
        assert_eq!(ast_expr.to_string(), str_expr.to_string(),);
    }
}
