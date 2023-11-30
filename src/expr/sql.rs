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
            | expr::function::Function::Pi
            | expr::function::Function::Least
            | expr::function::Function::Greatest
            | expr::function::Function::Coalesce
            | expr::function::Function::Rtrim
            | expr::function::Function::Ltrim
            | expr::function::Function::Substr
            | expr::function::Function::SubstrWithSize
            | expr::function::Function::Ceil
            | expr::function::Function::Floor
            | expr::function::Function::Sign
            | expr::function::Function::RegexpContains
            | expr::function::Function::RegexpReplace
            | expr::function::Function::Unhex
            | expr::function::Function::Encode
            | expr::function::Function::Decode
            | expr::function::Function::Newid
            | expr::function::Function::Dayname
            | expr::function::Function::DateFormat
            | expr::function::Function::Quarter
            | expr::function::Function::DatetimeDiff
            | expr::function::Function::Date
            | expr::function::Function::FromUnixtime
            | expr::function::Function::UnixTimestamp => ast::Expr::Function(ast::Function {
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
            expr::function::Function::RegexpExtract => ast::Expr::Function(ast::Function {
                name: ast::ObjectName(vec![ast::Ident::new(function.to_string())]),
                args: vec![arguments[0].clone(), arguments[1].clone(), arguments[2].clone(), arguments[3].clone()]
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
            expr::function::Function::Round
            | expr::function::Function::Trunc => ast::Expr::Function(ast::Function {
                name: ast::ObjectName(vec![ast::Ident::new(function.to_string())]),
                args: arguments
                    .into_iter()
                    .filter_map(|e| (e!=ast::Expr::Value(ast::Value::Number("1".to_string(), false))).then_some(ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(e))))
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
            expr::function::Function::CastAsFloat => ast::Expr::Cast {
                expr: arguments[0].clone().into(),
                data_type: DataType::float().into(),
                format: None,
            },
            expr::function::Function::CastAsInteger => ast::Expr::Cast {
                expr: arguments[0].clone().into(),
                data_type: DataType::integer().into(),
                format: None,
            },
            expr::function::Function::CastAsBoolean => ast::Expr::Cast {
                expr: arguments[0].clone().into(),
                data_type: DataType::boolean().into(),
                format: None,
            },
            expr::function::Function::CastAsDateTime => ast::Expr::Cast {
                expr: arguments[0].clone().into(),
                data_type: DataType::date_time().into(),
                format: None,
            },
            expr::function::Function::CastAsDate => ast::Expr::Cast {
                expr: arguments[0].clone().into(),
                data_type: DataType::date().into(),
                format: None,
            },
            expr::function::Function::CastAsTime => ast::Expr::Cast {
                expr: arguments[0].clone().into(),
                data_type: DataType::time().into(),
                format: None,
            },
            expr::function::Function::CurrentDate
            | expr::function::Function::CurrentTime
            | expr::function::Function::CurrentTimestamp => ast::Expr::Function(ast::Function {
                name: ast::ObjectName(vec![ast::Ident::new(function.to_string())]),
                args: vec![],
                over: None,
                distinct: false,
                special: true,
                order_by: vec![],
                filter: None,
                null_treatment: None,
            }),
            expr::function::Function::ExtractYear => ast::Expr::Extract {field: ast::DateTimeField::Year, expr: arguments[0].clone().into()},
            expr::function::Function::ExtractMonth => ast::Expr::Extract {field: ast::DateTimeField::Month, expr: arguments[0].clone().into()},
            expr::function::Function::ExtractDay => ast::Expr::Extract {field: ast::DateTimeField::Day, expr: arguments[0].clone().into()},
            expr::function::Function::ExtractHour => ast::Expr::Extract {field: ast::DateTimeField::Hour, expr: arguments[0].clone().into()},
            expr::function::Function::ExtractMinute => ast::Expr::Extract {field: ast::DateTimeField::Minute, expr: arguments[0].clone().into()},
            expr::function::Function::ExtractSecond => ast::Expr::Extract {field: ast::DateTimeField::Second, expr: arguments[0].clone().into()},
            expr::function::Function::ExtractMicrosecond => ast::Expr::Extract {field: ast::DateTimeField::Microsecond, expr: arguments[0].clone().into()},
            expr::function::Function::ExtractMillisecond => ast::Expr::Extract {field: ast::DateTimeField::Millisecond, expr: arguments[0].clone().into()},
            expr::function::Function::ExtractDow => ast::Expr::Extract {field: ast::DateTimeField::Dow, expr: arguments[0].clone().into()},
            expr::function::Function::ExtractWeek => ast::Expr::Extract {field: ast::DateTimeField::Week, expr: arguments[0].clone().into()},
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
            expr::aggregate::Aggregate::MeanDistinct => ast::Expr::Function(ast::Function {
                name: ast::ObjectName(vec![ast::Ident {
                    value: String::from("avg"),
                    quote_style: None,
                }]),
                args: vec![ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(
                    argument,
                ))],
                over: None,
                distinct: true,
                special: false,
                order_by: vec![],
                filter: None,
                null_treatment: None,
            }),
            expr::aggregate::Aggregate::CountDistinct => ast::Expr::Function(ast::Function {
                name: ast::ObjectName(vec![ast::Ident {
                    value: String::from("count"),
                    quote_style: None,
                }]),
                args: vec![ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(
                    argument,
                ))],
                over: None,
                distinct: true,
                special: false,
                order_by: vec![],
                filter: None,
                null_treatment: None,
            }),
            expr::aggregate::Aggregate::SumDistinct => ast::Expr::Function(ast::Function {
                name: ast::ObjectName(vec![ast::Ident {
                    value: String::from("sum"),
                    quote_style: None,
                }]),
                args: vec![ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(
                    argument,
                ))],
                over: None,
                distinct: true,
                special: false,
                order_by: vec![],
                filter: None,
                null_treatment: None,
            }),
            expr::aggregate::Aggregate::StdDistinct => ast::Expr::Function(ast::Function {
                name: ast::ObjectName(vec![ast::Ident {
                    value: String::from("stddev"),
                    quote_style: None,
                }]),
                args: vec![ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(
                    argument,
                ))],
                over: None,
                distinct: true,
                special: false,
                order_by: vec![],
                filter: None,
                null_treatment: None,
            }),
            expr::aggregate::Aggregate::VarDistinct => ast::Expr::Function(ast::Function {
                name: ast::ObjectName(vec![ast::Ident {
                    value: String::from("variance"),
                    quote_style: None,
                }]),
                args: vec![ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(
                    argument,
                ))],
                over: None,
                distinct: true,
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

        // Newid
        let ast_expr: ast::Expr = parse_expr("newid()").unwrap();
        println!("ast::expr = {ast_expr}");
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        assert_eq!(ast_expr, gen_expr);

        // encode
        let ast_expr: ast::Expr = parse_expr("encode(col1, col2)").unwrap();
        println!("ast::expr = {ast_expr}");
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        assert_eq!(ast_expr, gen_expr);

        // decode
        let ast_expr: ast::Expr = parse_expr("decode(col1, col2)").unwrap();
        println!("ast::expr = {ast_expr}");
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        assert_eq!(ast_expr, gen_expr);

        // unhex
        let ast_expr: ast::Expr = parse_expr("unhex(col1)").unwrap();
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

    #[test]
    fn test_substr() {
        let str_expr = "substr(a, 5, 2)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(ast_expr, gen_expr);

        let str_expr = "\nsubstr(a, 5)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(ast_expr, gen_expr);

        let str_expr = "\nsubstring(a from 5 for 2)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(gen_expr, parse_expr("substr(a, 5, 2)").unwrap());

        let str_expr = "\nsubstring(a from 5)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(gen_expr, parse_expr("substr(a, 5)").unwrap());

        let str_expr = "\nsubstring(a for 5)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(gen_expr, parse_expr("substr(a, 0, 5)").unwrap());
    }

    #[test]
    fn test_cast() {
        let str_expr = "cast(a as varchar)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(ast_expr, gen_expr);

        let str_expr = "cast(a as bigint)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(ast_expr, gen_expr);

        let str_expr = "cast(a as boolean)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(ast_expr, gen_expr);

        let str_expr = "cast(a as float)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(ast_expr, gen_expr);

        let str_expr = "cast(a as date)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(ast_expr, gen_expr);

        let str_expr = "cast(a as time)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(ast_expr, gen_expr);

        let str_expr = "cast(a as timestamp)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(ast_expr, gen_expr);
    }

    #[test]
    fn test_ceil() {
        let str_expr = "ceil(a)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(ast_expr.to_string().to_lowercase(), gen_expr.to_string().to_lowercase());
    }


    #[test]
    fn test_floor() {
        let str_expr = "floor(a)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(ast_expr.to_string().to_lowercase(), gen_expr.to_string().to_lowercase());
    }

    #[test]
    fn test_round() {
        let str_expr = "round(a, 2)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(ast_expr.to_string().to_lowercase(), gen_expr.to_string().to_lowercase());
    }

    #[test]
    fn test_trunc() {
        let str_expr = "trunc(a, 4)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(ast_expr.to_string().to_lowercase(), gen_expr.to_string().to_lowercase());
    }

    #[test]
    fn test_sign() {
        let str_expr = "sign(a)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(ast_expr, gen_expr);
    }

    #[test]
    fn test_square() {
        let str_expr = "square(x)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        let true_expr = parse_expr("pow(x, 2)").unwrap();
        assert_eq!(gen_expr, true_expr);
    }

    #[test]
    fn test_log() {
        let str_expr = "ln(x)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(ast_expr, gen_expr);

        let str_expr = "log(x)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(ast_expr, gen_expr);

        let str_expr = "log(b, x)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        let true_expr = parse_expr("(log(x)) / ((log(b)))").unwrap();
        assert_eq!(gen_expr, true_expr);

        let str_expr = "log10(x)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        let true_expr = parse_expr("(log(10)) / ((log(x)))").unwrap();
        assert_eq!(gen_expr, true_expr);

        let str_expr = "log2(x)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        let true_expr = parse_expr("(log(2)) / ((log(x)))").unwrap();
        assert_eq!(gen_expr, true_expr);
    }

    #[test]
    fn test_trigo() {
        let str_expr = "sin(x)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(ast_expr, gen_expr);

        let str_expr = "cos(x)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(ast_expr, gen_expr);

        let str_expr = "tan(x)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        let true_expr = parse_expr("(sin(x)) / ((cos(x)))").unwrap();
        assert_eq!(gen_expr, true_expr);
    }

    #[test]
    fn test_random() {
        let str_expr = "random()";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(ast_expr, gen_expr);
    }

    #[test]
    fn test_pi() {
        let str_expr = "pi()";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(ast_expr, gen_expr);
    }

    #[test]
    fn test_degrees() {
        let str_expr = "degrees(100)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        let true_expr = parse_expr("(100) * (((180) / ((pi()))))").unwrap();
        assert_eq!(gen_expr, true_expr);
    }

    #[test]
    fn test_regexp_contains() {
        let str_expr = "regexp_contains(col1, col2)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(ast_expr, gen_expr);
    }

    #[test]
    fn test_regexp_extract() {
        let str_expr = "regexp_extract(value, regexp)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        let true_expr = parse_expr("regexp_extract(value, regexp, 0, 1)").unwrap();
        assert_eq!(gen_expr, true_expr);

        let str_expr = "regexp_substr(value, regexp)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        let true_expr = parse_expr("regexp_extract(value, regexp, 0, 1)").unwrap();
        assert_eq!(gen_expr, true_expr);


        let str_expr = "regexp_extract(value, regexp, position)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        let true_expr = parse_expr("regexp_extract(value, regexp, position, 1)").unwrap();
        assert_eq!(gen_expr, true_expr);

        let str_expr = "regexp_substr(value, regexp, position)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        let true_expr = parse_expr("regexp_extract(value, regexp, position, 1)").unwrap();
        assert_eq!(gen_expr, true_expr);

        let str_expr = "regexp_extract(value, regexp, position, occurrence)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(gen_expr, ast_expr);
    }

    #[test]
    fn test_regexp_replace() {
        let str_expr = "regexp_replace(value, regexp, replacement)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        let true_expr = parse_expr("regexp_replace(value, regexp, replacement)").unwrap();
        assert_eq!(gen_expr, true_expr);
    }

    #[test]
    fn test_current() {
        // CURRENT_DATE
        let str_expr = "current_date";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(gen_expr, ast_expr);

        // CURRENT_TIME
        let str_expr = "current_time";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(gen_expr, ast_expr);

        // CURRENT_TIMESTAMP
        let str_expr = "current_timestamp";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(gen_expr, ast_expr);
    }

    #[test]
    fn test_extract() {
        // EXTRACT(YEAR FROM col1)
        let str_expr = "extract(year from col1)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(gen_expr, ast_expr);

        // EXTRACT(MONTH FROM col1)
        let str_expr = "extract(month from col1)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(gen_expr, ast_expr);

        // EXTRACT(DAY FROM col1)
        let str_expr = "extract(day from col1)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(gen_expr, ast_expr);

        // EXTRACT(HOUR FROM col1)
        let str_expr = "extract(hour from col1)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(gen_expr, ast_expr);

        // EXTRACT(MINUTE FROM col1)
        let str_expr = "extract(minute from col1)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(gen_expr, ast_expr);

        // EXTRACT(SECOND FROM col1)
        let str_expr = "extract(second from col1)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(gen_expr, ast_expr);

        // EXTRACT(MICROSECOND FROM col1)
        let str_expr = "extract(microsecond from col1)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(gen_expr, ast_expr);

        // EXTRACT(MILLISECOND FROM col1)
        let str_expr = "extract(millisecond from col1)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(gen_expr, ast_expr);

        // EXTRACT(DOW FROM col1)
        let str_expr = "extract(dow from col1)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(gen_expr, ast_expr);

        // EXTRACT(WEEK FROM col1)
        let str_expr = "extract(week from col1)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(gen_expr, ast_expr);
    }

    #[test]
    fn test_datetime_functions() {
        // DAYNAME
        let str_expr = "dayname(col1)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(gen_expr, ast_expr);

        // DATE_FORMAT
        let str_expr = "date_format(col1, format_date)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(gen_expr, ast_expr);

        // Quarter
        let str_expr = "quarter(col1)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(gen_expr, ast_expr);

        // datetime_diff
        let str_expr = "datetime_diff(col1, col2, col3)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(gen_expr, ast_expr);

        // date
        let str_expr = "date(col1)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(gen_expr, ast_expr);

        // from_unixtime
        let str_expr = "from_unixtime(col1, '%Y-%m-%d')";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(gen_expr, ast_expr);

        let str_expr = "from_unixtime(col1)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        let true_expr = parse_expr("from_unixtime(col1, '%Y-%m-%d %H:%i:%S')").unwrap();
        assert_eq!(gen_expr, true_expr);

        // unix_timestamp
        let str_expr = "unix_timestamp(col1)";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        assert_eq!(gen_expr, ast_expr);

        let str_expr = "unix_timestamp()";
        let ast_expr: ast::Expr = parse_expr(str_expr).unwrap();
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let gen_expr = ast::Expr::from(&expr);
        println!("ast::expr = {gen_expr}");
        let true_expr = parse_expr("unix_timestamp(current_timestamp)").unwrap();
        assert_eq!(gen_expr, true_expr);
    }
}
