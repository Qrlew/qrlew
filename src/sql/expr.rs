//! This file provides tools for converting a sqlparser::ast::Expr
//! into the corresponding Qrlew expression.
//! Example: `Expr::try_from(sql_parser_expr)`

use super::{Error, Result};
use crate::{
    builder::With,
    expr::{identifier::Identifier, Expr, Value},
    hierarchy::{Hierarchy, Path},
    visitor::{self, Acceptor, Dependencies, Visited},
};
use itertools::Itertools;
use sqlparser::{
    ast,
    dialect::{Dialect, GenericDialect},
    parser::Parser,
    tokenizer::Tokenizer,
};
use std::{iter, result, str::FromStr};

// A few conversions

impl Path for ast::Ident {
    fn path(self) -> Vec<String> {
        self.value.path()
    }
}

impl Path for Vec<ast::Ident> {
    fn path(self) -> Vec<String> {
        self.into_iter().map(|i| i.value).collect()
    }
}

// Implement a visitor for SQL Expr

/// Implement the Acceptor trait
impl<'a> Acceptor<'a> for ast::Expr {
    // TODO fill in the sub exprs
    /// When the number of dependencies is variable, because of Vecs or Optional arguments, the order is preseved in the output vec
    fn dependencies(&'a self) -> Dependencies<'a, Self> {
        match self {
            ast::Expr::Identifier(_) => Dependencies::empty(),
            ast::Expr::CompoundIdentifier(_) => Dependencies::empty(),
            ast::Expr::JsonAccess {
                left,
                operator: _,
                right,
            } => Dependencies::from([left.as_ref(), right.as_ref()]),
            ast::Expr::CompositeAccess { expr, key: _ } => Dependencies::from([expr.as_ref()]),
            ast::Expr::IsFalse(expr) => Dependencies::from([expr.as_ref()]),
            ast::Expr::IsNotFalse(expr) => Dependencies::from([expr.as_ref()]),
            ast::Expr::IsTrue(expr) => Dependencies::from([expr.as_ref()]),
            ast::Expr::IsNotTrue(expr) => Dependencies::from([expr.as_ref()]),
            ast::Expr::IsNull(expr) => Dependencies::from([expr.as_ref()]),
            ast::Expr::IsNotNull(expr) => Dependencies::from([expr.as_ref()]),
            ast::Expr::IsUnknown(expr) => Dependencies::from([expr.as_ref()]),
            ast::Expr::IsNotUnknown(expr) => Dependencies::from([expr.as_ref()]),
            ast::Expr::IsDistinctFrom(left, right) => {
                Dependencies::from([left.as_ref(), right.as_ref()])
            }
            ast::Expr::IsNotDistinctFrom(left, right) => {
                Dependencies::from([left.as_ref(), right.as_ref()])
            }
            ast::Expr::InList {
                expr,
                list,
                negated: _,
            } => iter::once(expr.as_ref()).chain(list.iter()).collect(),
            ast::Expr::InSubquery {
                expr,
                subquery: _,
                negated: _,
            } => Dependencies::from([expr.as_ref()]),
            ast::Expr::InUnnest {
                expr,
                array_expr,
                negated: _,
            } => Dependencies::from([expr.as_ref(), array_expr.as_ref()]),
            ast::Expr::Between {
                expr,
                negated: _,
                low,
                high,
            } => Dependencies::from([expr.as_ref(), low.as_ref(), high.as_ref()]),
            ast::Expr::BinaryOp { left, op: _, right } => {
                Dependencies::from([left.as_ref(), right.as_ref()])
            }
            ast::Expr::Like {
                negated: _,
                expr,
                pattern,
                escape_char: _,
            } => Dependencies::from([expr.as_ref(), pattern.as_ref()]),
            ast::Expr::ILike {
                negated: _,
                expr,
                pattern,
                escape_char: _,
            } => Dependencies::from([expr.as_ref(), pattern.as_ref()]),
            ast::Expr::SimilarTo {
                negated: _,
                expr,
                pattern,
                escape_char: _,
            } => Dependencies::from([expr.as_ref(), pattern.as_ref()]),
            ast::Expr::AnyOp(expr) => Dependencies::from([expr.as_ref()]),
            ast::Expr::AllOp(expr) => Dependencies::from([expr.as_ref()]),
            ast::Expr::UnaryOp { op: _, expr } => Dependencies::from([expr.as_ref()]),
            ast::Expr::Cast { expr, data_type: _ } => Dependencies::from([expr.as_ref()]),
            ast::Expr::TryCast { expr, data_type: _ } => Dependencies::from([expr.as_ref()]),
            ast::Expr::SafeCast { expr, data_type: _ } => Dependencies::from([expr.as_ref()]),
            ast::Expr::AtTimeZone {
                timestamp,
                time_zone: _,
            } => Dependencies::from([timestamp.as_ref()]),
            ast::Expr::Extract { field: _, expr } => Dependencies::from([expr.as_ref()]),
            ast::Expr::Ceil { expr, field: _ } => Dependencies::from([expr.as_ref()]),
            ast::Expr::Floor { expr, field: _ } => Dependencies::from([expr.as_ref()]),
            ast::Expr::Position { expr, r#in } => {
                Dependencies::from([expr.as_ref(), r#in.as_ref()])
            }
            ast::Expr::Substring {
                expr,
                substring_from,
                substring_for,
            } => vec![Some(expr), substring_from.as_ref(), substring_for.as_ref()]
                .iter()
                .filter_map(|expr| expr.map(AsRef::as_ref))
                .collect(),
            ast::Expr::Trim {
                expr,
                trim_where: _,
                trim_what,
            } => vec![Some(expr), trim_what.as_ref()]
                .iter()
                .filter_map(|expr| expr.map(AsRef::as_ref))
                .collect(),
            ast::Expr::Overlay {
                expr,
                overlay_what,
                overlay_from,
                overlay_for,
            } => vec![
                Some(expr),
                Some(overlay_what),
                Some(overlay_from),
                overlay_for.as_ref(),
            ]
            .iter()
            .filter_map(|expr| expr.map(AsRef::as_ref))
            .collect(),
            ast::Expr::Collate { expr, collation: _ } => Dependencies::from([expr.as_ref()]),
            ast::Expr::Nested(expr) => Dependencies::from([expr.as_ref()]),
            ast::Expr::Value(_) => Dependencies::empty(),
            ast::Expr::TypedString {
                data_type: _,
                value: _,
            } => Dependencies::empty(),
            ast::Expr::MapAccess { column, keys } => {
                iter::once(column.as_ref()).chain(keys.iter()).collect()
            }
            ast::Expr::Function(function) => function
                .args
                .iter()
                .map(|arg| match arg {
                    ast::FunctionArg::Named { name: _, arg } => arg,
                    ast::FunctionArg::Unnamed(arg) => arg,
                })
                .filter_map(|arg| match arg {
                    ast::FunctionArgExpr::Expr(expr) => Some(expr),
                    _ => None,
                })
                .collect(),
            ast::Expr::AggregateExpressionWithFilter { expr, filter } => {
                Dependencies::from([expr.as_ref(), filter.as_ref()])
            }
            ast::Expr::Case {
                operand,
                conditions,
                results,
                else_result,
            } => operand
                .as_ref()
                .into_iter()
                .map(AsRef::as_ref)
                .chain(conditions.iter())
                .chain(results.iter())
                .chain(else_result.as_ref().into_iter().map(AsRef::as_ref))
                .collect(),
            ast::Expr::Exists {
                subquery: _,
                negated: _,
            } => Dependencies::empty(),
            ast::Expr::Subquery(_) => Dependencies::empty(),
            ast::Expr::ArraySubquery(_) => Dependencies::empty(),
            ast::Expr::ListAgg(_) => Dependencies::empty(),
            ast::Expr::ArrayAgg(_) => Dependencies::empty(),
            ast::Expr::GroupingSets(exprs) => exprs.iter().flat_map(|exprs| exprs.iter()).collect(),
            ast::Expr::Cube(exprs) => exprs.iter().flat_map(|exprs| exprs.iter()).collect(),
            ast::Expr::Rollup(exprs) => exprs.iter().flat_map(|exprs| exprs.iter()).collect(),
            ast::Expr::Tuple(exprs) => exprs.iter().collect(),
            ast::Expr::ArrayIndex { obj, indexes } => {
                iter::once(obj.as_ref()).chain(indexes.iter()).collect()
            }
            ast::Expr::Array(_) => Dependencies::empty(),
            ast::Expr::Interval(_) => Dependencies::empty(),
            ast::Expr::MatchAgainst {
                columns,
                match_value,
                opt_search_modifier,
            } => Dependencies::empty(),
            ast::Expr::IntroducedString { introducer, value } => Dependencies::empty(),
        }
    }
}

/// A Visitor for the type Expr
pub trait Visitor<'a, T: Clone> {
    // Quasi-expressions
    fn qualified_wildcard(&self, idents: &'a Vec<ast::Ident>) -> T;
    fn wildcard(&self) -> T;
    // Proper expressions
    fn identifier(&self, ident: &'a ast::Ident) -> T;
    fn compound_identifier(&self, idents: &'a Vec<ast::Ident>) -> T;
    fn binary_op(&self, left: T, op: &'a ast::BinaryOperator, right: T) -> T;
    fn unary_op(&self, op: &'a ast::UnaryOperator, expr: T) -> T;
    fn value(&self, value: &'a ast::Value) -> T;
    fn function(&self, function: &'a ast::Function, args: Vec<FunctionArg<T>>) -> T;
    fn case(&self, operand: Option<T>, conditions: Vec<T>, results: Vec<T>, else_result: Option<T>) -> T;
}

// For the visitor to be more convenient, we create a few auxiliary objects

/// Mirrors ast::FunctionArgs but only contains image of ast::FunctionArgExpr by the visitor
pub enum FunctionArg<T> {
    Named { name: ast::Ident, arg: T },
    Unnamed(T),
}

/// Unpack the visited expressions of visitor::Visitor to ease the writing of Visitor
impl<'a, T: Clone, V: Visitor<'a, T>> visitor::Visitor<'a, ast::Expr, T> for V {
    fn visit(&self, acceptor: &'a ast::Expr, dependencies: Visited<'a, ast::Expr, T>) -> T {
        match acceptor {
            ast::Expr::Identifier(ident) => self.identifier(ident),
            ast::Expr::CompoundIdentifier(idents) => self.compound_identifier(idents),
            ast::Expr::JsonAccess {
                left,
                operator,
                right,
            } => todo!(),
            ast::Expr::CompositeAccess { expr, key } => todo!(),
            ast::Expr::IsFalse(_) => todo!(),
            ast::Expr::IsNotFalse(_) => todo!(),
            ast::Expr::IsTrue(_) => todo!(),
            ast::Expr::IsNotTrue(_) => todo!(),
            ast::Expr::IsNull(_) => todo!(),
            ast::Expr::IsNotNull(_) => todo!(),
            ast::Expr::IsUnknown(_) => todo!(),
            ast::Expr::IsNotUnknown(_) => todo!(),
            ast::Expr::IsDistinctFrom(_, _) => todo!(),
            ast::Expr::IsNotDistinctFrom(_, _) => todo!(),
            ast::Expr::InList {
                expr,
                list,
                negated,
            } => todo!(),
            ast::Expr::InSubquery {
                expr,
                subquery,
                negated,
            } => todo!(),
            ast::Expr::InUnnest {
                expr,
                array_expr,
                negated,
            } => todo!(),
            ast::Expr::Between {
                expr,
                negated,
                low,
                high,
            } => todo!(),
            ast::Expr::BinaryOp { left, op, right } => self.binary_op(
                dependencies.get(left).clone(),
                op,
                dependencies.get(right).clone(),
            ),
            ast::Expr::Like {
                negated,
                expr,
                pattern,
                escape_char,
            } => todo!(),
            ast::Expr::ILike {
                negated,
                expr,
                pattern,
                escape_char,
            } => todo!(),
            ast::Expr::SimilarTo {
                negated,
                expr,
                pattern,
                escape_char,
            } => todo!(),
            ast::Expr::AnyOp(_) => todo!(),
            ast::Expr::AllOp(_) => todo!(),
            ast::Expr::UnaryOp { op, expr } => self.unary_op(op, dependencies.get(expr).clone()),
            ast::Expr::Cast { expr, data_type } => todo!(),
            ast::Expr::TryCast { expr, data_type } => todo!(),
            ast::Expr::SafeCast { expr, data_type } => todo!(),
            ast::Expr::AtTimeZone {
                timestamp,
                time_zone,
            } => todo!(),
            ast::Expr::Extract { field, expr } => todo!(),
            ast::Expr::Ceil { expr, field } => todo!(),
            ast::Expr::Floor { expr, field } => todo!(),
            ast::Expr::Position { expr, r#in } => todo!(),
            ast::Expr::Substring {
                expr,
                substring_from,
                substring_for,
            } => todo!(),
            ast::Expr::Trim {
                expr,
                trim_where,
                trim_what,
            } => todo!(),
            ast::Expr::Overlay {
                expr,
                overlay_what,
                overlay_from,
                overlay_for,
            } => todo!(),
            ast::Expr::Collate { expr, collation } => todo!(),
            ast::Expr::Nested(expr) => dependencies.get(expr).clone(),
            ast::Expr::Value(value) => self.value(value),
            ast::Expr::TypedString { data_type, value } => todo!(),
            ast::Expr::MapAccess { column, keys } => todo!(),
            ast::Expr::Function(function) => self.function(function, {
                let mut result = Vec::new();
                for function_arg in function.args.iter() {
                    result.push(match function_arg {
                        ast::FunctionArg::Named { name, arg } => FunctionArg::Named {
                            name: name.clone(),
                            arg: match arg {
                                ast::FunctionArgExpr::Expr(e) => dependencies.get(e).clone(),
                                ast::FunctionArgExpr::QualifiedWildcard(idents) => {
                                    self.qualified_wildcard(&idents.0)
                                }
                                ast::FunctionArgExpr::Wildcard => self.wildcard(),
                            },
                        },
                        ast::FunctionArg::Unnamed(arg) => FunctionArg::Unnamed(match arg {
                            ast::FunctionArgExpr::Expr(e) => dependencies.get(e).clone(),
                            ast::FunctionArgExpr::QualifiedWildcard(idents) => {
                                self.qualified_wildcard(&idents.0)
                            }
                            ast::FunctionArgExpr::Wildcard => self.wildcard(),
                        }),
                    });
                }
                result
            }),
            ast::Expr::AggregateExpressionWithFilter { expr, filter } => todo!(),
            ast::Expr::Case {
                operand,
                conditions,
                results,
                else_result,
            } => self.case(
                operand.clone().map(|x| dependencies.get(&*x).clone()),
                conditions.iter().map(|x| dependencies.get(x).clone()).collect(),
                results.iter().map(|x| dependencies.get(x).clone()).collect(),
                else_result.clone().map(|x| dependencies.get(&*x).clone()),
            ),
            ast::Expr::Exists { subquery, negated } => todo!(),
            ast::Expr::Subquery(_) => todo!(),
            ast::Expr::ArraySubquery(_) => todo!(),
            ast::Expr::ListAgg(_) => todo!(),
            ast::Expr::ArrayAgg(_) => todo!(),
            ast::Expr::GroupingSets(_) => todo!(),
            ast::Expr::Cube(_) => todo!(),
            ast::Expr::Rollup(_) => todo!(),
            ast::Expr::Tuple(_) => todo!(),
            ast::Expr::ArrayIndex { obj, indexes } => todo!(),
            ast::Expr::Array(_) => todo!(),
            ast::Expr::Interval(_) => todo!(),
            ast::Expr::MatchAgainst {
                columns,
                match_value,
                opt_search_modifier,
            } => todo!(),
            ast::Expr::IntroducedString { introducer, value } => todo!(),
        }
    }
}

/// A simple SQL expression parser with dialect
pub fn parse_expr_with_dialect<D: Dialect>(expr: &str, dialect: D) -> Result<ast::Expr> {
    let mut tokenizer = Tokenizer::new(&dialect, expr);
    let tokens = tokenizer.tokenize()?;
    let mut parser = Parser::new(&dialect).with_tokens(tokens);
    let expr = parser.parse_expr()?;
    Ok(expr)
}

/// A simple SQL expression parser to test the code
pub fn parse_expr(expr: &str) -> Result<ast::Expr> {
    parse_expr_with_dialect(expr, GenericDialect)
}

/// A simple display Visitor
pub struct DisplayVisitor;

impl<'a> Visitor<'a, String> for DisplayVisitor {
    fn qualified_wildcard(&self, idents: &'a Vec<ast::Ident>) -> String {
        format!("{}.*", idents.iter().join("."))
    }

    fn wildcard(&self) -> String {
        format!("*")
    }

    fn identifier(&self, ident: &'a ast::Ident) -> String {
        format!("{}", ident)
    }

    fn compound_identifier(&self, idents: &'a Vec<ast::Ident>) -> String {
        format!("{}", idents.iter().join("."))
    }

    fn binary_op(&self, left: String, op: &'a ast::BinaryOperator, right: String) -> String {
        format!("({} {} {})", left, op, right)
    }

    fn unary_op(&self, op: &'a ast::UnaryOperator, expr: String) -> String {
        format!("{} ({})", op, expr)
    }

    fn value(&self, value: &'a ast::Value) -> String {
        format!("{}", value)
    }

    fn function(&self, function: &'a ast::Function, args: Vec<FunctionArg<String>>) -> String {
        format!(
            "{}({})",
            function.name,
            args.into_iter()
                .map(|function_arg| match function_arg {
                    FunctionArg::Named { name, arg } => format!("{name}: {arg}"),
                    FunctionArg::Unnamed(arg) => format!("{arg}"),
                })
                .join(", ")
        )
    }

    fn case(&self, operand: Option<String>, conditions: Vec<String>, results: Vec<String>, else_result: Option<String>) -> String {
        let mut case_str = "CASE ".to_string();
        if let Some(op) = operand {case_str.push_str(&format!("{} ", op))};
        conditions.iter()
            .zip(results.iter())
            .for_each(|(c, r)| case_str.push_str(&format!("WHEN {} THEN {} ", c, r)));
        if let Some(r) = else_result {case_str.push_str(&format!("ELSE {} ", r))};
        case_str.push_str("END");
        case_str
    }

}

/// A simple ast::Expr -> Expr conversion Visitor
pub struct TryIntoExprVisitor<'a>(&'a Hierarchy<String>); // With name remapping

/// Implement conversion from Ident to Identifier
impl From<&ast::Ident> for Identifier {
    fn from(value: &ast::Ident) -> Self {
        value.value.clone().into()
    }
}

/// Implement conversion from Ident vector to Identifier
impl From<&Vec<ast::Ident>> for Identifier {
    fn from(value: &Vec<ast::Ident>) -> Self {
        value.into_iter().map(|i| i.value.clone()).collect()
    }
}

impl<'a> Visitor<'a, Result<Expr>> for TryIntoExprVisitor<'a> {
    fn qualified_wildcard(&self, idents: &'a Vec<ast::Ident>) -> Result<Expr> {
        todo!()
    }

    fn wildcard(&self) -> Result<Expr> {
        todo!()
    }

    fn identifier(&self, ident: &'a ast::Ident) -> Result<Expr> {
        let name = self
            .0
            .get(&ident.cloned())
            .cloned()
            .unwrap_or_else(|| ident.value.clone());
        Ok(Expr::col(name))
    }

    fn compound_identifier(&self, idents: &'a Vec<ast::Ident>) -> Result<Expr> {
        let name = self
            .0
            .get(&idents.cloned())
            .cloned()
            .unwrap_or_else(|| idents.split_last().unwrap().0.value.clone());
        Ok(Expr::col(name))
    }

    fn binary_op(
        &self,
        left: Result<Expr>,
        op: &'a ast::BinaryOperator,
        right: Result<Expr>,
    ) -> Result<Expr> {
        let left = left?;
        let right = right?;
        Ok(match op {
            ast::BinaryOperator::Plus => Expr::plus(left, right),
            ast::BinaryOperator::Minus => Expr::minus(left, right),
            ast::BinaryOperator::Multiply => Expr::multiply(left, right),
            ast::BinaryOperator::Divide => Expr::divide(left, right),
            ast::BinaryOperator::Modulo => Expr::modulo(left, right),
            ast::BinaryOperator::StringConcat => Expr::string_concat(left, right),
            ast::BinaryOperator::Gt => Expr::gt(left, right),
            ast::BinaryOperator::Lt => Expr::lt(left, right),
            ast::BinaryOperator::GtEq => Expr::gt_eq(left, right),
            ast::BinaryOperator::LtEq => Expr::lt_eq(left, right),
            ast::BinaryOperator::Spaceship => todo!(),
            ast::BinaryOperator::Eq => Expr::eq(left, right),
            ast::BinaryOperator::NotEq => Expr::not_eq(left, right),
            ast::BinaryOperator::And => Expr::and(left, right),
            ast::BinaryOperator::Or => Expr::or(left, right),
            ast::BinaryOperator::Xor => Expr::xor(left, right),
            ast::BinaryOperator::BitwiseOr => Expr::bitwise_or(left, right),
            ast::BinaryOperator::BitwiseAnd => Expr::bitwise_and(left, right),
            ast::BinaryOperator::BitwiseXor => Expr::bitwise_xor(left, right),
            ast::BinaryOperator::PGBitwiseXor => todo!(),
            ast::BinaryOperator::PGBitwiseShiftLeft => todo!(),
            ast::BinaryOperator::PGBitwiseShiftRight => todo!(),
            ast::BinaryOperator::PGRegexMatch => todo!(),
            ast::BinaryOperator::PGRegexIMatch => todo!(),
            ast::BinaryOperator::PGRegexNotMatch => todo!(),
            ast::BinaryOperator::PGRegexNotIMatch => todo!(),
            ast::BinaryOperator::PGCustomBinaryOperator(_) => todo!(),
            ast::BinaryOperator::PGExp => todo!(),
            ast::BinaryOperator::DuckIntegerDivide => todo!(),
            ast::BinaryOperator::MyIntegerDivide => todo!(),
            ast::BinaryOperator::Custom(_) => todo!(),
        })
    }

    fn unary_op(&self, op: &'a ast::UnaryOperator, expr: Result<Expr>) -> Result<Expr> {
        let expr = expr?;
        Ok(match op {
            ast::UnaryOperator::Plus => todo!(),
            ast::UnaryOperator::Minus => Expr::opposite(expr),
            ast::UnaryOperator::Not => Expr::not(expr),
            ast::UnaryOperator::PGBitwiseNot => todo!(),
            ast::UnaryOperator::PGSquareRoot => todo!(),
            ast::UnaryOperator::PGCubeRoot => todo!(),
            ast::UnaryOperator::PGPostfixFactorial => todo!(),
            ast::UnaryOperator::PGPrefixFactorial => todo!(),
            ast::UnaryOperator::PGAbs => todo!(),
        })
    }

    fn value(&self, value: &'a ast::Value) -> Result<Expr> {
        Ok(match value {
            ast::Value::Number(number, _) => {
                let x: f64 = FromStr::from_str(number)?;
                Expr::val(x)
            }
            ast::Value::SingleQuotedString(_) => todo!(),
            ast::Value::EscapedStringLiteral(_) => todo!(),
            ast::Value::NationalStringLiteral(_) => todo!(),
            ast::Value::HexStringLiteral(_) => todo!(),
            ast::Value::DoubleQuotedString(_) => todo!(),
            ast::Value::Boolean(_) => todo!(),
            ast::Value::Null => todo!(),
            ast::Value::Placeholder(_) => todo!(),
            ast::Value::UnQuotedString(_) => todo!(),
            ast::Value::DollarQuotedString(_) => todo!(),
            ast::Value::SingleQuotedByteStringLiteral(_) => todo!(),
            ast::Value::DoubleQuotedByteStringLiteral(_) => todo!(),
            ast::Value::RawStringLiteral(_) => todo!(),
        })
    }

    fn function(
        &self,
        function: &'a ast::Function,
        args: Vec<FunctionArg<Result<Expr>>>,
    ) -> Result<Expr> {
        // All args are unnamed for now
        let flat_args: Result<Vec<Expr>> = args
            .into_iter()
            .map(|function_arg| match function_arg {
                FunctionArg::Named { name: _, arg } => arg,
                FunctionArg::Unnamed(arg) => arg,
            })
            .collect();
        let flat_args = flat_args?;
        let function_name: &str = &function.name.0.iter().join(".").to_lowercase();
        Ok(match function_name {
            // Functions Opposite, Not, Exp, Ln, Log, Abs, Sin, Cos
            "opposite" => Expr::opposite(flat_args[0].clone()),
            "not" => Expr::not(flat_args[0].clone()),
            "exp" => Expr::exp(flat_args[0].clone()),
            "ln" => Expr::ln(flat_args[0].clone()),
            "log" => Expr::log(flat_args[0].clone()),
            "abs" => Expr::abs(flat_args[0].clone()),
            "sin" => Expr::sin(flat_args[0].clone()),
            "cos" => Expr::cos(flat_args[0].clone()),
            "sqrt" => Expr::sqrt(flat_args[0].clone()),
            "pow" => Expr::pow(flat_args[0].clone(), flat_args[1].clone()),
            "power" => Expr::pow(flat_args[0].clone(), flat_args[1].clone()),
            "lower" => Expr::lower(flat_args[0].clone()),
            "upper" => Expr::upper(flat_args[0].clone()),
            "char_length" => Expr::char_length(flat_args[0].clone()),
            "position" => Expr::position(flat_args[0].clone(), flat_args[1].clone()),
            // Aggregates
            "min" => Expr::min(flat_args[0].clone()),
            "max" => Expr::max(flat_args[0].clone()),
            "count" => Expr::count(flat_args[0].clone()),
            "avg" => Expr::mean(flat_args[0].clone()),
            "sum" => Expr::sum(flat_args[0].clone()),
            "variance" => Expr::var(flat_args[0].clone()),
            "stddev" => Expr::std(flat_args[0].clone()),
            _ => todo!(),
        })
    }

    fn case(&self, operand: Option<Result<Expr>>, conditions: Vec<Result<Expr>>, results: Vec<Result<Expr>>, else_result: Option<Result<Expr>>) -> Result<Expr> {
        let when_exprs = match operand {
            Some(op) => {
                conditions.iter()
                    .map(|x| self.binary_op(op.clone(), &ast::BinaryOperator::Eq, x.clone()))
                    .collect::<Result<Vec<Expr>>>()?
            },
            None => conditions.into_iter().collect::<Result<Vec<Expr>>>()?,
        };
        let then_exprs = results.into_iter().collect::<Result<Vec<Expr>>>()?;
        let mut case_expr = match else_result {
            Some(r) => r?,
            None => Expr::Value(Value::none()),
        };
        for (w,t) in when_exprs.iter().rev().zip(then_exprs.iter().rev()) {
            case_expr = Expr::case(w.clone(), t.clone(), case_expr.clone());
        }
        Ok(case_expr)
    }

}

// A struct holding a query and a context for conversion to Relation
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct ExprWithNames<'a>(&'a ast::Expr, &'a Hierarchy<String>);

impl<'a> ExprWithNames<'a> {
    pub fn new(expr: &'a ast::Expr, names: &'a Hierarchy<String>) -> Self {
        ExprWithNames(expr, names)
    }
}

impl<'a> With<&'a Hierarchy<String>, ExprWithNames<'a>> for &'a ast::Expr {
    fn with(self, input: &'a Hierarchy<String>) -> ExprWithNames<'a> {
        ExprWithNames::new(self, input)
    }
}

/// Based on the TryIntoExprVisitor implement the TryFrom trait
impl<'a> TryFrom<ExprWithNames<'a>> for Expr {
    type Error = Error;

    fn try_from(value: ExprWithNames<'a>) -> result::Result<Self, Self::Error> {
        let ExprWithNames(expr, names) = value;
        expr.accept(TryIntoExprVisitor(names))
    }
}

/// Based on the TryIntoExprVisitor implement the TryFrom trait
impl<'a> TryFrom<&'a ast::Expr> for Expr {
    type Error = Error;

    fn try_from(value: &'a ast::Expr) -> result::Result<Self, Self::Error> {
        Expr::try_from(value.with(&Hierarchy::empty()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{data_type::DataType, expr::dot::display};
    use std::convert::TryFrom;

    #[test]
    fn test_try_into_expr() {
        let ast_expr: ast::Expr = parse_expr("exp(a*cos(SIN(x) + 2*a + b))").unwrap();
        println!("ast::expr = {ast_expr}");
        let expr = Expr::try_from(ast_expr.with(&Hierarchy::empty())).unwrap();
        println!("expr = {}", expr);
    }

    #[test]
    fn test_try_into_expr_iter() {
        let expr = parse_expr("exp(a*cos(SIN(x) + 2*a + b))").unwrap();
        for (x, t) in expr.iter_with(TryIntoExprVisitor(&Hierarchy::empty())) {
            println!("{x} ({})", t.unwrap());
        }
    }

    #[test]
    fn test_display_iter() {
        let expr = parse_expr("(exp(a*max(sin(x) + 2*a + b)))").unwrap();
        for (x, t) in expr.iter_with(DisplayVisitor) {
            println!("{x} ({t})");
        }
    }

    #[test]
    fn test_parsing() {
        let expr = parse_expr("exp(a*max(sin(x) + 2*a + b))").unwrap();
        for x in expr.iter() {
            println!("{x}");
        }
    }

    #[ignore]
    #[test]
    fn test_try_into_expr_dot() {
        let ast_expr: ast::Expr = parse_expr("exp(a*cos(SIN(x) + 2*a + b))/count(c)").unwrap();
        println!("ast::expr = {ast_expr}");
        let expr = Expr::try_from(&ast_expr).unwrap();
        println!("expr = {}", expr);
        let data_type = DataType::structured([
            ("a", DataType::float_interval(1.1, 2.1)),
            ("b", DataType::float_values([-1., 2., 3.])),
            ("x", DataType::float()),
            ("c", DataType::list(DataType::Any, 1, 10)),
        ]);
        display(&expr, data_type);
    }

    #[test]
    fn test_try_into_expr_with_names() {
        let ast_expr: ast::Expr =
            parse_expr("log(exp(table_1.a*cos(SIN(table_2.x)) + 2*table_2.a + table_1.b))")
                .unwrap();
        println!("ast::expr = {ast_expr}");
        let expr = Expr::try_from(ast_expr.with(&Hierarchy::from([
            (["schema", "table_1", "a"], "a".into()),
            (["schema", "table_1", "b"], "b".into()),
        ])))
        .unwrap();
        println!("expr = {}", expr);
    }

    #[test]
    fn test_case() {
        let ast_expr: ast::Expr = parse_expr("CASE WHEN a > 5 THEN 5 WHEN a < 2 THEN 2 ELSE a END").unwrap();
        println!("ast::expr = {ast_expr}");
        let expr = Expr::try_from(ast_expr.with(&Hierarchy::empty())).unwrap();
        println!("expr = {}", expr);
        for (x, t) in ast_expr.iter_with(DisplayVisitor) {
            println!("{x} ({t})");
        }
        let true_expr = expr!(case(gt(a, 5), 5, case(lt(a, 2), 2, a)));
        assert_eq!(true_expr.to_string(), expr.to_string());

        let ast_expr: ast::Expr = parse_expr("CASE WHEN a > 5 THEN 5 WHEN a < 2 THEN 2 END").unwrap();
        println!("\nast::expr = {ast_expr}");
        let expr = Expr::try_from(ast_expr.with(&Hierarchy::empty())).unwrap();
        println!("expr = {}", expr);

        let ast_expr: ast::Expr = parse_expr("CASE a WHEN 5 THEN a + 3 WHEN 2 THEN a -4 ELSE a END").unwrap();
        println!("\nast::expr = {ast_expr}");
        let expr = Expr::try_from(ast_expr.with(&Hierarchy::empty())).unwrap();
        println!("expr = {}", expr);
    }
}
