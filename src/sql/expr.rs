//! This file provides tools for converting an ast::Expr
//! into the corresponding Qrlew expression.
//! Example: `Expr::try_from(sql_parser_expr)`

use super::{Error, Result};
use crate::{
    builder::{WithContext, WithoutContext},
    expr::{identifier::Identifier, Expr, Value},
    hierarchy::{Hierarchy, Path},
    namer,
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
            ast::Expr::JsonAccess { value, path: _ } => Dependencies::from([value.as_ref()]),
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
            ast::Expr::AnyOp {
                left,
                compare_op: _,
                right,
            } => Dependencies::from([left.as_ref(), right.as_ref()]),
            ast::Expr::AllOp {
                left,
                compare_op: _,
                right,
            } => Dependencies::from([left.as_ref(), right.as_ref()]),
            ast::Expr::UnaryOp { op: _, expr } => Dependencies::from([expr.as_ref()]),
            ast::Expr::Cast {
                expr,
                data_type: _,
                format: _,
                kind: _,
            } => Dependencies::from([expr.as_ref()]),
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
                special: _,
            } => vec![Some(expr), substring_from.as_ref(), substring_for.as_ref()]
                .iter()
                .filter_map(|expr| expr.map(AsRef::as_ref))
                .collect(),
            ast::Expr::Trim {
                expr,
                trim_where: _,
                trim_what,
                trim_characters: _,
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
            ast::Expr::MapAccess { column, keys: _ } => Dependencies::from([column.as_ref()]),
            ast::Expr::Function(function) => match &function.args {
                ast::FunctionArguments::None => Dependencies::empty(),
                ast::FunctionArguments::Subquery(_) => Dependencies::empty(),
                ast::FunctionArguments::List(list_args) => list_args
                    .args
                    .iter()
                    .map(|arg| match arg {
                        ast::FunctionArg::Named {
                            name: _,
                            arg,
                            operator: _,
                        } => arg,
                        ast::FunctionArg::Unnamed(arg) => arg,
                    })
                    .filter_map(|arg| match arg {
                        ast::FunctionArgExpr::Expr(expr) => Some(expr),
                        _ => None,
                    })
                    .collect(),
            },
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
                columns: _,
                match_value: _,
                opt_search_modifier: _,
            } => Dependencies::empty(),
            ast::Expr::IntroducedString {
                introducer: _,
                value: _,
            } => Dependencies::empty(),
            ast::Expr::RLike {
                negated: _,
                expr: _,
                pattern: _,
                regexp: _,
            } => todo!(),
            ast::Expr::Struct {
                values: _,
                fields: _,
            } => todo!(),
            ast::Expr::Named { expr: _, name: _ } => todo!(),
            ast::Expr::Convert {
                expr: _,
                data_type: _,
                charset: _,
                target_before_value: _,
                styles: _,
            } => todo!(),
            ast::Expr::Wildcard => todo!(),
            ast::Expr::QualifiedWildcard(_) => todo!(),
            ast::Expr::Dictionary(_) => Dependencies::empty(),
            ast::Expr::OuterJoin(expr) => Dependencies::from([expr.as_ref()]),
            ast::Expr::Prior(expr) => Dependencies::from([expr.as_ref()]),
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
    fn case(
        &self,
        operand: Option<T>,
        conditions: Vec<T>,
        results: Vec<T>,
        else_result: Option<T>,
    ) -> T;
    fn position(&self, expr: T, r#in: T) -> T;
    fn in_list(&self, expr: T, list: Vec<T>) -> T;
    fn trim(&self, expr: T, trim_where: &Option<ast::TrimWhereField>, trim_what: Option<T>) -> T;
    fn substring(&self, expr: T, substring_from: Option<T>, substring_for: Option<T>) -> T;
    fn ceil(&self, expr: T, field: &'a ast::DateTimeField) -> T;
    fn floor(&self, expr: T, field: &'a ast::DateTimeField) -> T;
    fn cast(&self, expr: T, data_type: &'a ast::DataType) -> T;
    fn extract(&self, field: &'a ast::DateTimeField, expr: T) -> T;
    fn like(&self, expr: T, pattern: T) -> T;
    fn ilike(&self, expr: T, pattern: T) -> T;
    fn is(&self, expr: T, value: Option<bool>) -> T;
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
            ast::Expr::JsonAccess { value: _, path: _ } => todo!(),
            ast::Expr::CompositeAccess { expr: _, key: _ } => todo!(),
            ast::Expr::IsFalse(expr) => self.is(
                self.cast(dependencies.get(expr).clone(), &ast::DataType::Boolean),
                Some(false),
            ),
            ast::Expr::IsNotFalse(expr) => self.unary_op(
                &ast::UnaryOperator::Not,
                self.is(
                    self.cast(dependencies.get(expr).clone(), &ast::DataType::Boolean),
                    Some(false),
                ),
            ),
            ast::Expr::IsTrue(expr) => self.is(
                self.cast(dependencies.get(expr).clone(), &ast::DataType::Boolean),
                Some(true),
            ),
            ast::Expr::IsNotTrue(expr) => self.unary_op(
                &ast::UnaryOperator::Not,
                self.is(
                    self.cast(dependencies.get(expr).clone(), &ast::DataType::Boolean),
                    Some(true),
                ),
            ),
            ast::Expr::IsNull(expr) => self.is(dependencies.get(expr).clone(), None),
            ast::Expr::IsNotNull(expr) => self.unary_op(
                &ast::UnaryOperator::Not,
                self.is(dependencies.get(expr).clone(), None),
            ),
            ast::Expr::IsUnknown(_) => todo!(),
            ast::Expr::IsNotUnknown(_) => todo!(),
            ast::Expr::IsDistinctFrom(_, _) => todo!(),
            ast::Expr::IsNotDistinctFrom(_, _) => todo!(),
            ast::Expr::InList {
                expr,
                list,
                negated,
            } => {
                let in_expr = self.in_list(
                    dependencies.get(expr).clone(),
                    list.iter().map(|x| dependencies.get(x).clone()).collect(),
                );
                if *negated {
                    self.unary_op(&ast::UnaryOperator::Not, in_expr)
                } else {
                    in_expr
                }
            }
            ast::Expr::InSubquery {
                expr: _,
                subquery: _,
                negated: _,
            } => todo!(),
            ast::Expr::InUnnest {
                expr: _,
                array_expr: _,
                negated: _,
            } => todo!(),
            ast::Expr::Between {
                expr,
                negated,
                low,
                high,
            } => {
                let x = self.binary_op(
                    self.binary_op(
                        dependencies.get(expr).clone(),
                        &ast::BinaryOperator::GtEq,
                        dependencies.get(low).clone(),
                    ),
                    &ast::BinaryOperator::And,
                    self.binary_op(
                        dependencies.get(expr).clone(),
                        &ast::BinaryOperator::LtEq,
                        dependencies.get(high).clone(),
                    ),
                );
                if *negated {
                    self.unary_op(&ast::UnaryOperator::Not, x)
                } else {
                    x
                }
            }
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
            } => {
                if escape_char.is_some() {
                    todo!()
                };
                let x = self.like(
                    dependencies.get(expr).clone(),
                    dependencies.get(pattern).clone(),
                );
                if *negated {
                    self.unary_op(&ast::UnaryOperator::Not, x)
                } else {
                    x
                }
            }
            ast::Expr::ILike {
                negated,
                expr,
                pattern,
                escape_char,
            } => {
                if escape_char.is_some() {
                    todo!()
                };
                let x = self.ilike(
                    dependencies.get(expr).clone(),
                    dependencies.get(pattern).clone(),
                );
                if *negated {
                    self.unary_op(&ast::UnaryOperator::Not, x)
                } else {
                    x
                }
            }
            ast::Expr::SimilarTo {
                negated: _,
                expr: _,
                pattern: _,
                escape_char: _,
            } => todo!(),
            ast::Expr::AnyOp {
                left: _,
                compare_op: _,
                right: _,
            } => {
                todo!()
            }
            ast::Expr::AllOp {
                left: _,
                compare_op: _,
                right: _,
            } => {
                todo!()
            }
            ast::Expr::UnaryOp { op, expr } => self.unary_op(op, dependencies.get(expr).clone()),
            ast::Expr::Cast {
                expr,
                data_type,
                format: _,
                kind: _,
            } => self.cast(dependencies.get(expr).clone(), data_type),
            ast::Expr::AtTimeZone {
                timestamp: _,
                time_zone: _,
            } => todo!(),
            ast::Expr::Extract { field, expr } => {
                self.extract(field, dependencies.get(expr).clone())
            }
            ast::Expr::Ceil { expr, field } => self.ceil(dependencies.get(expr).clone(), field),
            ast::Expr::Floor { expr, field } => self.floor(dependencies.get(expr).clone(), field),
            ast::Expr::Position { expr, r#in } => self.position(
                dependencies.get(expr).clone(),
                dependencies.get(r#in).clone(),
            ),
            ast::Expr::Substring {
                expr,
                substring_from,
                substring_for,
                special: _,
            } => self.substring(
                dependencies.get(expr).clone(),
                substring_from
                    .as_ref()
                    .map(|x| dependencies.get(x.as_ref()).clone()),
                substring_for
                    .as_ref()
                    .map(|x| dependencies.get(x.as_ref()).clone()),
            ),
            ast::Expr::Trim {
                expr,
                trim_where,
                trim_what,
                trim_characters,
            } => {
                let trim_what = match (trim_what, trim_characters) {
                    (None, None) => None,
                    (Some(x), None) => Some(x.as_ref()),
                    (None, Some(_v)) => todo!(),
                    _ => todo!(),
                };
                self.trim(
                    dependencies.get(expr).clone(),
                    trim_where,
                    trim_what.map(|x| dependencies.get(x).clone()),
                )
            }
            ast::Expr::Overlay {
                expr: _,
                overlay_what: _,
                overlay_from: _,
                overlay_for: _,
            } => todo!(),
            ast::Expr::Collate {
                expr: _,
                collation: _,
            } => todo!(),
            ast::Expr::Nested(expr) => dependencies.get(expr).clone(),
            ast::Expr::Value(value) => self.value(value),
            ast::Expr::TypedString {
                data_type: _,
                value: _,
            } => todo!(),
            ast::Expr::MapAccess { column: _, keys: _ } => todo!(),
            ast::Expr::Function(function) => self.function(function, {
                let mut result = vec![];
                let function_args = match &function.args {
                    ast::FunctionArguments::None => vec![],
                    ast::FunctionArguments::Subquery(_) => vec![],
                    ast::FunctionArguments::List(arg_list) => arg_list.args.iter().collect(),
                };
                for function_arg in function_args.iter() {
                    result.push(match function_arg {
                        ast::FunctionArg::Named {
                            name,
                            arg,
                            operator: _,
                        } => FunctionArg::Named {
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
            ast::Expr::Case {
                operand,
                conditions,
                results,
                else_result,
            } => self.case(
                operand.clone().map(|x| dependencies.get(&*x).clone()),
                conditions
                    .iter()
                    .map(|x| dependencies.get(x).clone())
                    .collect(),
                results
                    .iter()
                    .map(|x| dependencies.get(x).clone())
                    .collect(),
                else_result.clone().map(|x| dependencies.get(&*x).clone()),
            ),
            ast::Expr::Exists {
                subquery: _,
                negated: _,
            } => todo!(),
            ast::Expr::Subquery(_) => todo!(),
            ast::Expr::GroupingSets(_) => todo!(),
            ast::Expr::Cube(_) => todo!(),
            ast::Expr::Rollup(_) => todo!(),
            ast::Expr::Tuple(_) => todo!(),
            ast::Expr::ArrayIndex { obj: _, indexes: _ } => todo!(),
            ast::Expr::Array(_) => todo!(),
            ast::Expr::Interval(_) => todo!(),
            ast::Expr::MatchAgainst {
                columns: _,
                match_value: _,
                opt_search_modifier: _,
            } => todo!(),
            ast::Expr::IntroducedString {
                introducer: _,
                value: _,
            } => todo!(),
            ast::Expr::RLike {
                negated: _,
                expr: _,
                pattern: _,
                regexp: _,
            } => todo!(),
            ast::Expr::Struct {
                values: _,
                fields: _,
            } => todo!(),
            ast::Expr::Named { expr: _, name: _ } => todo!(),
            ast::Expr::Convert {
                expr: _,
                data_type: _,
                charset: _,
                target_before_value: _,
                styles: _,
            } => todo!(),
            ast::Expr::Wildcard => todo!(),
            ast::Expr::QualifiedWildcard(_) => todo!(),
            ast::Expr::Dictionary(_) => todo!(),
            ast::Expr::OuterJoin(_) => todo!(),
            ast::Expr::Prior(_) => todo!(),
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

    fn case(
        &self,
        operand: Option<String>,
        conditions: Vec<String>,
        results: Vec<String>,
        else_result: Option<String>,
    ) -> String {
        let mut case_str = "CASE ".to_string();
        if let Some(op) = operand {
            case_str.push_str(&format!("{} ", op))
        };
        conditions
            .iter()
            .zip(results.iter())
            .for_each(|(c, r)| case_str.push_str(&format!("WHEN {} THEN {} ", c, r)));
        if let Some(r) = else_result {
            case_str.push_str(&format!("ELSE {} ", r))
        };
        case_str.push_str("END");
        case_str
    }

    fn position(&self, expr: String, r#in: String) -> String {
        format!("POSITION({} IN {})", expr, r#in)
    }

    fn in_list(&self, expr: String, list: Vec<String>) -> String {
        format!(
            "{} IN ({})",
            expr,
            list.into_iter().map(|x| format!("{x}")).join(", ")
        )
    }

    fn trim(
        &self,
        expr: String,
        trim_where: &Option<ast::TrimWhereField>,
        trim_what: Option<String>,
    ) -> String {
        format!(
            "TRIM ({} {} FROM {})",
            trim_where.map(|w| w.to_string()).unwrap_or("".to_string()),
            expr,
            trim_what.unwrap_or("".to_string()),
        )
    }

    fn substring(
        &self,
        expr: String,
        substring_from: Option<String>,
        substring_for: Option<String>,
    ) -> String {
        format!(
            "SUBSTRING ({} {} {})",
            expr,
            substring_from
                .map(|s| format!("FROM {}", s))
                .unwrap_or("".to_string()),
            substring_for
                .map(|s| format!("FOR {}", s))
                .unwrap_or("".to_string()),
        )
    }

    fn ceil(&self, expr: String, field: &'a ast::DateTimeField) -> String {
        format!(
            "CEIL ({}{})",
            expr,
            if matches!(field, ast::DateTimeField::NoDateTime) {
                "".to_string()
            } else {
                format!(", {field}")
            }
        )
    }

    fn floor(&self, expr: String, field: &'a ast::DateTimeField) -> String {
        format!(
            "FLOOR ({}{})",
            expr,
            if matches!(field, ast::DateTimeField::NoDateTime) {
                "".to_string()
            } else {
                format!(", {field}")
            }
        )
    }

    fn cast(&self, expr: String, data_type: &ast::DataType) -> String {
        format!("CAST ({} AS {})", expr, data_type)
    }

    fn extract(&self, field: &'a ast::DateTimeField, expr: String) -> String {
        format!("EXTRACT({} FROM {})", field, expr)
    }

    fn like(&self, expr: String, pattern: String) -> String {
        format!("{} LIKE {}", expr, pattern)
    }

    fn ilike(&self, expr: String, pattern: String) -> String {
        format!("{} ILIKE {}", expr, pattern)
    }

    fn is(&self, expr: String, value: Option<bool>) -> String {
        format!(
            "{} IS {}",
            expr,
            value
                .map(|b| b.to_string().to_uppercase())
                .unwrap_or("NULL".to_string())
        )
    }
}

/// A simple ast::Expr -> Expr conversion Visitor
pub struct TryIntoExprVisitor<'a>(&'a Hierarchy<Identifier>); // With columns remapping

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
    fn qualified_wildcard(&self, _idents: &'a Vec<ast::Ident>) -> Result<Expr> {
        todo!()
    }

    fn wildcard(&self) -> Result<Expr> {
        Ok(Expr::val(1))
    }

    fn identifier(&self, ident: &'a ast::Ident) -> Result<Expr> {
        let column = self.0.get(&ident.cloned()).cloned().unwrap_or_else(|| {
            if let Some(_) = ident.quote_style {
                ident.value.clone().into()
            } else {
                ident.value.to_lowercase().clone().into()
            }
        });
        Ok(Expr::Column(column))
    }

    fn compound_identifier(&self, idents: &'a Vec<ast::Ident>) -> Result<Expr> {
        let column = self
            .0
            .get(&idents.cloned())
            .cloned()
            .unwrap_or_else(|| idents.iter().map(|i| i.value.clone()).collect());
        Ok(Expr::Column(column))
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
            ast::BinaryOperator::PGOverlap => todo!(),
            ast::BinaryOperator::PGLikeMatch => todo!(),
            ast::BinaryOperator::PGILikeMatch => todo!(),
            ast::BinaryOperator::PGNotLikeMatch => todo!(),
            ast::BinaryOperator::PGNotILikeMatch => todo!(),
            ast::BinaryOperator::PGStartsWith => todo!(),
            ast::BinaryOperator::Arrow => todo!(),
            ast::BinaryOperator::LongArrow => todo!(),
            ast::BinaryOperator::HashArrow => todo!(),
            ast::BinaryOperator::HashLongArrow => todo!(),
            ast::BinaryOperator::AtAt => todo!(),
            ast::BinaryOperator::AtArrow => todo!(),
            ast::BinaryOperator::ArrowAt => todo!(),
            ast::BinaryOperator::HashMinus => todo!(),
            ast::BinaryOperator::AtQuestion => todo!(),
            ast::BinaryOperator::Question => todo!(),
            ast::BinaryOperator::QuestionAnd => todo!(),
            ast::BinaryOperator::QuestionPipe => todo!(),
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
            ast::Value::SingleQuotedString(v) => Expr::val(v.to_string()),
            ast::Value::EscapedStringLiteral(_) => todo!(),
            ast::Value::NationalStringLiteral(_) => todo!(),
            ast::Value::HexStringLiteral(_) => todo!(),
            ast::Value::DoubleQuotedString(_) => todo!(),
            ast::Value::Boolean(b) => Expr::val(*b),
            ast::Value::Null => Expr::val(None),
            ast::Value::Placeholder(_) => todo!(),
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
        let distinct: bool = match &function.args {
            ast::FunctionArguments::List(func_arg_list)
                if func_arg_list.duplicate_treatment == Some(ast::DuplicateTreatment::Distinct) =>
            {
                true
            }
            _ => false,
        };
        Ok(match function_name {
            // Math Functions
            "opposite" => Expr::opposite(flat_args[0].clone()),
            "not" => Expr::not(flat_args[0].clone()),
            "exp" => Expr::exp(flat_args[0].clone()),
            "ln" => Expr::ln(flat_args[0].clone()),
            "log" => {
                if flat_args.len() == 1 {
                    Expr::log(flat_args[0].clone())
                } else {
                    Expr::divide(
                        Expr::log(flat_args[1].clone()),
                        Expr::log(flat_args[0].clone()),
                    )
                }
            }
            "log2" => Expr::divide(Expr::log(Expr::val(2)), Expr::log(flat_args[0].clone())),
            "log10" => Expr::divide(Expr::log(Expr::val(10)), Expr::log(flat_args[0].clone())),
            "abs" => Expr::abs(flat_args[0].clone()),
            "sin" => Expr::sin(flat_args[0].clone()),
            "cos" => Expr::cos(flat_args[0].clone()),
            "tan" => Expr::divide(
                Expr::sin(flat_args[0].clone()),
                Expr::cos(flat_args[0].clone()),
            ),
            "sqrt" => Expr::sqrt(flat_args[0].clone()),
            "pow" => Expr::pow(flat_args[0].clone(), flat_args[1].clone()),
            "power" => Expr::pow(flat_args[0].clone(), flat_args[1].clone()),
            "square" => Expr::pow(flat_args[0].clone(), Expr::val(2)),
            "md5" => Expr::md5(flat_args[0].clone()),
            "coalesce" => {
                let (first, vec) = flat_args.split_first().unwrap();
                vec.iter()
                    .fold(first.clone(), |acc, x| Expr::coalesce(acc, x.clone()))
            }
            "ltrim" => self.trim(
                Ok(flat_args[0].clone()),
                &Some(ast::TrimWhereField::Leading),
                (flat_args.len() > 1).then_some(Ok(flat_args[1].clone())),
            )?,
            "rtrim" => self.trim(
                Ok(flat_args[0].clone()),
                &Some(ast::TrimWhereField::Trailing),
                (flat_args.len() > 1).then_some(Ok(flat_args[1].clone())),
            )?,
            "btrim" => self.trim(
                Ok(flat_args[0].clone()),
                &Some(ast::TrimWhereField::Both),
                (flat_args.len() > 1).then_some(Ok(flat_args[1].clone())),
            )?,
            "round" => {
                let precision = if flat_args.len() > 1 {
                    flat_args[1].clone()
                } else {
                    Expr::val(0)
                };
                Expr::round(flat_args[0].clone(), precision)
            }
            "trunc" | "truncate" => {
                let precision = if flat_args.len() > 1 {
                    flat_args[1].clone()
                } else {
                    Expr::val(0)
                };
                Expr::trunc(flat_args[0].clone(), precision)
            }
            "sign" => Expr::sign(flat_args[0].clone()),
            "random" | "rand" => Expr::random(namer::new_id("UNIFORM_SAMPLING")),
            "pi" => Expr::pi(),
            "degrees" => Expr::multiply(
                flat_args[0].clone(),
                Expr::divide(Expr::val(180.), Expr::pi()),
            ),
            "choose" => Expr::choose(
                flat_args[0].clone(),
                Expr::val(Value::list(
                    flat_args
                        .iter()
                        .skip(1)
                        .map(|x| Value::try_from(x.clone()).map_err(|e| Error::other(e)))
                        .collect::<Result<Vec<_>>>()?,
                )),
            ),
            // String functions
            "lower" => Expr::lower(flat_args[0].clone()),
            "upper" => Expr::upper(flat_args[0].clone()),
            "char_length" => Expr::char_length(flat_args[0].clone()),
            "concat" => Expr::concat(flat_args.clone()),
            "substr" => {
                if flat_args.len() > 2 {
                    Expr::substr_with_size(
                        flat_args[0].clone(),
                        flat_args[1].clone(),
                        flat_args[2].clone(),
                    )
                } else {
                    Expr::substr(flat_args[0].clone(), flat_args[1].clone())
                }
            }
            "regexp_contains" => Expr::regexp_contains(flat_args[0].clone(), flat_args[1].clone()),
            "regexp_extract" | "regexp_substr" => {
                let position = if flat_args.len() > 2 {
                    flat_args[2].clone()
                } else {
                    Expr::val(0)
                };
                let occurrence = if flat_args.len() > 3 {
                    flat_args[3].clone()
                } else {
                    Expr::val(1)
                };
                Expr::regexp_extract(
                    flat_args[0].clone(),
                    flat_args[1].clone(),
                    position,
                    occurrence,
                )
            }
            "regexp_replace" => Expr::regexp_replace(
                flat_args[0].clone(),
                flat_args[1].clone(),
                flat_args[2].clone(),
            ),
            "newid" => Expr::newid(),
            "encode" => Expr::encode(flat_args[0].clone(), flat_args[1].clone()),
            "decode" => Expr::decode(flat_args[0].clone(), flat_args[1].clone()),
            "unhex" | "from_hex" => Expr::unhex(flat_args[0].clone()),
            // Date functions
            "current_date" => Expr::current_date(),
            "current_time" => Expr::current_time(),
            "current_timestamp" => Expr::current_timestamp(),
            "dayname" => Expr::dayname(flat_args[0].clone()),
            "date_format" => Expr::date_format(flat_args[0].clone(), flat_args[1].clone()),
            "quarter" => Expr::quarter(flat_args[0].clone()),
            "datetime_diff" => Expr::datetime_diff(
                flat_args[0].clone(),
                flat_args[1].clone(),
                flat_args[2].clone(),
            ),
            "date" => Expr::date(flat_args[0].clone()),
            "from_unixtime" => {
                let format = if flat_args.len() > 1 {
                    flat_args[1].clone()
                } else {
                    Expr::val("%Y-%m-%d %H:%i:%S".to_string())
                };
                Expr::from_unixtime(flat_args[0].clone(), format)
            }
            "unix_timestamp" => {
                let arg = if flat_args.len() > 0 {
                    flat_args[0].clone()
                } else {
                    Expr::current_timestamp()
                };
                Expr::unix_timestamp(arg)
            }
            "greatest" => Expr::greatest(flat_args[0].clone(), flat_args[1].clone()),
            "least" => Expr::least(flat_args[0].clone(), flat_args[1].clone()),
            // Aggregates
            "min" => Expr::min(flat_args[0].clone()),
            "max" => Expr::max(flat_args[0].clone()),
            "count" if distinct => Expr::count_distinct(flat_args[0].clone()),
            "count" => Expr::count(flat_args[0].clone()),
            "avg" if distinct => Expr::mean_distinct(flat_args[0].clone()),
            "avg" => Expr::mean(flat_args[0].clone()),
            "sum" if distinct => Expr::sum_distinct(flat_args[0].clone()),
            "sum" => Expr::sum(flat_args[0].clone()),
            "variance" if distinct => Expr::var_distinct(flat_args[0].clone()),
            "variance" => Expr::var(flat_args[0].clone()),
            "stddev" if distinct => Expr::std_distinct(flat_args[0].clone()),
            "stddev" => Expr::std(flat_args[0].clone()),
            _ => todo!(),
        })
    }

    fn case(
        &self,
        operand: Option<Result<Expr>>,
        conditions: Vec<Result<Expr>>,
        results: Vec<Result<Expr>>,
        else_result: Option<Result<Expr>>,
    ) -> Result<Expr> {
        let when_exprs = match operand {
            Some(op) => conditions
                .iter()
                .map(|x| self.binary_op(op.clone(), &ast::BinaryOperator::Eq, x.clone()))
                .collect::<Result<Vec<Expr>>>()?,
            None => conditions.into_iter().collect::<Result<Vec<Expr>>>()?,
        };
        let then_exprs = results.into_iter().collect::<Result<Vec<Expr>>>()?;
        let mut case_expr = match else_result {
            Some(r) => r?,
            None => Expr::Value(Value::unit()),
        };
        for (w, t) in when_exprs.iter().rev().zip(then_exprs.iter().rev()) {
            case_expr = Expr::case(w.clone(), t.clone(), case_expr.clone());
        }
        Ok(case_expr)
    }

    fn position(&self, expr: Result<Expr>, r#in: Result<Expr>) -> Result<Expr> {
        Ok(Expr::position(expr?, r#in?))
    }

    fn in_list(&self, expr: Result<Expr>, list: Vec<Result<Expr>>) -> Result<Expr> {
        let list: Result<Vec<Value>> = list
            .into_iter()
            .map(|r| {
                r.map(|x| Value::try_from(x).map_err(|e| Error::other(e)))
                    .and_then(|x| x)
                    .map_err(|e| Error::other(e))
            })
            .collect();
        Ok(Expr::in_list(expr?, Expr::val(Value::list(list?))))
    }

    fn trim(
        &self,
        expr: Result<Expr>,
        trim_where: &Option<ast::TrimWhereField>,
        trim_what: Option<Result<Expr>>,
    ) -> Result<Expr> {
        let trim_what = trim_what.unwrap_or(Ok(Expr::val(" ".to_string())));
        Ok(match trim_where {
            Some(ast::TrimWhereField::Leading) => Expr::ltrim(expr?, trim_what?),
            Some(ast::TrimWhereField::Trailing) => Expr::rtrim(expr?, trim_what?),
            Some(ast::TrimWhereField::Both) | None => {
                Expr::ltrim(Expr::rtrim(expr?, trim_what.clone()?), trim_what?)
            }
        })
    }

    fn substring(
        &self,
        expr: Result<Expr>,
        substring_from: Option<Result<Expr>>,
        substring_for: Option<Result<Expr>>,
    ) -> Result<Expr> {
        let substring_from = substring_from.unwrap_or(Ok(Expr::val(0)));
        substring_for
            .map(|x| {
                Ok(Expr::substr_with_size(
                    expr.clone()?,
                    substring_from.clone()?,
                    x?,
                ))
            })
            .unwrap_or(Ok(Expr::substr(expr.clone()?, substring_from.clone()?)))
    }

    fn ceil(&self, expr: Result<Expr>, field: &'a ast::DateTimeField) -> Result<Expr> {
        if !matches!(field, ast::DateTimeField::NoDateTime) {
            todo!()
        }
        Ok(Expr::ceil(expr.clone()?))
    }

    fn floor(&self, expr: Result<Expr>, field: &'a ast::DateTimeField) -> Result<Expr> {
        if !matches!(field, ast::DateTimeField::NoDateTime) {
            todo!()
        }
        Ok(Expr::floor(expr.clone()?))
    }

    fn cast(&self, expr: Result<Expr>, data_type: &'a ast::DataType) -> Result<Expr> {
        Ok(match data_type {
            //Text
            ast::DataType::Character(_)
            | ast::DataType::Char(_)
            | ast::DataType::CharacterVarying(_)
            | ast::DataType::CharVarying(_)
            | ast::DataType::Varchar(_)
            | ast::DataType::Nvarchar(_)
            | ast::DataType::Uuid
            | ast::DataType::CharacterLargeObject(_)
            | ast::DataType::CharLargeObject(_)
            | ast::DataType::Clob(_)
            | ast::DataType::Text
            | ast::DataType::String(_) => Expr::cast_as_text(expr.clone()?),
            //Bytes
            ast::DataType::Binary(_)
            | ast::DataType::Varbinary(_)
            | ast::DataType::Blob(_)
            | ast::DataType::Bytes(_)
            | ast::DataType::Bytea => todo!(),
            //Float
            ast::DataType::Numeric(_)
            | ast::DataType::Decimal(_)
            | ast::DataType::BigNumeric(_)
            | ast::DataType::BigDecimal(_)
            | ast::DataType::Dec(_)
            | ast::DataType::Float(_)
            | ast::DataType::Float4
            | ast::DataType::Float64
            | ast::DataType::Real
            | ast::DataType::Float8
            | ast::DataType::Double
            | ast::DataType::DoublePrecision => Expr::cast_as_float(expr.clone()?),
            // Integer
            ast::DataType::TinyInt(_)
            | ast::DataType::UnsignedTinyInt(_)
            | ast::DataType::Int2(_)
            | ast::DataType::UnsignedInt2(_)
            | ast::DataType::SmallInt(_)
            | ast::DataType::UnsignedSmallInt(_)
            | ast::DataType::MediumInt(_)
            | ast::DataType::UnsignedMediumInt(_)
            | ast::DataType::Int(_)
            | ast::DataType::Int4(_)
            | ast::DataType::Int64
            | ast::DataType::Integer(_)
            | ast::DataType::UnsignedInt(_)
            | ast::DataType::UnsignedInt4(_)
            | ast::DataType::UnsignedInteger(_)
            | ast::DataType::BigInt(_)
            | ast::DataType::UnsignedBigInt(_)
            | ast::DataType::Int8(_)
            | ast::DataType::UnsignedInt8(_) => Expr::cast_as_integer(expr.clone()?),
            // Boolean
            ast::DataType::Bool | ast::DataType::Boolean => Expr::cast_as_boolean(expr.clone()?),
            // Date
            ast::DataType::Date => Expr::cast_as_date(expr.clone()?),
            // Time
            ast::DataType::Time(_, _) => Expr::cast_as_time(expr.clone()?),
            // DateTime
            ast::DataType::Datetime(_) | ast::DataType::Timestamp(_, _) => {
                Expr::cast_as_date_time(expr.clone()?)
            }
            ast::DataType::Interval => todo!(),
            ast::DataType::JSON => todo!(),
            ast::DataType::Regclass => todo!(),
            ast::DataType::Custom(_, _) => todo!(),
            ast::DataType::Array(_) => todo!(),
            ast::DataType::Enum(_) => todo!(),
            ast::DataType::Set(_) => todo!(),
            ast::DataType::Struct(_) => todo!(),
            ast::DataType::JSONB => todo!(),
            ast::DataType::Unspecified => todo!(),
        })
    }

    fn extract(&self, field: &'a ast::DateTimeField, expr: Result<Expr>) -> Result<Expr> {
        Ok(match field {
            ast::DateTimeField::Year => Expr::extract_year(expr.clone()?),
            ast::DateTimeField::Month => Expr::extract_month(expr.clone()?),
            ast::DateTimeField::Week(_) => Expr::extract_week(expr.clone()?),
            ast::DateTimeField::Day => Expr::extract_day(expr.clone()?),
            ast::DateTimeField::Hour => Expr::extract_hour(expr.clone()?),
            ast::DateTimeField::Minute => Expr::extract_minute(expr.clone()?),
            ast::DateTimeField::Second => Expr::extract_second(expr.clone()?),
            ast::DateTimeField::Dow => Expr::extract_dow(expr.clone()?),
            ast::DateTimeField::Microsecond => Expr::extract_microsecond(expr.clone()?),
            ast::DateTimeField::Millisecond => Expr::extract_millisecond(expr.clone()?),
            _ => todo!(),
        })
    }

    fn like(&self, expr: Result<Expr>, pattern: Result<Expr>) -> Result<Expr> {
        Ok(Expr::like(expr.clone()?, pattern.clone()?))
    }

    fn ilike(&self, expr: Result<Expr>, pattern: Result<Expr>) -> Result<Expr> {
        Ok(Expr::ilike(expr.clone()?, pattern.clone()?))
    }

    fn is(&self, expr: Result<Expr>, value: Option<bool>) -> Result<Expr> {
        Ok(match value {
            Some(b) => Expr::is_bool(expr.clone()?, Expr::val(b)),
            None => Expr::is_null(expr.clone()?),
        })
    }
}

/// Based on the TryIntoExprVisitor implement the TryFrom trait
impl<'a> TryFrom<WithContext<&'a ast::Expr, &'a Hierarchy<Identifier>>> for Expr {
    type Error = Error;

    fn try_from(
        value: WithContext<&'a ast::Expr, &'a Hierarchy<Identifier>>,
    ) -> result::Result<Self, Self::Error> {
        let WithContext { object, context } = value;
        object.accept(TryIntoExprVisitor(context))
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
    use crate::{builder::WithContext, data_type::DataType, display::Dot};
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
        WithContext {
            object: &expr,
            context: data_type,
        }
        .display_dot()
        .unwrap();
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
        let ast_expr: ast::Expr =
            parse_expr("CASE WHEN a > 5 THEN 5 WHEN a < 2 THEN 2 ELSE a END").unwrap();
        println!("ast::expr = {ast_expr}");
        let expr = Expr::try_from(ast_expr.with(&Hierarchy::empty())).unwrap();
        println!("expr = {}", expr);
        for (x, t) in ast_expr.iter_with(DisplayVisitor) {
            println!("{x} ({t})");
        }
        let true_expr = expr!(case(gt(a, 5), 5, case(lt(a, 2), 2, a)));
        assert_eq!(true_expr.to_string(), expr.to_string());
        assert_eq!(
            ast::Expr::from(&expr).to_string(),
            String::from(
                "CASE WHEN (a) > (5) THEN 5 ELSE CASE WHEN (a) < (2) THEN 2 ELSE a END END"
            )
        );

        let ast_expr: ast::Expr =
            parse_expr("CASE WHEN a > 5 THEN 5 WHEN a < 2 THEN 2 END").unwrap();
        println!("\nast::expr = {ast_expr}");
        let expr = Expr::try_from(ast_expr.with(&Hierarchy::empty())).unwrap();
        println!("expr = {}", expr);
        
        
        assert_eq!(
            ast::Expr::from(&expr).to_string(),
            String::from(
                "CASE WHEN (a) > (5) THEN 5 ELSE CASE WHEN (a) < (2) THEN 2 ELSE NULL END END"
            )
        );

        let ast_expr: ast::Expr =
            parse_expr("CASE a WHEN 5 THEN a + 3 WHEN 2 THEN a -4 ELSE a END").unwrap();
        println!("\nast::expr = {ast_expr}");
        let expr = Expr::try_from(ast_expr.with(&Hierarchy::empty())).unwrap();
        println!("expr = {}", expr);
        assert_eq!(
            ast::Expr::from(&expr).to_string(),
            String::from(
                "CASE WHEN (a) = (5) THEN (a) + (3) ELSE CASE WHEN (a) = (2) THEN (a) - (4) ELSE a END END"
            )
        );
    }

    #[test]
    fn test_in_list() {
        // IN
        let ast_expr: ast::Expr = parse_expr("a in (3, 4, 5)").unwrap();
        println!("ast::expr = {ast_expr}");
        let expr = Expr::try_from(ast_expr.with(&Hierarchy::empty())).unwrap();
        println!("expr = {}", expr);
        for (x, t) in ast_expr.iter_with(DisplayVisitor) {
            println!("{x} ({t})");
        }
        let true_expr = Expr::in_list(Expr::col("a"), Expr::list([3, 4, 5]));
        assert_eq!(true_expr.to_string(), expr.to_string());
        assert_eq!(expr.to_string(), String::from("(a in (3, 4, 5))"));

        // NOT IN
        let ast_expr: ast::Expr = parse_expr("a not in (3, 4, 5)").unwrap();
        println!("ast::expr = {ast_expr}");
        let expr = Expr::try_from(ast_expr.with(&Hierarchy::empty())).unwrap();
        println!("expr = {}", expr);
        for (x, t) in ast_expr.iter_with(DisplayVisitor) {
            println!("{x} ({t})");
        }
        let true_expr = Expr::not(Expr::in_list(Expr::col("a"), Expr::list([3, 4, 5])));
        assert_eq!(true_expr.to_string(), expr.to_string());
        assert_eq!(expr.to_string(), String::from("(not (a in (3, 4, 5)))"));
    }

    #[test]
    fn test_coalesce() {
        let ast_expr: ast::Expr = parse_expr("coalesce(col1, col2, col3, 'default')").unwrap();
        println!("ast::expr = {ast_expr}");
        let expr = Expr::try_from(ast_expr.with(&Hierarchy::empty())).unwrap();
        println!("expr = {}", expr);
        for (x, t) in ast_expr.iter_with(DisplayVisitor) {
            println!("{x} ({t})");
        }
        let true_expr = Expr::coalesce(
            Expr::coalesce(
                Expr::coalesce(Expr::col("col1"), Expr::col("col2")),
                Expr::col("col3"),
            ),
            Expr::val("default".to_string()),
        );
        assert_eq!(true_expr.to_string(), expr.to_string());
        assert_eq!(
            expr.to_string(),
            String::from("coalesce(coalesce(coalesce(col1, col2), col3), default)")
        );
    }

    #[test]
    fn test_trim() {
        // TODO: TRIM(LEADING|TRAILING|BOTH FROM string) does not work in SQLParser

        // TRIM(LEADING 'a' FROM string)
        let ast_expr: ast::Expr = parse_expr("TRIM(LEADING 'a' FROM col1)").unwrap();
        println!("ast::expr = {ast_expr}");
        let expr = Expr::try_from(ast_expr.with(&Hierarchy::empty())).unwrap();
        println!("expr = {}", expr);
        for (x, t) in ast_expr.iter_with(DisplayVisitor) {
            println!("{x} ({t})");
        }
        let true_expr = Expr::ltrim(Expr::col("col1"), Expr::val("a".to_string()));
        assert_eq!(true_expr.to_string(), expr.to_string());
        assert_eq!(expr.to_string(), String::from("ltrim(col1, a)"));

        // LTRIM(string, "a")
        let ast_expr: ast::Expr = parse_expr("LTRIM(col1, 'a')").unwrap();
        println!("\nast::expr = {ast_expr}");
        let expr = Expr::try_from(ast_expr.with(&Hierarchy::empty())).unwrap();
        println!("expr = {}", expr);
        for (x, t) in ast_expr.iter_with(DisplayVisitor) {
            println!("{x} ({t})");
        }
        let true_expr = Expr::ltrim(Expr::col("col1"), Expr::val("a".to_string()));
        assert_eq!(true_expr.to_string(), expr.to_string());
        assert_eq!(expr.to_string(), String::from("ltrim(col1, a)"));

        // TRIM(TRAILING "a" FROM string)
        let ast_expr: ast::Expr = parse_expr("TRIM(TRAILING 'a' FROM col1)").unwrap();
        println!("\nast::expr = {ast_expr}");
        let expr = Expr::try_from(ast_expr.with(&Hierarchy::empty())).unwrap();
        println!("expr = {}", expr);
        for (x, t) in ast_expr.iter_with(DisplayVisitor) {
            println!("{x} ({t})");
        }
        let true_expr = Expr::rtrim(Expr::col("col1"), Expr::val("a".to_string()));
        assert_eq!(true_expr.to_string(), expr.to_string());
        assert_eq!(expr.to_string(), String::from("rtrim(col1, a)"));

        // RTRIM(string, "a")
        let ast_expr: ast::Expr = parse_expr("RTRIM(col1, 'a')").unwrap();
        println!("\nast::expr = {ast_expr}");
        let expr = Expr::try_from(ast_expr.with(&Hierarchy::empty())).unwrap();
        println!("expr = {}", expr);
        for (x, t) in ast_expr.iter_with(DisplayVisitor) {
            println!("{x} ({t})");
        }
        let true_expr = Expr::rtrim(Expr::col("col1"), Expr::val("a".to_string()));
        assert_eq!(true_expr.to_string(), expr.to_string());
        assert_eq!(expr.to_string(), String::from("rtrim(col1, a)"));

        // TRIM(BOTH "a" FROM string)
        let ast_expr: ast::Expr = parse_expr("TRIM(BOTH 'a' FROM col1)").unwrap();
        println!("\nast::expr = {ast_expr}");
        let expr = Expr::try_from(ast_expr.with(&Hierarchy::empty())).unwrap();
        println!("expr = {}", expr);
        for (x, t) in ast_expr.iter_with(DisplayVisitor) {
            println!("{x} ({t})");
        }
        let true_expr = Expr::ltrim(
            Expr::rtrim(Expr::col("col1"), Expr::val("a".to_string())),
            Expr::val("a".to_string()),
        );
        assert_eq!(true_expr.to_string(), expr.to_string());
        assert_eq!(expr.to_string(), String::from("ltrim(rtrim(col1, a), a)"));

        // BTRIM(string, "a")
        let ast_expr: ast::Expr = parse_expr("BTRIM(col1, 'a')").unwrap();
        println!("\nast::expr = {ast_expr}");
        let expr = Expr::try_from(ast_expr.with(&Hierarchy::empty())).unwrap();
        println!("expr = {}", expr);
        for (x, t) in ast_expr.iter_with(DisplayVisitor) {
            println!("{x} ({t})");
        }
        let true_expr = Expr::ltrim(
            Expr::rtrim(Expr::col("col1"), Expr::val("a".to_string())),
            Expr::val("a".to_string()),
        );
        assert_eq!(true_expr.to_string(), expr.to_string());
        assert_eq!(expr.to_string(), String::from("ltrim(rtrim(col1, a), a)"));

        // TRIM(string)
        let ast_expr: ast::Expr = parse_expr("TRIM(col1)").unwrap();
        println!("\nast::expr = {ast_expr}");
        let expr = Expr::try_from(ast_expr.with(&Hierarchy::empty())).unwrap();
        println!("expr = {}", expr);
        for (x, t) in ast_expr.iter_with(DisplayVisitor) {
            println!("{x} ({t})");
        }
        let true_expr = Expr::ltrim(
            Expr::rtrim(Expr::col("col1"), Expr::val(" ".to_string())),
            Expr::val(" ".to_string()),
        );
        assert_eq!(true_expr.to_string(), expr.to_string());
        assert_eq!(expr.to_string(), String::from("ltrim(rtrim(col1,  ),  )"));

        // TRIM("a" FROM string)
        let ast_expr: ast::Expr = parse_expr("TRIM('a' FROM col1)").unwrap();
        println!("\nast::expr = {ast_expr}");
        let expr = Expr::try_from(ast_expr.with(&Hierarchy::empty())).unwrap();
        println!("expr = {}", expr);
        for (x, t) in ast_expr.iter_with(DisplayVisitor) {
            println!("{x} ({t})");
        }
        let true_expr = Expr::ltrim(
            Expr::rtrim(Expr::col("col1"), Expr::val("a".to_string())),
            Expr::val("a".to_string()),
        );
        assert_eq!(true_expr.to_string(), expr.to_string());
        assert_eq!(expr.to_string(), String::from("ltrim(rtrim(col1, a), a)"));
    }
}
