//! This module contains everything needed to parse an SQL query and build a Relation out of it
//!

pub mod expr;
pub mod query_names;
pub mod reader;
pub mod relation;
pub mod visitor;
pub mod writer;

use sqlparser::ast as sql;

// I would put here the abstact AST Visitor.
// Then in expr.rs module we write an implementation of the abstract visitor for Qrlew expr

pub trait Visitor<'a, T> {
    fn identifier(&self, identifier: &'a sql::Ident) -> T;
    fn compound_identifier(&self, qident: &'a Vec<sql::Ident>) -> T;
    fn unary_op(&self, op: &'a sql::UnaryOperator, expr: &'a Box<sql::Expr>) -> T;
    fn binary_op(
        &self,
        left: &'a Box<sql::Expr>,
        op: &'a sql::BinaryOperator,
        right: &'a Box<sql::Expr>,
    ) -> T;
}

use std::{
    convert::Infallible,
    error, fmt,
    num::{ParseFloatError, ParseIntError},
    result,
};

use sqlparser::parser::ParserError;
use sqlparser::tokenizer::TokenizerError;

// Error management

#[derive(Debug, Clone)]
pub enum Error {
    ParsingError(String),
    Other(String),
}

impl Error {
    pub fn parsing_error(input: impl fmt::Display) -> Error {
        Error::ParsingError(format!("Cannot parse {}", input))
    }
    pub fn other<T: fmt::Display>(desc: T) -> Error {
        Error::Other(desc.to_string())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::ParsingError(input) => writeln!(f, "ParsingError: {}", input),
            Error::Other(err) => writeln!(f, "{}", err),
        }
    }
}

impl error::Error for Error {}

impl From<Infallible> for Error {
    fn from(err: Infallible) -> Self {
        Error::Other(err.to_string())
    }
}
impl From<TokenizerError> for Error {
    fn from(err: TokenizerError) -> Self {
        Error::Other(err.to_string())
    }
}
impl From<ParserError> for Error {
    fn from(err: ParserError) -> Self {
        Error::Other(err.to_string())
    }
}
impl From<ParseIntError> for Error {
    fn from(err: ParseIntError) -> Self {
        Error::Other(err.to_string())
    }
}
impl From<ParseFloatError> for Error {
    fn from(err: ParseFloatError) -> Self {
        Error::Other(err.to_string())
    }
}
impl From<crate::relation::Error> for Error {
    fn from(err: crate::relation::Error) -> Self {
        Error::Other(err.to_string())
    }
}

pub type Result<T> = result::Result<T, Error>;

// Import a few functions
pub use expr::{parse_expr, parse_expr_with_dialect};
pub use relation::{parse, parse_with_dialect};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        builder::With,
        io::{postgresql, Database},
        relation::{display, Relation},
    };
    use colored::Colorize;
    use itertools::Itertools;
    use sqlparser::ast;
    #[cfg(feature = "sqlite")]
    use crate::io::sqlite;
    
    #[ignore]
    #[test]
    fn test_display() {
        let database = postgresql::test_database();
        println!("database {} = {}", database.name(), database.relations());
        for tab in database.tables() {
            println!("schema {} = {}", tab, tab.schema);
        }
        for query in [
            "SELECT 1+count(y) as a, sum(1+x) as b FROM table_2",
            "
            WITH t1 AS (SELECT a,d FROM table_1 WHERE a>4),
            t2 AS (SELECT * FROM table_2)
            SELECT max(a), sum(d) FROM t1 INNER JOIN t2 ON t1.d = t2.x CROSS JOIN table_2 GROUP BY t2.y, t1.a",
            "
            WITH t1 AS (SELECT a,d FROM table_1),
            t2 AS (SELECT * FROM table_2)
            SELECT * FROM t1 INNER JOIN t2 ON t1.d = t2.x INNER JOIN table_2 ON t1.d=table_2.x ORDER BY t1.a LIMIT 10",
        ] {
            let relation = Relation::try_from(parse(query).unwrap().with(&database.relations())).unwrap();
            display(&relation);
        }
    }
}
