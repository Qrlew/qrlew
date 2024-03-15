//! # SQL parsing and conversion into Relation
//!
//! This module contains everything needed to parse an SQL query and build a Relation out of it
//!

pub mod expr;
pub mod query_names;
pub mod reader;
pub mod relation;
pub mod visitor;
pub mod writer;

use crate::{ast, relation::Variant as _};

// I would put here the abstact AST Visitor.
// Then in expr.rs module we write an implementation of the abstract visitor for Qrlew expr

pub trait Visitor<'a, T> {
    fn identifier(&self, identifier: &'a ast::Ident) -> T;
    fn compound_identifier(&self, qident: &'a Vec<ast::Ident>) -> T;
    fn unary_op(&self, op: &'a ast::UnaryOperator, expr: &'a Box<ast::Expr>) -> T;
    fn binary_op(
        &self,
        left: &'a Box<ast::Expr>,
        op: &'a ast::BinaryOperator,
        right: &'a Box<ast::Expr>,
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
    #[cfg(feature = "sqlite")]
    use crate::io::sqlite;
    use crate::{
        ast,
        builder::With,
        display::Dot,
        io::{postgresql, Database},
        relation::Relation, DataType,
    };
    use colored::Colorize;
    use itertools::Itertools;
    use sqlparser::dialect::BigQueryDialect;

    #[test]
    fn test_display_test() {
        let database = postgresql::test_database();
        let relations = database.relations();
        let query = r#"
        with aa as (SELECT x AS ahah FROM table2 t1 JOIN table2 t1 USING (x))
        SELECT * FROM aa ORDER BY ahah
        "#;
        let relation = Relation::try_from(parse(query).unwrap().with(&relations)).unwrap();
        relation.display_dot().unwrap();
        let relation_query: &str = &ast::Query::from(&relation).to_string();
        println!("{}",relation_query);
    }

    #[test]
    fn test_display() {
        let database = postgresql::test_database();
        println!("database {} = {}", database.name(), database.relations());
        for tab in database.tables() {
            println!("schema {} = {}", tab, tab.schema());
        }
        for query in [
            "SELECT 1+count(y) as a, sum(1+x) as b FROM table_2",
            "
            WITH t1 AS (SELECT a,d FROM table_1 WHERE a>4),
            t2 AS (SELECT * FROM table_2)
            SELECT max(a), sum(d) FROM t1 INNER JOIN t2 ON t1.d = t2.x CROSS JOIN table_2 GROUP BY t2.y, t1.a",
            "
            WITH t1 AS (SELECT a, d FROM table_1),
            t2 AS (SELECT * FROM table_2)
            SELECT * FROM t1 INNER JOIN t2 ON t1.d = t2.x INNER JOIN table_2 ON t1.d=table_2.x ORDER BY t1.a LIMIT 10",
        ] {
            let relation = Relation::try_from(parse(query).unwrap().with(&database.relations())).unwrap();
            relation.display_dot().unwrap();
        }
    }

    #[test]
    fn test_queries() {
        let mut database = postgresql::test_database();

        for query in [
            "SELECT CAST(a AS text) FROM table_1", // float => text
            "SELECT CAST(b AS text) FROM table_1", // integer => text
            "SELECT CAST(c AS text) FROM table_1", // date => text
            "SELECT CAST(z AS text) FROM table_2", // text => text
            "SELECT CAST(x AS float) FROM table_2", // integer => float
            "SELECT CAST('true' AS boolean) FROM table_2", // integer => float
            "SELECT CEIL(3 * b), FLOOR(3 * b), TRUNC(3 * b), ROUND(3 * b) FROM table_1",
            "SELECT SUM(DISTINCT a), SUM(a) FROM table_1"
        ] {
            let res1 = database.query(query).unwrap();
            let relation = Relation::try_from(parse(query).unwrap().with(&database.relations())).unwrap();
            let relation_query: &str = &ast::Query::from(&relation).to_string();
            println!("{query} => {relation_query}");
            let res2 = database.query(relation_query).unwrap();
            assert_eq!(res1, res2);
        }
    }
}
