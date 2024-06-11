//! # SQL parsing and conversion into Relation
//!
//! This module contains everything needed to parse an SQL query and build a Relation out of it
//!

pub mod expr;
pub mod query_names;
pub mod query_aliases;
pub mod reader;
pub mod relation;
pub mod visitor;
pub mod writer;

use crate::ast;

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
    use crate::{
        ast,
        builder::With,
        display::Dot,
        io::{postgresql, Database},
        relation::{Relation, Variant as _},
    };

    #[test]
    fn test_display() {
        let database = postgresql::test_database();
        println!("database {} = {}", database.name(), database.relations());
        for tab in database.tables() {
            println!("schema {} = {}", tab, tab.schema());
        }
        for query in [
            // "SELECT 1+count(y) as a, sum(1+x) as b FROM table_2",
            // "
            // WITH t1 AS (SELECT a,d FROM table_1 WHERE a>4),
            // t2 AS (SELECT * FROM table_2)
            // SELECT max(a), sum(d) FROM t1 INNER JOIN t2 ON t1.d = t2.x CROSS JOIN table_2 GROUP BY t2.y, t1.a",
            "
            WITH t1 AS (SELECT a, d FROM table_1),
            t2 AS (SELECT * FROM table_2)
            SELECT * FROM t1 INNER JOIN t2 ON t1.d = t2.x INNER JOIN table_2 ON t1.d=table_2.x ORDER BY t1.a LIMIT 10",
        ] {
            let relation = Relation::try_from(parse(query).unwrap().with(&database.relations())).unwrap();
            relation.display_dot().unwrap();
            let relation_query: &str = &ast::Query::from(&relation).to_string();
            println!("QUERY:\n{}", relation_query);
            let relation = Relation::try_from(parse(relation_query).unwrap().with(&database.relations())).unwrap();
            relation.display_dot().unwrap();
        }
    }

    #[test]
    fn test_queries() {
        let mut database = postgresql::test_database();

        for query in [
            "SELECT CAST(a AS text) FROM table_1",         // float => text
            "SELECT CAST(b AS text) FROM table_1",         // integer => text
            "SELECT CAST(c AS text) FROM table_1",         // date => text
            "SELECT CAST(z AS text) FROM table_2",         // text => text
            "SELECT CAST(x AS float) FROM table_2",        // integer => float
            "SELECT CAST('true' AS boolean) FROM table_2", // integer => float
            "SELECT CEIL(3 * b), FLOOR(3 * b), TRUNC(3 * b), ROUND(3 * b) FROM table_1",
            "SELECT SUM(DISTINCT a), SUM(a) FROM table_1",
        ] {
            let res1 = database.query(query).unwrap();
            let relation =
                Relation::try_from(parse(query).unwrap().with(&database.relations())).unwrap();
            let relation_query: &str = &ast::Query::from(&relation).to_string();
            println!("{query} => {relation_query}");
            let res2 = database.query(relation_query).unwrap();
            assert_eq!(res1, res2);
        }
    }

    #[test]
    fn test_parsing_complex_cte_query_with_column_aliases() {
        let database = postgresql::test_database();
        let query: &str = r#"
        WITH
        "map_3kfp" ("a", "d") AS (
            SELECT
            "a" AS "a",
            "d" AS "d"
            FROM
            "table_1"
        ),
        "map_l_fx" ("x", "y", "z") AS (
            SELECT
            "x" AS "x",
            "y" AS "y",
            "z" AS "z"
            FROM
            "table_2"
        ),
        "join_pgtf" (
            "field_yc23",
            "field_yep7",
            "field_dzgn",
            "field_pwch",
            "field_w_o_"
        ) AS (
            SELECT
            *
            FROM
            "map_3kfp" AS "_LEFT_"
            JOIN "map_l_fx" AS "_RIGHT_" ON ("_LEFT_"."d") = ("_RIGHT_"."x")
        ),
        "join_uvlf" (
            "field_l5nj",
            "field_bn3m",
            "field_m0fl",
            "field_v3e7",
            "field_z8oi",
            "field_dzgn",
            "field_pwch",
            "field_w_o_"
        ) AS (
            SELECT
            *
            FROM
            "join_pgtf" AS "_LEFT_"
            JOIN "table_2" AS "_RIGHT_" ON ("_LEFT_"."field_yep7") = ("_RIGHT_"."x")
        ),
        "map_hfqo" (
            "a",
            "d",
            "field_m0fl",
            "field_v3e7",
            "field_z8oi",
            "field_dzgn",
            "field_pwch",
            "field_w_o_"
        ) AS (
            SELECT
            "field_l5nj" AS "a",
            "field_bn3m" AS "d",
            "field_m0fl" AS "field_m0fl",
            "field_v3e7" AS "field_v3e7",
            "field_z8oi" AS "field_z8oi",
            "field_dzgn" AS "field_dzgn",
            "field_pwch" AS "field_pwch",
            "field_w_o_" AS "field_w_o_"
            FROM
            "join_uvlf"
        ),
        "map_m5ph" (
            "a",
            "d",
            "field_m0fl",
            "field_v3e7",
            "field_z8oi",
            "field_dzgn",
            "field_pwch",
            "field_w_o_"
        ) AS (
            SELECT
            "a" AS "a",
            "d" AS "d",
            "field_m0fl" AS "field_m0fl",
            "field_v3e7" AS "field_v3e7",
            "field_z8oi" AS "field_z8oi",
            "field_dzgn" AS "field_dzgn",
            "field_pwch" AS "field_pwch",
            "field_w_o_" AS "field_w_o_"
            FROM
            "map_hfqo"
            ORDER BY
            "a" ASC
            LIMIT
            10
        )
        SELECT
        *
        FROM
        "map_m5ph"
        LIMIT
        10
        "#;
        let query = parse(query).unwrap();
        println!("QUERY: {}", query);
        let binding = database.relations();
        let qwr = query.with(&binding);
        let relation = Relation::try_from(qwr).unwrap();
        relation.display_dot().unwrap();
    }


    #[test]
    fn test_parsing_simple_cte_query_with_column_aliases() {
        let database = postgresql::test_database();
        let query: &str = r#"
        WITH
        "tab1" ("my_new_name_a", "my_new_name_b") AS (
            SELECT * FROM "table_1"
        ),
        "tab2" ("my_new_new_name_a") AS (
            SELECT "my_new_name_a" FROM "tab1"
        )
        SELECT "my_new_new_name_a" FROM "tab2"
        "#;
        let query = parse(query).unwrap();
        println!("QUERY: {}", query);
        let binding = database.relations();
        let qwr = query.with(&binding);
        let relation = Relation::try_from(qwr).unwrap();
        relation.display_dot().unwrap();
    }
}
