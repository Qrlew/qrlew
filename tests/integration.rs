use itertools::Itertools;
use colored::Colorize;
use sqlparser::ast;
use qrlew::{
    Relation, With,
    relation::display,
    sql::parse,
    io::{Database, postgresql},
};
#[cfg(feature = "sqlite")]
use qrlew::io::sqlite;

pub fn test_rewritten_eq<D: Database>(database: &mut D, query: &str) -> bool {
    let relations = database.relations();
    let relation = Relation::try_from(parse(query).unwrap().with(&relations)).unwrap();
    let rewriten_query: &str = &ast::Query::from(&relation).to_string();
    // DEBUG
    // display(&relation);
    // Displaying the test for DEBUG purpose
    println!(
        "{}\n{}",
        format!("{query}").red(),
        database
            .query(query)
            .unwrap()
            .iter()
            .map(ToString::to_string)
            .join("\n")
    );
    println!(
        "{}\n{}",
        format!("{rewriten_query}").yellow(),
        database
            .query(rewriten_query)
            .unwrap()
            .iter()
            .map(ToString::to_string)
            .join("\n")
    );
    database.eq(query, rewriten_query)
}

const QUERIES: &[&str] = &[
    "SELECT AVG(x) as a FROM table_2",
    "SELECT 1+count(y) as a, sum(1+x) as b FROM table_2",
    "SELECT 1+SUM(a), count(b) FROM table_1",
    "SELECT 1+SUM(a), count(b) FROM table_1 WHERE a>4",
    "SELECT 1+SUM(a), count(b) FROM table_1 GROUP BY d",
    "SELECT 1+SUM(a), count(b) FROM table_1 WHERE d>4 GROUP BY d",
    "SELECT 1+SUM(a), count(b), d FROM table_1 GROUP BY d",
    "SELECT sum(a) FROM table_1 JOIN table_2 ON table_1.d = table_2.x",
    
    "WITH t1 AS (SELECT a,d FROM table_1),
    t2 AS (SELECT * FROM table_2)
    SELECT sum(a) FROM t1 JOIN t2 ON t1.d = t2.x",
    
    "WITH t1 AS (SELECT a,d FROM table_1 WHERE a>4),
    t2 AS (SELECT * FROM table_2)
    SELECT max(a), sum(d) FROM t1 INNER JOIN t2 ON t1.d = t2.x CROSS JOIN table_2",
    
    "WITH t1 AS (SELECT a,d FROM table_1),
    t2 AS (SELECT * FROM table_2)
    SELECT * FROM t1 INNER JOIN t2 ON t1.d = t2.x INNER JOIN table_2 ON t1.d=table_2.x ORDER BY t1.a, t2.x, t2.y, t2.z",
    // Test LIMIT
    "WITH t1 AS (SELECT a,d FROM table_1),
    t2 AS (SELECT * FROM table_2)
    SELECT * FROM t1 INNER JOIN t2 ON t1.d = t2.x INNER JOIN table_2 ON t1.d=table_2.x ORDER BY t1.a, t2.x, t2.y, t2.z LIMIT 17",
    // Test UNION
    "SELECT 1*a FROM table_1 UNION SELECT 1*x FROM table_2",
    // Test no UNION with CTEs
    "WITH t1 AS (SELECT a,d FROM table_1),
    t2 AS (SELECT x,y FROM table_2)
    SELECT * FROM t1",
    // Test UNION with CTEs
    "WITH t1 AS (SELECT 1*a, 1*d FROM table_1),
    t2 AS (SELECT 0.1*x as a, 2*x as b FROM table_2)
    SELECT * FROM t1 UNION SELECT * FROM t2",
    // Some joins
    "SELECT * FROM primary_table LEFT JOIN secondary_table on id=primary_id WHERE amount>0"
];

#[cfg(feature = "sqlite")]
const SQLITE_QUERIES: &[&str] = &["SELECT AVG(b) as n, count(b) as d FROM table_1"];

#[cfg(feature = "sqlite")]
#[test]
fn test_on_sqlite() {
    let mut database = sqlite::test_database();
    println!("database {} = {}", database.name(), database.relations());
    for tab in database.tables() {
        println!("schema {} = {}", tab, tab.schema);
    }
    for &query in SQLITE_QUERIES.iter().chain(QUERIES) {
        assert!(test_rewritten_eq(&mut database, query));
    }
}
// This should work: https://www.db-fiddle.com/f/ouKSHjkEk29zWY5PN2YmjZ/10

const POSTGRESQL_QUERIES: &[&str] = &[
    "SELECT AVG(b) as n, count(b) as d FROM table_1",
    // Test MD5
    "SELECT MD5(z) FROM table_2 LIMIT 10",
    "SELECT CONCAT(x,y,z) FROM table_2 LIMIT 11",
];

#[test]
fn test_on_postgresql() {
    let mut database = postgresql::test_database();
    println!("database {} = {}", database.name(), database.relations());
    for tab in database.tables() {
        println!("schema {} = {}", tab, tab.schema);
    }
    for &query in POSTGRESQL_QUERIES.iter().chain(QUERIES) {
        assert!(test_rewritten_eq(&mut database, query));
    }
}