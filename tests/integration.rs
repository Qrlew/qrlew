//! # Integration tests
//!
//! Various queries are tested against their version rewriten to Relation + re-rewriten.

use colored::Colorize;
use itertools::Itertools;
#[cfg(feature = "sqlite")]
use qrlew::io::sqlite;
#[cfg(feature = "mssql")]
use qrlew::io::mssql;
use qrlew::{
    ast,
    expr,
    io::{postgresql, Database},
    relation::Variant as _,
    sql::parse,
    Relation, With, dialect_translation::{RelationToQueryTranslator, RelationWithTranslator, postgres::PostgresTranslator},
};

pub fn test_eq<D: Database>(database: &mut D, query1: &str, query2: &str) -> bool {
    println!(
        "{}\n{}",
        format!("{query1}").red(),
        database
            .query(query1)
            .unwrap()
            .iter()
            .map(ToString::to_string)
            .join("\n")
    );
    println!("{}", format!("{query2}").yellow());
    println!(
        "{}",
        database
            .query(query2)
            .unwrap()
            .iter()
            .map(ToString::to_string)
            .join("\n")
    );
    database.eq(query1, query2)
}

pub fn test_rewritten_eq<D: Database>(database: &mut D, query: &str) -> bool {
    let relations = database.relations();
    let relation = Relation::try_from(parse(query).unwrap().with(&relations)).unwrap();
    let rewriten_query: &str = &ast::Query::from(&relation).to_string();
    // relation.display_dot().unwrap();
    test_eq(database, query, rewriten_query)
}

pub fn test_execute<D: Database, T: RelationToQueryTranslator>(database: &mut D, query: &str, translator: T) {
    let relations = database.relations();
    let relation = Relation::try_from(parse(query).unwrap().with(&relations)).unwrap();
    let relaiton_with_tranlator = RelationWithTranslator(&relation, translator);
    let rewriten_query: &str = &ast::Query::from(relaiton_with_tranlator).to_string();
    println!("Original query: \n{}\n", query);
    println!("Translated query: \n{}\n", rewriten_query);
    println!(
        "{}",
        database
            .query(rewriten_query)
            .unwrap()
            .iter()
            .map(ToString::to_string)
            .join("\n")
    );
    println!("===== Done =====");
    // TODO test rebuild relation from rewriten_query
}

const QUERIES: &[&str] = &[
    "SELECT AVG(x) as a FROM table_2",
    "SELECT 1+count(y) as a, sum(1+x) as b FROM table_2",
    "SELECT 1+SUM(a), count(b) FROM table_1",
    // Some WHERE
    "SELECT 1+SUM(a), count(b) FROM table_1 WHERE a>4",
    "SELECT SUM(a), count(b) FROM table_1 WHERE a>4",
    // Some GROUP BY
    "SELECT 1+SUM(a), count(b) FROM table_1 GROUP BY d",
    "SELECT count(b) FROM table_1 GROUP BY CEIL(d)",
    "SELECT CEIL(d) AS d_ceiled, count(b) FROM table_1 GROUP BY CEIL(d)",
    "SELECT CEIL(d) AS d_ceiled, count(b) FROM table_1 GROUP BY d_ceiled",
    // Some WHERE and GROUP BY
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
    // Test LIMIT // OFFSET
    "WITH t1 AS (SELECT a,d FROM table_1),
    t2 AS (SELECT * FROM table_2)
    SELECT * FROM t1 INNER JOIN t2 ON t1.d = t2.x INNER JOIN table_2 ON t1.d=table_2.x ORDER BY t1.a, t2.x, t2.y, t2.z LIMIT 17",
    "WITH t1 AS (SELECT a,d FROM table_1),
    t2 AS (SELECT * FROM table_2)
    SELECT * FROM t1 INNER JOIN t2 ON t1.d = t2.x INNER JOIN table_2 ON t1.d=table_2.x ORDER BY t1.a, t2.x, t2.y, t2.z OFFSET 5",
    "WITH t1 AS (SELECT a,d FROM table_1),
    t2 AS (SELECT * FROM table_2)
    SELECT * FROM t1 INNER JOIN t2 ON t1.d = t2.x INNER JOIN table_2 ON t1.d=table_2.x ORDER BY t1.a, t2.x, t2.y, t2.z LIMIT 17 OFFSET 5",

    "SELECT CASE a WHEN 5 THEN 0 ELSE a END FROM table_1",
    "SELECT CASE WHEN a < 5 THEN 0 WHEN a < 3 THEN 3 ELSE a END FROM table_1",
    "SELECT CASE WHEN a < 5 THEN 0 WHEN a < 3 THEN 3 END FROM table_1",
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
    "SELECT * FROM order_table LEFT JOIN item_table on id=order_id WHERE price>10",
    // Some string functions
    "SELECT UPPER(z) FROM table_2 LIMIT 5",
    "SELECT LOWER(z) FROM table_2 LIMIT 5",
    // ORDER BY
    "SELECT d, COUNT(*) AS my_count FROM table_1 GROUP BY d ORDER BY d",
    "SELECT d, COUNT(*) AS my_count FROM table_1 GROUP BY d ORDER BY d DESC",
    "SELECT d, COUNT(*) AS my_count FROM table_1 GROUP BY d ORDER BY my_count",
    "SELECT d, COUNT(*) AS my_count FROM table_1 GROUP BY d ORDER BY my_count",
    // DISTINCT
    "SELECT DISTINCT COUNT(*) FROM table_1 GROUP BY d",
    "SELECT DISTINCT c, d FROM table_1",
    "SELECT c, COUNT(DISTINCT d) AS count_d, SUM(DISTINCT d) AS sum_d FROM table_1 GROUP BY c ORDER BY c",
    "SELECT SUM(DISTINCT a) AS s1 FROM table_1 GROUP BY c HAVING COUNT(*) > 5;",
    // using joins
    "WITH t1 AS (SELECT a, b, c FROM table_1 WHERE a > 5), t2 AS (SELECT a, d, c FROM table_1 WHERE a < 7) SELECT * FROM t1 INNER JOIN t2 USING(a)",
    "WITH t1 AS (SELECT a, b, c FROM table_1 WHERE a > 5), t2 AS (SELECT a, d, c FROM table_1 WHERE a < 7) SELECT * FROM t1 LEFT JOIN t2 USING(a)",
    "WITH t1 AS (SELECT a, b, c FROM table_1 WHERE a > 5), t2 AS (SELECT a, d, c FROM table_1 WHERE a < 7) SELECT * FROM t1 RIGHT JOIN t2 USING(a)",
    "WITH t1 AS (SELECT a, b, c FROM table_1 WHERE a > 5), t2 AS (SELECT a, d, c FROM table_1 WHERE a < 7) SELECT * FROM t1 FULL JOIN t2 USING(a)",
    // natural joins
    "WITH t1 AS (SELECT a, b, c FROM table_1 WHERE a > 5), t2 AS (SELECT a, d, c FROM table_1 WHERE a < 7 LIMIT 10) SELECT * FROM t1 NATURAL INNER JOIN t2",
    "WITH t1 AS (SELECT a, b, c FROM table_1 WHERE a > 5), t2 AS (SELECT a, d, c FROM table_1 WHERE a < 7 LIMIT 10) SELECT * FROM t1 NATURAL LEFT JOIN t2",
    "WITH t1 AS (SELECT a, b, c FROM table_1 WHERE a > 5), t2 AS (SELECT a, d, c FROM table_1 WHERE a < 7 LIMIT 10) SELECT * FROM t1 NATURAL RIGHT JOIN t2",
    "WITH t1 AS (SELECT a, b, c FROM table_1 WHERE a > 5), t2 AS (SELECT a, d, c FROM table_1 WHERE a < 7 LIMIT 10) SELECT * FROM t1 NATURAL FULL JOIN t2",
    "SELECT a, SUM(a) FROM table_1 GROUP BY a"
];

#[cfg(feature = "sqlite")]
const SQLITE_QUERIES: &[&str] = &["SELECT AVG(b) as n, count(b) as d FROM table_1"];

#[cfg(feature = "sqlite")]
#[test]
fn test_on_sqlite() {

    let mut database = sqlite::test_database();
    println!("database {} = {}", database.name(), database.relations());
    for tab in database.tables() {
        println!("schema {} = {}", tab, tab.schema());
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
    // Some joins
    "SELECT CHAR_LENGTH(z) AS char_length FROM table_2 LIMIT 1",
    "SELECT POSITION('o' IN z) AS char_length FROM table_2 LIMIT 5",
];

#[test]
fn test_on_postgresql() {
    let mut database = postgresql::test_database();
    println!("database {} = {}", database.name(), database.relations());
    for tab in database.tables() {
        println!("schema {} = {}", tab, tab.schema());
    }
    for &query in QUERIES.iter().chain(POSTGRESQL_QUERIES) {
        assert!(test_rewritten_eq(&mut database, query));
    }
}

#[cfg(feature = "mssql")]
const QUERIES_FOR_MSSQL: &[&str] = &[
    "SELECT AVG(x) as a FROM table_2",
    "SELECT 1+count(y) as a, sum(1+x) as b FROM table_2",
    "SELECT 1+SUM(a), count(b) FROM table_1",
    // Some WHERE
    "SELECT 1+SUM(a), count(b) FROM table_1 WHERE a>4",
    "SELECT SUM(a), count(b) FROM table_1 WHERE a>4",
    // Some GROUP BY
    "SELECT 1+SUM(a), count(b) FROM table_1 GROUP BY d",
    "SELECT count(b) FROM table_1 GROUP BY CEIL(d)",
    "SELECT CEIL(d) AS d_ceiled, count(b) FROM table_1 GROUP BY CEIL(d)",
    // "SELECT CEIL(d) AS d_ceiled, count(b) FROM table_1 GROUP BY d_ceiled",
    // Some WHERE and GROUP BY
    "SELECT 1+SUM(a), count(b) FROM table_1 WHERE d>4 GROUP BY d",
    "SELECT 1+SUM(a), count(b), d FROM table_1 GROUP BY d",
    "SELECT sum(a) FROM table_1 JOIN table_2 ON table_1.d = table_2.x",
    "WITH t1 AS (SELECT a,d FROM table_1),
    t2 AS (SELECT * FROM table_2)
    SELECT sum(a) FROM t1 JOIN t2 ON t1.d = t2.x",
    "WITH t1 AS (SELECT a,d FROM table_1 WHERE a>4),
    t2 AS (SELECT * FROM table_2)
    SELECT max(a), sum(d) FROM t1 INNER JOIN t2 ON t1.d = t2.x CROSS JOIN table_2",
    // The ORDER BY clause is invalid in views, inline functions, derived tables, subqueries, and common table expressions, unless TOP, OFFSET or FOR XML is also specified.
    // "WITH t1 AS (SELECT a,d FROM table_1),
    // t2 AS (SELECT * FROM table_2)
    // SELECT * FROM t1 INNER JOIN t2 ON t1.d = t2.x INNER JOIN table_2 ON t1.d=table_2.x ORDER BY t1.a, t2.x, t2.y, t2.z",
    // Test LIMIT
    // Test LIMIT
    "WITH t1 AS (SELECT a,d FROM table_1),
    t2 AS (SELECT * FROM table_2)
    SELECT * FROM t1 INNER JOIN t2 ON t1.d = t2.x INNER JOIN table_2 ON t1.d=table_2.x ORDER BY t1.a, t2.x, t2.y, t2.z LIMIT 17",
    "SELECT CASE a WHEN 5 THEN 0 ELSE a END FROM table_1",
    "SELECT CASE WHEN a < 5 THEN 0 WHEN a < 3 THEN 3 ELSE a END FROM table_1",
    "SELECT CASE WHEN a < 5 THEN 0 WHEN a < 3 THEN 3 END FROM table_1",
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
    "SELECT * FROM order_table LEFT JOIN item_table on id=order_id WHERE price>10",
    // Some string functions
    //"SELECT UPPER(z) FROM table_2 LIMIT 5",
    //"SELECT LOWER(z) FROM table_2 LIMIT 5",
    // ORDER BY
    // The ORDER BY clause is invalid in views, inline functions, derived tables, subqueries, and common table expressions, unless TOP, OFFSET or FOR XML is also specified.
    //"SELECT d, COUNT(*) AS my_count FROM table_1 GROUP BY d ORDER BY d",
    //"SELECT d, COUNT(*) AS my_count FROM table_1 GROUP BY d ORDER BY d DESC",
    //"SELECT d, COUNT(*) AS my_count FROM table_1 GROUP BY d ORDER BY my_count",
    //"SELECT d, COUNT(*) AS my_count FROM table_1 GROUP BY d ORDER BY my_count",
    // DISTINCT
    // Some bug somewhere. Error not informative: panicked at src/expr/mod.rs:1029:18: Option::unwrap()` on a `None` value
    //"SELECT DISTINCT COUNT(*) FROM table_1 GROUP BY d", 
    "SELECT DISTINCT c, d FROM table_1",
    "SELECT c, COUNT(DISTINCT d) AS count_d, SUM(DISTINCT d) AS sum_d FROM table_1 GROUP BY c ORDER BY c",
    "SELECT SUM(DISTINCT a) AS s1 FROM table_1 GROUP BY c HAVING COUNT(*) > 5;",
    // using joins
    "WITH t1 AS (SELECT a, b, c FROM table_1 WHERE a > 5), t2 AS (SELECT a, d, c FROM table_1 WHERE a < 7) SELECT * FROM t1 INNER JOIN t2 USING(a)",
    "WITH t1 AS (SELECT a, b, c FROM table_1 WHERE a > 5), t2 AS (SELECT a, d, c FROM table_1 WHERE a < 7) SELECT * FROM t1 LEFT JOIN t2 USING(a)",
    "WITH t1 AS (SELECT a, b, c FROM table_1 WHERE a > 5), t2 AS (SELECT a, d, c FROM table_1 WHERE a < 7) SELECT * FROM t1 RIGHT JOIN t2 USING(a)",
    "WITH t1 AS (SELECT a, b, c FROM table_1 WHERE a > 5), t2 AS (SELECT a, d, c FROM table_1 WHERE a < 7) SELECT * FROM t1 FULL JOIN t2 USING(a)",
    // natural joins
    "WITH t1 AS (SELECT a, b, c FROM table_1 WHERE a > 5), t2 AS (SELECT a, d, c FROM table_1 WHERE a < 7 LIMIT 10) SELECT * FROM t1 NATURAL INNER JOIN t2",
    "WITH t1 AS (SELECT a, b, c FROM table_1 WHERE a > 5), t2 AS (SELECT a, d, c FROM table_1 WHERE a < 7 LIMIT 10) SELECT * FROM t1 NATURAL LEFT JOIN t2",
    "WITH t1 AS (SELECT a, b, c FROM table_1 WHERE a > 5), t2 AS (SELECT a, d, c FROM table_1 WHERE a < 7 LIMIT 10) SELECT * FROM t1 NATURAL RIGHT JOIN t2",
    "WITH t1 AS (SELECT a, b, c FROM table_1 WHERE a > 5), t2 AS (SELECT a, d, c FROM table_1 WHERE a < 7 LIMIT 10) SELECT * FROM t1 NATURAL FULL JOIN t2",
];


#[cfg(feature = "mssql")]
#[test]
fn test_on_mssql() {
    // In this test we construct relations from QUERIES and we execute
    // the translated queries
    use qrlew::dialect_translation::mssql::MSSQLTranslator;

    let mut database = mssql::test_database();
    println!("database {} = {}", database.name(), database.relations());
    for tab in database.tables() {
        println!("schema {} = {}", tab, tab.schema());
    }
    for &query in QUERIES_FOR_MSSQL.iter() {
        test_execute(&mut database, query, MSSQLTranslator);
    }
}

#[test]
fn test_distinct_aggregates() {
    let mut database = postgresql::test_database();
    let table = database
        .relations()
        .get(&["table_1".to_string()])
        .unwrap()
        .as_ref()
        .clone();

    let true_query = "SELECT COUNT(DISTINCT d) AS count_d, SUM(DISTINCT d) AS sum_d FROM table_1";
    let column = "d";
    let group_by = vec![];
    let aggregates = vec![
        ("count_d", expr::aggregate::Aggregate::Count),
        ("sum_d", expr::aggregate::Aggregate::Sum),
    ];
    let distinct_rel = table
        .clone()
        .distinct_aggregates(column, group_by, aggregates);
    let rewriten_query: &str = &ast::Query::from(&distinct_rel).to_string();
    assert!(test_eq(&mut database, true_query, rewriten_query));

    let true_query = "SELECT c, COUNT(DISTINCT d) AS count_d, SUM(DISTINCT d) AS sum_d FROM table_1 GROUP BY c ORDER BY c";
    let column = "d";
    let group_by = vec!["c"];
    let aggregates = vec![
        ("count_d", expr::aggregate::Aggregate::Count),
        ("sum_d", expr::aggregate::Aggregate::Sum),
    ];
    let distinct_rel = table.distinct_aggregates(column, group_by, aggregates);
    let rewriten_query: &str = &ast::Query::from(&distinct_rel).to_string();
    assert!(test_eq(&mut database, true_query, rewriten_query));
}
