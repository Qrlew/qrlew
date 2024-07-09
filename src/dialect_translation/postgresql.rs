use super::{function_builder, QueryToRelationTranslator, RelationToQueryTranslator};
use sqlparser::{ast, dialect::PostgreSqlDialect};

#[derive(Clone, Copy)]
pub struct PostgreSqlTranslator;

impl RelationToQueryTranslator for PostgreSqlTranslator {
    fn first(&self, expr: ast::Expr) -> ast::Expr {
        expr
    }

    fn mean(&self, expr: ast::Expr) -> ast::Expr {
        function_builder("AVG", vec![expr], false)
    }

    fn var(&self, expr: ast::Expr) -> ast::Expr {
        function_builder("VARIANCE", vec![expr], false)
    }

    fn std(&self, expr: ast::Expr) -> ast::Expr {
        function_builder("STDDEV", vec![expr], false)
    }

    fn trunc(&self, exprs: Vec<ast::Expr>) -> ast::Expr {
        // TRUNC in postgres has a problem:
        // In TRUNC(double_precision_number, precision) if precision is specified it fails
        // If it is not specified it passes considering precision = 0.
        // SELECT TRUNC(CAST (0.12 AS DOUBLE PRECISION), 0) fails
        // SELECT TRUNC(CAST (0.12 AS DOUBLE PRECISION)) passes.
        // Here we check precision, if it is 0 we remove it (such that the precision is implicit).
        let func_args_list = ast::FunctionArgumentList {
            duplicate_treatment: None,
            args: exprs
                .into_iter()
                .filter_map(|e| {
                    (e != ast::Expr::Value(ast::Value::Number("0".to_string(), false)))
                        .then_some(ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(e)))
                })
                .collect(),
            clauses: vec![],
        };
        ast::Expr::Function(ast::Function {
            name: ast::ObjectName(vec![ast::Ident::from("TRUNC")]),
            args: ast::FunctionArguments::List(func_args_list),
            over: None,
            filter: None,
            null_treatment: None,
            within_group: vec![],
        })
    }

    fn round(&self, exprs: Vec<ast::Expr>) -> ast::Expr {
        // Same as TRUNC
        // what if I wanted to do round(0, 0)
        let func_args_list = ast::FunctionArgumentList {
            duplicate_treatment: None,
            args: exprs
                .into_iter()
                .filter_map(|e| {
                    (e != ast::Expr::Value(ast::Value::Number("0".to_string(), false)))
                        .then_some(ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(e)))
                })
                .collect(),
            clauses: vec![],
        };
        ast::Expr::Function(ast::Function {
            name: ast::ObjectName(vec![ast::Ident::from("ROUND")]),
            args: ast::FunctionArguments::List(func_args_list),
            over: None,
            filter: None,
            null_treatment: None,
            within_group: vec![],
        })
    }
}

impl QueryToRelationTranslator for PostgreSqlTranslator {
    type D = PostgreSqlDialect;

    fn dialect(&self) -> Self::D {
        PostgreSqlDialect {}
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::sql::Result;
    use crate::{
        builder::Ready,
        data_type::DataType,
        dialect_translation::RelationWithTranslator,
        hierarchy::Hierarchy,
        io::{postgresql, Database as _},
        relation::{schema::Schema, Relation},
        sql::{parse_with_dialect, relation::QueryWithRelations},
    };
    use std::sync::Arc;
    use std::fs;

    // fn assert_same_query_str(query_1: &str, query_2: &str) {
    //     let a_no_whitespace: String = query_1.chars().filter(|c| !c.is_whitespace()).collect();
    //     let b_no_whitespace: String = query_2.chars().filter(|c| !c.is_whitespace()).collect();
    //     assert_eq!(a_no_whitespace, b_no_whitespace);
    // }

    #[test]
    fn test_query() -> Result<()> {
        let translator = PostgreSqlTranslator;
        let query_str = "SELECT POSITION('o' IN z) AS col FROM table_2";
        let query = parse_with_dialect(query_str, translator.dialect())?;
        println!("{:?}", query);
        Ok(())
    }

    #[test]
    fn test_map() -> Result<()> {
        let schema: Schema = vec![
            ("a", DataType::float()),
            ("b", DataType::float_interval(-2., 2.)),
            ("c", DataType::float()),
            ("d", DataType::float_interval(0., 1.)),
        ]
        .into_iter()
        .collect();
        let table = Relation::table()
            .name("tab")
            .schema(schema.clone())
            .size(100)
            .build();
        let relations = Hierarchy::from([(["schema", "table"], Arc::new(table))]);

        let query_str = "SELECT log(table.d + 1) FROM schema.table";
        let translator = PostgreSqlTranslator;
        let query = parse_with_dialect(query_str, translator.dialect())?;
        let query_with_relation = QueryWithRelations::new(&query, &relations);
        let relation = Relation::try_from((query_with_relation, translator))?;

        // let retranslated: ast::Query::from()
        print!("{}", relation);
        Ok(())
    }

    #[test]
    fn test_table_special() -> Result<()> {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        let query_str =
            r#"SELECT "Id", NORMAL_COL, "Na.Me" FROM "MY SPECIAL TABLE" ORDER BY "Id" "#;
        let translator = PostgreSqlTranslator;
        let query = parse_with_dialect(query_str, translator.dialect())?;
        let query_with_relation = QueryWithRelations::new(&query, &relations);
        let relation = Relation::try_from((query_with_relation, translator))?;
        println!("\n {} \n", relation);
        let rel_with_traslator = RelationWithTranslator(&relation, translator);
        let translated = ast::Query::from(rel_with_traslator);
        print!("{}", translated);
        _ = database
            .query(translated.to_string().as_str())
            .unwrap()
            .iter()
            .map(ToString::to_string);
        Ok(())
    }

    #[test]
    fn test_relation_to_query_with_null_field() -> Result<()> {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        let query_str = r#"SELECT CASE WHEN (1) > (2) THEN 1 ELSE NULL END AS "_PRIVACY_UNIT_", a AS a FROM table_1"#;
        let translator = PostgreSqlTranslator;
        let query = parse_with_dialect(query_str, translator.dialect())?;
        let query_with_relation = QueryWithRelations::new(&query, &relations);
        let relation = Relation::try_from((query_with_relation, translator))?;
        println!("\n {} \n", relation);
        let rel_with_traslator = RelationWithTranslator(&relation, translator);
        let translated = ast::Query::from(rel_with_traslator);
        print!("{}", translated);
        _ = database
            .query(translated.to_string().as_str())
            .unwrap()
            .iter()
            .map(ToString::to_string);
        Ok(())
    }

    #[test]
    fn test_extract() -> Result<()> {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        let query_str = r#"SELECT extract(EPOCH FROM c) as my_epoch, extract(YEAR FROM c) as my_year, extract(WEEK FROM c) as my_week, extract(DOW FROM c) as my_dow FROM table_1"#;
        let translator = PostgreSqlTranslator;
        let query = parse_with_dialect(query_str, translator.dialect())?;
        let query_with_relation = QueryWithRelations::new(&query, &relations);
        let relation = Relation::try_from((query_with_relation, translator))?;
        println!("\n {} \n", relation);
        let rel_with_traslator = RelationWithTranslator(&relation, translator);
        let translated = ast::Query::from(rel_with_traslator);
        print!("{}", translated);
        _ = database
            .query(translated.to_string().as_str())
            .unwrap()
            .iter()
            .map(ToString::to_string);
        Ok(())
    }

    #[test]
    fn test_flatten_case() -> Result<()> {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        let query_str = r#"
        SELECT
            CASE
                WHEN (a BETWEEN 0.0 AND 0.2) THEN 0.0
                WHEN (a BETWEEN 0.2 AND 0.4) THEN 0.2
                WHEN (a BETWEEN 0.4 AND 0.6) THEN 0.4
                WHEN (a BETWEEN 0.6 AND 0.8) THEN 0.6
                WHEN (a BETWEEN 0.8 AND 1.0) THEN 0.8
                ELSE 1.0
            END AS my_case
        FROM table_1
        "#;
        let translator = PostgreSqlTranslator;
        let query = parse_with_dialect(query_str, translator.dialect())?;
        println!("Query: \n{}", query);
        let query_with_relation = QueryWithRelations::new(&query, &relations);
        let relation = Relation::try_from((query_with_relation, translator))?;
        let rel_with_traslator = RelationWithTranslator(&relation, translator);
        let translated = ast::Query::from(rel_with_traslator);
        println!("FROM RELATION \n {} \n", translated);
        _ = database
            .query(translated.to_string().as_str())
            .unwrap()
            .iter()
            .map(ToString::to_string);
        Ok(())
    }

    #[test]
    fn test_flatten_case_with_operand() -> Result<()> {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        let query_str = r#"
        SELECT
            CASE z
                WHEN 'Foo' THEN 0.0
                WHEN 'Bar' THEN 0.2
                ELSE 10.0
            END AS my_case
        FROM table_2
        "#;
        let translator = PostgreSqlTranslator;
        let query = parse_with_dialect(query_str, translator.dialect())?;
        println!("Query: \n{}", query);
        let query_with_relation = QueryWithRelations::new(&query, &relations);
        let relation = Relation::try_from((query_with_relation, translator))?;
        let rel_with_traslator = RelationWithTranslator(&relation, translator);
        let translated = ast::Query::from(rel_with_traslator);
        println!("FROM RELATION \n {} \n", translated);
        _ = database
            .query(translated.to_string().as_str())
            .unwrap()
            .iter()
            .map(ToString::to_string);
        Ok(())
    }


    #[test]
    fn test_complex_case_query() -> Result<()> {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        // Specify the path to the file
        let path = "src/dialect_translation/complex_case_query.txt";

        // Read the file contents into a string
        let query_str = fs::read_to_string(path).unwrap();
        let translator = PostgreSqlTranslator;
        let query = parse_with_dialect(&query_str[..], translator.dialect())?;
        println!("Parsed Query: \n{}", query);
        let query_with_relation = QueryWithRelations::new(&query, &relations);
        let relation = Relation::try_from((query_with_relation, translator))?;
        let rel_with_traslator = RelationWithTranslator(&relation, translator);
        let rewritten = ast::Query::from(rel_with_traslator);
        println!("Rewritten Query: \n{}", rewritten);
        _ = database
            .query(rewritten.to_string().as_str())
            .unwrap()
            .iter()
            .map(ToString::to_string);
        Ok(())
    }
}
