use std::sync::Arc;

use crate::{
    hierarchy::Hierarchy,
    relation::sql::FromRelationVisitor,
    sql::{
        parse_with_dialect, query_names::IntoQueryNamesVisitor,
    },
    visitor::Acceptor,
    Relation, expr,
};

use super::{RelationToQueryTranslator, QueryToRelationTranslator, function_builder};
use sqlparser::{ast, dialect::PostgreSqlDialect};

use crate::sql::{Error, Result};
#[derive(Clone, Copy)]
pub struct PostgresTranslator;

impl RelationToQueryTranslator for PostgresTranslator {
    fn first(&self, expr: &expr::Expr) -> ast::Expr {
        ast::Expr::from(expr)
    }

    fn mean(&self, expr: &expr::Expr) -> ast::Expr {
        let arg = self.expr(expr);
        function_builder("AVG", vec![arg], false)
    }

    fn var(&self, expr: &expr::Expr) -> ast::Expr {
        let arg = self.expr(expr);
        function_builder("VARIANCE", vec![arg], false,)
    }

    fn std(&self, expr: &expr::Expr) -> ast::Expr {
        let arg = self.expr(expr);
        function_builder("STDDEV", vec![arg], false)
    }

    fn trunc(&self, exprs: Vec<&expr::Expr>) -> ast::Expr {
        // TRUNC in postgres has a problem:
        // In TRUNC(double_precision_number, precision) if precision is specified it fails
        // If it is not specified it passes considering precision = 0.
        // SELECT TRUNC(CAST (0.12 AS DOUBLE PRECISION), 0) fails
        // SELECT TRUNC(CAST (0.12 AS DOUBLE PRECISION)) passes.
        // Here we check precision, if it is 0 we remove it (such that the precision is implicit).
        let ast_exprs: Vec<ast::Expr> = exprs.into_iter().map(|expr| self.expr(expr)).collect();
        ast::Expr::Function(ast::Function {
            name: ast::ObjectName(vec![ast::Ident::from("TRUNC")]),
            args: ast_exprs
                .into_iter()
                .filter_map(
                    |e| (e!=ast::Expr::Value(ast::Value::Number("0".to_string(), false))
                ).then_some(ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(e))))
                .collect(),
            over: None,
            distinct: false,
            special: false,
            order_by: vec![],
            filter: None,
            null_treatment: None,
        })
    }

    fn round(&self, exprs: Vec<&expr::Expr>) -> ast::Expr {
        // Same as TRUNC
        // what if I wanted to do round(0, 0)
        let ast_exprs: Vec<ast::Expr> = exprs.into_iter().map(|expr| self.expr(expr)).collect();
        ast::Expr::Function(ast::Function {
            name: ast::ObjectName(vec![ast::Ident::from("ROUND")]),
            args: ast_exprs
                .into_iter()
                .filter_map(
                    |e| (e!=ast::Expr::Value(ast::Value::Number("0".to_string(), false))
                ).then_some(ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(e))))
                .collect(),
            over: None,
            distinct: false,
            special: false,
            order_by: vec![],
            filter: None,
            null_treatment: None,
        })
    }

    fn position(&self, exprs: Vec<&expr::Expr>) -> ast::Expr {
        assert!(exprs.len() == 2);
        let ast_exprs: Vec<ast::Expr> = exprs.into_iter().map(|expr| self.expr(expr)).collect();
        ast::Expr::Position { expr: Box::new(ast_exprs[0].clone()), r#in: Box::new(ast_exprs[1].clone()) }
    }
}

impl QueryToRelationTranslator for PostgresTranslator {
    type D = PostgreSqlDialect;

    fn dialect(&self) -> Self::D {
        PostgreSqlDialect {}
    }
}

// // It should not exists outside the dialect translator module
// struct PostgresQueryWithRelation<'a>(ast::Query, &'a Hierarchy<Arc<Relation>>);

// impl<'a> PostgresQueryWithRelation<'a> {
//     // Not public. Can't create a PostgresQueryWithRelation from any Query
//     // you can only create PostgresQueryWithRelation using the try_from.
//     fn new(query: ast::Query, relations: &'a Hierarchy<Arc<Relation>>) -> Self {
//         PostgresQueryWithRelation(query, relations)
//     }
// }

// impl<'a> TryFrom<(&'a str, &'a Hierarchy<Arc<Relation>>)> for PostgresQueryWithRelation<'a> {
//     type Error = Error;

//     fn try_from(value: (&'a str, &'a Hierarchy<Arc<Relation>>)) -> Result<Self> {
//         let (query, relations) = value;
//         let translator = PostgresTranslator;
//         let ast = parse_with_dialect(query, translator.dialect())?;
//         Ok(PostgresQueryWithRelation::new(ast, relations))
//     }
// }

// impl<'a> TryFrom<PostgresQueryWithRelation<'a>> for Relation {
//     type Error = Error;

//     fn try_from(value: PostgresQueryWithRelation<'a>) -> Result<Self> {
//         // Pull values from the object
//         let PostgresQueryWithRelation(query, relations) = value;
//         // Visit the query to get query names
//         let query_names = query.accept(IntoQueryNamesVisitor);
//         // Visit for conversion
//         query
//             .accept(TryIntoRelationVisitor::new(
//                 relations,
//                 query_names,
//                 PostgresTranslator,
//             ))
//             .map(|r| r.as_ref().clone())
//     }
// }

// pub struct RelationWithPostgresTranslator<'a>(pub &'a Relation, pub PostgresTranslator);

// impl<'a> From<RelationWithPostgresTranslator<'a>> for ast::Query {
//     fn from(value: RelationWithPostgresTranslator) -> Self {
//         let RelationWithPostgresTranslator(relation, translator) = value;
//         relation.accept(FromRelationVisitor::new(translator))
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        builder::{Ready, With},
        data_type::{DataType, Value as _},
        display::Dot,
        expr::Expr,
        namer,
        relation::{schema::Schema, Relation},
        sql::{parse, relation::QueryWithRelations},
    };
    use std::sync::Arc;

    #[test]
    fn test_query() -> Result<()> {
        let translator = PostgresTranslator;
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

        let query_str = "SELECT exp(table.a) FROM schema.table";
        let translator = PostgresTranslator;
        let query = parse_with_dialect(query_str, translator.dialect())?;
        let query_with_relation = QueryWithRelations::new(&query, &relations);
        let relation = Relation::try_from((query_with_relation, translator))?;

        // let retranslated: ast::Query::from()
        print!("{}", relation);
        Ok(())
    }
}
