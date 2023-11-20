use std::sync::Arc;

use crate::{
    hierarchy::Hierarchy,
    relation::sql::FromRelationVisitor,
    sql::{
        parse_with_dialect, query_names::IntoQueryNamesVisitor, relation::TryIntoRelationVisitor,
    },
    visitor::Acceptor,
    Relation,
};

use super::{IntoDialectTranslator, IntoRelationTranslator};
use sqlparser::{ast, dialect::PostgreSqlDialect};

use crate::sql::{Error, Result};
#[derive(Clone, Copy)]
pub struct PostgresTranslator;

impl IntoDialectTranslator for PostgresTranslator {}

impl IntoRelationTranslator for PostgresTranslator {
    type D = PostgreSqlDialect;

    fn dialect(&self) -> Self::D {
        PostgreSqlDialect {}
    }
}

// It should not exists outside the dialect translator module
struct PostgresQueryWithRelation<'a>(ast::Query, &'a Hierarchy<Arc<Relation>>);

impl<'a> PostgresQueryWithRelation<'a> {
    // Not public. Can't create a PostgresQueryWithRelation from any Query
    // you can only create PostgresQueryWithRelation using the try_from.
    fn new(query: ast::Query, relations: &'a Hierarchy<Arc<Relation>>) -> Self {
        PostgresQueryWithRelation(query, relations)
    }
}

impl<'a> TryFrom<(&'a str, &'a Hierarchy<Arc<Relation>>)> for PostgresQueryWithRelation<'a> {
    type Error = Error;

    fn try_from(value: (&'a str, &'a Hierarchy<Arc<Relation>>)) -> Result<Self> {
        let (query, relations) = value;
        let translator = PostgresTranslator;
        let ast = parse_with_dialect(query, translator.dialect())?;
        Ok(PostgresQueryWithRelation::new(ast, relations))
    }
}

impl<'a> TryFrom<PostgresQueryWithRelation<'a>> for Relation {
    type Error = Error;

    fn try_from(value: PostgresQueryWithRelation<'a>) -> Result<Self> {
        // Pull values from the object
        let PostgresQueryWithRelation(query, relations) = value;
        // Visit the query to get query names
        let query_names = query.accept(IntoQueryNamesVisitor);
        // Visit for conversion
        query
            .accept(TryIntoRelationVisitor::new(
                relations,
                query_names,
                PostgresTranslator,
            ))
            .map(|r| r.as_ref().clone())
    }
}

pub struct RelationWithPostgresTranslator<'a>(pub &'a Relation, pub PostgresTranslator);

impl<'a> From<RelationWithPostgresTranslator<'a>> for ast::Query {
    fn from(value: RelationWithPostgresTranslator) -> Self {
        let RelationWithPostgresTranslator(relation, translator) = value;
        relation.accept(FromRelationVisitor::new(translator))
    }
}

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
