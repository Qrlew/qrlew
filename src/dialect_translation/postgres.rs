use crate::{relation::sql::FromRelationVisitor, visitor::Acceptor, Relation};

use super::IntoDialectTranslator;
use sqlparser::{ast, dialect::PostgreSqlDialect};

pub struct PostgresTranslator;

impl IntoDialectTranslator for PostgresTranslator {
    type D = PostgreSqlDialect;

    fn dialect(&self) -> Self::D {
        PostgreSqlDialect {}
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
        sql::parse,
    };
    use std::sync::Arc;

    #[test]
    fn test_xxx() {}
}
