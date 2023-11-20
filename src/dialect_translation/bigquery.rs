use crate::{relation::sql::FromRelationVisitor, visitor::Acceptor, Relation};

use super::IntoDialectTranslator;
use sqlparser::{ast, dialect::BigQueryDialect};

pub struct BigQueryTranlator;

impl IntoDialectTranslator for BigQueryTranlator {
    // type D = BigQueryDialect;

    // fn dialect(&self) -> Self::D {
    //     BigQueryDialect {}
    // }
}

pub struct RelationWithBigQueryTranlator<'a>(pub &'a Relation, pub BigQueryTranlator);

impl<'a> From<RelationWithBigQueryTranlator<'a>> for ast::Query {
    fn from(value: RelationWithBigQueryTranlator) -> Self {
        let RelationWithBigQueryTranlator(relation, translator) = value;
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
