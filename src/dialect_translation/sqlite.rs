use crate::{relation::sql::FromRelationVisitor, visitor::Acceptor, Relation};

use super::IntoDialectTranslator;
use sqlparser::{ast, dialect::SQLiteDialect};
#[derive(Clone, Copy)]
pub struct SQLiteTranslator;

impl IntoDialectTranslator for SQLiteTranslator {}

// pub struct RelationWithSQLiteTranslator<'a>(pub &'a Relation, pub &'a SQLiteTranslator);

// impl<'a> From<RelationWithSQLiteTranslator<'a>> for ast::Query {
//     fn from(value: RelationWithSQLiteTranslator) -> Self {
//         let RelationWithSQLiteTranslator(relation, translator) = value;
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
        sql::parse,
    };
    use std::sync::Arc;

    #[test]
    fn test_xxx() {}
}
