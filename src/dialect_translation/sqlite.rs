use crate::{relation::sql::FromRelationVisitor, visitor::Acceptor, Relation};

use super::RelationToQueryTranslator;
use sqlparser::{ast, dialect::SQLiteDialect};
#[derive(Clone, Copy)]
pub struct SQLiteTranslator;

impl RelationToQueryTranslator for SQLiteTranslator {}

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
}
