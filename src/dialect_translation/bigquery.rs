use std::sync::Arc;

use crate::{
    expr,
    hierarchy::Hierarchy,
    relation::sql::FromRelationVisitor,
    sql::{parse_with_dialect, query_names::IntoQueryNamesVisitor},
    visitor::Acceptor,
    Relation,
};

use super::{function_builder, QueryToRelationTranslator, RelationToQueryTranslator};
use sqlparser::{ast, dialect::BigQueryDialect};

use crate::sql::{Error, Result};
#[derive(Clone, Copy)]
pub struct BigQueryTranslator;

impl RelationToQueryTranslator for BigQueryTranslator {}

impl QueryToRelationTranslator for BigQueryTranslator {
    type D = BigQueryDialect;

    fn dialect(&self) -> Self::D {
        BigQueryDialect {}
    }
}
