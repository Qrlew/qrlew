use crate::{
    builder::{Ready, With, WithIterator},
    expr::{AggregateColumn, Expr, Identifier},
    hierarchy::Hierarchy,
    relation::{Join, Map, Reduce, Relation, Table, Values, Variant as _},
};
use std::{error, fmt, ops::Deref, result, sync::Arc};

pub const SYNTHETIC_PREFIX: &str = "_SYNTHETIC_";

#[derive(Debug, Clone)]
pub enum Error {
    NoSyntheticData(String),
    Other(String),
}

impl Error {
    pub fn no_synthetic_data(table: impl fmt::Display) -> Error {
        Error::NoSyntheticData(format!("{} has no SD", table))
    }
    pub fn other(value: impl fmt::Display) -> Error {
        Error::Other(format!("{} has no SD", value))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NoSyntheticData(desc) => {
                writeln!(f, "NoSyntheticData: {}", desc)
            }
            Error::Other(err) => writeln!(f, "{}", err),
        }
    }
}

impl error::Error for Error {}

pub type Result<T> = result::Result<T, Error>;

#[derive(Clone, Debug)]
pub struct SDRelation(pub Relation);

impl From<SDRelation> for Relation {
    fn from(value: SDRelation) -> Self {
        value.0
    }
}

impl From<Relation> for SDRelation {
    fn from(value: Relation) -> Self {
        SDRelation(value)
    }
}

impl Deref for SDRelation {
    type Target = Relation;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Implements the synthetic data equivalent of tables
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct SyntheticData {
    synthetic_paths: Hierarchy<Identifier>,
}

impl SyntheticData {
    pub fn new(synthetic_paths: Hierarchy<Identifier>) -> SyntheticData {
        SyntheticData { synthetic_paths }
    }

    pub fn synthetic_prefix() -> &'static str {
        SYNTHETIC_PREFIX
    }

    /// Table sd equivalent
    pub fn table(&self, table: &Table) -> Result<SDRelation> {
        let relation: Relation = Relation::table()
            .name(format!(
                "{}{}",
                SyntheticData::synthetic_prefix(),
                table.name()
            ))
            .path(
                self.synthetic_paths
                    .get(table.path())
                    .ok_or(Error::no_synthetic_data(table))?
                    .clone(),
            )
            .size(table.size().iter().last().ok_or(Error::other(table))?[0])
            .schema(table.schema().clone())
            .build();
        Ok(relation.into())
    }
}
