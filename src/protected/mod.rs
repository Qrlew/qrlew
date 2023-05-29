use std::{fmt, error, result, collections::HashMap};
use crate::{
    expr::Expr,
    relation::{Table, Relation, Visitor},
    builder::With,
};

#[derive(Debug, Clone)]
pub enum Error {
    NotProtectedEntityPreserving(String),
    Other(String),
}

impl Error {
    pub fn not_protected_entity_preserving(relation: impl fmt::Display) -> Error {
        Error::NotProtectedEntityPreserving(format!("{} is not PEP", relation))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NotProtectedEntityPreserving(desc) => writeln!(f, "NotProtectedEntityPreserving: {}", desc),
            Error::Other(err) => writeln!(f, "{}", err),
        }
    }
}

impl error::Error for Error {}

pub type Result<T> = result::Result<T, Error>;

/// A wrapper to compute Relation protection
#[derive(Clone, Debug)]
pub struct Protection<'a> {
    /// The protected entity definition
    protected_entity: HashMap<&'a Table, Expr>
}

impl<'a> Protection<'a> {
    pub fn new<T: IntoIterator<Item=(&'a Table, Expr)>>(protected_entity: T) -> Self {
        Protection { protected_entity: HashMap::from_iter(protected_entity) }
    }

    pub fn empty() -> Self {
        Protection::new([])
    }
}

impl<'a> With<(&'a Table, Expr)> for Protection<'a> {
    fn with(mut self, input: (&'a Table, Expr)) -> Self {
        self.protected_entity.insert(input.0, input.1);
        self
    }
}


impl<'a> Visitor<'a, Result<Relation>> for Protection<'a> {
    fn table(&self, table: &'a Table) -> Result<Relation> {
        todo!()
    }

    fn map(&self, map: &'a crate::relation::Map, input: Result<Relation>) -> Result<Relation> {
        todo!()
    }

    fn reduce(&self, reduce: &'a crate::relation::Reduce, input: Result<Relation>) -> Result<Relation> {
        todo!()
    }

    fn join(&self, join: &'a crate::relation::Join, left: Result<Relation>, right: Result<Relation>) -> Result<Relation> {
        todo!()
    }
}