use std::{fmt, error, result, collections::HashMap};
use crate::{
    expr::Expr,
    relation::{Table, Relation, Visitor},
    builder::With, visitor::Acceptor,
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

pub const PEID: &str = "_PROTECTED_ENTITY_ID_";

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Default)]
pub enum Strategy {
    /// Protect only when it does not affect the meaning of the original query.
    /// Fail otherwise.
    #[default]
    Soft,
    /// Protect at all cost.
    /// Will succeede most of the time.
    Hard,
}

/// A wrapper to compute Relation protection
#[derive(Clone, Debug)]
pub struct Protection<'a> {
    /// The protected entity definition
    protected_entity: HashMap<&'a Table, Expr>,
    /// Strategy used
    strategy: Strategy,
}

impl<'a> Protection<'a> {
    pub fn new<T: IntoIterator<Item=(&'a Table, Expr)>>(protected_entity: T, strategy: Strategy) -> Self {
        Protection { protected_entity: HashMap::from_iter(protected_entity), strategy }
    }

    pub fn empty() -> Self {
        Protection::new([], Strategy::default())
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
        match self.protected_entity.get(table) {
            Some(expr) => Ok(Relation::from(table.clone()).with_computed_field(PEID, expr.clone())),
            None => Ok(table.clone().into()),
        }
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

    fn set(&self, set: &'a crate::relation::Set, left: Result<Relation>, right: Result<Relation>) -> Result<Relation> {
        todo!()
    }
}

impl Relation {
    /// Add protection
    pub fn protect<'a, T: IntoIterator<Item=(&'a Table, Expr)>+'a>(self, protected_entity: T) -> Result<Relation> {
        self.accept(Protection::new(protected_entity, Strategy::Soft))
    }

    /// Force protection
    pub fn force_protect<'a, T: IntoIterator<Item=(&'a Table, Expr)>+'a>(self, protected_entity: T) -> Relation {
        self.accept(Protection::new(protected_entity, Strategy::Hard)).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        relation::display,
        sql::parse,
        io::{Database, postgresql},
    };

    #[ignore]
    #[test]
    fn test_table_protection() {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        let table = relations.get(&["table_1".into()]).unwrap().as_ref().clone();
        // Table
        let table = table.protect([(&database.tables()[0], expr!(a))]).unwrap();
        display(&table);
    }

    #[ignore]
    #[test]
    fn test_relation_protection() {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        let table = relations.get(&["table_1".into()]).unwrap().as_ref().clone();
        let relation = Relation::try_from(parse("SELECT * FROM table_1").unwrap().with(&relations)).unwrap();
        // Table
        let table = table.with_computed_field("peid", expr!(a+b));
        let relation = relation.with_computed_field("peid", expr!(cos(a)));
    }
}