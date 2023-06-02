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

/// A visitor to compute Relation protection
#[derive(Clone, Debug)]
pub struct ProtectVisitor<F: Fn(&Table) -> Relation> {
    /// The protected entity definition
    protect_tables: F,
    /// Strategy used
    strategy: Strategy,
}

impl<F: Fn(&Table) -> Relation> ProtectVisitor<F> {
    pub fn new(protect_tables: F, strategy: Strategy) -> Self {
        ProtectVisitor { protect_tables, strategy }
    }
}

/// Build a visitor from exprs
pub fn protect_visitor_from_exprs<'a, A: AsRef<[(&'a Table, Expr)]>+'a>(protected_entity: A, strategy: Strategy) -> ProtectVisitor<impl Fn(&Table) -> Relation> {
    ProtectVisitor::new(move |table: &Table| {
        match protected_entity.as_ref().iter().find_map(|(t, e)| (table==*t).then(|| e.clone())) {
            Some(expr) => Relation::from(table.clone()).with_computed_field(PEID, expr.clone()),
            None => table.clone().into(),
        }
    }, strategy)
}

// /// Build a visitor from exprs and relation
// pub fn protect_visitor_from_exprs<'a, A: AsRef<[(&'a Table, Expr)]>+'a>(protected_entity: A, strategy: Strategy) -> ProtectVisitor<impl Fn(&Table) -> Relation> {
//     ProtectVisitor::new(move |table: &Table| {
//         match protected_entity.as_ref().iter().find_map(|(t, e)| (table==*t).then(|| e.clone())) {
//             Some(expr) => Relation::from(table.clone()).with_computed_field(PEID, expr.clone()),
//             None => table.clone().into(),
//         }
//     }, strategy)
// }

impl<'a, F: Fn(&Table) -> Relation> Visitor<'a, Result<Relation>> for ProtectVisitor<F> {
    fn table(&self, table: &'a Table) -> Result<Relation> {
        Ok((self.protect_tables)(table))
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
    pub fn protect<F: Fn(&Table) -> Relation>(self, protect_tables: F) -> Result<Relation> {
        self.accept(ProtectVisitor::new(protect_tables, Strategy::Soft))
    }

    /// Add protection
    pub fn protect_from_exprs<'a, A: AsRef<[(&'a Table, Expr)]>+'a>(self, protected_entity: A) -> Result<Relation> {
        self.accept(protect_visitor_from_exprs(protected_entity, Strategy::Soft))
    }

    /// Force protection
    pub fn force_protect<F: Fn(&Table) -> Relation>(self, protect_tables: F) -> Relation {
        self.accept(ProtectVisitor::new(protect_tables, Strategy::Hard)).unwrap()
    }

    /// Force protection
    pub fn force_protect_from_exprs<'a, A: AsRef<[(&'a Table, Expr)]>+'a>(self, protected_entity: A) -> Result<Relation> {
        self.accept(protect_visitor_from_exprs(protected_entity, Strategy::Soft))
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
        let table = table.protect_from_exprs([(&database.tables()[0], expr!(md5(a)))]).unwrap();
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