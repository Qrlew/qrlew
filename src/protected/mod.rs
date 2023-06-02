use std::{fmt, error, result, collections::HashMap};
use crate::{
    expr::Expr,
    relation::{Table, Map, Reduce, Join, Set, Relation, Visitor, Variant as _},
    builder::{With, Ready},
    visitor::Acceptor,
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
            Some(expr) => Relation::from(table.clone()).identity_with_field(PEID, expr.clone()),
            None => table.clone().into(),
        }
    }, strategy)
}

/// Build a visitor from exprs
pub fn protect_visitor_from_field_paths<'a>(protected_entity: &'a[&'a[(&'a str, &'a str)]], strategy: Strategy) -> ProtectVisitor<impl Fn(&Table) -> Relation+'a> {
    ProtectVisitor::new(move |table: &Table| {
        match protected_entity.into_iter().find(|&&tabs_cols| tabs_cols.get(0).map(|(tab, _col)| table.name()==*tab).unwrap_or(false)) {
            Some([(_tab, col)]) => Relation::from(table.clone()).with_field(PEID, Expr::col(*col)),
            Some(tabs_cols) => todo!(),// TODO implement this
            None => table.clone().into(),
        }
    }, strategy)
}

impl<'a, F: Fn(&Table) -> Relation> Visitor<'a, Result<Relation>> for ProtectVisitor<F> {
    fn table(&self, table: &'a Table) -> Result<Relation> {
        Ok((self.protect_tables)(table))
    }

    fn map(&self, map: &'a Map, input: Result<Relation>) -> Result<Relation> {
        let builder = Relation::map().with((PEID, Expr::col(PEID))).with(map.clone()).input(input?);
        Ok(builder.build())
    }

    fn reduce(&self, reduce: &'a Reduce, input: Result<Relation>) -> Result<Relation> {
        match self.strategy {
            Strategy::Soft => Err(Error::not_protected_entity_preserving(reduce)),
            Strategy::Hard => {
                let builder = Relation::reduce().with_group_by_column(PEID).with(reduce.clone()).group_by(Expr::col(PEID)).input(input?);
                Ok(builder.build())
            },
        }
    }

    fn join(&self, join: &'a crate::relation::Join, left: Result<Relation>, right: Result<Relation>) -> Result<Relation> {
        match self.strategy {
            Strategy::Soft => Err(Error::not_protected_entity_preserving(join)),
            Strategy::Hard => {
                let Join { name, operator, .. } = join;
                let left = left?;
                let right = right?;
                let builder = Relation::join().name(name).operator(operator.clone())
                    .on(Expr::eq(Expr::qcol(left.name(), PEID), Expr::qcol(right.name(), PEID)))
                    .left(left)
                    .right(right);
                Ok(builder.build())
            },
        }
    }

    fn set(&self, set: &'a crate::relation::Set, left: Result<Relation>, right: Result<Relation>) -> Result<Relation> {
        let Set { name, operator, quantifier, .. } = set;
        let builder = Relation::set().name(name).operator(operator.clone()).quantifier(quantifier.clone()).left(left?).right(right?);
        Ok(builder.build())
    }
}

impl Relation {
    /// Add protection
    pub fn protect_from_visitor<F: Fn(&Table) -> Relation>(self, protect_visitor: ProtectVisitor<F>) -> Result<Relation> {
        self.accept(protect_visitor)
    }

    /// Add protection
    pub fn protect<F: Fn(&Table) -> Relation>(self, protect_tables: F) -> Result<Relation> {
        self.accept(ProtectVisitor::new(protect_tables, Strategy::Soft))
    }

    /// Add protection
    pub fn protect_from_exprs<'a, A: AsRef<[(&'a Table, Expr)]>+'a>(self, protected_entity: A) -> Result<Relation> {
        self.accept(protect_visitor_from_exprs(protected_entity, Strategy::Soft))
    }

    /// Add protection
    pub fn protect_from_field_paths<'a>(self, protected_entity: &'a[&'a[(&'a str, &'a str)]]) -> Result<Relation> {
        self.accept(protect_visitor_from_field_paths(protected_entity, Strategy::Soft))
    }

    /// Force protection
    pub fn force_protect<F: Fn(&Table) -> Relation>(self, protect_tables: F) -> Relation {
        self.accept(ProtectVisitor::new(protect_tables, Strategy::Hard)).unwrap()
    }

    /// Force protection
    pub fn force_protect_from_exprs<'a, A: AsRef<[(&'a Table, Expr)]>+'a>(self, protected_entity: A) -> Relation {
        self.accept(protect_visitor_from_exprs(protected_entity, Strategy::Hard)).unwrap()
    }

     /// Force protection
    pub fn force_protect_from_field_paths<'a>(self, protected_entity: &'a[&'a[(&'a str, &'a str)]]) -> Relation {
        self.accept(protect_visitor_from_field_paths(protected_entity, Strategy::Hard)).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        relation::{display, Variant},
        sql::parse,
        io::{Database, postgresql},
    };
    use sqlparser::ast;

    #[test]
    fn test_table_protection() {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        let table = relations.get(&["table_1".into()]).unwrap().as_ref().clone();
        // Table
        let table = table.protect_from_exprs([(&database.tables()[0], expr!(md5(a)))]).unwrap();
        println!("Schema protected = {}", table.schema());
        assert_eq!(table.schema()[0].name(), PEID)
    }

    #[test]
    fn test_table_protection_from_field_paths() {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        let table = relations.get(&["secondary_table".into()]).unwrap().as_ref().clone();
        // Table
        let table = table.protect_from_field_paths(&[&[("secondary_table", "primary_id")]]).unwrap();
        println!("Schema protected = {}", table.schema());
        assert_eq!(table.schema()[0].name(), PEID)
    }

    #[ignore]
    #[test]
    fn test_relation_protection() {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        let relation = Relation::try_from(parse("SELECT sum(amount) FROM secondary_table GROUP BY primary_id").unwrap().with(&relations)).unwrap();
        // let relation = Relation::try_from(parse("SELECT * FROM primary_table").unwrap().with(&relations)).unwrap();
        // Table
        let relation = relation.force_protect_from_field_paths(&[&[("primary_table", "id")], &[("secondary_table", "primary_id"), ("primary_table", "id"), ("primary_table", "id")]]);
        display(&relation);
        println!("Schema protected = {}", relation.schema());
        assert_eq!(relation.schema()[0].name(), PEID)
    }

    // #[ignore]
    // #[test]
    // fn test_relation_protection() {
    //     let mut database = postgresql::test_database();
    //     let relations = database.relations();
    //     let table = relations.get(&["table_1".into()]).unwrap().as_ref().clone();
    //     let relation = Relation::try_from(parse("SELECT * FROM table_1").unwrap().with(&relations)).unwrap();
    //     // Table
    //     let table = table.with_computed_field("peid", expr!(a+b));
    //     let relation = relation.with_computed_field("peid", expr!(cos(a)));
    //     // Print a few rows
    //     for row in database.query(&ast::Query::from(&relation).to_string()).unwrap() {
    //         println!("{row}");
    //     }
    // }
}