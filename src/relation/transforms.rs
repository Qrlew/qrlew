//! A few transforms for relations
//! 

use itertools::Itertools;

use super::{Relation, Map, display, Variant as _};
use crate::{
    expr::Expr,
    builder::{With, Ready, WithIterator},
    hierarchy::Hierarchy,
};

impl Map {
    pub fn with_field(self, name: &str, expr: Expr) -> Map {
        Relation::map().with((name, expr)).with(self).build()
    }

    pub fn filter_fields<P: Fn(&str) -> bool>(self, predicate: P) -> Map {
        Relation::map().filter_with(self, predicate).build()
    }
}

impl Relation {
    /// Add a field that derives from existing fields
    pub fn identity_with_field(self, name: &str, expr: Expr) -> Relation {
        Relation::map()
            .with((name, expr))
            .with_iter(self.schema().iter().map(|f| (f.name(), Expr::col(f.name()))))
            .input(self)
            .build()
    }

    /// Add a field that derives from input fields
    pub fn with_field(self, name: &str, expr: Expr) -> Relation {
        match self {
            // Simply add a column on Maps
            Relation::Map(map) => map.with_field(name, expr).into(),
            relation => relation.identity_with_field(name, expr),
        }
    }

    /// Add a field designated with a foreign relation and a field
    pub fn with_foreign_field(self, name: &str, foreign: Relation, on: (&str, &str), field: &str) -> Relation {
        let names: Vec<String> = self.schema().iter().map(|f| f.name().to_string()).collect();
        let join: Relation = Relation::join().inner()
            .on(Expr::eq(Expr::qcol(foreign.name(), on.0), Expr::qcol(self.name(), on.1)))
            .left(foreign)
            .right(self)
            .build();
        Relation::map()
            .with_iter(join.schema().iter().zip(join.input_fields()).filter_map(|(o, i)| {
                if names.contains(&i.name().to_string()) {
                    Some((i.name(), Expr::col(o.name())))
                } else if field==i.name() {
                    Some((name, Expr::col(o.name())))
                } else {
                    None
                }
            }))
            .input(join)
            .build()
    }

    /// Add a field designated with a "fiald path"
    pub fn with_field_path(self, name: &str, path: &[(&str, &str)]) -> Relation {//TODO implement this
        let schema = self.schema().clone();
        // path.iter().chunks(2).fold(self, ||)
        todo!()
    }

    pub fn filter_fields<P: Fn(&str) -> bool>(self, predicate: P) -> Relation {
        match self {
            Relation::Map(map) => map.filter_fields(predicate).into(),
            relation => {
                Relation::map()
                    .with_iter(relation.schema().iter().filter_map(|f| predicate(f.name()).then_some((f.name(), Expr::col(f.name())))))
                    .input(relation)
                    .build()
            }
        }
        
    }

    
}

impl With<(&str, Expr)> for Relation {
    fn with(self, (name, expr): (&str, Expr)) -> Self {
        self.identity_with_field(name, expr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        sql::parse,
        io::{Database, postgresql},
    };

    #[test]
    fn test_with_computed_field() {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        let table = relations.get(&["table_1".into()]).unwrap().as_ref().clone();
        let relation = Relation::try_from(parse("SELECT * FROM table_1").unwrap().with(&relations)).unwrap();
        // Table
        assert!(table.schema()[0].name()!="peid");
        let table = table.identity_with_field("peid", expr!(a+b));
        assert!(table.schema()[0].name()=="peid");
        // Relation
        assert!(relation.schema()[0].name()!="peid");
        let relation = relation.identity_with_field("peid", expr!(cos(a)));
        assert!(relation.schema()[0].name()=="peid");
    }

    #[ignore]
    #[test]
    fn test_filter_fields() {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        let relation = Relation::try_from(parse("SELECT * FROM table_1").unwrap().with(&relations)).unwrap();
        let relation = relation.with_field("peid", expr!(cos(a)));
        display(&relation);
        let relation = relation.filter_fields(|n| n!="peid");
        display(&relation);
    }

    #[ignore]
    #[test]
    fn test_foreign_field() {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        let secondary = Relation::try_from(parse("SELECT * FROM secondary_table").unwrap().with(&relations)).unwrap();
        let primary = relations.get(&["primary_table".to_string()]).unwrap().as_ref();
        let relation = secondary.with_foreign_field("peid", primary.clone(), ("primary_id", "id"), "id");
        assert!(relation.schema()[0].name()!="peid");
        let relation = relation.filter_fields(|n| n!="peid");
        assert!(relation.schema()[0].name()!="peid");
    }
}