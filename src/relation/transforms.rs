//! A few transforms for relations
//! 

use std::{
    rc::Rc,
    iter::once,
};
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
    pub fn with_foreign_field(self, name: &str, foreign: Rc<Relation>, on: (&str, &str), field: &str) -> Relation {
        let left_size = foreign.schema().len();
        let names: Vec<String> = self.schema().iter().map(|f| f.name().to_string()).collect();
        let join: Relation = Relation::join().inner()
            .on(Expr::eq(Expr::qcol(foreign.name(), on.0), Expr::qcol(self.name(), on.1)))
            .left(foreign)
            .right(self)
            .build();
        let left: Vec<_> = join.schema().iter().zip(join.input_fields()).take(left_size).collect();
        let right: Vec<_> = join.schema().iter().zip(join.input_fields()).skip(left_size).collect();
        Relation::map()
            .with_iter(left.into_iter().find_map(|(o, i)| {
                (field==i.name()).then_some((name, Expr::col(o.name())))
            }))
            .with_iter(right.into_iter().filter_map(|(o, i)| {
                names.contains(&i.name().to_string()).then_some((i.name(), Expr::col(o.name())))
            }))
            
            .input(join)
            .build()
    }

    /// Add a field designated with a "fiald path"
    pub fn with_field_path(self, name: &str, relations: &Hierarchy<Rc<Relation>>, path: &[(&str, (&str, &str))], field: &str) -> Relation {//TODO implement this
        if path.is_empty() {
            self.identity_with_field(name, Expr::col(field))
        } else {
            let path: Vec<((Rc<Relation>, (&str, &str)), (&str, &str))> = path.iter()
                .map(|(foreign_key, (primary_table, primary_key))| (relations.get(&[primary_table.to_string()]).unwrap().clone(), (*foreign_key, *primary_key)))
                .zip(path.iter().skip(1).map(|(foreign_key, (_, primary_key))| (*foreign_key, *primary_key)).chain(once((field, name))))
                .collect();
            // Build the relation following the path to compute the new field
            path.into_iter().fold(self, |relation, ((next, on), (field, name))| relation.with_foreign_field(name, next, on, field))
        }
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
    use colored::Colorize;
    use sqlparser::ast;
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

    #[test]
    fn test_filter_fields() {
        let database = postgresql::test_database();
        let relations = database.relations();
        let relation = Relation::try_from(parse("SELECT * FROM table_1").unwrap().with(&relations)).unwrap();
        let relation = relation.with_field("peid", expr!(cos(a)));
        assert!(relation.schema()[0].name()=="peid");
        let relation = relation.filter_fields(|n| n!="peid");
        assert!(relation.schema()[0].name()!="peid");
    }

    #[test]
    fn test_foreign_field() {
        let database = postgresql::test_database();
        let relations = database.relations();
        let orders = Relation::try_from(parse("SELECT * FROM order_table").unwrap().with(&relations)).unwrap();
        let user = relations.get(&["user_table".to_string()]).unwrap().as_ref();
        let relation = orders.with_foreign_field("peid", Rc::new(user.clone()), ("user_id", "id"), "id");
        assert!(relation.schema()[0].name()=="peid");
        let relation = relation.filter_fields(|n| n!="peid");
        assert!(relation.schema()[0].name()!="peid");
    }

    #[ignore]//TODO fix the ON in the JOIN
    #[test]
    fn test_field_path() {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        // Link orders to users
        let orders = relations.get(&["order_table".to_string()]).unwrap().as_ref();
        let relation = orders.clone().with_field_path("peid", &relations, &[("user_id", ("user_table", "id"))], "id");
        assert!(relation.schema()[0].name()=="peid");
        // Link items to orders
        let items = relations.get(&["item_table".to_string()]).unwrap().as_ref();
        let relation = items.clone().with_field_path("peid", &relations, &[("order_id", ("order_table", "id")), ("user_id", ("user_table", "id"))], "id");
        assert!(relation.schema()[0].name()=="peid");
        // Produce the query
        let query: &str = &ast::Query::from(&relation).to_string();
        println!("{query}");
        println!(
            "{}\n{}",
            format!("{query}").yellow(),
            database
                .query(query)
                .unwrap()
                .iter()
                .map(ToString::to_string)
                .join("\n")
        );
        let relation = relation.filter_fields(|n| n!="peid");
        assert!(relation.schema()[0].name()!="peid");
    }
}