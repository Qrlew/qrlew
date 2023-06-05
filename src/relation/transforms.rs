//! A few transforms for relations
//! 

use std::{
    rc::Rc,
    ops::Deref,
};

use super::{Relation, Map, Variant as _};
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

// A few utility objects
#[derive(Clone, Debug)]
pub struct Step<'a> {
    pub referring_id: &'a str,
    pub referred_relation: &'a str,
    pub referred_id: &'a str,
}

impl<'a> From<(&'a str, &'a str, &'a str)> for Step<'a> {
    fn from((referring_id, referred_relation, referred_id): (&'a str, &'a str, &'a str)) -> Self {
        Step {
            referring_id,
            referred_relation,
            referred_id,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Path<'a>(pub Vec<Step<'a>>);

impl<'a> Deref for Path<'a> {
    type Target = Vec<Step<'a>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> FromIterator<&'a(&'a str, &'a str, &'a str)> for Path<'a> {
    fn from_iter<T: IntoIterator<Item = &'a (&'a str, &'a str, &'a str)>>(iter: T) -> Self {
        Path(iter.into_iter().map(|(referring_id, referred_relation, referred_id)| Step {
            referring_id,
            referred_relation,
            referred_id,
        }).collect())
    }
}

impl<'a> IntoIterator for Path<'a> {
    type Item = Step<'a>;
    type IntoIter = <Vec<Step<'a>> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

/// A link to a relation and a field to keep with a new name
#[derive(Clone, Debug)]
pub struct ReferredField<'a> {
    pub referring_id: &'a str,
    pub referred_relation: &'a str,
    pub referred_id: &'a str,
    pub referred_field: &'a str,
    pub referred_field_name: &'a str,
}

#[derive(Clone, Debug)]
pub struct FieldPath<'a>(pub Vec<ReferredField<'a>>);

impl<'a> FieldPath<'a> {
    pub fn from_path(path: Path<'a>, referred_field: &'a str, referred_field_name: &'a str) -> Self {
        let mut field_path = FieldPath(Vec::new());
        let mut last_step: Option<Step> = None;
        // Fill the vec
        for step in path {
            if let Some(last_step) = &mut last_step {
                field_path.0.push(ReferredField {
                    referring_id: last_step.referring_id,
                    referred_relation: last_step.referred_relation,
                    referred_id: last_step.referred_id,
                    referred_field: step.referring_id,
                    referred_field_name,
                });
                *last_step = Step {
                    referring_id: referred_field_name,
                    referred_relation: step.referred_relation,
                    referred_id: step.referred_id,
                };
            } else {
                last_step = Some(step);
            }
        }
        if let Some(last_step) = last_step {
            field_path.0.push(ReferredField {
                referring_id: last_step.referring_id,
                referred_relation: last_step.referred_relation,
                referred_id: last_step.referred_id,
                referred_field,
                referred_field_name,
            });
        }
        field_path
    }
}

impl<'a> Deref for FieldPath<'a> {
    type Target = Vec<ReferredField<'a>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> IntoIterator for FieldPath<'a> {
    type Item = ReferredField<'a>;
    type IntoIter = <Vec<ReferredField<'a>> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
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
    pub fn with_referred_field<'a>(self, referring_id: &'a str, referred_relation: Rc<Relation>, referred_id: &'a str, referred_field: &'a str, referred_field_name: &'a str) -> Relation {
        let left_size = referred_relation.schema().len();
        let names: Vec<String> = self.schema().iter()
            .map(|f| f.name().to_string())
            .filter(|name| name != referred_field_name)//TODO remove this
            .collect();
        let join: Relation = Relation::join().inner()
            .on(Expr::eq(Expr::qcol(self.name(), referring_id), Expr::qcol(referred_relation.name(), referred_id)))
            .left(referred_relation)
            .right(self)
            .build();
        let left: Vec<_> = join.schema().iter().zip(join.input_fields()).take(left_size).collect();
        let right: Vec<_> = join.schema().iter().zip(join.input_fields()).skip(left_size).collect();
        Relation::map()
            .with_iter(left.into_iter().find_map(|(o, i)| {
                (referred_field==i.name()).then_some((referred_field_name, Expr::col(o.name())))
            }))
            .with_iter(right.into_iter().filter_map(|(o, i)| {
                names.contains(&i.name().to_string()).then_some((i.name(), Expr::col(o.name())))
            }))
            .input(join)
            .build()
    }

    /// Add a field designated with a "fiald path"
    pub fn with_field_path<'a>(self, relations: &'a Hierarchy<Rc<Relation>>, path: &'a [(&'a str, &'a str, &'a str)], referred_field: &'a str, referred_field_name: &'a str) -> Relation {
        if path.is_empty() {
            self.identity_with_field(referred_field_name, Expr::col(referred_field))
        } else {
            let path = Path::from_iter(path);
            let field_path = FieldPath::from_path(path, referred_field, referred_field_name);
            // Build the relation following the path to compute the new field
            field_path.into_iter().fold(self, |relation, ReferredField { referring_id, referred_relation, referred_id, referred_field, referred_field_name }|  {
                relation.with_referred_field(referring_id,
                    relations.get(&[referred_relation.to_string()]).unwrap().clone(),
                    referred_id,
                    referred_field,
                    referred_field_name
                )
            })
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
    use itertools::Itertools;
    use super::*;
    use crate::{
        relation::display,
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
    fn test_referred_field() {
        let database = postgresql::test_database();
        let relations = database.relations();
        let orders = Relation::try_from(parse("SELECT * FROM order_table").unwrap().with(&relations)).unwrap();
        let user = relations.get(&["user_table".to_string()]).unwrap().as_ref();
        let relation = orders.with_referred_field("user_id", Rc::new(user.clone()), "id", "id", "peid");
        assert!(relation.schema()[0].name()=="peid");
        let relation = relation.filter_fields(|n| n!="peid");
        assert!(relation.schema()[0].name()!="peid");
    }

    #[test]
    fn test_field_path() {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        // Link orders to users
        let orders = relations.get(&["order_table".to_string()]).unwrap().as_ref();
        let relation = orders.clone().with_field_path(&relations, &[("user_id", "user_table", "id")], "id", "peid");
        assert!(relation.schema()[0].name()=="peid");
        // Link items to orders
        let items = relations.get(&["item_table".to_string()]).unwrap().as_ref();
        let relation = items.clone().with_field_path(&relations, &[("order_id", "order_table", "id"), ("user_id", "user_table", "id")], "name", "peid");
        assert!(relation.schema()[0].name()=="peid");
        // Produce the query
        // display(&relation);
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