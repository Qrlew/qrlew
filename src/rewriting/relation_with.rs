use std::{ops::Deref, sync::Arc, hash::Hash, fmt::Debug};
use crate::{
    relation::{Relation, Table, Map, Reduce, Join, Set},
    visitor::{self, Visited, Acceptor, Dependencies},
    builder::With,
};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct RelationWith<'a, Attributes> {
    relation: &'a Relation,
    with: Attributes,
    inputs: Vec<Arc<RelationWith<'a, Attributes>>>,
}

impl<'a, Attributes> RelationWith<'a, Attributes> {
    /// A basic builder
    pub fn new(relation: &'a Relation, with: Attributes, inputs: Vec<Arc<RelationWith<'a, Attributes>>>) -> Self {
        RelationWith {
            relation,
            with,
            inputs,
        }
    }
    /// Access attribbutes read-only
    pub fn with(&self) -> &Attributes {
        &self.with
    }
    /// Access attribbutes
    pub fn with_mut(&mut self) -> &mut Attributes {
        &mut self.with
    }
}

impl<'a, Attributes> Deref for RelationWith<'a, Attributes> {
    type Target = Relation;

    fn deref(&self) -> &Self::Target {
        &self.relation
    }
}

/// Create a Relation with default Attributes
struct WithAttributesVisitor<Attributes: Clone>(Attributes);

impl<'a, Attributes: Clone> visitor::Visitor<'a, Relation, Arc<RelationWith<'a, Attributes>>> for WithAttributesVisitor<Attributes> {
    fn visit(&self, acceptor: &'a Relation, mut dependencies: Visited<'a, Relation, Arc<RelationWith<'a, Attributes>>>) -> Arc<RelationWith<'a, Attributes>> {
        Arc::new(RelationWith::new(acceptor, self.0.clone(), acceptor.inputs().into_iter().map(|r| dependencies.pop(r)).collect()))
    }
}

/// Create a Relation wrapper
impl<'a, Attributes: Clone> With<Attributes, RelationWith<'a, Attributes>> for &'a Relation {
    fn with(self, input: Attributes) -> RelationWith<'a, Attributes> {
        (*self.accept(WithAttributesVisitor(input))).clone()
    }
}

// Implements Acceptor, Visitor and derive an iterator and a few other Visitor driven functions

/// Implement the Acceptor trait
impl<'a, Attributes: 'a+Clone+Debug+Hash+Eq> Acceptor<'a> for RelationWith<'a, Attributes> {
    fn dependencies(&'a self) -> Dependencies<'a, Self> {
        // A relation depends on its inputs
        self.inputs.iter().map(AsRef::as_ref).collect()
    }
}

impl<'a, Attributes> IntoIterator for &'a RelationWith<'a, Attributes> {
    type Item = &'a Relation;
    type IntoIter = visitor::Iter<'a, Relation>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// Visitors

// /// A Visitor for the type Expr
// pub trait Visitor<'a, T: Clone> {
//     fn table(&self, table: &'a Table) -> T;
//     fn map(&self, map: &'a Map, input: T) -> T;
//     fn reduce(&self, reduce: &'a Reduce, input: T) -> T;
//     fn join(&self, join: &'a Join, left: T, right: T) -> T;
//     fn set(&self, set: &'a Set, left: T, right: T) -> T;
//     fn values(&self, values: &'a Values) -> T;
// }

// /// Implement a specific visitor to dispatch the dependencies more easily
// impl<'a, T: Clone, V: Visitor<'a, T>> visitor::Visitor<'a, Relation, T> for V {
//     fn visit(&self, acceptor: &'a Relation, dependencies: Visited<'a, Relation, T>) -> T {
//         match acceptor {
//             Relation::Table(table) => self.table(table),
//             Relation::Map(map) => self.map(map, dependencies.get(&map.input).clone()),
//             Relation::Reduce(reduce) => {
//                 self.reduce(reduce, dependencies.get(&reduce.input).clone())
//             }
//             Relation::Join(join) => self.join(
//                 join,
//                 dependencies.get(&join.left).clone(),
//                 dependencies.get(&join.right).clone(),
//             ),
//             Relation::Set(set) => self.set(
//                 set,
//                 dependencies.get(&set.left).clone(),
//                 dependencies.get(&set.right).clone(),
//             ),
//             Relation::Values(values) => self.values(values),
//         }
//     }
// }