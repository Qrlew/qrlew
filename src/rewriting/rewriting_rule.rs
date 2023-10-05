//! For now a simple definition of Property

use std::ops::Deref;

use crate::{
    relation::{Relation, Table, Map, Reduce, Join, Set, Values},
    visitor::{self, Visited},
};

/// A simple Property object to tag Relations properties
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum Property {
    Public,
    Published,
    ProtectedEntityPreserving,
    DifferentiallyPrivate,
}

/// Relation rewriting rule
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct RewritingRule {
    /// Property requirements on inputs
    inputs: Vec<Property>,
    /// Rewriting output property
    output: Property,
}

pub type RelationWithRewritingRules<'a> = super::relation_with::RelationWith<'a, Vec<RewritingRule>>;

// TODO Write a map method RelationWithRules -> RelationWithRules
// TODO Write a rewrite method RelationWithRules -> Relation
// TODO Write a dot method RelationWithRules -> Dot

// Visitors

/// A Visitor for the type Expr
pub trait Visitor<'a, T: Clone> {
    fn table(&self, table: &'a Table, rewriting_rules: Vec<RewritingRule>) -> T;
    fn map(&self, map: &'a Map, input: T, rewriting_rules: Vec<RewritingRule>) -> T;
    fn reduce(&self, reduce: &'a Reduce, input: T, rewriting_rules: Vec<RewritingRule>) -> T;
    fn join(&self, join: &'a Join, left: T, right: T, rewriting_rules: Vec<RewritingRule>) -> T;
    fn set(&self, set: &'a Set, left: T, right: T, rewriting_rules: Vec<RewritingRule>) -> T;
    fn values(&self, values: &'a Values, rewriting_rules: Vec<RewritingRule>) -> T;
}

/// Implement a specific visitor to dispatch the dependencies more easily
impl<'a, T: Clone, V: Visitor<'a, T>> visitor::Visitor<'a, RelationWithRewritingRules<'a>, T> for V {
    fn visit(&self, acceptor: &'a RelationWithRewritingRules<'a>, dependencies: Visited<'a, RelationWithRewritingRules<'a>, T>) -> T {
        match acceptor.relation() {
            Relation::Table(table) => self.table(table, acceptor.with().clone()),
            Relation::Map(map) => self.map(map, dependencies.get(acceptor.inputs()[0].as_ref()).clone(), acceptor.with().clone()),
            Relation::Reduce(reduce) => {
                self.reduce(reduce, dependencies.get(acceptor.inputs()[0].as_ref()).clone(), acceptor.with().clone())
            }
            Relation::Join(join) => self.join(
                join,
                dependencies.get(acceptor.inputs()[0].as_ref()).clone(),
                dependencies.get(acceptor.inputs()[1].as_ref()).clone(),
                acceptor.with().clone(),
            ),
            Relation::Set(set) => self.set(
                set,
                dependencies.get(acceptor.inputs()[0].as_ref()).clone(),
                dependencies.get(acceptor.inputs()[1].as_ref()).clone(),
                acceptor.with().clone(),
            ),
            Relation::Values(values) => self.values(values, acceptor.with().clone()),
        }
    }
}