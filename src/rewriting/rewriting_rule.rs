//! For now a simple definition of Property

use std::{
    sync::Arc,
    ops::Deref,
    marker::PhantomData,
};

use crate::{
    relation::{Relation, Table, Map, Reduce, Join, Set, Values},
    visitor::{Visitor, Dependencies, Visited, Acceptor},
    rewriting::relation_with_attributes::RelationWithAttributes,
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

pub type RelationWithRewritingRules<'a> = super::relation_with_attributes::RelationWithAttributes<'a, Vec<RewritingRule>>;



// TODO Write a map method Relation -> RelationWithRules
// TODO Write a map method RelationWithRules -> RelationWithRules
// TODO Write a rewrite method RelationWithRules -> Relation
// TODO Write a dot method RelationWithRules -> Dot

// Visitors

/// A Visitor to set RR
pub trait SetRewritingRules<'a> {
    fn table(&self, table: &'a Table) -> Vec<RewritingRule>;
    fn map(&self, map: &'a Map, input: &'a RelationWithRewritingRules<'a>) -> Vec<RewritingRule>;
    fn reduce(&self, reduce: &'a Reduce, input: &'a RelationWithRewritingRules<'a>) -> Vec<RewritingRule>;
    fn join(&self, join: &'a Join, left: &'a RelationWithRewritingRules<'a>, right: &'a RelationWithRewritingRules<'a>) -> Vec<RewritingRule>;
    fn set(&self, set: &'a Set, left: &'a RelationWithRewritingRules<'a>, right: &'a RelationWithRewritingRules<'a>) -> Vec<RewritingRule>;
    fn values(&self, values: &'a Values) -> Vec<RewritingRule>;
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct SetRewritingRulesVisitor<'a, S: SetRewritingRules<'a>>(S, PhantomData<&'a S>);

/// Implement Visitor for all SetRewritingRulesVisitors
impl<'a, S: SetRewritingRules<'a>> Visitor<'a, Relation, Arc<RelationWithRewritingRules<'a>>> for SetRewritingRulesVisitor<'a, S> {
    fn visit(&self, acceptor: &'a Relation, dependencies: Visited<'a, Relation, Arc<RelationWithRewritingRules<'a>>>) -> Arc<RelationWithRewritingRules<'a>> {
        let rewriting_rules = match acceptor {
            Relation::Table(table) => self.0.table(table),
            Relation::Map(map) => self.0.map(map, dependencies.get(map.input())),
            Relation::Reduce(reduce) => self.0.reduce(reduce, dependencies.get(reduce.input())),
            Relation::Join(join) => self.0.join(join, dependencies.get(join.left()), dependencies.get(join.right())),
            Relation::Set(set) => self.0.set(set, dependencies.get(set.left()), dependencies.get(set.right())),
            Relation::Values(values) => self.0.values(values),
        };
    }
}

/// A Visitor for the type Expr
pub trait MapRewritingRulesVisitor<'a> {
    fn table(&self, table: &'a Table, rewriting_rules: &'a[RewritingRule]) -> Vec<RewritingRule>;
    fn map(&self, map: &'a Map, rewriting_rules: &'a[RewritingRule], input: RelationWithRewritingRules<'a>) -> Vec<RewritingRule>;
    fn reduce(&self, reduce: &'a Reduce, rewriting_rules: &'a[RewritingRule], input: RelationWithRewritingRules<'a>) -> Vec<RewritingRule>;
    fn join(&self, join: &'a Join, rewriting_rules: &'a[RewritingRule], left: RelationWithRewritingRules<'a>, right: RelationWithRewritingRules<'a>) -> Vec<RewritingRule>;
    fn set(&self, set: &'a Set, rewriting_rules: &'a[RewritingRule], left: RelationWithRewritingRules<'a>, right: RelationWithRewritingRules<'a>) -> Vec<RewritingRule>;
    fn values(&self, values: &'a Values, rewriting_rules: &'a[RewritingRule]) -> Vec<RewritingRule>;
}

// /// Implement a specific visitor to dispatch the dependencies more easily
// impl<'a, V: MapRewritingRulesVisitor<'a>> Visitor<'a, RelationWithRewritingRules<'a>, RelationWithRewritingRules<'a>> for V {
//     // fn visit(&self, acceptor: &'a RelationWithRewritingRules<'a>, dependencies: Visited<'a, RelationWithRewritingRules<'a>, T>) -> T {
//     //     match acceptor.relation() {
//     //         Relation::Table(table) => self.table(table, acceptor.with().clone()),
//     //         Relation::Map(map) => self.map(map, dependencies.get(acceptor.inputs()[0].as_ref()).clone(), acceptor.with().clone()),
//     //         Relation::Reduce(reduce) => {
//     //             self.reduce(reduce, dependencies.get(acceptor.inputs()[0].as_ref()).clone(), acceptor.with().clone())
//     //         }
//     //         Relation::Join(join) => self.join(
//     //             join,
//     //             dependencies.get(acceptor.inputs()[0].as_ref()).clone(),
//     //             dependencies.get(acceptor.inputs()[1].as_ref()).clone(),
//     //             acceptor.with().clone(),
//     //         ),
//     //         Relation::Set(set) => self.set(
//     //             set,
//     //             dependencies.get(acceptor.inputs()[0].as_ref()).clone(),
//     //             dependencies.get(acceptor.inputs()[1].as_ref()).clone(),
//     //             acceptor.with().clone(),
//     //         ),
//     //         Relation::Values(values) => self.values(values, acceptor.with().clone()),
//     //     }
//     // }
// }


// /// Implement a specific visitor to dispatch the dependencies more easily
// impl<'a, T: Clone, V: Visitor<'a, T>> visitor::Visitor<'a, RelationWithRewritingRules<'a>, T> for V {
//     fn visit(&self, acceptor: &'a RelationWithRewritingRules<'a>, dependencies: Visited<'a, RelationWithRewritingRules<'a>, T>) -> T {
//         match acceptor.relation() {
//             Relation::Table(table) => self.table(table, acceptor.with().clone()),
//             Relation::Map(map) => self.map(map, dependencies.get(acceptor.inputs()[0].as_ref()).clone(), acceptor.with().clone()),
//             Relation::Reduce(reduce) => {
//                 self.reduce(reduce, dependencies.get(acceptor.inputs()[0].as_ref()).clone(), acceptor.with().clone())
//             }
//             Relation::Join(join) => self.join(
//                 join,
//                 dependencies.get(acceptor.inputs()[0].as_ref()).clone(),
//                 dependencies.get(acceptor.inputs()[1].as_ref()).clone(),
//                 acceptor.with().clone(),
//             ),
//             Relation::Set(set) => self.set(
//                 set,
//                 dependencies.get(acceptor.inputs()[0].as_ref()).clone(),
//                 dependencies.get(acceptor.inputs()[1].as_ref()).clone(),
//                 acceptor.with().clone(),
//             ),
//             Relation::Values(values) => self.values(values, acceptor.with().clone()),
//         }
//     }
// }

// Compute a RelationWithRules from a Relation
