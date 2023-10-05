use std::{ops::Deref, sync::Arc, hash::Hash, fmt::Debug};
use crate::{
    relation::Relation,
    visitor::{self, Acceptor, Dependencies, Visited},
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
    /// Access relation read-only
    pub fn relation(&self) -> &Relation {
        &self.relation
    }
    /// Access attributes read-only
    pub fn with(&self) -> &Attributes {
        &self.with
    }
    /// Access attributes
    pub fn with_mut(&mut self) -> &mut Attributes {
        &mut self.with
    }
    /// Access attributes
    pub fn inputs(&self) -> &[Arc<RelationWith<'a, Attributes>>] {
        &self.inputs
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
