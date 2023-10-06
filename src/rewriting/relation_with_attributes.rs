use std::{ops::Deref, sync::Arc, hash::Hash, fmt::Debug, marker::PhantomData};
use crate::{
    relation::Relation,
    visitor::{self, Acceptor, Dependencies, Visited},
};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct RelationWithAttributes<'a, Attributes> {
    relation: &'a Relation,
    attributes: Attributes,
    inputs: Vec<Arc<RelationWithAttributes<'a, Attributes>>>,
}

impl<'a, Attributes> RelationWithAttributes<'a, Attributes> {
    /// A basic builder
    pub fn new(relation: &'a Relation, attributes: Attributes, inputs: Vec<Arc<RelationWithAttributes<'a, Attributes>>>) -> Self {
        RelationWithAttributes {
            relation,
            attributes,
            inputs,
        }
    }
    /// Access relation read-only
    pub fn relation(&self) -> &Relation {
        &self.relation
    }
    /// Access attributes read-only
    pub fn attributes(&self) -> &Attributes {
        &self.attributes
    }
    /// Access attributes
    pub fn attributes_mut(&mut self) -> &mut Attributes {
        &mut self.attributes
    }
    /// Access attributes
    pub fn inputs(&self) -> &[Arc<RelationWithAttributes<'a, Attributes>>] {
        &self.inputs
    }
}

impl<'a, Attributes> Deref for RelationWithAttributes<'a, Attributes> {
    type Target = Relation;

    fn deref(&self) -> &Self::Target {
        &self.relation
    }
}

/// Create a Relation with clone Attributes
struct WithCloneAttributesVisitor<Attributes: Clone>(Attributes);

impl<'a, Attributes: Clone> visitor::Visitor<'a, Relation, Arc<RelationWithAttributes<'a, Attributes>>> for WithCloneAttributesVisitor<Attributes> {
    fn visit(&self, acceptor: &'a Relation, mut dependencies: Visited<'a, Relation, Arc<RelationWithAttributes<'a, Attributes>>>) -> Arc<RelationWithAttributes<'a, Attributes>> {
        Arc::new(RelationWithAttributes::new(acceptor, self.0.clone(), acceptor.inputs().into_iter().map(|r| dependencies.pop(r)).collect()))
    }
}

/// Create a Relation with default Attributes
struct WithDefaultAttributesVisitor<Attributes: Default>(PhantomData<Attributes>);

impl<'a, Attributes: Default> visitor::Visitor<'a, Relation, Arc<RelationWithAttributes<'a, Attributes>>> for WithDefaultAttributesVisitor<PhantomData<Attributes>> {
    fn visit(&self, acceptor: &'a Relation, mut dependencies: Visited<'a, Relation, Arc<RelationWithAttributes<'a, Attributes>>>) -> Arc<RelationWithAttributes<'a, Attributes>> {
        Arc::new(RelationWithAttributes::new(acceptor, Attributes::default(), acceptor.inputs().into_iter().map(|r| dependencies.pop(r)).collect()))
    }
}

impl Relation {
    /// Add attributes to Relation
    pub fn with_attributes<'a, Attributes: Clone>(&'a self, attributes: Attributes) -> RelationWithAttributes<'a, Attributes> {
        (*self.accept(WithCloneAttributesVisitor(attributes))).clone()
    }
    /// Add attributes to Relation
    pub fn with_default_attributes<'a, Attributes: Clone+Default>(&'a self) -> RelationWithAttributes<'a, Attributes> {
        (*self.accept(WithDefaultAttributesVisitor(PhantomData))).clone()
    }
}

// Implements Acceptor, Visitor and derive an iterator and a few other Visitor driven functions

/// Implement the Acceptor trait
impl<'a, Attributes: 'a+Clone+Debug+Hash+Eq> Acceptor<'a> for RelationWithAttributes<'a, Attributes> {
    fn dependencies(&'a self) -> Dependencies<'a, Self> {
        // A relation depends on its inputs
        self.inputs.iter().map(AsRef::as_ref).collect()
    }
}

impl<'a, Attributes> IntoIterator for &'a RelationWithAttributes<'a, Attributes> {
    type Item = &'a Relation;
    type IntoIter = visitor::Iter<'a, Relation>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
