use std::{ops::Deref, sync::Arc, hash::Hash, fmt::Debug, marker::PhantomData};
use crate::{
    relation::{Relation, Table, Map, Reduce, Join, Set, Values},
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

// Visitors

/// A Visitor to set RR
pub trait SetAttributesVisitor<'a, Attributes: 'a+Clone+Debug+Hash+Eq> {
    fn table(&self, table: &'a Table) -> Attributes;
    fn map(&self, map: &'a Map, input: Arc<RelationWithAttributes<'a, Attributes>>) -> Attributes;
    fn reduce(&self, reduce: &'a Reduce, input: Arc<RelationWithAttributes<'a, Attributes>>) -> Attributes;
    fn join(&self, join: &'a Join, left: Arc<RelationWithAttributes<'a, Attributes>>, right: Arc<RelationWithAttributes<'a, Attributes>>) -> Attributes;
    fn set(&self, set: &'a Set, left: Arc<RelationWithAttributes<'a, Attributes>>, right: Arc<RelationWithAttributes<'a, Attributes>>) -> Attributes;
    fn values(&self, values: &'a Values) -> Attributes;
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct SetAttributesVisitorWrapper<'a, Attributes: 'a+Clone+Debug+Hash+Eq, S: SetAttributesVisitor<'a, Attributes>>(S, PhantomData<(&'a Attributes, &'a S)>);

/// Implement the visitor trait
impl<'a, Attributes: 'a+Clone+Debug+Hash+Eq, S: SetAttributesVisitor<'a, Attributes>> visitor::Visitor<'a, Relation, Arc<RelationWithAttributes<'a, Attributes>>> for SetAttributesVisitorWrapper<'a, Attributes, S> {
    fn visit(&self, acceptor: &'a Relation, dependencies: Visited<'a, Relation, Arc<RelationWithAttributes<'a, Attributes>>>) -> Arc<RelationWithAttributes<'a, Attributes>> {
        let rewriting_rules = match acceptor {
            Relation::Table(table) => self.0.table(table),
            Relation::Map(map) => self.0.map(map, dependencies.get(map.input()).clone()),
            Relation::Reduce(reduce) => self.0.reduce(reduce, dependencies.get(reduce.input()).clone()),
            Relation::Join(join) => self.0.join(join, dependencies.get(join.left()).clone(), dependencies.get(join.right()).clone()),
            Relation::Set(set) => self.0.set(set, dependencies.get(set.left()).clone(), dependencies.get(set.right()).clone()),
            Relation::Values(values) => self.0.values(values),
        };
        let inputs: Vec<Arc<RelationWithAttributes<'a, Attributes>>> = acceptor.inputs().into_iter().map(|input| dependencies.get(input).clone()).collect();
        Arc::new(RelationWithAttributes::new(acceptor, rewriting_rules, inputs))
    }
}

impl Relation {
    /// Take a relation and set rewriting rules
    pub fn set_attributes<'a, Attributes: 'a+Clone+Debug+Hash+Eq, S: 'a+SetAttributesVisitor<'a, Attributes>>(&'a self, set_attributes_visitor: S) -> RelationWithAttributes<'a, Attributes> {
        (*self.accept(SetAttributesVisitorWrapper(set_attributes_visitor, PhantomData))).clone()
    }
}

/// A Visitor to update RRs
struct MapAttributesVisitor<'a, A: 'a+Clone+Debug, B: Clone, Map: Fn(&'a RelationWithAttributes<'a, A>)->B>(Map, PhantomData<(&'a A, B)>);

impl<'a, A: 'a+Clone+Debug+Hash+Eq, B: Clone, Map: Fn(&'a RelationWithAttributes<'a, A>)->B> visitor::Visitor<'a, RelationWithAttributes<'a, A>, Arc<RelationWithAttributes<'a, B>>> for MapAttributesVisitor<'a, A, B, Map> {
    fn visit(&self, acceptor: &'a RelationWithAttributes<'a, A>, mut dependencies: Visited<'a, RelationWithAttributes<'a, A>, Arc<RelationWithAttributes<'a, B>>>) -> Arc<RelationWithAttributes<'a, B>> {
        Arc::new(RelationWithAttributes::new(acceptor.relation(), self.0(acceptor), acceptor.inputs().into_iter().map(|r| dependencies.pop(r)).collect()))
    }
}

impl<'a, A: 'a+Clone+Debug+Hash+Eq> RelationWithAttributes<'a, A> {
    /// Add attributes to Relation
    pub fn map_attributes<B: Clone, Map: Fn(&'a RelationWithAttributes<'a, A>)->B>(&'a self, map: Map) -> RelationWithAttributes<'a, B> {
        (*self.accept(MapAttributesVisitor(map, PhantomData))).clone()
    }
}
