//! For now a simple definition of Property
use std::{
    fmt,
    sync::Arc,
    ops::Deref,
    marker::PhantomData,
    collections::HashSet,
};

use itertools::Itertools;

use crate::{
    relation::{Relation, Table, Map, Reduce, Join, Set, Values},
    visitor::{Visitor, Dependencies, Visited, Acceptor},
    rewriting::relation_with_attributes::RelationWithAttributes,
};

/// A simple Property object to tag Relations properties
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum Property {
    Private,
    SyntheticData,
    ProtectedEntityPreserving,
    DifferentiallyPrivate,
    Published,
    Public,
}

impl fmt::Display for Property {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Property::Private => write!(f, "Priv"),
            Property::SyntheticData => write!(f, "SD"),
            Property::ProtectedEntityPreserving => write!(f, "PEP"),
            Property::DifferentiallyPrivate => write!(f, "DP"),
            Property::Published => write!(f, "Pubd"),
            Property::Public => write!(f, "Pub"),
        }
    }
}

/// Relation rewriting rule
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct RewritingRule {
    /// Property requirements on inputs
    inputs: Vec<Property>,
    /// Rewriting output property
    output: Property,
}

impl RewritingRule {
    pub fn new(inputs: Vec<Property>, output: Property) -> RewritingRule {
        RewritingRule {inputs, output}
    }
    /// Read inputs
    pub fn inputs(&self) -> &[Property] {
        &self.inputs
    }
    /// Read output
    pub fn output(&self) -> &Property {
        &self.output
    }
}

impl fmt::Display for RewritingRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.inputs.len() {
            0 => write!(f, "{}", self.output),
            1 => write!(f, "{} → {}", self.inputs[0], self.output),
            _ => write!(f, "{} → {}", self.inputs.iter().join(", "), self.output),
        }
    }
}

/// A Relation with rewriting rules attached
pub type RelationWithRewritingRules<'a> = super::relation_with_attributes::RelationWithAttributes<'a, Vec<RewritingRule>>;
/// A Relation with rewriting rules attached
pub type RelationWithRewritingRule<'a> = super::relation_with_attributes::RelationWithAttributes<'a, RewritingRule>;
/// A Relation with a proven property attached
pub type RelationWithProperty<'a> = super::relation_with_attributes::RelationWithAttributes<'a, Vec<RewritingRule>>;

// TODO Write a rewrite method RelationWithRules -> Relation

// Visitors

/// A Visitor to set RR
pub trait SetRewritingRulesVisitor<'a> {
    fn table(&self, table: &'a Table) -> Vec<RewritingRule>;
    fn map(&self, map: &'a Map, input: Arc<RelationWithRewritingRules<'a>>) -> Vec<RewritingRule>;
    fn reduce(&self, reduce: &'a Reduce, input: Arc<RelationWithRewritingRules<'a>>) -> Vec<RewritingRule>;
    fn join(&self, join: &'a Join, left: Arc<RelationWithRewritingRules<'a>>, right: Arc<RelationWithRewritingRules<'a>>) -> Vec<RewritingRule>;
    fn set(&self, set: &'a Set, left: Arc<RelationWithRewritingRules<'a>>, right: Arc<RelationWithRewritingRules<'a>>) -> Vec<RewritingRule>;
    fn values(&self, values: &'a Values) -> Vec<RewritingRule>;
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct SetRewritingRulesVisitorWrapper<'a, S: SetRewritingRulesVisitor<'a>>(S, PhantomData<&'a S>);

/// Implement the visitor trait
impl<'a, S: SetRewritingRulesVisitor<'a>> Visitor<'a, Relation, Arc<RelationWithRewritingRules<'a>>> for SetRewritingRulesVisitorWrapper<'a, S> {
    fn visit(&self, acceptor: &'a Relation, dependencies: Visited<'a, Relation, Arc<RelationWithRewritingRules<'a>>>) -> Arc<RelationWithRewritingRules<'a>> {
        let rewriting_rules = match acceptor {
            Relation::Table(table) => self.0.table(table),
            Relation::Map(map) => self.0.map(map, dependencies.get(map.input()).clone()),
            Relation::Reduce(reduce) => self.0.reduce(reduce, dependencies.get(reduce.input()).clone()),
            Relation::Join(join) => self.0.join(join, dependencies.get(join.left()).clone(), dependencies.get(join.right()).clone()),
            Relation::Set(set) => self.0.set(set, dependencies.get(set.left()).clone(), dependencies.get(set.right()).clone()),
            Relation::Values(values) => self.0.values(values),
        };
        let inputs: Vec<Arc<RelationWithRewritingRules<'a>>> = acceptor.inputs().into_iter().map(|input| dependencies.get(input).clone()).collect();
        Arc::new(RelationWithAttributes::new(acceptor, rewriting_rules, inputs))
    }
}

impl Relation {
    /// Take a relation and set rewriting rules
    pub fn set_rewriting_rules<'a, S: SetRewritingRulesVisitor<'a>+'a>(&'a self, set_rewriting_rules_visitor: S) -> RelationWithRewritingRules<'a> {
        (*self.accept(SetRewritingRulesVisitorWrapper(set_rewriting_rules_visitor, PhantomData))).clone()
    }
}

/// A Visitor to update RRs
pub trait MapRewritingRulesVisitor<'a> {
    fn table(&self, table: &'a Table, rewriting_rules: &'a[RewritingRule]) -> Vec<RewritingRule>;
    fn map(&self, map: &'a Map, rewriting_rules: &'a[RewritingRule], input: Arc<RelationWithRewritingRules<'a>>) -> Vec<RewritingRule>;
    fn reduce(&self, reduce: &'a Reduce, rewriting_rules: &'a[RewritingRule], input: Arc<RelationWithRewritingRules<'a>>) -> Vec<RewritingRule>;
    fn join(&self, join: &'a Join, rewriting_rules: &'a[RewritingRule], left: Arc<RelationWithRewritingRules<'a>>, right: Arc<RelationWithRewritingRules<'a>>) -> Vec<RewritingRule>;
    fn set(&self, set: &'a Set, rewriting_rules: &'a[RewritingRule], left: Arc<RelationWithRewritingRules<'a>>, right: Arc<RelationWithRewritingRules<'a>>) -> Vec<RewritingRule>;
    fn values(&self, values: &'a Values, rewriting_rules: &'a[RewritingRule]) -> Vec<RewritingRule>;
}

/// Implement the visitor trait
impl<'a, V: MapRewritingRulesVisitor<'a>> Visitor<'a, RelationWithRewritingRules<'a>, Arc<RelationWithRewritingRules<'a>>> for V {
    fn visit(&self, acceptor: &'a RelationWithRewritingRules<'a>, dependencies: Visited<'a, RelationWithRewritingRules<'a>, Arc<RelationWithRewritingRules<'a>>>) -> Arc<RelationWithRewritingRules<'a>> {
        let rewriting_rules = match acceptor.relation() {
            Relation::Table(table) => self.table(table, acceptor.attributes()),
            Relation::Map(map) => self.map(map, acceptor.attributes(), dependencies.get(acceptor.inputs()[0].deref()).clone()),
            Relation::Reduce(reduce) => self.reduce(reduce, acceptor.attributes(), dependencies.get(acceptor.inputs()[0].deref()).clone()),
            Relation::Join(join) => self.join(join, acceptor.attributes(), dependencies.get(acceptor.inputs()[0].deref()).clone(), dependencies.get(acceptor.inputs()[1].deref()).clone()),
            Relation::Set(set) => self.set(set, acceptor.attributes(), dependencies.get(acceptor.inputs()[0].deref()).clone(), dependencies.get(acceptor.inputs()[1].deref()).clone()),
            Relation::Values(values) => self.values(values, acceptor.attributes()),
        };
        let inputs: Vec<Arc<RelationWithRewritingRules<'a>>> = acceptor.inputs().into_iter().map(|input| dependencies.get(input).clone()).collect();
        Arc::new(RelationWithAttributes::new(acceptor.relation(), rewriting_rules, inputs))
    }
}

impl<'a> RelationWithRewritingRules<'a> {
    /// Change rewriting rules
    pub fn map_rewriting_rules<M: MapRewritingRulesVisitor<'a>+'a>(&'a self, map_rewriting_rules_visitor: M) -> RelationWithRewritingRules<'a> {
        (*self.accept(map_rewriting_rules_visitor)).clone()
    }
}

/// A Visitor to select one RR
pub trait SelectRewritingRuleVisitor<'a> {
    fn table(&self, table: &'a Table, rewriting_rules: &'a[RewritingRule]) -> RelationWithRewritingRule<'a>;
    fn map(&self, map: &'a Map, rewriting_rules: &'a[RewritingRule], input: RelationWithRewritingRule<'a>) -> RelationWithRewritingRule<'a>;
    fn reduce(&self, reduce: &'a Reduce, rewriting_rules: &'a[RewritingRule], input: RelationWithRewritingRule<'a>) -> RelationWithRewritingRule<'a>;
    fn join(&self, join: &'a Join, rewriting_rules: &'a[RewritingRule], left: RelationWithRewritingRule<'a>, right: RelationWithRewritingRule<'a>) -> RelationWithRewritingRule<'a>;
    fn set(&self, set: &'a Set, rewriting_rules: &'a[RewritingRule], left: RelationWithRewritingRule<'a>, right: RelationWithRewritingRule<'a>) -> RelationWithRewritingRule<'a>;
    fn values(&self, values: &'a Values, rewriting_rules: &'a[RewritingRule]) -> RelationWithRewritingRule<'a>;
}

/// Implement the visitor trait
impl<'a, V: SelectRewritingRuleVisitor<'a>> Visitor<'a, RelationWithRewritingRules<'a>, Vec<RelationWithRewritingRule<'a>>> for V {
    fn visit(&self, acceptor: &'a RelationWithRewritingRules<'a>, dependencies: Visited<'a, RelationWithRewritingRules<'a>, Vec<RelationWithRewritingRule<'a>>>) -> Vec<RelationWithRewritingRule<'a>> {
        match acceptor.relation() {
            Relation::Table(table) => vec![self.table(table, acceptor.attributes())],
            Relation::Map(map) => dependencies.get(acceptor.inputs()[0].deref()).into_iter()
                .map(|rwrr| self.map(map, acceptor.attributes(), rwrr.clone())).collect(),
            Relation::Reduce(reduce) => dependencies.get(acceptor.inputs()[0].deref()).into_iter()
                .map(|rwrr| self.reduce(reduce, acceptor.attributes(), rwrr.clone())).collect(),
            Relation::Join(join) => dependencies.get(acceptor.inputs()[0].deref()).into_iter()
                .flat_map(|left| dependencies.get(acceptor.inputs()[1].deref()).into_iter().map(move |right| (left, right)))
                .map(|(left, right)| self.join(join, acceptor.attributes(), left.clone(), right.clone())).collect(),
            Relation::Set(set) => dependencies.get(acceptor.inputs()[0].deref()).into_iter()
                .flat_map(|left| dependencies.get(acceptor.inputs()[1].deref()).into_iter().map(move |right| (left, right)))
                .map(|(left, right)| self.set(set, acceptor.attributes(), left.clone(), right.clone())).collect(),
            Relation::Values(values) => vec![self.values(values, acceptor.attributes())],
        }
    }
}

// # Implement various rewriting rules visitors

/// A basic rewriting rule setter
struct BaseRewritingRulesSetter;// TODO implement this properly

impl<'a> SetRewritingRulesVisitor<'a> for BaseRewritingRulesSetter {
    fn table(&self, table: &'a Table) -> Vec<RewritingRule> {
        vec![
            RewritingRule::new(vec![], Property::Private),
            RewritingRule::new(vec![], Property::SyntheticData),
            RewritingRule::new(vec![], Property::ProtectedEntityPreserving),
        ]
    }

    fn map(&self, map: &'a Map, input: Arc<RelationWithRewritingRules<'a>>) -> Vec<RewritingRule> {
        vec![
            RewritingRule::new(vec![Property::DifferentiallyPrivate], Property::Published),
            RewritingRule::new(vec![Property::Published], Property::Published),
            RewritingRule::new(vec![Property::Public], Property::Public),
            RewritingRule::new(vec![Property::ProtectedEntityPreserving], Property::ProtectedEntityPreserving),
            RewritingRule::new(vec![Property::SyntheticData], Property::SyntheticData),
        ]
    }

    fn reduce(&self, reduce: &'a Reduce, input: Arc<RelationWithRewritingRules<'a>>) -> Vec<RewritingRule> {
        vec![
            RewritingRule::new(vec![Property::Published], Property::Published),
            RewritingRule::new(vec![Property::Public], Property::Public),
            RewritingRule::new(vec![Property::ProtectedEntityPreserving], Property::DifferentiallyPrivate),
            RewritingRule::new(vec![Property::SyntheticData], Property::SyntheticData),
        ]
    }

    fn join(&self, join: &'a Join, left: Arc<RelationWithRewritingRules<'a>>, right: Arc<RelationWithRewritingRules<'a>>) -> Vec<RewritingRule> {
        vec![
            RewritingRule::new(vec![Property::Published, Property::Published], Property::Published),
            RewritingRule::new(vec![Property::Public, Property::Public], Property::Public),
            RewritingRule::new(vec![Property::ProtectedEntityPreserving, Property::ProtectedEntityPreserving], Property::ProtectedEntityPreserving),
            RewritingRule::new(vec![Property::SyntheticData, Property::SyntheticData], Property::SyntheticData),
        ]
    }

    fn set(&self, set: &'a Set, left: Arc<RelationWithRewritingRules<'a>>, right: Arc<RelationWithRewritingRules<'a>>) -> Vec<RewritingRule> {
        vec![
            RewritingRule::new(vec![Property::Published, Property::Published], Property::Published),
            RewritingRule::new(vec![Property::Public, Property::Public], Property::Public),
            RewritingRule::new(vec![Property::ProtectedEntityPreserving, Property::ProtectedEntityPreserving], Property::ProtectedEntityPreserving),
            RewritingRule::new(vec![Property::SyntheticData, Property::SyntheticData], Property::SyntheticData),
        ]
    }

    fn values(&self, values: &'a Values) -> Vec<RewritingRule> {
        vec![
            RewritingRule::new(vec![], Property::SyntheticData),
            RewritingRule::new(vec![], Property::Public),
        ]
    }
}

/// A basic rewriting rule eliminator
struct BaseRewritingRulesEliminator;// TODO implement this properly

impl<'a> MapRewritingRulesVisitor<'a> for BaseRewritingRulesEliminator {
    fn table(&self, table: &'a Table, rewriting_rules: &'a[RewritingRule]) -> Vec<RewritingRule> {
        rewriting_rules.into_iter().cloned().collect()
    }

    fn map(&self, map: &'a Map, rewriting_rules: &'a[RewritingRule], input: Arc<RelationWithRewritingRules<'a>>) -> Vec<RewritingRule> {
        let input_properties: HashSet<&Property> = input.attributes().into_iter().map(|rr| rr.output()).collect();
        println!("MAP {}", rewriting_rules.into_iter().filter(|rr| input_properties.contains(&rr.inputs()[0])).cloned().join(", "));
        rewriting_rules.into_iter().filter(|rr| input_properties.contains(&rr.inputs()[0])).cloned().collect()
    }

    fn reduce(&self, reduce: &'a Reduce, rewriting_rules: &'a[RewritingRule], input: Arc<RelationWithRewritingRules<'a>>) -> Vec<RewritingRule> {
        let input_properties: HashSet<&Property> = input.attributes().into_iter().map(|rr| rr.output()).collect();
        println!("REDUCE {}", rewriting_rules.into_iter().filter(|rr| input_properties.contains(&rr.inputs()[0])).cloned().join(", "));
        rewriting_rules.into_iter().filter(|rr| input_properties.contains(&rr.inputs()[0])).cloned().collect()
    }

    fn join(&self, join: &'a Join, rewriting_rules: &'a[RewritingRule], left: Arc<RelationWithRewritingRules<'a>>, right: Arc<RelationWithRewritingRules<'a>>) -> Vec<RewritingRule> {
        let left_properties: HashSet<&Property> = left.attributes().into_iter().map(|rr| rr.output()).collect();
        let right_properties: HashSet<&Property> = right.attributes().into_iter().map(|rr| rr.output()).collect();
        rewriting_rules.into_iter().filter(|rr| left_properties.contains(&rr.inputs()[0]) && right_properties.contains(&rr.inputs()[1])).cloned().collect()
    }

    fn set(&self, set: &'a Set, rewriting_rules: &'a[RewritingRule], left: Arc<RelationWithRewritingRules<'a>>, right: Arc<RelationWithRewritingRules<'a>>) -> Vec<RewritingRule> {
        let left_properties: HashSet<&Property> = left.attributes().into_iter().map(|rr| rr.output()).collect();
        let right_properties: HashSet<&Property> = right.attributes().into_iter().map(|rr| rr.output()).collect();
        rewriting_rules.into_iter().filter(|rr| left_properties.contains(&rr.inputs()[0]) && right_properties.contains(&rr.inputs()[1])).cloned().collect()
    }

    fn values(&self, values: &'a Values, rewriting_rules: &'a[RewritingRule]) -> Vec<RewritingRule> {
        rewriting_rules.into_iter().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ast,
        builder::With,
        display::Dot,
        io::{postgresql, Database},
        sql::parse,
        Relation,
    };

    #[test]
    fn test_set_rewriting_rules() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        // for (p, r) in relations.iter() {
        //     println!("{} -> {r}", p.into_iter().join("."))
        // }

        let query = parse(
            "SELECT order_id, sum(price) AS sum_price,
        count(price) AS count_price,
        avg(price) AS mean_price
        FROM item_table WHERE order_id IN (1,2,3,4,5,6,7,8,9,10) GROUP BY order_id",
        )
        .unwrap();
        let relation = Relation::try_from(query.with(&relations)).unwrap();
        relation.display_dot().unwrap();
        // Add rewritting rules
        let relation_with_rules = relation.set_rewriting_rules(BaseRewritingRulesSetter);
        relation_with_rules.display_dot().unwrap();
        let relation_with_rules = relation_with_rules.map_rewriting_rules(BaseRewritingRulesEliminator);
        relation_with_rules.display_dot().unwrap();
    }
}