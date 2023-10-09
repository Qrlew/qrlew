//! For now a simple definition of Property

use std::{
    fmt,
    sync::Arc,
    ops::Deref,
    marker::PhantomData,
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
    Public,
    Published,
    ProtectedEntityPreserving,
    DifferentiallyPrivate,
}

impl fmt::Display for Property {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Property::Public => write!(f, "Public"),
            Property::Published => write!(f, "Published"),
            Property::ProtectedEntityPreserving => write!(f, "PEP"),
            Property::DifferentiallyPrivate => write!(f, "DP"),
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
            1 => write!(f, "{}→{}", self.inputs[0], self.output),
            _ => write!(f, "{}→{}", self.inputs.iter().join(", "), self.output),
        }
    }
}

pub type RelationWithRewritingRules<'a> = super::relation_with_attributes::RelationWithAttributes<'a, Vec<RewritingRule>>;

// TODO Write a rewrite method RelationWithRules -> Relation
// TODO Write a dot method RelationWithRules -> Dot

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

/// Implement Visitor for all SetRewritingRulesVisitors
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

/// A Visitor for the type Expr
pub trait MapRewritingRulesVisitor<'a> {
    fn table(&self, table: &'a Table, rewriting_rules: &'a[RewritingRule]) -> Vec<RewritingRule>;
    fn map(&self, map: &'a Map, rewriting_rules: &'a[RewritingRule], input: Arc<RelationWithRewritingRules<'a>>) -> Vec<RewritingRule>;
    fn reduce(&self, reduce: &'a Reduce, rewriting_rules: &'a[RewritingRule], input: Arc<RelationWithRewritingRules<'a>>) -> Vec<RewritingRule>;
    fn join(&self, join: &'a Join, rewriting_rules: &'a[RewritingRule], left: Arc<RelationWithRewritingRules<'a>>, right: Arc<RelationWithRewritingRules<'a>>) -> Vec<RewritingRule>;
    fn set(&self, set: &'a Set, rewriting_rules: &'a[RewritingRule], left: Arc<RelationWithRewritingRules<'a>>, right: Arc<RelationWithRewritingRules<'a>>) -> Vec<RewritingRule>;
    fn values(&self, values: &'a Values, rewriting_rules: &'a[RewritingRule]) -> Vec<RewritingRule>;
}

/// Implement a specific visitor to dispatch the dependencies more easily
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
        Arc::new(RelationWithAttributes::new(acceptor.relation(), rewriting_rules, acceptor.inputs().into_iter().cloned().collect()))
    }
}

impl<'a> RelationWithRewritingRules<'a> {
    /// Change rewriting rules
    pub fn map_rewriting_rules<M: MapRewritingRulesVisitor<'a>+'a>(&'a self, map_rewriting_rules_visitor: M) -> RelationWithRewritingRules<'a> {
        (*self.accept(map_rewriting_rules_visitor)).clone()
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

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

        for (p, r) in relations.iter() {
            println!("{} -> {r}", p.into_iter().join("."))
        }

        let query = parse(
            "SELECT order_id, sum(price) AS sum_price,
        count(price) AS count_price,
        avg(price) AS mean_price
        FROM item_table WHERE order_id IN (1,2,3,4,5,6,7,8,9,10) GROUP BY order_id",
        )
        .unwrap();
        let relation = Relation::try_from(query.with(&relations)).unwrap();
        relation.display_dot().unwrap();

        struct SimpleRewritingRules;
        impl<'a> SetRewritingRulesVisitor<'a> for SimpleRewritingRules {
            fn table(&self, table: &'a Table) -> Vec<RewritingRule> {
                vec![RewritingRule::new(vec![], Property::ProtectedEntityPreserving)]
            }

            fn map(&self, map: &'a Map, input: Arc<RelationWithRewritingRules<'a>>) -> Vec<RewritingRule> {
                vec![
                    RewritingRule::new(vec![Property::DifferentiallyPrivate], Property::Published),
                    RewritingRule::new(vec![Property::Published], Property::Published),
                    RewritingRule::new(vec![Property::Public], Property::Public),
                    RewritingRule::new(vec![Property::ProtectedEntityPreserving], Property::ProtectedEntityPreserving),
                ]
            }

            fn reduce(&self, reduce: &'a Reduce, input: Arc<RelationWithRewritingRules<'a>>) -> Vec<RewritingRule> {
                vec![
                    RewritingRule::new(vec![Property::Published], Property::Published),
                    RewritingRule::new(vec![Property::Public], Property::Public),
                    RewritingRule::new(vec![Property::ProtectedEntityPreserving], Property::DifferentiallyPrivate)
                ]
            }

            fn join(&self, join: &'a Join, left: Arc<RelationWithRewritingRules<'a>>, right: Arc<RelationWithRewritingRules<'a>>) -> Vec<RewritingRule> {
                vec![
                    RewritingRule::new(vec![Property::Published, Property::Published], Property::Published),
                    RewritingRule::new(vec![Property::Public, Property::Public], Property::Public),
                    RewritingRule::new(vec![Property::ProtectedEntityPreserving, Property::ProtectedEntityPreserving], Property::ProtectedEntityPreserving)
                ]
            }

            fn set(&self, set: &'a Set, left: Arc<RelationWithRewritingRules<'a>>, right: Arc<RelationWithRewritingRules<'a>>) -> Vec<RewritingRule> {
                vec![
                    RewritingRule::new(vec![Property::Published, Property::Published], Property::Published),
                    RewritingRule::new(vec![Property::Public, Property::Public], Property::Public),
                    RewritingRule::new(vec![Property::ProtectedEntityPreserving, Property::ProtectedEntityPreserving], Property::ProtectedEntityPreserving)
                ]
            }

            fn values(&self, values: &'a Values) -> Vec<RewritingRule> {
                vec![RewritingRule::new(vec![], Property::Public)]
            }
        }

        // Add rewritting rules
        let relation_with_rules = relation.set_rewriting_rules(SimpleRewritingRules);
        println!("{:#?}", relation_with_rules);
    }
}