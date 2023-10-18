//! For now a simple definition of Property
use std::{
    collections::HashSet,
    fmt::{self, Arguments},
    marker::PhantomData,
    ops::Deref,
    sync::Arc,
};

use itertools::Itertools;

use crate::{
    builder::{Ready, With},
    differential_privacy::budget::Budget,
    hierarchy::Hierarchy,
    protection::{protected_entity::ProtectedEntity, Protection},
    relation::{Join, Map, Reduce, Relation, Set, Table, Values, Variant as _},
    rewriting::relation_with_attributes::RelationWithAttributes,
    visitor::{Acceptor, Dependencies, Visited, Visitor},
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

/// Possible parameters for RewritingRules
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Parameters {
    None,
    Budget(Budget),
    ProtectedEntity(ProtectedEntity),
}

/// Relation rewriting rule
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct RewritingRule {
    /// Property requirements on inputs
    inputs: Vec<Property>,
    /// Rewriting output property
    output: Property,
    /// Parameters
    parameters: Parameters,
}

impl RewritingRule {
    pub fn new(inputs: Vec<Property>, output: Property, parameters: Parameters) -> RewritingRule {
        RewritingRule {
            inputs,
            output,
            parameters,
        }
    }
    /// Read inputs
    pub fn inputs(&self) -> &[Property] {
        &self.inputs
    }
    /// Read output
    pub fn output(&self) -> &Property {
        &self.output
    }
    /// Read output
    pub fn parameters(&self) -> &Parameters {
        &self.parameters
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
pub type RelationWithRewritingRules<'a> =
    super::relation_with_attributes::RelationWithAttributes<'a, Vec<RewritingRule>>;
/// A Relation with rewriting rules attached
pub type RelationWithRewritingRule<'a> =
    super::relation_with_attributes::RelationWithAttributes<'a, RewritingRule>;
/// A Relation with a proven property attached
pub type RelationWithProperty<'a> =
    super::relation_with_attributes::RelationWithAttributes<'a, Vec<RewritingRule>>;

// TODO Write a rewrite method RelationWithRules -> Relation

// Visitors

/// A Visitor to set RR
pub trait SetRewritingRulesVisitor<'a> {
    fn table(&self, table: &'a Table) -> Vec<RewritingRule>;
    fn map(&self, map: &'a Map, input: Arc<RelationWithRewritingRules<'a>>) -> Vec<RewritingRule>;
    fn reduce(
        &self,
        reduce: &'a Reduce,
        input: Arc<RelationWithRewritingRules<'a>>,
    ) -> Vec<RewritingRule>;
    fn join(
        &self,
        join: &'a Join,
        left: Arc<RelationWithRewritingRules<'a>>,
        right: Arc<RelationWithRewritingRules<'a>>,
    ) -> Vec<RewritingRule>;
    fn set(
        &self,
        set: &'a Set,
        left: Arc<RelationWithRewritingRules<'a>>,
        right: Arc<RelationWithRewritingRules<'a>>,
    ) -> Vec<RewritingRule>;
    fn values(&self, values: &'a Values) -> Vec<RewritingRule>;
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct SetRewritingRulesVisitorWrapper<'a, S: SetRewritingRulesVisitor<'a>>(S, PhantomData<&'a S>);
/// Implement the visitor trait
impl<'a, S: SetRewritingRulesVisitor<'a>> Visitor<'a, Relation, Arc<RelationWithRewritingRules<'a>>>
    for SetRewritingRulesVisitorWrapper<'a, S>
{
    fn visit(
        &self,
        acceptor: &'a Relation,
        dependencies: Visited<'a, Relation, Arc<RelationWithRewritingRules<'a>>>,
    ) -> Arc<RelationWithRewritingRules<'a>> {
        let rewriting_rules = match acceptor {
            Relation::Table(table) => self.0.table(table),
            Relation::Map(map) => self.0.map(map, dependencies.get(map.input()).clone()),
            Relation::Reduce(reduce) => self
                .0
                .reduce(reduce, dependencies.get(reduce.input()).clone()),
            Relation::Join(join) => self.0.join(
                join,
                dependencies.get(join.left()).clone(),
                dependencies.get(join.right()).clone(),
            ),
            Relation::Set(set) => self.0.set(
                set,
                dependencies.get(set.left()).clone(),
                dependencies.get(set.right()).clone(),
            ),
            Relation::Values(values) => self.0.values(values),
        };
        let inputs: Vec<Arc<RelationWithRewritingRules<'a>>> = acceptor
            .inputs()
            .into_iter()
            .map(|input| dependencies.get(input).clone())
            .collect();
        Arc::new(RelationWithAttributes::new(
            acceptor,
            rewriting_rules,
            inputs,
        ))
    }
}

impl Relation {
    /// Take a relation and set rewriting rules
    pub fn set_rewriting_rules<'a, S: 'a + SetRewritingRulesVisitor<'a>>(
        &'a self,
        set_rewriting_rules_visitor: S,
    ) -> RelationWithRewritingRules<'a> {
        (*self.accept(SetRewritingRulesVisitorWrapper(
            set_rewriting_rules_visitor,
            PhantomData,
        )))
        .clone()
    }
}

/// A Visitor to update RRs
pub trait MapRewritingRulesVisitor<'a> {
    fn table(&self, table: &'a Table, rewriting_rules: &'a [RewritingRule]) -> Vec<RewritingRule>;
    fn map(
        &self,
        map: &'a Map,
        rewriting_rules: &'a [RewritingRule],
        input: Arc<RelationWithRewritingRules<'a>>,
    ) -> Vec<RewritingRule>;
    fn reduce(
        &self,
        reduce: &'a Reduce,
        rewriting_rules: &'a [RewritingRule],
        input: Arc<RelationWithRewritingRules<'a>>,
    ) -> Vec<RewritingRule>;
    fn join(
        &self,
        join: &'a Join,
        rewriting_rules: &'a [RewritingRule],
        left: Arc<RelationWithRewritingRules<'a>>,
        right: Arc<RelationWithRewritingRules<'a>>,
    ) -> Vec<RewritingRule>;
    fn set(
        &self,
        set: &'a Set,
        rewriting_rules: &'a [RewritingRule],
        left: Arc<RelationWithRewritingRules<'a>>,
        right: Arc<RelationWithRewritingRules<'a>>,
    ) -> Vec<RewritingRule>;
    fn values(
        &self,
        values: &'a Values,
        rewriting_rules: &'a [RewritingRule],
    ) -> Vec<RewritingRule>;
}
/// Implement the visitor trait
impl<'a, V: MapRewritingRulesVisitor<'a>>
    Visitor<'a, RelationWithRewritingRules<'a>, Arc<RelationWithRewritingRules<'a>>> for V
{
    fn visit(
        &self,
        acceptor: &'a RelationWithRewritingRules<'a>,
        dependencies: Visited<
            'a,
            RelationWithRewritingRules<'a>,
            Arc<RelationWithRewritingRules<'a>>,
        >,
    ) -> Arc<RelationWithRewritingRules<'a>> {
        let rewriting_rules = match acceptor.relation() {
            Relation::Table(table) => self.table(table, acceptor.attributes()),
            Relation::Map(map) => self.map(
                map,
                acceptor.attributes(),
                dependencies.get(acceptor.inputs()[0].deref()).clone(),
            ),
            Relation::Reduce(reduce) => self.reduce(
                reduce,
                acceptor.attributes(),
                dependencies.get(acceptor.inputs()[0].deref()).clone(),
            ),
            Relation::Join(join) => self.join(
                join,
                acceptor.attributes(),
                dependencies.get(acceptor.inputs()[0].deref()).clone(),
                dependencies.get(acceptor.inputs()[1].deref()).clone(),
            ),
            Relation::Set(set) => self.set(
                set,
                acceptor.attributes(),
                dependencies.get(acceptor.inputs()[0].deref()).clone(),
                dependencies.get(acceptor.inputs()[1].deref()).clone(),
            ),
            Relation::Values(values) => self.values(values, acceptor.attributes()),
        };
        let inputs: Vec<Arc<RelationWithRewritingRules<'a>>> = acceptor
            .inputs()
            .into_iter()
            .map(|input| dependencies.get(input).clone())
            .collect();
        Arc::new(RelationWithAttributes::new(
            acceptor.relation(),
            rewriting_rules,
            inputs,
        ))
    }
}

impl<'a> RelationWithRewritingRules<'a> {
    /// Change rewriting rules
    pub fn map_rewriting_rules<M: MapRewritingRulesVisitor<'a> + 'a>(
        &'a self,
        map_rewriting_rules_visitor: M,
    ) -> RelationWithRewritingRules<'a> {
        (*self.accept(map_rewriting_rules_visitor)).clone()
    }
}

/// A Visitor to select one RR
pub trait SelectRewritingRuleVisitor<'a> {
    fn table(&self, table: &'a Table, rewriting_rules: &'a [RewritingRule]) -> Vec<RewritingRule>;
    fn map(
        &self,
        map: &'a Map,
        rewriting_rules: &'a [RewritingRule],
        input: &RelationWithRewritingRule<'a>,
    ) -> Vec<RewritingRule>;
    fn reduce(
        &self,
        reduce: &'a Reduce,
        rewriting_rules: &'a [RewritingRule],
        input: &RelationWithRewritingRule<'a>,
    ) -> Vec<RewritingRule>;
    fn join(
        &self,
        join: &'a Join,
        rewriting_rules: &'a [RewritingRule],
        left: &RelationWithRewritingRule<'a>,
        right: &RelationWithRewritingRule<'a>,
    ) -> Vec<RewritingRule>;
    fn set(
        &self,
        set: &'a Set,
        rewriting_rules: &'a [RewritingRule],
        left: &RelationWithRewritingRule<'a>,
        right: &RelationWithRewritingRule<'a>,
    ) -> Vec<RewritingRule>;
    fn values(
        &self,
        values: &'a Values,
        rewriting_rules: &'a [RewritingRule],
    ) -> Vec<RewritingRule>;
}
/// Implement the visitor trait
impl<'a, V: SelectRewritingRuleVisitor<'a>>
    Visitor<'a, RelationWithRewritingRules<'a>, Vec<Arc<RelationWithRewritingRule<'a>>>> for V
{
    fn visit(
        &self,
        acceptor: &'a RelationWithRewritingRules<'a>,
        dependencies: Visited<
            'a,
            RelationWithRewritingRules<'a>,
            Vec<Arc<RelationWithRewritingRule<'a>>>,
        >,
    ) -> Vec<Arc<RelationWithRewritingRule<'a>>> {
        match acceptor.relation() {
            Relation::Table(table) => self
                .table(table, acceptor.attributes())
                .into_iter()
                .map(|rr| {
                    Arc::new(RelationWithRewritingRule::new(
                        acceptor.relation(),
                        rr,
                        vec![],
                    ))
                })
                .collect(),
            Relation::Map(map) => dependencies
                .get(acceptor.inputs()[0].deref())
                .into_iter()
                .flat_map(|input| {
                    self.map(map, acceptor.attributes(), input.deref())
                        .into_iter()
                        .map(|rr| {
                            Arc::new(RelationWithRewritingRule::new(
                                acceptor.relation(),
                                rr,
                                vec![input.clone()],
                            ))
                        })
                })
                .collect(),
            Relation::Reduce(reduce) => dependencies
                .get(acceptor.inputs()[0].deref())
                .into_iter()
                .flat_map(|input| {
                    self.reduce(reduce, acceptor.attributes(), input.deref())
                        .into_iter()
                        .map(|rr| {
                            Arc::new(RelationWithRewritingRule::new(
                                acceptor.relation(),
                                rr,
                                vec![input.clone()],
                            ))
                        })
                })
                .collect(),
            Relation::Join(join) => dependencies
                .get(acceptor.inputs()[0].deref())
                .into_iter()
                .flat_map(|left| {
                    dependencies
                        .get(acceptor.inputs()[1].deref())
                        .into_iter()
                        .map(move |right| (left, right))
                })
                .flat_map(|(left, right)| {
                    self.join(join, acceptor.attributes(), left.deref(), right.deref())
                        .into_iter()
                        .map(|rr| {
                            Arc::new(RelationWithRewritingRule::new(
                                acceptor.relation(),
                                rr,
                                vec![left.clone(), right.clone()],
                            ))
                        })
                })
                .collect(),
            Relation::Set(set) => dependencies
                .get(acceptor.inputs()[0].deref())
                .into_iter()
                .flat_map(|left| {
                    dependencies
                        .get(acceptor.inputs()[1].deref())
                        .into_iter()
                        .map(move |right| (left, right))
                })
                .flat_map(|(left, right)| {
                    self.set(set, acceptor.attributes(), left.deref(), right.deref())
                        .into_iter()
                        .map(|rr| {
                            Arc::new(RelationWithRewritingRule::new(
                                acceptor.relation(),
                                rr,
                                vec![left.clone(), right.clone()],
                            ))
                        })
                })
                .collect(),
            Relation::Values(values) => self
                .values(values, acceptor.attributes())
                .into_iter()
                .map(|rr| {
                    Arc::new(RelationWithRewritingRule::new(
                        acceptor.relation(),
                        rr,
                        vec![],
                    ))
                })
                .collect(),
        }
    }
}

impl<'a> RelationWithRewritingRules<'a> {
    /// Change rewriting rules
    pub fn select_rewriting_rules<S: SelectRewritingRuleVisitor<'a> + 'a>(
        &'a self,
        select_rewriting_rules_visitor: S,
    ) -> Vec<RelationWithRewritingRule<'a>> {
        self.accept(select_rewriting_rules_visitor)
            .into_iter()
            .map(|rwrr| (*rwrr).clone())
            .collect()
    }
}
/// Implement the visitor trait
impl<'a> From<&'a RelationWithRewritingRule<'a>> for RelationWithRewritingRules<'a> {
    fn from(value: &'a RelationWithRewritingRule<'a>) -> Self {
        value.map_attributes(|rwrr| vec![rwrr.attributes().clone()])
    }
}

/// A Visitor to rewrite a RelationWithRewritingRule
pub trait RewriteVisitor<'a> {
    fn table(&self, table: &'a Table, rewriting_rule: &'a RewritingRule) -> Arc<Relation>;
    fn map(
        &self,
        map: &'a Map,
        rewriting_rule: &'a RewritingRule,
        rewritten_input: Arc<Relation>,
    ) -> Arc<Relation>;
    fn reduce(
        &self,
        reduce: &'a Reduce,
        rewriting_rule: &'a RewritingRule,
        rewritten_input: Arc<Relation>,
    ) -> Arc<Relation>;
    fn join(
        &self,
        join: &'a Join,
        rewriting_rule: &'a RewritingRule,
        rewritten_left: Arc<Relation>,
        rewritten_right: Arc<Relation>,
    ) -> Arc<Relation>;
    fn set(
        &self,
        set: &'a Set,
        rewriting_rule: &'a RewritingRule,
        rewritten_left: Arc<Relation>,
        rewritten_right: Arc<Relation>,
    ) -> Arc<Relation>;
    fn values(&self, values: &'a Values, rewriting_rule: &'a RewritingRule) -> Arc<Relation>;
}
/// Implement the visitor trait
impl<'a, V: RewriteVisitor<'a>> Visitor<'a, RelationWithRewritingRule<'a>, Arc<Relation>> for V {
    fn visit(
        &self,
        acceptor: &'a RelationWithRewritingRule<'a>,
        dependencies: Visited<'a, RelationWithRewritingRule<'a>, Arc<Relation>>,
    ) -> Arc<Relation> {
        match acceptor.relation() {
            Relation::Table(table) => self.table(table, acceptor.attributes()),
            Relation::Map(map) => self.map(
                map,
                acceptor.attributes(),
                dependencies.get(acceptor.inputs()[0].deref()).clone(),
            ),
            Relation::Reduce(reduce) => self.reduce(
                reduce,
                acceptor.attributes(),
                dependencies.get(acceptor.inputs()[0].deref()).clone(),
            ),
            Relation::Join(join) => self.join(
                join,
                acceptor.attributes(),
                dependencies.get(acceptor.inputs()[0].deref()).clone(),
                dependencies.get(acceptor.inputs()[1].deref()).clone(),
            ),
            Relation::Set(set) => self.set(
                set,
                acceptor.attributes(),
                dependencies.get(acceptor.inputs()[0].deref()).clone(),
                dependencies.get(acceptor.inputs()[1].deref()).clone(),
            ),
            Relation::Values(values) => self.values(values, acceptor.attributes()),
        }
    }
}

impl<'a> RelationWithRewritingRule<'a> {
    /// Rewrite the RelationWithRewritingRule
    pub fn rewrite<V: RewriteVisitor<'a>>(&'a self, rewrite_visitor: V) -> Relation {
        (*self.accept(rewrite_visitor)).clone()
    }
}

// # Implement various rewriting rules visitors

/// A basic rewriting rule setter
struct BaseRewritingRulesSetter {
    protected_entity: ProtectedEntity,
    budget: Budget,
}
// TODO implement this properly

impl BaseRewritingRulesSetter {
    pub fn new(protected_entity: ProtectedEntity, budget: Budget) -> BaseRewritingRulesSetter {
        BaseRewritingRulesSetter {
            protected_entity,
            budget,
        }
    }
}

impl<'a> SetRewritingRulesVisitor<'a> for BaseRewritingRulesSetter {
    fn table(&self, table: &'a Table) -> Vec<RewritingRule> {
        vec![
            RewritingRule::new(vec![], Property::Private, Parameters::None),
            RewritingRule::new(vec![], Property::SyntheticData, Parameters::None),
            RewritingRule::new(
                vec![],
                Property::ProtectedEntityPreserving,
                Parameters::ProtectedEntity(self.protected_entity.clone()),
            ),
        ]
    }

    fn map(&self, map: &'a Map, input: Arc<RelationWithRewritingRules<'a>>) -> Vec<RewritingRule> {
        vec![
            RewritingRule::new(vec![Property::Public], Property::Public, Parameters::None),
            RewritingRule::new(
                vec![Property::Published],
                Property::Published,
                Parameters::None,
            ),
            RewritingRule::new(
                vec![Property::DifferentiallyPrivate],
                Property::Published,
                Parameters::None,
            ),
            RewritingRule::new(
                vec![Property::ProtectedEntityPreserving],
                Property::ProtectedEntityPreserving,
                Parameters::None,
            ),
            RewritingRule::new(
                vec![Property::SyntheticData],
                Property::SyntheticData,
                Parameters::None,
            ),
        ]
    }

    fn reduce(
        &self,
        reduce: &'a Reduce,
        input: Arc<RelationWithRewritingRules<'a>>,
    ) -> Vec<RewritingRule> {
        vec![
            RewritingRule::new(vec![Property::Public], Property::Public, Parameters::None),
            RewritingRule::new(
                vec![Property::Published],
                Property::Published,
                Parameters::None,
            ),
            RewritingRule::new(vec![Property::Public], Property::Public, Parameters::None),
            RewritingRule::new(
                vec![Property::ProtectedEntityPreserving],
                Property::DifferentiallyPrivate,
                Parameters::None,
            ),
            RewritingRule::new(
                vec![Property::SyntheticData],
                Property::SyntheticData,
                Parameters::None,
            ),
            RewritingRule::new(
                vec![Property::SyntheticData],
                Property::Published,
                Parameters::None,
            ),
        ]
    }

    fn join(
        &self,
        join: &'a Join,
        left: Arc<RelationWithRewritingRules<'a>>,
        right: Arc<RelationWithRewritingRules<'a>>,
    ) -> Vec<RewritingRule> {
        vec![
            RewritingRule::new(
                vec![Property::Public, Property::Public],
                Property::Public,
                Parameters::None,
            ),
            RewritingRule::new(
                vec![Property::Published, Property::Published],
                Property::Published,
                Parameters::None,
            ),
            RewritingRule::new(
                vec![Property::Published, Property::ProtectedEntityPreserving],
                Property::ProtectedEntityPreserving,
                Parameters::None,
            ),
            RewritingRule::new(
                vec![Property::ProtectedEntityPreserving, Property::Published],
                Property::ProtectedEntityPreserving,
                Parameters::None,
            ),
            RewritingRule::new(
                vec![
                    Property::DifferentiallyPrivate,
                    Property::ProtectedEntityPreserving,
                ],
                Property::ProtectedEntityPreserving,
                Parameters::None,
            ),
            RewritingRule::new(
                vec![
                    Property::ProtectedEntityPreserving,
                    Property::DifferentiallyPrivate,
                ],
                Property::ProtectedEntityPreserving,
                Parameters::None,
            ),
            RewritingRule::new(
                vec![
                    Property::ProtectedEntityPreserving,
                    Property::ProtectedEntityPreserving,
                ],
                Property::ProtectedEntityPreserving,
                Parameters::None,
            ),
            RewritingRule::new(
                vec![Property::SyntheticData, Property::SyntheticData],
                Property::SyntheticData,
                Parameters::None,
            ),
        ]
    }

    fn set(
        &self,
        set: &'a Set,
        left: Arc<RelationWithRewritingRules<'a>>,
        right: Arc<RelationWithRewritingRules<'a>>,
    ) -> Vec<RewritingRule> {
        vec![
            RewritingRule::new(
                vec![Property::Public, Property::Public],
                Property::Public,
                Parameters::None,
            ),
            RewritingRule::new(
                vec![Property::Published, Property::Published],
                Property::Published,
                Parameters::None,
            ),
            RewritingRule::new(
                vec![
                    Property::ProtectedEntityPreserving,
                    Property::ProtectedEntityPreserving,
                ],
                Property::ProtectedEntityPreserving,
                Parameters::None,
            ),
            RewritingRule::new(
                vec![Property::SyntheticData, Property::SyntheticData],
                Property::SyntheticData,
                Parameters::None,
            ),
        ]
    }

    fn values(&self, values: &'a Values) -> Vec<RewritingRule> {
        vec![
            RewritingRule::new(vec![], Property::SyntheticData, Parameters::None),
            RewritingRule::new(vec![], Property::Public, Parameters::None),
        ]
    }
}

/// A basic rewriting rule eliminator
struct BaseRewritingRulesEliminator; // TODO implement this properly

impl<'a> MapRewritingRulesVisitor<'a> for BaseRewritingRulesEliminator {
    fn table(&self, table: &'a Table, rewriting_rules: &'a [RewritingRule]) -> Vec<RewritingRule> {
        rewriting_rules.into_iter().cloned().collect()
    }

    fn map(
        &self,
        map: &'a Map,
        rewriting_rules: &'a [RewritingRule],
        input: Arc<RelationWithRewritingRules<'a>>,
    ) -> Vec<RewritingRule> {
        let input_properties: HashSet<&Property> = input
            .attributes()
            .into_iter()
            .map(|rr| rr.output())
            .collect();
        rewriting_rules
            .into_iter()
            .filter(|rr| input_properties.contains(&rr.inputs()[0]))
            .cloned()
            .collect()
    }

    fn reduce(
        &self,
        reduce: &'a Reduce,
        rewriting_rules: &'a [RewritingRule],
        input: Arc<RelationWithRewritingRules<'a>>,
    ) -> Vec<RewritingRule> {
        let input_properties: HashSet<&Property> = input
            .attributes()
            .into_iter()
            .map(|rr| rr.output())
            .collect();
        rewriting_rules
            .into_iter()
            .filter(|rr| input_properties.contains(&rr.inputs()[0]))
            .cloned()
            .collect()
    }

    fn join(
        &self,
        join: &'a Join,
        rewriting_rules: &'a [RewritingRule],
        left: Arc<RelationWithRewritingRules<'a>>,
        right: Arc<RelationWithRewritingRules<'a>>,
    ) -> Vec<RewritingRule> {
        let left_properties: HashSet<&Property> = left
            .attributes()
            .into_iter()
            .map(|rr| rr.output())
            .collect();
        let right_properties: HashSet<&Property> = right
            .attributes()
            .into_iter()
            .map(|rr| rr.output())
            .collect();
        rewriting_rules
            .into_iter()
            .filter(|rr| {
                left_properties.contains(&rr.inputs()[0])
                    && right_properties.contains(&rr.inputs()[1])
            })
            .cloned()
            .collect()
    }

    fn set(
        &self,
        set: &'a Set,
        rewriting_rules: &'a [RewritingRule],
        left: Arc<RelationWithRewritingRules<'a>>,
        right: Arc<RelationWithRewritingRules<'a>>,
    ) -> Vec<RewritingRule> {
        let left_properties: HashSet<&Property> = left
            .attributes()
            .into_iter()
            .map(|rr| rr.output())
            .collect();
        let right_properties: HashSet<&Property> = right
            .attributes()
            .into_iter()
            .map(|rr| rr.output())
            .collect();
        rewriting_rules
            .into_iter()
            .filter(|rr| {
                left_properties.contains(&rr.inputs()[0])
                    && right_properties.contains(&rr.inputs()[1])
            })
            .cloned()
            .collect()
    }

    fn values(
        &self,
        values: &'a Values,
        rewriting_rules: &'a [RewritingRule],
    ) -> Vec<RewritingRule> {
        rewriting_rules.into_iter().cloned().collect()
    }
}

/// A basic rewriting rule selector
struct BaseRewritingRulesSelector; // TODO implement this properly

impl<'a> SelectRewritingRuleVisitor<'a> for BaseRewritingRulesSelector {
    fn table(&self, table: &'a Table, rewriting_rules: &'a [RewritingRule]) -> Vec<RewritingRule> {
        rewriting_rules.into_iter().cloned().collect()
    }

    fn map(
        &self,
        map: &'a Map,
        rewriting_rules: &'a [RewritingRule],
        input: &RelationWithRewritingRule<'a>,
    ) -> Vec<RewritingRule> {
        rewriting_rules
            .into_iter()
            .filter(|rr| rr.inputs()[0] == *input.attributes().output())
            .cloned()
            .collect()
    }

    fn reduce(
        &self,
        reduce: &'a Reduce,
        rewriting_rules: &'a [RewritingRule],
        input: &RelationWithRewritingRule<'a>,
    ) -> Vec<RewritingRule> {
        rewriting_rules
            .into_iter()
            .filter(|rr| rr.inputs()[0] == *input.attributes().output())
            .cloned()
            .collect()
    }

    fn join(
        &self,
        join: &'a Join,
        rewriting_rules: &'a [RewritingRule],
        left: &RelationWithRewritingRule<'a>,
        right: &RelationWithRewritingRule<'a>,
    ) -> Vec<RewritingRule> {
        rewriting_rules
            .into_iter()
            .filter(|rr| {
                rr.inputs()[0] == *left.attributes().output()
                    && rr.inputs()[1] == *right.attributes().output()
            })
            .cloned()
            .collect()
    }

    fn set(
        &self,
        set: &'a Set,
        rewriting_rules: &'a [RewritingRule],
        left: &RelationWithRewritingRule<'a>,
        right: &RelationWithRewritingRule<'a>,
    ) -> Vec<RewritingRule> {
        rewriting_rules
            .into_iter()
            .filter(|rr| {
                rr.inputs()[0] == *left.attributes().output()
                    && rr.inputs()[1] == *right.attributes().output()
            })
            .cloned()
            .collect()
    }

    fn values(
        &self,
        values: &'a Values,
        rewriting_rules: &'a [RewritingRule],
    ) -> Vec<RewritingRule> {
        rewriting_rules.into_iter().cloned().collect()
    }
}

struct BaseRewriter<'a>(&'a Hierarchy<Arc<Relation>>); // TODO implement this properly

impl<'a> RewriteVisitor<'a> for BaseRewriter<'a> {
    fn table(&self, table: &'a Table, rewriting_rule: &'a RewritingRule) -> Arc<Relation> {
        Arc::new(
            match (rewriting_rule.output(), rewriting_rule.parameters()) {
                (Property::Private, _) => table.clone().into(),
                (
                    Property::ProtectedEntityPreserving,
                    Parameters::ProtectedEntity(protected_entity),
                ) => {
                    let protection = Protection::new(
                        self.0,
                        protected_entity.clone(),
                        crate::protection::Strategy::Hard,
                    );
                    protection.table(table).unwrap().into()
                }
                (Property::DifferentiallyPrivate, _) => table.clone().into(),
                (Property::Published, _) => table.clone().into(),
                (Property::Public, _) => table.clone().into(),
                _ => table.clone().into(),
            },
        )
    }

    fn map(
        &self,
        map: &'a Map,
        rewriting_rule: &'a RewritingRule,
        rewritten_input: Arc<Relation>,
    ) -> Arc<Relation> {
        Arc::new(
            match (
                rewriting_rule.inputs()[0],
                rewriting_rule.output(),
                rewriting_rule.parameters(),
            ) {
                (
                    Property::ProtectedEntityPreserving,
                    Property::ProtectedEntityPreserving,
                    Parameters::ProtectedEntity(protected_entity),
                ) => {
                    let protection = Protection::new(
                        self.0,
                        protected_entity.clone(),
                        crate::protection::Strategy::Hard,
                    );
                    protection
                        .map(map, (*rewritten_input).clone().try_into().unwrap())
                        .unwrap()
                        .into()
                }
                _ => Relation::map()
                    .with(map.clone())
                    .input(rewritten_input)
                    .build(),
            },
        )
    }

    fn reduce(
        &self,
        reduce: &'a Reduce,
        rewriting_rule: &'a RewritingRule,
        rewritten_input: Arc<Relation>,
    ) -> Arc<Relation> {
        Arc::new(
            match (
                rewriting_rule.inputs()[0],
                rewriting_rule.output(),
                rewriting_rule.parameters(),
            ) {
                (
                    Property::ProtectedEntityPreserving,
                    Property::ProtectedEntityPreserving,
                    Parameters::ProtectedEntity(protected_entity),
                ) => {
                    let protection = Protection::new(
                        self.0,
                        protected_entity.clone(),
                        crate::protection::Strategy::Hard,
                    );
                    protection
                        .reduce(reduce, (*rewritten_input).clone().try_into().unwrap())
                        .unwrap()
                        .into()
                }
                _ => Relation::reduce()
                    .with(reduce.clone())
                    .input(rewritten_input)
                    .build(),
            },
        )
    }

    fn join(
        &self,
        join: &'a Join,
        rewriting_rule: &'a RewritingRule,
        rewritten_left: Arc<Relation>,
        rewritten_right: Arc<Relation>,
    ) -> Arc<Relation> {
        // TODO this is awfully ugly! change that quickly!
        // println!("DEBUG LEFT {}", rewritten_left.schema());
        // println!("DEBUG LEFT {}", join.left().schema());
        // println!("DEBUG RIGHT {}", rewritten_right.schema());
        // println!("DEBUG RIGHT {}", join.right().schema());
        let names: Vec<_> = join.schema().iter().map(|f| f.name().to_string()).collect();
        // let left
        Arc::new(
            Relation::join()
                .with(join.clone())
                // .left_names(names[0..rewritten_left])
                .left(rewritten_left)
                .right(rewritten_right)
                .build(),
        )
    }

    fn set(
        &self,
        set: &'a Set,
        rewriting_rule: &'a RewritingRule,
        rewritten_left: Arc<Relation>,
        rewritten_right: Arc<Relation>,
    ) -> Arc<Relation> {
        Arc::new(
            Relation::set()
                .with(set.clone())
                .left(rewritten_left)
                .right(rewritten_right)
                .build(),
        )
    }

    fn values(&self, values: &'a Values, rewriting_rule: &'a RewritingRule) -> Arc<Relation> {
        Arc::new(values.clone().into())
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
        protection::protected_entity,
        sql::parse,
        Relation,
    };

    #[test]
    fn test_set_eliminate_select_rewriting_rules() {
        let database = postgresql::test_database();
        let relations = database.relations();
        // Print relations with paths
        for (p, r) in relations.iter() {
            println!("{} -> {r}", p.into_iter().join("."))
        }
        let query = parse(
            "SELECT order_id, price FROM item_table WHERE order_id IN (1,2,3,4,5,6,7,8,9,10)",
        )
        .unwrap();
        let protected_entity = ProtectedEntity::from(vec![
            (
                "item_table",
                vec![
                    ("order_id", "order_table", "id"),
                    ("user_id", "user_table", "id"),
                ],
                "name",
            ),
            (
                "order_table",
                vec![("user_id", "user_table", "id")],
                "name",
            ),
            ("user_table", vec![], "name"),
        ]);
        let budget = Budget::new(1., 1e-3);
        let relation = Relation::try_from(query.with(&relations)).unwrap();
        relation.display_dot().unwrap();
        // Add rewritting rules
        let relation_with_rules =
            relation.set_rewriting_rules(BaseRewritingRulesSetter::new(protected_entity, budget));
        relation_with_rules.display_dot().unwrap();
        let relation_with_rules =
            relation_with_rules.map_rewriting_rules(BaseRewritingRulesEliminator);
        relation_with_rules.display_dot().unwrap();
        for rwrr in relation_with_rules.select_rewriting_rules(BaseRewritingRulesSelector) {
            rwrr.display_dot().unwrap();
            rwrr.rewrite(BaseRewriter(&relations))
                .display_dot()
                .unwrap();
        }
    }

    #[test]
    fn test_set_eliminate_select_rewriting_rules_aggregation() {
        let database = postgresql::test_database();
        let relations = database.relations();
        // Print relations with paths
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
        let protected_entity = ProtectedEntity::from(vec![
            (
                "item_table",
                vec![
                    ("order_id", "order_table", "id"),
                    ("user_id", "user_table", "id"),
                ],
                "name",
            ),
            (
                "order_table",
                vec![("user_id", "user_table", "id")],
                "name",
            ),
            ("user_table", vec![], "name"),
        ]);
        let budget = Budget::new(1., 1e-3);
        let relation = Relation::try_from(query.with(&relations)).unwrap();
        relation.display_dot().unwrap();
        // Add rewritting rules
        let relation_with_rules =
            relation.set_rewriting_rules(BaseRewritingRulesSetter::new(protected_entity, budget));
        relation_with_rules.display_dot().unwrap();
        let relation_with_rules =
            relation_with_rules.map_rewriting_rules(BaseRewritingRulesEliminator);
        relation_with_rules.display_dot().unwrap();
        for rwrr in relation_with_rules.select_rewriting_rules(BaseRewritingRulesSelector) {
            rwrr.display_dot().unwrap();
            rwrr.rewrite(BaseRewriter(&relations))
                .display_dot()
                .unwrap();
        }
    }

    #[test]
    fn test_set_eliminate_select_rewriting_rules_complex_query() {
        let database = postgresql::test_database();
        let relations = database.relations();
        let query = parse(r#"
        WITH order_avg_price (order_id, avg_price) AS (SELECT order_id, avg(price) AS avg_price FROM item_table GROUP BY order_id),
        order_std_price (order_id, std_price) AS (SELECT order_id, 2*stddev(price) AS std_price FROM item_table GROUP BY order_id),
        normalized_prices AS (SELECT order_avg_price.order_id, (item_table.price-order_avg_price.avg_price)/(0.1+order_std_price.std_price) AS normalized_price
            FROM item_table JOIN order_avg_price ON item_table.order_id=order_avg_price.order_id JOIN order_std_price ON item_table.order_id=order_std_price.order_id)
        SELECT order_id, sum(normalized_price) FROM normalized_prices GROUP BY order_id
        "#,
        ).unwrap();
        let protected_entity = ProtectedEntity::from(vec![
            (
                "item_table",
                vec![
                    ("order_id", "order_table", "id"),
                    ("user_id", "user_table", "id"),
                ],
                "name",
            ),
            (
                "order_table",
                vec![("user_id", "user_table", "id")],
                "name",
            ),
            ("user_table", vec![], "name"),
        ]);
        let budget = Budget::new(1., 1e-3);
        let relation = Relation::try_from(query.with(&relations)).unwrap();
        relation.display_dot().unwrap();
        // Add rewritting rules
        let relation_with_rules =
            relation.set_rewriting_rules(BaseRewritingRulesSetter::new(protected_entity, budget));
        relation_with_rules.display_dot().unwrap();
        let relation_with_rules =
            relation_with_rules.map_rewriting_rules(BaseRewritingRulesEliminator);
        relation_with_rules.display_dot().unwrap();
        for rwrr in relation_with_rules.select_rewriting_rules(BaseRewritingRulesSelector) {
            rwrr.display_dot().unwrap();
            rwrr.rewrite(BaseRewriter(&relations))
                .display_dot()
                .unwrap();
        }
    }
}
