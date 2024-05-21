//! For now a simple definition of Property
use std::{collections::HashSet, fmt, marker::PhantomData, ops::Deref, sync::Arc};

use itertools::Itertools;

use crate::{
    builder::{Ready, With},
    differential_privacy::{DpParameters, DpEvent},
    hierarchy::Hierarchy,
    privacy_unit_tracking::{privacy_unit::PrivacyUnit, PrivacyUnitTracking},
    expr::aggregate::Aggregate,
    relation::{Join, Map, Reduce, Relation, Set, Table, Values, Variant as _},
    rewriting::relation_with_attributes::RelationWithAttributes,
    synthetic_data::SyntheticData,
    visitor::{Acceptor, Visited, Visitor}, display::Dot,
};

/// A simple Property object to tag Relations properties
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum Property {
    Private,
    SyntheticData,
    PrivacyUnitPreserving,
    DifferentiallyPrivate,
    Published,
    Public,
}

impl fmt::Display for Property {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Property::Private => write!(f, "Priv"),
            Property::SyntheticData => write!(f, "SD"),
            Property::PrivacyUnitPreserving => write!(f, "PUP"),
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
    SyntheticData(SyntheticData),
    DifferentialPrivacy(DpParameters),
    PrivacyUnit(PrivacyUnit),
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

#[derive(Clone, Debug, PartialEq)]
pub struct RelationWithDpEvent {
    relation: Arc<Relation>,
    dp_event: DpEvent,
}

impl RelationWithDpEvent {
    pub fn relation(&self) -> &Relation {
        self.relation.deref()
    }

    pub fn dp_event(&self) -> &DpEvent {
        &self.dp_event
    }
}

impl From<RelationWithDpEvent> for (Relation, DpEvent) {
    fn from(value: RelationWithDpEvent) -> Self {
        let RelationWithDpEvent {
            relation,
            dp_event,
        } = value;
        (relation.deref().clone(), dp_event)
    }
}

impl From<(Arc<Relation>, DpEvent)> for RelationWithDpEvent {
    fn from(value: (Arc<Relation>, DpEvent)) -> Self {
        RelationWithDpEvent {
            relation: value.0,
            dp_event: value.1,
        }
    }
}

/// A Visitor to rewrite a RelationWithRewritingRule
pub trait RewriteVisitor<'a> {
    fn table(
        &self,
        table: &'a Table,
        rewriting_rule: &'a RewritingRule,
    ) -> RelationWithDpEvent;
    fn map(
        &self,
        map: &'a Map,
        rewriting_rule: &'a RewritingRule,
        rewritten_input: RelationWithDpEvent,
    ) -> RelationWithDpEvent;
    fn reduce(
        &self,
        reduce: &'a Reduce,
        rewriting_rule: &'a RewritingRule,
        rewritten_input: RelationWithDpEvent,
    ) -> RelationWithDpEvent;
    fn join(
        &self,
        join: &'a Join,
        rewriting_rule: &'a RewritingRule,
        rewritten_left: RelationWithDpEvent,
        rewritten_right: RelationWithDpEvent,
    ) -> RelationWithDpEvent;
    fn set(
        &self,
        set: &'a Set,
        rewriting_rule: &'a RewritingRule,
        rewritten_left: RelationWithDpEvent,
        rewritten_right: RelationWithDpEvent,
    ) -> RelationWithDpEvent;
    fn values(
        &self,
        values: &'a Values,
        rewriting_rule: &'a RewritingRule,
    ) -> RelationWithDpEvent;
}
/// Implement the visitor trait
impl<'a, V: RewriteVisitor<'a>> Visitor<'a, RelationWithRewritingRule<'a>, RelationWithDpEvent>
    for V
{
    fn visit(
        &self,
        acceptor: &'a RelationWithRewritingRule<'a>,
        dependencies: Visited<'a, RelationWithRewritingRule<'a>, RelationWithDpEvent>,
    ) -> RelationWithDpEvent {
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
    pub fn rewrite<V: RewriteVisitor<'a>>(
        &'a self,
        rewrite_visitor: V,
    ) -> RelationWithDpEvent {
        self.accept(rewrite_visitor)
    }
}

// # Implement various rewriting rules visitors

/// A basic rewriting rule setter
pub struct RewritingRulesSetter<'a> {
    relations: &'a Hierarchy<Arc<Relation>>,
    synthetic_data: Option<SyntheticData>,
    privacy_unit: PrivacyUnit,
    dp_parameters: DpParameters,
}

impl<'a> RewritingRulesSetter<'a> {
    pub fn new(
        relations: &'a Hierarchy<Arc<Relation>>,
        synthetic_data: Option<SyntheticData>,
        privacy_unit: PrivacyUnit,
        dp_parameters: DpParameters,
    ) -> RewritingRulesSetter {
        RewritingRulesSetter {
            relations,
            synthetic_data,
            privacy_unit,
            dp_parameters,
        }
    }
}

impl<'a> SetRewritingRulesVisitor<'a> for RewritingRulesSetter<'a> {
    fn table(&self, table: &'a Table) -> Vec<RewritingRule> {
        let mut rewriting_rules = if self
            .privacy_unit
            .iter()
            .find(|(name, _field_path)| table.name() == self.relations[name.as_str()].name())
            .is_some()
        {
            vec![
                RewritingRule::new(vec![], Property::Private, Parameters::None),
                RewritingRule::new(
                    vec![],
                    Property::PrivacyUnitPreserving,
                    Parameters::PrivacyUnit(self.privacy_unit.clone()),
                ),
            ]
        } else {
            vec![
                RewritingRule::new(
                    vec![],
                    Property::Public,
                    Parameters::None,
                ),
            ]
        };
        if let Some(synthetic_data) = &self.synthetic_data {
            rewriting_rules.push(
                RewritingRule::new(
                    vec![],
                    Property::SyntheticData,
                    Parameters::SyntheticData(synthetic_data.clone()),
                )
            )
        }
        rewriting_rules
    }

    fn map(&self, map: &'a Map, input: Arc<RelationWithRewritingRules<'a>>) -> Vec<RewritingRule> {
        let mut rewriting_rules = vec![
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
                vec![Property::PrivacyUnitPreserving],
                Property::PrivacyUnitPreserving,
                Parameters::PrivacyUnit(self.privacy_unit.clone()),
            ),
        ];
        if let Some(synthetic_data) = &self.synthetic_data {
            rewriting_rules.push(
                RewritingRule::new(
                    vec![Property::SyntheticData],
                    Property::SyntheticData,
                    Parameters::SyntheticData(synthetic_data.clone()),
                )
            )
        }
        rewriting_rules
    }

    fn reduce(
        &self,
        reduce: &'a Reduce,
        input: Arc<RelationWithRewritingRules<'a>>,
    ) -> Vec<RewritingRule> {
        let mut rewriting_rules = vec![
            RewritingRule::new(vec![Property::Public], Property::Public, Parameters::None),
            RewritingRule::new(
                vec![Property::Published],
                Property::Published,
                Parameters::None,
            )
        ];
        if let Some(synthetic_data) = &self.synthetic_data {
            rewriting_rules.push(
                RewritingRule::new(
                    vec![Property::SyntheticData],
                    Property::SyntheticData,
                    Parameters::SyntheticData(synthetic_data.clone()),
                )
            );
            // rewriting_rules.push(
            //     RewritingRule::new(
            //         vec![Property::SyntheticData],
            //         Property::Published,
            //         Parameters::None,
            //     )
            // );
        }
        // We can compile into DP only if the aggregations are supported
        if reduce.aggregate().iter().all(|f| {
            match f.aggregate() {
                Aggregate::Mean |
                Aggregate::MeanDistinct |
                Aggregate::Count |
                Aggregate::CountDistinct |
                Aggregate::Sum |
                Aggregate::SumDistinct |
                Aggregate::Std |
                Aggregate::StdDistinct |
                Aggregate::Var |
                Aggregate::VarDistinct => true,
                Aggregate::Min |
                Aggregate::Max |
                Aggregate::Median |
                Aggregate::First |
                Aggregate::Last |
                Aggregate::Quantile(_) |
                Aggregate::Quantiles(_) => reduce.group_by().contains(f.column()),
                _ => false,
            }
        }) {
            rewriting_rules.push(
                RewritingRule::new(
            vec![Property::PrivacyUnitPreserving],
            Property::DifferentiallyPrivate,
                    Parameters::DifferentialPrivacy(self.dp_parameters.clone()),
                )
            );
            rewriting_rules.push(
                RewritingRule::new(
            vec![Property::PrivacyUnitPreserving],
            Property::PrivacyUnitPreserving,
                    Parameters::PrivacyUnit(self.privacy_unit.clone()),
                )
            )
        }
        rewriting_rules
    }

    fn join(
        &self,
        join: &'a Join,
        left: Arc<RelationWithRewritingRules<'a>>,
        right: Arc<RelationWithRewritingRules<'a>>,
    ) -> Vec<RewritingRule> {
        let mut rewriting_rules = vec![
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
                vec![Property::Public, Property::PrivacyUnitPreserving],
                Property::PrivacyUnitPreserving,
                Parameters::PrivacyUnit(self.privacy_unit.clone()),
            ),
            RewritingRule::new(
                vec![Property::PrivacyUnitPreserving, Property::Public],
                Property::PrivacyUnitPreserving,
                Parameters::PrivacyUnit(self.privacy_unit.clone()),
            ),
            RewritingRule::new(
                vec![Property::Published, Property::PrivacyUnitPreserving],
                Property::PrivacyUnitPreserving,
                Parameters::PrivacyUnit(self.privacy_unit.clone()),
            ),
            RewritingRule::new(
                vec![
                    Property::DifferentiallyPrivate,
                    Property::PrivacyUnitPreserving,
                ],
                Property::PrivacyUnitPreserving,
                Parameters::PrivacyUnit(self.privacy_unit.clone()),
            ),
            RewritingRule::new(
                vec![Property::PrivacyUnitPreserving, Property::Published],
                Property::PrivacyUnitPreserving,
                Parameters::PrivacyUnit(self.privacy_unit.clone()),
            ),
            RewritingRule::new(
                vec![
                    Property::PrivacyUnitPreserving,
                    Property::DifferentiallyPrivate,
                ],
                Property::PrivacyUnitPreserving,
                Parameters::PrivacyUnit(self.privacy_unit.clone()),
            ),
            RewritingRule::new(
                vec![
                    Property::PrivacyUnitPreserving,
                    Property::PrivacyUnitPreserving,
                ],
                Property::PrivacyUnitPreserving,
                Parameters::PrivacyUnit(self.privacy_unit.clone()),
            ),
        ];
        if let Some(synthetic_data) = &self.synthetic_data {
            rewriting_rules.push(
                RewritingRule::new(
                    vec![Property::SyntheticData, Property::SyntheticData],
                    Property::SyntheticData,
                    Parameters::SyntheticData(synthetic_data.clone()),
                )
            )
        }
        rewriting_rules
    }

    fn set(
        &self,
        set: &'a Set,
        left: Arc<RelationWithRewritingRules<'a>>,
        right: Arc<RelationWithRewritingRules<'a>>,
    ) -> Vec<RewritingRule> {
        let mut rewriting_rules = vec![
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
                    Property::PrivacyUnitPreserving,
                    Property::PrivacyUnitPreserving,
                ],
                Property::PrivacyUnitPreserving,
                Parameters::PrivacyUnit(self.privacy_unit.clone()),
            ),
        ];
        if let Some(synthetic_data) = &self.synthetic_data {
            rewriting_rules.push(
                RewritingRule::new(
                    vec![Property::SyntheticData, Property::SyntheticData],
                    Property::SyntheticData,
                    Parameters::SyntheticData(synthetic_data.clone()),
                )
            )
        }
        rewriting_rules
    }

    fn values(&self, values: &'a Values) -> Vec<RewritingRule> {
        let mut rewriting_rules = vec![
            RewritingRule::new(vec![], Property::Public, Parameters::None),
        ];
        if let Some(synthetic_data) = &self.synthetic_data {
            rewriting_rules.push(
                RewritingRule::new(
                    vec![],
                    Property::SyntheticData,
                    Parameters::SyntheticData(synthetic_data.clone()),
                )
            )
        }
        rewriting_rules
    }
}

/// A basic rewriting rule eliminator
pub struct RewritingRulesEliminator;

impl<'a> MapRewritingRulesVisitor<'a> for RewritingRulesEliminator {
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
pub struct RewritingRulesSelector;

impl<'a> SelectRewritingRuleVisitor<'a> for RewritingRulesSelector {
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

/// Compute the number of DP ops
pub struct BudgetDispatcher;

impl<'a> Visitor<'a, RelationWithRewritingRule<'a>, usize> for BudgetDispatcher {
    fn visit(
        &self,
        acceptor: &'a RelationWithRewritingRule<'a>,
        dependencies: Visited<'a, RelationWithRewritingRule<'a>, usize>,
    ) -> usize {
        acceptor.inputs().iter().fold(
            match acceptor.attributes().output() {
                Property::DifferentiallyPrivate => 1,
                _ => 0,
            },
            |sum, rwrr| sum + dependencies.get(rwrr.deref()),
        )
    }
}

/// Compute the score
pub struct Score;

impl<'a> Visitor<'a, RelationWithRewritingRule<'a>, f64> for Score {
    fn visit(
        &self,
        acceptor: &'a RelationWithRewritingRule<'a>,
        dependencies: Visited<'a, RelationWithRewritingRule<'a>, f64>,
    ) -> f64 {
        acceptor.inputs().iter().fold(
            match acceptor.attributes().output() {
                Property::SyntheticData => 1.,
                Property::PrivacyUnitPreserving => 2.,
                Property::DifferentiallyPrivate => 5.,
                Property::Published => 1.,
                Property::Public => 10.,
                _ => 0.,
            },
            |sum, rwrr| sum + dependencies.get(rwrr.deref()),
        )
    }
}

pub struct Rewriter<'a>(&'a Hierarchy<Arc<Relation>>); // TODO implement this properly

impl<'a> Rewriter<'a> {
    pub fn new(relations: &'a Hierarchy<Arc<Relation>>) -> Rewriter<'a> {
        Rewriter(relations)
    }
}

impl<'a> RewriteVisitor<'a> for Rewriter<'a> {
    fn table(
        &self,
        table: &'a Table,
        rewriting_rule: &'a RewritingRule,
    ) -> RelationWithDpEvent {
        println!("DEBUG: RewriteVisitor, table ");
        let relation = Arc::new(
            match (rewriting_rule.output(), rewriting_rule.parameters()) {
                (Property::Private, _) => table.clone().into(),
                (Property::SyntheticData, Parameters::SyntheticData(synthetic_data)) => {
                    synthetic_data.table(table).unwrap().into()
                }
                (Property::PrivacyUnitPreserving, Parameters::PrivacyUnit(privacy_unit)) => {
                    let privacy_unit_tracking = PrivacyUnitTracking::new(
                        self.0,
                        privacy_unit.clone(),
                        crate::privacy_unit_tracking::Strategy::Soft,
                    );
                    privacy_unit_tracking.table(table).unwrap().into()
                }
                (Property::DifferentiallyPrivate, _) => table.clone().into(),
                (Property::Published, _) => table.clone().into(),
                (Property::Public, _) => table.clone().into(),
                _ => table.clone().into(),
            },
        );
        println!("END DEBUG: RewriteVisitor, table ");
        (relation, DpEvent::no_op()).into()
    }

    fn map(
        &self,
        map: &'a Map,
        rewriting_rule: &'a RewritingRule,
        rewritten_input: RelationWithDpEvent,
    ) -> RelationWithDpEvent {
        println!("DEBUG: RewriteVisitor, map ");
        let (relation_input, dp_event_input) = rewritten_input.into();
        let relation: Arc<Relation> = Arc::new(
            match (
                rewriting_rule.inputs(),
                rewriting_rule.output(),
                rewriting_rule.parameters(),
            ) {
                (
                    [Property::PrivacyUnitPreserving],
                    Property::PrivacyUnitPreserving,
                    Parameters::PrivacyUnit(privacy_unit),
                ) => {
                    let privacy_unit_tracking = PrivacyUnitTracking::new(
                        self.0,
                        privacy_unit.clone(),
                        crate::privacy_unit_tracking::Strategy::Soft,
                    );
                    relation_input.display_dot().unwrap();
                    privacy_unit_tracking
                        .map(map, relation_input.try_into().unwrap())
                        .unwrap()
                        .into()
                }
                _ => Relation::map()
                    .with(map.clone())
                    .input(relation_input)
                    .build(),
            },
        );
        println!("END DEBUG: RewriteVisitor, map ");
        (relation, dp_event_input).into()
    }

    fn reduce(
        &self,
        reduce: &'a Reduce,
        rewriting_rule: &'a RewritingRule,
        rewritten_input: RelationWithDpEvent,
    ) -> RelationWithDpEvent {
        println!("DEBUG: RewriteVisitor, reduce ");
        let (relation_input, mut dp_event_input) = rewritten_input.into();
        let relation = Arc::new(
            match (
                rewriting_rule.inputs(),
                rewriting_rule.output(),
                rewriting_rule.parameters(),
            ) {
                (
                    [Property::PrivacyUnitPreserving],
                    Property::DifferentiallyPrivate,
                    Parameters::DifferentialPrivacy(dp_parameters),
                ) => {
                    let (dp_relation, dp_event) = dp_parameters
                        .reduce(reduce, relation_input.try_into().unwrap())
                        .unwrap()
                        .into();
                    dp_event_input = dp_event_input.compose(dp_event);
                    dp_relation
                }
                (
                    [Property::PrivacyUnitPreserving],
                    Property::PrivacyUnitPreserving,
                    Parameters::PrivacyUnit(privacy_unit),
                ) => {
                    let privacy_unit_tracking = PrivacyUnitTracking::new(
                        self.0,
                        privacy_unit.clone(),
                        crate::privacy_unit_tracking::Strategy::Hard,
                    );
                    privacy_unit_tracking
                        .reduce(reduce, relation_input.try_into().unwrap())
                        .unwrap()
                        .into()
                }
                _ => Relation::reduce()
                    .with(reduce.clone())
                    .input(relation_input)
                    .build(),
            },
        );
        println!("END DEBUG: RewriteVisitor, map ");
        (relation, dp_event_input).into()
    }

    fn join(
        &self,
        join: &'a Join,
        rewriting_rule: &'a RewritingRule,
        rewritten_left: RelationWithDpEvent,
        rewritten_right: RelationWithDpEvent,
    ) -> RelationWithDpEvent {
        println!("DEBUG: RewriteVisitor, join ");
        let (relation_left, dp_event_left) = rewritten_left.into();
        let (relation_right, dp_event_right) = rewritten_right.into();
        let relation: Arc<Relation> = Arc::new(
            match (
                rewriting_rule.inputs(),
                rewriting_rule.output(),
                rewriting_rule.parameters(),
            ) {
                (
                    [Property::PrivacyUnitPreserving, Property::PrivacyUnitPreserving],
                    Property::PrivacyUnitPreserving,
                    Parameters::PrivacyUnit(privacy_unit),
                ) => {
                    let privacy_unit_tracking = PrivacyUnitTracking::new(
                        self.0,
                        privacy_unit.clone(),
                        crate::privacy_unit_tracking::Strategy::Hard,
                    );
                    privacy_unit_tracking
                        .join(
                            join,
                            relation_left.try_into().unwrap(),
                            relation_right.try_into().unwrap(),
                        )
                        .unwrap()
                        .into()
                }
                (
                    [Property::Published, Property::PrivacyUnitPreserving],
                    Property::PrivacyUnitPreserving,
                    Parameters::PrivacyUnit(privacy_unit),
                ) | (
                    [Property::DifferentiallyPrivate, Property::PrivacyUnitPreserving],
                    Property::PrivacyUnitPreserving,
                    Parameters::PrivacyUnit(privacy_unit),
                ) => {
                    let privacy_unit_tracking = PrivacyUnitTracking::new(
                        self.0,
                        privacy_unit.clone(),
                        crate::privacy_unit_tracking::Strategy::Hard,
                    );
                    privacy_unit_tracking
                        .join_left_published(
                            join,
                            relation_left.try_into().unwrap(),
                            relation_right.try_into().unwrap(),
                        )
                        .unwrap()
                        .into()
                }
                (
                    [Property::PrivacyUnitPreserving, Property::Published],
                    Property::PrivacyUnitPreserving,
                    Parameters::PrivacyUnit(privacy_unit),
                ) | (
                    [Property::PrivacyUnitPreserving, Property::DifferentiallyPrivate],
                    Property::PrivacyUnitPreserving,
                    Parameters::PrivacyUnit(privacy_unit),
                ) => {
                    let privacy_unit_tracking = PrivacyUnitTracking::new(
                        self.0,
                        privacy_unit.clone(),
                        crate::privacy_unit_tracking::Strategy::Hard,
                    );
                    privacy_unit_tracking
                        .join_right_published(
                            join,
                            relation_left.try_into().unwrap(),
                            relation_right.try_into().unwrap(),
                        )
                        .unwrap()
                        .into()
                }
                _ => Relation::join()
                    .with(join.clone())
                    // .left_names(names[0..rewritten_left])
                    .left(relation_left)
                    .right(relation_right)
                    .build(),
            },
        );
        println!("END DEBUG: RewriteVisitor, join ");
        (relation, dp_event_left.compose(dp_event_right)).into()
    }

    fn set(
        &self,
        set: &'a Set,
        rewriting_rule: &'a RewritingRule,
        rewritten_left: RelationWithDpEvent,
        rewritten_right: RelationWithDpEvent,
    ) -> RelationWithDpEvent {
        let (relation_left, dp_event_left) = rewritten_left.into();
        let (relation_right, dp_event_right) = rewritten_right.into();
        let relation: Arc<Relation> = Arc::new(
            Relation::set()
                .with(set.clone())
                .left(relation_left)
                .right(relation_right)
                .build(),
        );
        (relation, dp_event_left.compose(dp_event_right)).into()
    }

    fn values(
        &self,
        values: &'a Values,
        rewriting_rule: &'a RewritingRule,
    ) -> RelationWithDpEvent {
        (Arc::new(values.clone().into()), DpEvent::no_op()).into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        builder::With,
        display::Dot,
        expr::Identifier,
        io::{postgresql, Database},
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
        let synthetic_data = SyntheticData::new(Hierarchy::from([
            (vec!["item_table"], Identifier::from("item_table")),
            (vec!["order_table"], Identifier::from("order_table")),
            (vec!["user_table"], Identifier::from("user_table")),
        ]));
        let privacy_unit = PrivacyUnit::from(vec![
            (
                "item_table",
                vec![
                    ("order_id", "order_table", "id"),
                    ("user_id", "user_table", "id"),
                ],
                "name",
            ),
            ("order_table", vec![("user_id", "user_table", "id")], "name"),
            ("user_table", vec![], "name"),
        ]);
        let dp_parameters = DpParameters::from_epsilon_delta(1., 1e-3);
        let relation = Relation::try_from(query.with(&relations)).unwrap();
        relation.display_dot().unwrap();
        // Add rewritting rules
        let relation_with_rules = relation.set_rewriting_rules(RewritingRulesSetter::new(
            &relations,
            Some(synthetic_data),
            privacy_unit,
            dp_parameters,
        ));
        relation_with_rules.display_dot().unwrap();
        let relation_with_rules = relation_with_rules.map_rewriting_rules(RewritingRulesEliminator);
        relation_with_rules.display_dot().unwrap();
        for rwrr in relation_with_rules.select_rewriting_rules(RewritingRulesSelector) {
            rwrr.display_dot().unwrap();
            let num_dp = rwrr.accept(BudgetDispatcher);
            println!("DEBUG SPLIT BUDGET IN {}", num_dp);
            println!("DEBUG SCORE {}", rwrr.accept(Score));
            let relation_with_dp_event = rwrr.rewrite(Rewriter(&relations));
            println!(
                "PrivateQuery: {:?}",
                relation_with_dp_event.dp_event()
            );
            relation_with_dp_event
                .relation()
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
        count(*) AS count_price,
        avg(price) AS mean_price
        FROM item_table WHERE order_id IN (1,2,3,4,5,6,7,8,9,10) GROUP BY order_id",
        )
        .unwrap();
        let synthetic_data = SyntheticData::new(Hierarchy::from([
            (vec!["item_table"], Identifier::from("item_table")),
            (vec!["order_table"], Identifier::from("order_table")),
            (vec!["user_table"], Identifier::from("user_table")),
        ]));
        let privacy_unit = PrivacyUnit::from(vec![
            (
                "item_table",
                vec![
                    ("order_id", "order_table", "id"),
                    ("user_id", "user_table", "id"),
                ],
                "name",
            ),
            ("order_table", vec![("user_id", "user_table", "id")], "name"),
            ("user_table", vec![], "name"),
        ]);
        let dp_parameters = DpParameters::from_epsilon_delta(1., 1e-3);
        let relation = Relation::try_from(query.with(&relations)).unwrap();
        relation.display_dot().unwrap();
        // Add rewritting rules
        let relation_with_rules = relation.set_rewriting_rules(RewritingRulesSetter::new(
            &relations,
            Some(synthetic_data),
            privacy_unit,
            dp_parameters,
        ));
        relation_with_rules.display_dot().unwrap();
        let relation_with_rules = relation_with_rules.map_rewriting_rules(RewritingRulesEliminator);
        relation_with_rules.display_dot().unwrap();
        for rwrr in relation_with_rules.select_rewriting_rules(RewritingRulesSelector) {
            rwrr.display_dot().unwrap();
            let num_dp = rwrr.accept(BudgetDispatcher);
            println!("DEBUG SPLIT BUDGET IN {}", num_dp);
            println!("DEBUG SCORE {}", rwrr.accept(Score));
            let relation_with_dp_event = rwrr.rewrite(Rewriter(&relations));
            println!(
                "PrivateQuery: {:?}",
                relation_with_dp_event.dp_event()
            );
            relation_with_dp_event
                .relation()
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
        order_sum_abs_price (order_id, sum_abs_price) AS (SELECT order_id, sum(abs(price)) AS sum_abs_price FROM item_table GROUP BY order_id),
        normalized_prices AS (SELECT order_avg_price.order_id, (item_table.price-order_avg_price.avg_price)/(0.1+order_sum_abs_price.sum_abs_price) AS normalized_price
            FROM item_table JOIN order_avg_price ON item_table.order_id=order_avg_price.order_id JOIN order_sum_abs_price ON item_table.order_id=order_sum_abs_price.order_id)
        SELECT order_id, sum(normalized_price) FROM normalized_prices GROUP BY order_id
        "#,
        ).unwrap();
        let synthetic_data = SyntheticData::new(Hierarchy::from([
            (vec!["item_table"], Identifier::from("item_table")),
            (vec!["order_table"], Identifier::from("order_table")),
            (vec!["user_table"], Identifier::from("user_table")),
        ]));
        let privacy_unit = PrivacyUnit::from(vec![
            (
                "item_table",
                vec![
                    ("order_id", "order_table", "id"),
                    ("user_id", "user_table", "id"),
                ],
                "name",
            ),
            ("order_table", vec![("user_id", "user_table", "id")], "name"),
            ("user_table", vec![], "name"),
        ]);
        let dp_parameters = DpParameters::from_epsilon_delta(1., 1e-3);
        let relation = Relation::try_from(query.with(&relations)).unwrap();
        relation.display_dot().unwrap();
        // Add rewritting rules
        let relation_with_rules = relation.set_rewriting_rules(RewritingRulesSetter::new(
            &relations,
            Some(synthetic_data),
            privacy_unit,
            dp_parameters,
        ));
        relation_with_rules.display_dot().unwrap();
        let relation_with_rules = relation_with_rules.map_rewriting_rules(RewritingRulesEliminator);
        relation_with_rules.display_dot().unwrap();
        for rwrr in relation_with_rules.select_rewriting_rules(RewritingRulesSelector) {
            rwrr.display_dot().unwrap();
            let num_dp = rwrr.accept(BudgetDispatcher);
            println!("DEBUG SPLIT BUDGET IN {}", num_dp);
            println!("DEBUG SCORE {}", rwrr.accept(Score));
            let relation_with_dp_event = rwrr.rewrite(Rewriter(&relations));
            println!(
                "PrivateQuery: {:?}",
                relation_with_dp_event.dp_event()
            );
            relation_with_dp_event
                .relation()
                .display_dot()
                .unwrap();
        }
    }

    #[test]
    fn test_set_eliminate_select_rewriting_rules_synthetic() {
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
        let synthetic_data = SyntheticData::new(Hierarchy::from([
            (vec!["item_table"], Identifier::from("item_table")),
            (vec!["order_table"], Identifier::from("order_table")),
            (vec!["user_table"], Identifier::from("user_table")),
        ]));
        let privacy_unit = PrivacyUnit::from(vec![
            (
                "item_table",
                vec![
                    ("order_id", "order_table", "id"),
                    ("user_id", "user_table", "id"),
                ],
                "name",
            ),
            ("order_table", vec![("user_id", "user_table", "id")], "name"),
            ("user_table", vec![], "name"),
        ]);
        let dp_parameters = DpParameters::from_epsilon_delta(1., 1e-3);
        let relation = Relation::try_from(query.with(&relations)).unwrap();
        relation.display_dot().unwrap();
        // Add rewritting rules
        let relation_with_rules = relation.set_rewriting_rules(RewritingRulesSetter::new(
            &relations,
            Some(synthetic_data),
            privacy_unit,
            dp_parameters,
        ));
        relation_with_rules.display_dot().unwrap();
        let relation_with_rules = relation_with_rules.map_rewriting_rules(RewritingRulesEliminator);
        relation_with_rules.display_dot().unwrap();
        for rwrr in relation_with_rules.select_rewriting_rules(RewritingRulesSelector) {
            rwrr.display_dot().unwrap();
            let num_dp = rwrr.accept(BudgetDispatcher);
            println!("DEBUG SPLIT BUDGET IN {}", num_dp);
            println!("DEBUG SCORE {}", rwrr.accept(Score));
            let relation_with_dp_event = rwrr.rewrite(Rewriter(&relations));
            println!(
                "PrivateQuery: {:?}",
                relation_with_dp_event.dp_event()
            );
            relation_with_dp_event
                .relation()
                .display_dot()
                .unwrap();
        }
    }

    #[test]
    fn test_dp_supported_aggregations_query() {
        let database = postgresql::test_database();
        let relations = database.relations();
        let query = parse(r#"
            SELECT order_id, sum(price) FROM item_table GROUP BY order_id
        "#,
        ).unwrap();
        let synthetic_data = Some(SyntheticData::new(Hierarchy::from([
            (vec!["item_table"], Identifier::from("item_table")),
            (vec!["order_table"], Identifier::from("order_table")),
            (vec!["user_table"], Identifier::from("user_table")),
        ])));
        let privacy_unit = PrivacyUnit::from(vec![
            (
                "item_table",
                vec![
                    ("order_id", "order_table", "id"),
                    ("user_id", "user_table", "id"),
                ],
                "name",
            ),
            ("order_table", vec![("user_id", "user_table", "id")], "name"),
            ("user_table", vec![], "name"),
        ]);
        let dp_parameters = DpParameters::from_epsilon_delta(1., 1e-3);
        let relation = Relation::try_from(query.with(&relations)).unwrap();
        relation.display_dot().unwrap();
        // Add rewritting rules
        let relation_with_rules = relation.set_rewriting_rules(RewritingRulesSetter::new(
            &relations,
            synthetic_data,
            privacy_unit,
            dp_parameters,
        ));
        relation_with_rules.display_dot().unwrap();
        let relation_with_rules = relation_with_rules.map_rewriting_rules(RewritingRulesEliminator);
        relation_with_rules.display_dot().unwrap();
        for rwrr in relation_with_rules.select_rewriting_rules(RewritingRulesSelector) {
            rwrr.display_dot().unwrap();
            let num_dp = rwrr.accept(BudgetDispatcher);
            println!("\nDEBUG SPLIT BUDGET IN {}", num_dp);
            println!("DEBUG SCORE {}", rwrr.accept(Score));
            let relation_with_dp_event = rwrr.rewrite(Rewriter(&relations));
            println!(
                "PrivateQuery: {:?}",
                relation_with_dp_event.dp_event()
            );
            relation_with_dp_event
                .relation()
                .display_dot()
                .unwrap();
        }
    }

    #[test]
    fn test_no_synthetic_data() {
        let database = postgresql::test_database();
        let relations = database.relations();
        let query = parse(r#"
            SELECT order_id, sum(price) FROM item_table GROUP BY order_id
        "#,
        ).unwrap();
        let synthetic_data = None;
        let privacy_unit = PrivacyUnit::from(vec![
            (
                "item_table",
                vec![
                    ("order_id", "order_table", "id"),
                    ("user_id", "user_table", "id"),
                ],
                "name",
            ),
            ("order_table", vec![("user_id", "user_table", "id")], "name"),
            ("user_table", vec![], "name"),
        ]);
        let dp_parameters = DpParameters::from_epsilon_delta(1., 1e-3);
        let relation = Relation::try_from(query.with(&relations)).unwrap();
        relation.display_dot().unwrap();
        // Add rewritting rules
        let relation_with_rules = relation.set_rewriting_rules(RewritingRulesSetter::new(
            &relations,
            synthetic_data,
            privacy_unit,
            dp_parameters,
        ));
        relation_with_rules.display_dot().unwrap();
        let relation_with_rules = relation_with_rules.map_rewriting_rules(RewritingRulesEliminator);
        relation_with_rules.display_dot().unwrap();
        for rwrr in relation_with_rules.select_rewriting_rules(RewritingRulesSelector) {
            rwrr.display_dot().unwrap();
            let num_dp = rwrr.accept(BudgetDispatcher);
            println!("DEBUG SPLIT BUDGET IN {}", num_dp);
            println!("DEBUG SCORE {}", rwrr.accept(Score));
            let relation_with_dp_event = rwrr.rewrite(Rewriter(&relations));
            println!(
                "PrivateQuery: {:?}",
                relation_with_dp_event.dp_event()
            );
            relation_with_dp_event
                .relation()
                .display_dot()
                .unwrap();
        }
    }

    #[test]
    fn test_dp_unsupported_aggregations_query() {
        let database = postgresql::test_database();
        let relations = database.relations();
        let query = parse(r#"
            SELECT order_id, max(price) FROM item_table GROUP BY order_id
        "#,
        ).unwrap();
        let synthetic_data = SyntheticData::new(Hierarchy::from([
            (vec!["item_table"], Identifier::from("item_table")),
            (vec!["order_table"], Identifier::from("order_table")),
            (vec!["user_table"], Identifier::from("user_table")),
        ]));
        let privacy_unit = PrivacyUnit::from(vec![
            (
                "item_table",
                vec![
                    ("order_id", "order_table", "id"),
                    ("user_id", "user_table", "id"),
                ],
                "name",
            ),
            ("order_table", vec![("user_id", "user_table", "id")], "name"),
            ("user_table", vec![], "name"),
        ]);
        let dp_parameters = DpParameters::from_epsilon_delta(1., 1e-3);
        let relation = Relation::try_from(query.with(&relations)).unwrap();
        relation.display_dot().unwrap();
        // Add rewritting rules
        let relation_with_rules = relation.set_rewriting_rules(RewritingRulesSetter::new(
            &relations,
            Some(synthetic_data),
            privacy_unit,
            dp_parameters,
        ));
        relation_with_rules.display_dot().unwrap();
        let relation_with_rules = relation_with_rules.map_rewriting_rules(RewritingRulesEliminator);
        relation_with_rules.display_dot().unwrap();
        for rwrr in relation_with_rules.select_rewriting_rules(RewritingRulesSelector) {
            rwrr.display_dot().unwrap();
            let num_dp = rwrr.accept(BudgetDispatcher);
            println!("DEBUG SPLIT BUDGET IN {}", num_dp);
            println!("DEBUG SCORE {}", rwrr.accept(Score));
            let relation_with_dp_event = rwrr.rewrite(Rewriter(&relations));
            println!(
                "PrivateQuery: {:?}",
                relation_with_dp_event.dp_event()
            );
            relation_with_dp_event
                .relation()
                .display_dot()
                .unwrap();
        }
    }
}
