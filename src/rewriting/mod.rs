pub mod dot;
pub mod relation_with_attributes;
pub mod rewriting_rule;

use itertools::Itertools;
pub use relation_with_attributes::RelationWithAttributes;
pub use rewriting_rule::{
    Property, RelationWithPrivateQuery, RelationWithRewritingRule, RelationWithRewritingRules,
    RewritingRule,
};

use std::{error, fmt, result, sync::Arc};

use crate::{
    builder::{Ready, With},
    differential_privacy::{
        budget::Budget,
        private_query::{self, PrivateQuery},
    },
    expr::Identifier,
    hierarchy::Hierarchy,
    protection::{protected_entity::ProtectedEntity, Protection},
    relation::{Join, Map, Reduce, Relation, Set, Table, Values, Variant as _},
    synthetic_data::{self, SyntheticData},
    visitor::{Acceptor, Dependencies, Visited, Visitor},
};

use rewriting_rule::{
    Rewriter, RewritingRulesEliminator, RewritingRulesSelector,
    RewritingRulesSetter, Score,
};

#[derive(Debug)]
pub enum Error {
    UnreachableProperty(String),
    Other(String),
}

impl Error {
    pub fn unreachable_property(property: impl fmt::Display) -> Error {
        Error::UnreachableProperty(format!("{} is unreachable", property))
    }
    pub fn other(value: impl fmt::Display) -> Error {
        Error::Other(format!("Error with {}", value))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::UnreachableProperty(desc) => writeln!(f, "UnreachableProperty: {}", desc),
            Error::Other(err) => writeln!(f, "{}", err),
        }
    }
}

impl error::Error for Error {}

impl From<crate::io::Error> for Error {
    fn from(err: crate::io::Error) -> Self {
        Error::Other(err.to_string())
    }
}

pub type Result<T> = result::Result<T, Error>;

impl Relation {
    /// Rewrite the query so that the protected entity is tracked through the query.
    pub fn rewrite_as_protected_entity_preserving<'a>(
        &'a self,
        relations: &'a Hierarchy<Arc<Relation>>,
        synthetic_data: SyntheticData,
        protected_entity: ProtectedEntity,
        budget: Budget,
    ) -> Result<RelationWithPrivateQuery> {
        let relation_with_rules = self.set_rewriting_rules(RewritingRulesSetter::new(
            relations,
            synthetic_data,
            protected_entity,
            budget,
        ));
        let relation_with_rules =
            relation_with_rules.map_rewriting_rules(RewritingRulesEliminator);
        relation_with_rules
            .select_rewriting_rules(RewritingRulesSelector)
            .into_iter()
            .filter_map(|rwrr| match rwrr.attributes().output() {
                Property::Public | Property::ProtectedEntityPreserving => Some((
                    rwrr.rewrite(Rewriter::new(relations)),
                    rwrr.accept(Score),
                )),
                property => None,
            })
            .max_by_key(|&(_, value)| value.partial_cmp(&value).unwrap())
            .map(|(relation, _)| relation)
            .ok_or_else(|| Error::unreachable_property("protected_entity_preserving"))
    }
    /// Rewrite the query so that it is differentially private.
    pub fn rewrite_with_differential_privacy<'a>(
        &'a self,
        relations: &'a Hierarchy<Arc<Relation>>,
        synthetic_data: SyntheticData,
        protected_entity: ProtectedEntity,
        budget: Budget,
    ) -> Result<RelationWithPrivateQuery> {
        let relation_with_rules = self.set_rewriting_rules(RewritingRulesSetter::new(
            relations,
            synthetic_data,
            protected_entity,
            budget,
        ));
        let relation_with_rules =
            relation_with_rules.map_rewriting_rules(RewritingRulesEliminator);
        relation_with_rules
            .select_rewriting_rules(RewritingRulesSelector)
            .into_iter()
            .filter_map(|rwrr| match rwrr.attributes().output() {
                Property::Public | Property::Published | Property::DifferentiallyPrivate => Some((
                    rwrr.rewrite(Rewriter::new(relations)),
                    rwrr.accept(Score),
                )),
                property => None,
            })
            .max_by_key(|&(_, value)| value.partial_cmp(&value).unwrap())
            .map(|(relation, _)| relation)
            .ok_or_else(|| Error::unreachable_property("differential_privacy"))
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
        expr::Identifier,
        io::{postgresql, Database},
        sql::parse,
        Relation,
    };

    #[test]
    fn test_rewrite() {
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

        // Add rewritting rules
        let relation_with_rules: rewriting_rule::RelationWithRewritingRules =
            relation.with_default_attributes();
        println!("{:#?}", relation_with_rules);
    }

    #[test]
    fn test_rewrite_with_differential_privacy() {
        let database = postgresql::test_database();
        let relations = database.relations();
        let query = parse("SELECT order_id, sum(price) FROM item_table GROUP BY order_id").unwrap();
        let synthetic_data = SyntheticData::new(Hierarchy::from([
            (vec!["item_table"], Identifier::from("item_table")),
            (vec!["order_table"], Identifier::from("order_table")),
            (vec!["user_table"], Identifier::from("user_table")),
        ]));
        let protected_entity = ProtectedEntity::from(vec![
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
        let budget = Budget::new(1., 1e-3);
        let relation = Relation::try_from(query.with(&relations)).unwrap();
        let relation_with_private_query = relation
            .rewrite_with_differential_privacy(&relations, synthetic_data, protected_entity, budget)
            .unwrap();
        relation_with_private_query
            .relation()
            .display_dot()
            .unwrap();
        println!(
            "PrivateQuery = {}",
            relation_with_private_query.private_query()
        );
    }

    #[test]
    fn test_rewrite_as_protected_entity_preserving() {
        let database = postgresql::test_database();
        let relations = database.relations();
        let query = parse("SELECT order_id, price FROM item_table").unwrap();
        let synthetic_data = SyntheticData::new(Hierarchy::from([
            (vec!["item_table"], Identifier::from("item_table")),
            (vec!["order_table"], Identifier::from("order_table")),
            (vec!["user_table"], Identifier::from("user_table")),
        ]));
        let protected_entity = ProtectedEntity::from(vec![
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
        let budget = Budget::new(1., 1e-3);
        let relation = Relation::try_from(query.with(&relations)).unwrap();
        let relation_with_private_query = relation
            .rewrite_as_protected_entity_preserving(
                &relations,
                synthetic_data,
                protected_entity,
                budget,
            )
            .unwrap();
        relation_with_private_query
            .relation()
            .display_dot()
            .unwrap();
        println!(
            "PrivateQuery = {}",
            relation_with_private_query.private_query()
        );
    }
}
