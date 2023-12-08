pub mod dot;
pub mod relation_with_attributes;
pub mod rewriting_rule;

pub use relation_with_attributes::RelationWithAttributes;
pub use rewriting_rule::{
    Property, RelationWithPrivateQuery, RelationWithRewritingRule, RelationWithRewritingRules,
    RewritingRule,
};

use std::{error, fmt, result, sync::Arc};

use crate::{
    differential_privacy::budget::Budget,
    hierarchy::Hierarchy,
    privacy_unit_tracking::privacy_unit::PrivacyUnit,
    relation::Relation,
    synthetic_data::SyntheticData,
    visitor::Acceptor, display::Dot,
};

use rewriting_rule::{
    Rewriter, RewritingRulesEliminator, RewritingRulesSelector, RewritingRulesSetter, Score,
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
    /// Rewrite the query so that the privacy unit is tracked through the query.
    pub fn rewrite_as_privacy_unit_preserving<'a>(
        &'a self,
        relations: &'a Hierarchy<Arc<Relation>>,
        synthetic_data: SyntheticData,
        privacy_unit: PrivacyUnit,
        budget: Budget,
    ) -> Result<RelationWithPrivateQuery> {
        let relation_with_rules = self.set_rewriting_rules(RewritingRulesSetter::new(
            relations,
            synthetic_data,
            privacy_unit,
            budget,
        ));
        let relation_with_rules = relation_with_rules.map_rewriting_rules(RewritingRulesEliminator);
        relation_with_rules
            .select_rewriting_rules(RewritingRulesSelector)
            .into_iter()
            .filter_map(|rwrr| match rwrr.attributes().output() {
                Property::Public | Property::PrivacyUnitPreserving => {
                    Some((rwrr.rewrite(Rewriter::new(relations)), rwrr.accept(Score)))
                }
                property => None,
            })
            .max_by_key(|&(_, value)| value.partial_cmp(&value).unwrap())
            .map(|(relation, _)| relation)
            .ok_or_else(|| Error::unreachable_property("privacy_unit_preserving"))
    }
    /// Rewrite the query so that it is differentially private.
    pub fn rewrite_with_differential_privacy<'a>(
        &'a self,
        relations: &'a Hierarchy<Arc<Relation>>,
        synthetic_data: SyntheticData,
        privacy_unit: PrivacyUnit,
        budget: Budget,
    ) -> Result<RelationWithPrivateQuery> {
        let relation_with_rules = self.set_rewriting_rules(RewritingRulesSetter::new(
            relations,
            synthetic_data,
            privacy_unit,
            budget,
        ));
        let relation_with_rules = relation_with_rules.map_rewriting_rules(RewritingRulesEliminator);
        relation_with_rules
            .select_rewriting_rules(RewritingRulesSelector)
            .into_iter()
            .filter_map(|rwrr| match rwrr.attributes().output() {
                Property::Public | Property::Published | Property::DifferentiallyPrivate | Property::SyntheticData => {
                    Some((rwrr.rewrite(Rewriter::new(relations)), rwrr.accept(Score)))
                }
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
        builder::{Ready, With},
        display::Dot,
        expr::Identifier,
        io::{postgresql, Database},
        sql::parse,
        Relation,
        data_type::DataType,
        relation::{Schema, field::Constraint, Variant},
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
        let budget = Budget::new(1., 1e-3);
        let relation = Relation::try_from(query.with(&relations)).unwrap();
        let relation_with_private_query = relation
            .rewrite_with_differential_privacy(&relations, synthetic_data, privacy_unit, budget)
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
    fn test_rewrite_with_differential_privacy_with_row_privacy() {
        let database = postgresql::test_database();
        let relations = database.relations();
        let query = parse("SELECT order_id, sum(price) FROM item_table GROUP BY order_id").unwrap();
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
                PrivacyUnit::per_row_privacy(),
            ),
            ("order_table", vec![("user_id", "user_table", "id")], PrivacyUnit::per_row_privacy()),
            ("user_table", vec![], PrivacyUnit::per_row_privacy()),
        ]);
        let budget = Budget::new(1., 1e-3);
        let relation = Relation::try_from(query.with(&relations)).unwrap();
        let relation_with_private_query = relation
            .rewrite_with_differential_privacy(&relations, synthetic_data, privacy_unit, budget)
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
    fn test_rewrite_as_privacy_unit_preserving() {
        let database = postgresql::test_database();
        let relations = database.relations();
        let query = parse("SELECT order_id, price FROM item_table").unwrap();
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
        let budget = Budget::new(1., 1e-3);
        let relation = Relation::try_from(query.with(&relations)).unwrap();
        let relation_with_private_query = relation
            .rewrite_as_privacy_unit_preserving(&relations, synthetic_data, privacy_unit, budget)
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
    fn test_rewrite_as_privacy_unit_preserving_with_row_privacy() {
        let database = postgresql::test_database();
        let relations = database.relations();
        let query = parse("SELECT * FROM order_table").unwrap();
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
                PrivacyUnit::per_row_privacy(),
            ),
            ("order_table", vec![("user_id", "user_table", "id")], PrivacyUnit::per_row_privacy()),
            ("user_table", vec![], PrivacyUnit::per_row_privacy()),
        ]);
        let budget = Budget::new(1., 1e-3);
        let relation = Relation::try_from(query.with(&relations)).unwrap();
        let relation_with_private_query = relation
            .rewrite_as_privacy_unit_preserving(&relations, synthetic_data, privacy_unit, budget)
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
    fn test_retail() {
        let retail_transactions: Relation = Relation::table()
            .name("retail_transactions")
            .schema(
                vec![
                    ("household_id", DataType::integer()),
                    ("store_id", DataType::integer()),
                    ("basket_id", DataType::integer()),
                    ("product_id", DataType::integer()),
                    ("quantity", DataType::integer()),
                    ("sales_value", DataType::float()),
                    ("retail_disc", DataType::float()),
                    ("coupon_disc", DataType::float()),
                    ("coupon_match_disc", DataType::float()),
                    ("week", DataType::text()),
                    ("transaction_timestamp", DataType::date()),
                ]
                .into_iter()
                .collect::<Schema>()
            )
            .size(1000)
            .build();
        let retail_demographics: Relation = Relation::table()
            .name("retail_demographics")
            .schema(
                vec![
                    ("household_id", DataType::integer(), Some(Constraint::Unique)),
                    ("age", DataType::integer(), None),
                    ("income", DataType::float(), None),
                    ("home_ownership", DataType::text(), None),
                    ("marital_status", DataType::text(), None),
                    ("household_size", DataType::integer(), None),
                    ("household_comp", DataType::text(), None),
                    ("kids_count", DataType::integer(), None),
                ]
                .into_iter()
                .collect::<Schema>()
            )
            .size(10000)
            .build();
        let retail_products: Relation = Relation::table()
            .name("retail_products")
            .schema(
                vec![
                    ("product_id", DataType::integer(), Some(Constraint::Unique)),
                    ("manufacturer_id", DataType::integer(), None),
                    ("department", DataType::text(), None),
                    ("brand", DataType::text(), None),
                    ("product_category", DataType::text(), None),
                    ("product_type", DataType::text(), None),
                    ("package_size", DataType::text(), None),
                ]
                .into_iter()
                .collect::<Schema>()
            )
            .size(10000)
            .build();
        let relations: Hierarchy<Arc<Relation>> = vec![retail_transactions, retail_demographics, retail_products]
            .iter()
            .map(|t| (Identifier::from(t.name()), Arc::new(t.clone().into())))
            .collect();
        let synthetic_data = SyntheticData::new(Hierarchy::from([
            (vec!["retail_transactions"], Identifier::from("synthetic_retail_transactions")),
            (vec!["retail_demographics"], Identifier::from("synthetic_retail_demographics")),
            (vec!["retail_products"], Identifier::from("synthetic_retail_products")),
        ]));
        let privacy_unit = PrivacyUnit::from(vec![
            ("retail_demographics", vec![], "household_id"),
            ("retail_transactions", vec![("household_id","retail_demographics","household_id")], "household_id"),
        ]);
        let budget = Budget::new(1., 1e-3);

        let queries = [
            "SELECT COUNT(DISTINCT household_id) AS unique_customers FROM retail_transactions",
            "SELECT * FROM retail_transactions t1 INNER JOIN retail_transactions t2 ON t1.product_id = t2.product_id",
            "SELECT COUNT(*) FROM retail_transactions t INNER JOIN retail_products p ON t.product_id = p.product_id",
            "SELECT * FROM retail_transactions t INNER JOIN retail_products p ON t.product_id = p.product_id",
            "SELECT department, AVG(sales_value) AS average_sales FROM retail_transactions INNER JOIN retail_products ON retail_transactions.product_id = retail_products.product_id GROUP BY department",
            "SELECT * FROM retail_transactions INNER JOIN retail_products ON retail_transactions.product_id = retail_products.product_id",
            "WITH ranked_products AS (SELECT product_id, COUNT(*) AS my_count FROM retail_transactions GROUP BY product_id) SELECT product_id FROM ranked_products ORDER BY my_count",
            //"SELECT t.product_id, p.product_category, COUNT(*) AS purchase_count FROM retail_transactions t INNER JOIN retail_products p ON t.product_id = p.product_id WHERE t.transaction_timestamp < CAST('2023-02-01' AS date) GROUP BY t.product_id, p.product_category",
            "SELECT t.product_id, p.product_category, COUNT(*) AS purchase_count FROM retail_transactions t INNER JOIN retail_products p ON t.product_id = p.product_id WHERE t.transaction_timestamp > '2023-01-01' AND t.transaction_timestamp < '2023-02-01' GROUP BY t.product_id, p.product_category",
            "SELECT p.product_id, p.brand, COUNT(*) FROM retail_products p INNER JOIN retail_transactions t ON p.product_id = t.product_id GROUP BY p.product_id, p.brand",
        ];
        for query_str in queries {
            println!("\n{query_str}");
            let query = parse(query_str).unwrap();
            let relation = Relation::try_from(query.with(&relations)).unwrap();
            //relation.display_dot().unwrap();
            let dp_relation = relation.rewrite_with_differential_privacy(
                &relations,
                synthetic_data.clone(),
                privacy_unit.clone(),
                budget.clone()
            ).unwrap();
            //dp_relation.relation().display_dot().unwrap();
        }

    }
}
