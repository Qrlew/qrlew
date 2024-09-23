pub mod composition;
pub mod dot;
pub mod relation_with_attributes;
pub mod rewriting_rule;

pub use relation_with_attributes::RelationWithAttributes;
pub use rewriting_rule::{
    Property, RelationWithDpEvent, RelationWithRewritingRule, RelationWithRewritingRules,
    RewritingRule,
};

use std::{error, fmt, result, sync::Arc};

use crate::{
    differential_privacy::dp_parameters::DpParameters,
    hierarchy::Hierarchy,
    privacy_unit_tracking::{privacy_unit::PrivacyUnit, Strategy},
    relation::Relation,
    synthetic_data::SyntheticData,
    visitor::Acceptor,
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
    /// If a Strategy is not passed the Strategy::Hard will be used for the
    /// rewriting
    pub fn rewrite_as_privacy_unit_preserving<'a>(
        &'a self,
        relations: &'a Hierarchy<Arc<Relation>>,
        synthetic_data: Option<SyntheticData>,
        privacy_unit: PrivacyUnit,
        dp_parameters: DpParameters,
        strategy: Option<Strategy>,
    ) -> Result<RelationWithDpEvent> {
        let strategy = strategy.unwrap_or(Strategy::Hard);
        let relation_with_rules = self.set_rewriting_rules(RewritingRulesSetter::new(
            relations,
            synthetic_data,
            privacy_unit,
            dp_parameters,
            strategy,
        ));
        let relation_with_rules = relation_with_rules.map_rewriting_rules(RewritingRulesEliminator);
        relation_with_rules
            .select_rewriting_rules(RewritingRulesSelector)
            .into_iter()
            .filter_map(|rwrr| match rwrr.attributes().output() {
                Property::Public | Property::PrivacyUnitPreserving => {
                    Some((rwrr.rewrite(Rewriter::new(relations)), rwrr.accept(Score)))
                }
                _ => None,
            })
            .max_by(|&(_, x), &(_, y)| x.partial_cmp(&y).unwrap())
            .map(|(relation, _)| relation)
            .ok_or_else(|| Error::unreachable_property("privacy_unit_preserving"))
    }
    /// Rewrite the query so that it is differentially private.
    pub fn rewrite_with_differential_privacy<'a>(
        &'a self,
        relations: &'a Hierarchy<Arc<Relation>>,
        synthetic_data: Option<SyntheticData>,
        privacy_unit: PrivacyUnit,
        dp_parameters: DpParameters,
    ) -> Result<RelationWithDpEvent> {
        let relation_with_rules = self.set_rewriting_rules(RewritingRulesSetter::new(
            relations,
            synthetic_data,
            privacy_unit,
            dp_parameters,
            Strategy::Hard,
        ));
        let relation_with_rules = relation_with_rules.map_rewriting_rules(RewritingRulesEliminator);
        relation_with_rules
            .select_rewriting_rules(RewritingRulesSelector)
            .into_iter()
            .filter_map(|rwrr| match rwrr.attributes().output() {
                Property::Public
                | Property::Published
                | Property::DifferentiallyPrivate
                | Property::SyntheticData => {
                    Some((rwrr.rewrite(Rewriter::new(relations)), rwrr.accept(Score)))
                }
                _ => None,
            })
            .max_by(|&(_, x), &(_, y)| x.partial_cmp(&y).unwrap())
            .map(|(relation, _)| relation)
            .ok_or_else(|| Error::unreachable_property("differential_privacy"))
    }
}

#[cfg(test)]
mod tests {
    use chrono::NaiveDate;
    use itertools::Itertools;

    use super::*;
    use crate::{
        ast,
        builder::{Ready, With},
        data_type::DataType,
        display::Dot,
        expr::Identifier,
        io::{postgresql, Database},
        relation::{field::Constraint, Schema, Variant},
        sql::parse,
        Expr, Relation,
    };

    #[test]
    fn test_rewrite() {
        let database = postgresql::test_database();
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
        let mut database = postgresql::test_database();
        let relations = database.relations();

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
            ("table_1", vec![], PrivacyUnit::privacy_unit_row()),
        ]);
        let dp_parameters = DpParameters::from_epsilon_delta(1., 1e-3);

        let queries = [
            "SELECT city, COUNT(*) FROM order_table o JOIN user_table u ON(o.id=u.id) GROUP BY city ORDER BY city",
            "SELECT order_id, sum(price) FROM item_table GROUP BY order_id",
            "SELECT order_id, sum(price), sum(distinct price) FROM item_table GROUP BY order_id HAVING count(*) > 2",
            "SELECT order_id, sum(order_id) FROM item_table GROUP BY order_id",
            "SELECT order_id As my_order, sum(price) FROM item_table GROUP BY my_order",
            "SELECT order_id, MAX(order_id), sum(price) FROM item_table GROUP BY order_id",
            "WITH my_avg AS (SELECT AVG(price) AS avg_price, STDDEV(price) AS std_price FROM item_table WHERE price > 1.) SELECT AVG((price - avg_price) / std_price) FROM item_table CROSS JOIN my_avg WHERE std_price > 1.",
            "WITH my_avg AS (SELECT MIN(price) AS min_price, MAX(price) AS max_price FROM item_table WHERE price > 1.) SELECT AVG(price - min_price) FROM item_table CROSS JOIN my_avg",
        ];

        for q in queries {
            println!("=================================\n{q}");
            let query = parse(q).unwrap();
            let relation = Relation::try_from(query.with(&relations)).unwrap();
            relation.display_dot().unwrap();
            let relation_with_dp_event = relation
                .rewrite_with_differential_privacy(
                    &relations,
                    synthetic_data.clone(),
                    privacy_unit.clone(),
                    dp_parameters.clone(),
                )
                .unwrap();
            relation_with_dp_event.relation().display_dot().unwrap();
            let dp_query = ast::Query::from(&relation_with_dp_event.relation().clone()).to_string();
            println!("\n{dp_query}");
            _ = database
                .query(dp_query.as_str())
                .unwrap()
                .iter()
                .map(ToString::to_string)
                .join("\n");
        }
    }

    #[test]
    fn test_rewrite_with_differential_privacy_no_synthetic_data() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

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
            ("table_1", vec![], PrivacyUnit::privacy_unit_row()),
        ]);
        let dp_parameters = DpParameters::from_epsilon_delta(1., 1e-3);

        let queries = [
            "SELECT order_id, sum(price) FROM item_table GROUP BY order_id",
            "SELECT order_id, sum(price), sum(distinct price) FROM item_table GROUP BY order_id HAVING count(*) > 2",
            "SELECT order_id, sum(order_id) FROM item_table GROUP BY order_id",
            "SELECT order_id As my_order, sum(price) FROM item_table GROUP BY my_order",
            "SELECT order_id, MAX(order_id), sum(price) FROM item_table GROUP BY order_id",
            "WITH my_avg AS (SELECT AVG(price) AS avg_price, STDDEV(price) AS std_price FROM item_table WHERE price > 1.) SELECT AVG((price - avg_price) / std_price) FROM item_table CROSS JOIN my_avg WHERE std_price > 1.",
        ];

        for q in queries {
            println!("=================================\n{q}");
            let query = parse(q).unwrap();
            let relation = Relation::try_from(query.with(&relations)).unwrap();
            relation.display_dot().unwrap();
            let relation_with_dp_event = relation
                .rewrite_with_differential_privacy(
                    &relations,
                    synthetic_data.clone(),
                    privacy_unit.clone(),
                    dp_parameters.clone(),
                )
                .unwrap();
            relation_with_dp_event.relation().display_dot().unwrap();
            let dp_query = ast::Query::from(&relation_with_dp_event.relation().clone()).to_string();
            println!("\n{dp_query}");
            _ = database
                .query(dp_query.as_str())
                .unwrap()
                .iter()
                .map(ToString::to_string)
                .join("\n");

            // Test re-parsing
            let re_parsed = parse(dp_query.as_str()).unwrap();
            let relation = Relation::try_from(re_parsed.with(&relations)).unwrap();
            let query = ast::Query::from(&relation).to_string();
            _ = database
                .query(query.as_str())
                .unwrap()
                .iter()
                .map(ToString::to_string)
                .join("\n");
        }
    }

    #[test]
    fn test_rewrite_with_differential_privacy_with_row_privacy() {
        let database = postgresql::test_database();
        let relations = database.relations();
        let query = parse("SELECT order_id, sum(price) FROM item_table GROUP BY order_id").unwrap();
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
                PrivacyUnit::privacy_unit_row(),
            ),
            (
                "order_table",
                vec![("user_id", "user_table", "id")],
                PrivacyUnit::privacy_unit_row(),
            ),
            ("user_table", vec![], PrivacyUnit::privacy_unit_row()),
        ]);
        let dp_parameters = DpParameters::from_epsilon_delta(1., 1e-3);
        let relation = Relation::try_from(query.with(&relations)).unwrap();
        let relation_with_dp_event = relation
            .rewrite_with_differential_privacy(
                &relations,
                synthetic_data,
                privacy_unit,
                dp_parameters,
            )
            .unwrap();
        relation_with_dp_event.relation().display_dot().unwrap();
        println!("PrivateQuery = {}", relation_with_dp_event.dp_event());
    }

    #[test]
    fn test_rewrite_as_privacy_unit_preserving() {
        let database = postgresql::test_database();
        let relations = database.relations();
        let query = parse("SELECT order_id, price FROM item_table").unwrap();
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
        let relation_with_dp_event = relation
            .rewrite_as_privacy_unit_preserving(
                &relations,
                synthetic_data,
                privacy_unit,
                dp_parameters,
                None,
            )
            .unwrap();
        relation_with_dp_event.relation().display_dot().unwrap();
        println!("PrivateQuery = {}", relation_with_dp_event.dp_event());
    }

    #[test]
    fn test_rewrite_as_privacy_unit_preserving_with_row_privacy() {
        let database = postgresql::test_database();
        let relations = database.relations();
        let query = parse("SELECT * FROM order_table").unwrap();
        let synthetic_data = Some(SyntheticData::new(Hierarchy::from([
            (vec!["item_table"], Identifier::from("SYNTHETIC_item_table")),
            (
                vec!["order_table"],
                Identifier::from("SYNTHETIC_order_table"),
            ),
            (vec!["user_table"], Identifier::from("SYNTHETIC_user_table")),
        ])));
        let privacy_unit = PrivacyUnit::from(vec![
            (
                "item_table",
                vec![
                    ("order_id", "order_table", "id"),
                    ("user_id", "user_table", "id"),
                ],
                PrivacyUnit::privacy_unit_row(),
            ),
            (
                "order_table",
                vec![("user_id", "user_table", "id")],
                PrivacyUnit::privacy_unit_row(),
            ),
            ("user_table", vec![], PrivacyUnit::privacy_unit_row()),
        ]);
        let dp_parameters = DpParameters::from_epsilon_delta(1., 1e-3);
        let relation = Relation::try_from(query.with(&relations)).unwrap();
        relation.display_dot().unwrap();
        let new_rel = relation.clone().identity_with_field("Ones", Expr::val(1));
        new_rel.display_dot().unwrap();
        let relation_with_dp_event = relation
            .rewrite_as_privacy_unit_preserving(
                &relations,
                synthetic_data,
                privacy_unit,
                dp_parameters,
                None,
            )
            .unwrap();
        relation_with_dp_event.relation().display_dot().unwrap();
        println!("PrivateQuery = {}", relation_with_dp_event.dp_event());
    }

    #[test]
    fn test_retail() {
        let retail_transactions: Relation = Relation::table()
            .name("retail_transactions")
            .schema(
                vec![
                    ("household_id", DataType::integer()),
                    ("store_id", DataType::integer()),
                    ("basket_id", DataType::integer_interval(31198459904, 31950110720) ),
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
                .collect::<Schema>(),
            )
            .size(1000)
            .build();
        let retail_demographics: Relation = Relation::table()
            .name("retail_demographics")
            .schema(
                vec![
                    (
                        "household_id",
                        DataType::integer(),
                        Some(Constraint::Unique),
                    ),
                    //("Birthdate", DataType::date_interval(NaiveDate::from_ymd_opt(1933, 04, 02).unwrap(), NaiveDate::from_ymd_opt(2006, 02, 07).unwrap()), None),
                    ("Birthdate", DataType::date(), None),
                    ("age", DataType::integer(), None),
                    ("income", DataType::float(), None),
                    ("home_ownership", DataType::text(), None),
                    ("marital_status", DataType::text(), None),
                    ("household_size", DataType::integer(), None),
                    ("household_comp", DataType::text(), None),
                    ("kids_count", DataType::integer(), None),
                ]
                .into_iter()
                .collect::<Schema>(),
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
                .collect::<Schema>(),
            )
            .size(10000)
            .build();
        let relations: Hierarchy<Arc<Relation>> =
            vec![retail_transactions, retail_demographics, retail_products]
                .iter()
                .map(|t| (Identifier::from(t.name()), Arc::new(t.clone().into())))
                .collect();
        let synthetic_data = Some(SyntheticData::new(Hierarchy::from([
            (
                vec!["retail_transactions"],
                Identifier::from("synthetic_retail_transactions"),
            ),
            (
                vec!["retail_demographics"],
                Identifier::from("synthetic_retail_demographics"),
            ),
            (
                vec!["retail_products"],
                Identifier::from("synthetic_retail_products"),
            ),
        ])));
        let privacy_unit = PrivacyUnit::from(vec![
            ("retail_demographics", vec![], "household_id"),
            (
                "retail_transactions",
                vec![("household_id", "retail_demographics", "household_id")],
                "household_id",
            ),
        ]);
        let dp_parameters = DpParameters::from_epsilon_delta(1., 1e-3);

        let queries = [
            r#"
WITH
  anon_8 AS (
    SELECT
      "retail_demographics"."income" AS "Birthdate_1"
    FROM
      "retail_demographics"
  ),
  anon_7 AS (
    SELECT
      1.1574074074074073e-05 * CAST(
        EXTRACT(
          epoch
          FROM
            anon_8."Birthdate_1"
        ) AS FLOAT
      ) AS "Birthdate_1"
    FROM
      anon_8
  ),
  anon_5 AS (
    SELECT
      CASE
        WHEN (anon_6."Birthdate_1" BETWEEN -13423 AND -8192) THEN -13423
        WHEN (anon_6."Birthdate_1" BETWEEN -8192 AND -4096) THEN -8192
        WHEN (anon_6."Birthdate_1" BETWEEN -4096 AND -2048) THEN -4096
        WHEN (anon_6."Birthdate_1" BETWEEN -2048 AND -1024) THEN -2048
        WHEN (anon_6."Birthdate_1" BETWEEN -1024 AND -512) THEN -1024
        WHEN (anon_6."Birthdate_1" BETWEEN -512 AND -256) THEN -512
        WHEN (anon_6."Birthdate_1" BETWEEN -256 AND -128) THEN -256
        WHEN (anon_6."Birthdate_1" BETWEEN -128 AND -64) THEN -128
        WHEN (anon_6."Birthdate_1" BETWEEN -64 AND -32) THEN -64
        WHEN (anon_6."Birthdate_1" BETWEEN -32 AND -16) THEN -32
        WHEN (anon_6."Birthdate_1" BETWEEN -16 AND -8) THEN -16
        WHEN (anon_6."Birthdate_1" BETWEEN -8 AND -4) THEN -8
        WHEN (anon_6."Birthdate_1" BETWEEN -4 AND -2) THEN -4
        WHEN (anon_6."Birthdate_1" BETWEEN -2 AND -1) THEN -2
        WHEN (anon_6."Birthdate_1" BETWEEN -1 AND -0.5) THEN -1
        WHEN (anon_6."Birthdate_1" BETWEEN -0.5 AND -0.25) THEN -0.5
        WHEN (anon_6."Birthdate_1" BETWEEN -0.25 AND -0.125) THEN -0.25
        WHEN (anon_6."Birthdate_1" BETWEEN -0.125 AND -0.0625) THEN -0.125
        WHEN (anon_6."Birthdate_1" BETWEEN -0.0625 AND -0.03125) THEN -0.0625
        WHEN (
          anon_6."Birthdate_1" BETWEEN -0.03125 AND -0.015625
        ) THEN -0.03125
        WHEN (
          anon_6."Birthdate_1" BETWEEN -0.015625 AND -0.0078125
        ) THEN -0.015625
        WHEN (
          anon_6."Birthdate_1" BETWEEN -0.0078125 AND -0.00390625
        ) THEN -0.0078125
        WHEN (
          anon_6."Birthdate_1" BETWEEN -0.00390625 AND -0.001953125
        ) THEN -0.00390625
        WHEN (
          anon_6."Birthdate_1" BETWEEN -0.001953125 AND -0.0009765625
        ) THEN -0.001953125
        WHEN (
          anon_6."Birthdate_1" BETWEEN -0.0009765625 AND -0.00048828125
        ) THEN -0.0009765625
        WHEN (
          anon_6."Birthdate_1" BETWEEN -0.00048828125 AND -0.000244140625
        ) THEN -0.00048828125
        WHEN (
          anon_6."Birthdate_1" BETWEEN -0.000244140625 AND -0.0001220703125
        ) THEN -0.000244140625
        WHEN (
          anon_6."Birthdate_1" BETWEEN -0.0001220703125 AND -6.103515625e-05
        ) THEN -0.0001220703125
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.103515625e-05 AND -3.0517578125e-05
        ) THEN -6.103515625e-05
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.0517578125e-05 AND -1.52587890625e-05
        ) THEN -3.0517578125e-05
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.52587890625e-05 AND -7.62939453125e-06
        ) THEN -1.52587890625e-05
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.62939453125e-06 AND -3.814697265625e-06
        ) THEN -7.62939453125e-06
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.814697265625e-06 AND -1.9073486328125e-06
        ) THEN -3.814697265625e-06
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.9073486328125e-06 AND -9.5367431640625e-07
        ) THEN -1.9073486328125e-06
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.5367431640625e-07 AND -4.76837158203125e-07
        ) THEN -9.5367431640625e-07
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.76837158203125e-07 AND -2.384185791015625e-07
        ) THEN -4.76837158203125e-07
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.384185791015625e-07 AND -1.1920928955078125e-07
        ) THEN -2.384185791015625e-07
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.1920928955078125e-07 AND -5.960464477539063e-08
        ) THEN -1.1920928955078125e-07
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.960464477539063e-08 AND -2.9802322387695312e-08
        ) THEN -5.960464477539063e-08
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.9802322387695312e-08 AND -1.4901161193847656e-08
        ) THEN -2.9802322387695312e-08
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.4901161193847656e-08 AND -7.450580596923828e-09
        ) THEN -1.4901161193847656e-08
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.450580596923828e-09 AND -3.725290298461914e-09
        ) THEN -7.450580596923828e-09
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.725290298461914e-09 AND -1.862645149230957e-09
        ) THEN -3.725290298461914e-09
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.862645149230957e-09 AND -9.313225746154785e-10
        ) THEN -1.862645149230957e-09
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.313225746154785e-10 AND -4.656612873077393e-10
        ) THEN -9.313225746154785e-10
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.656612873077393e-10 AND -2.3283064365386963e-10
        ) THEN -4.656612873077393e-10
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.3283064365386963e-10 AND -1.1641532182693481e-10
        ) THEN -2.3283064365386963e-10
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.1641532182693481e-10 AND -5.820766091346741e-11
        ) THEN -1.1641532182693481e-10
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.820766091346741e-11 AND -2.9103830456733704e-11
        ) THEN -5.820766091346741e-11
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.9103830456733704e-11 AND -1.4551915228366852e-11
        ) THEN -2.9103830456733704e-11
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.4551915228366852e-11 AND -7.275957614183426e-12
        ) THEN -1.4551915228366852e-11
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.275957614183426e-12 AND -3.637978807091713e-12
        ) THEN -7.275957614183426e-12
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.637978807091713e-12 AND -1.8189894035458565e-12
        ) THEN -3.637978807091713e-12
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.8189894035458565e-12 AND -9.094947017729282e-13
        ) THEN -1.8189894035458565e-12
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.094947017729282e-13 AND -4.547473508864641e-13
        ) THEN -9.094947017729282e-13
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.547473508864641e-13 AND -2.2737367544323206e-13
        ) THEN -4.547473508864641e-13
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.2737367544323206e-13 AND -1.1368683772161603e-13
        ) THEN -2.2737367544323206e-13
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.1368683772161603e-13 AND -5.684341886080802e-14
        ) THEN -1.1368683772161603e-13
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.684341886080802e-14 AND -2.842170943040401e-14
        ) THEN -5.684341886080802e-14
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.842170943040401e-14 AND -1.4210854715202004e-14
        ) THEN -2.842170943040401e-14
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.4210854715202004e-14 AND -7.105427357601002e-15
        ) THEN -1.4210854715202004e-14
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.105427357601002e-15 AND -3.552713678800501e-15
        ) THEN -7.105427357601002e-15
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.552713678800501e-15 AND -1.7763568394002505e-15
        ) THEN -3.552713678800501e-15
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.7763568394002505e-15 AND -8.881784197001252e-16
        ) THEN -1.7763568394002505e-15
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.881784197001252e-16 AND -4.440892098500626e-16
        ) THEN -8.881784197001252e-16
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.440892098500626e-16 AND -2.220446049250313e-16
        ) THEN -4.440892098500626e-16
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.220446049250313e-16 AND -1.1102230246251565e-16
        ) THEN -2.220446049250313e-16
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.1102230246251565e-16 AND -5.551115123125783e-17
        ) THEN -1.1102230246251565e-16
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.551115123125783e-17 AND -2.7755575615628914e-17
        ) THEN -5.551115123125783e-17
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.7755575615628914e-17 AND -1.3877787807814457e-17
        ) THEN -2.7755575615628914e-17
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.3877787807814457e-17 AND -6.938893903907228e-18
        ) THEN -1.3877787807814457e-17
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.938893903907228e-18 AND -3.469446951953614e-18
        ) THEN -6.938893903907228e-18
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.469446951953614e-18 AND -1.734723475976807e-18
        ) THEN -3.469446951953614e-18
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.734723475976807e-18 AND -8.673617379884035e-19
        ) THEN -1.734723475976807e-18
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.673617379884035e-19 AND -4.336808689942018e-19
        ) THEN -8.673617379884035e-19
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.336808689942018e-19 AND -2.168404344971009e-19
        ) THEN -4.336808689942018e-19
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.168404344971009e-19 AND -1.0842021724855044e-19
        ) THEN -2.168404344971009e-19
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.0842021724855044e-19 AND -5.421010862427522e-20
        ) THEN -1.0842021724855044e-19
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.421010862427522e-20 AND -2.710505431213761e-20
        ) THEN -5.421010862427522e-20
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.710505431213761e-20 AND -1.3552527156068805e-20
        ) THEN -2.710505431213761e-20
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.3552527156068805e-20 AND -6.776263578034403e-21
        ) THEN -1.3552527156068805e-20
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.776263578034403e-21 AND -3.3881317890172014e-21
        ) THEN -6.776263578034403e-21
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.3881317890172014e-21 AND -1.6940658945086007e-21
        ) THEN -3.3881317890172014e-21
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.6940658945086007e-21 AND -8.470329472543003e-22
        ) THEN -1.6940658945086007e-21
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.470329472543003e-22 AND -4.235164736271502e-22
        ) THEN -8.470329472543003e-22
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.235164736271502e-22 AND -2.117582368135751e-22
        ) THEN -4.235164736271502e-22
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.117582368135751e-22 AND -1.0587911840678754e-22
        ) THEN -2.117582368135751e-22
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.0587911840678754e-22 AND -5.293955920339377e-23
        ) THEN -1.0587911840678754e-22
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.293955920339377e-23 AND -2.6469779601696886e-23
        ) THEN -5.293955920339377e-23
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.6469779601696886e-23 AND -1.3234889800848443e-23
        ) THEN -2.6469779601696886e-23
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.3234889800848443e-23 AND -6.617444900424222e-24
        ) THEN -1.3234889800848443e-23
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.617444900424222e-24 AND -3.308722450212111e-24
        ) THEN -6.617444900424222e-24
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.308722450212111e-24 AND -1.6543612251060553e-24
        ) THEN -3.308722450212111e-24
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.6543612251060553e-24 AND -8.271806125530277e-25
        ) THEN -1.6543612251060553e-24
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.271806125530277e-25 AND -4.1359030627651384e-25
        ) THEN -8.271806125530277e-25
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.1359030627651384e-25 AND -2.0679515313825692e-25
        ) THEN -4.1359030627651384e-25
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.0679515313825692e-25 AND -1.0339757656912846e-25
        ) THEN -2.0679515313825692e-25
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.0339757656912846e-25 AND -5.169878828456423e-26
        ) THEN -1.0339757656912846e-25
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.169878828456423e-26 AND -2.5849394142282115e-26
        ) THEN -5.169878828456423e-26
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.5849394142282115e-26 AND -1.2924697071141057e-26
        ) THEN -2.5849394142282115e-26
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.2924697071141057e-26 AND -6.462348535570529e-27
        ) THEN -1.2924697071141057e-26
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.462348535570529e-27 AND -3.2311742677852644e-27
        ) THEN -6.462348535570529e-27
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.2311742677852644e-27 AND -1.6155871338926322e-27
        ) THEN -3.2311742677852644e-27
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.6155871338926322e-27 AND -8.077935669463161e-28
        ) THEN -1.6155871338926322e-27
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.077935669463161e-28 AND -4.0389678347315804e-28
        ) THEN -8.077935669463161e-28
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.0389678347315804e-28 AND -2.0194839173657902e-28
        ) THEN -4.0389678347315804e-28
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.0194839173657902e-28 AND -1.0097419586828951e-28
        ) THEN -2.0194839173657902e-28
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.0097419586828951e-28 AND -5.048709793414476e-29
        ) THEN -1.0097419586828951e-28
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.048709793414476e-29 AND -2.524354896707238e-29
        ) THEN -5.048709793414476e-29
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.524354896707238e-29 AND -1.262177448353619e-29
        ) THEN -2.524354896707238e-29
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.262177448353619e-29 AND -6.310887241768095e-30
        ) THEN -1.262177448353619e-29
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.310887241768095e-30 AND -3.1554436208840472e-30
        ) THEN -6.310887241768095e-30
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.1554436208840472e-30 AND -1.5777218104420236e-30
        ) THEN -3.1554436208840472e-30
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.5777218104420236e-30 AND -7.888609052210118e-31
        ) THEN -1.5777218104420236e-30
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.888609052210118e-31 AND -3.944304526105059e-31
        ) THEN -7.888609052210118e-31
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.944304526105059e-31 AND -1.9721522630525295e-31
        ) THEN -3.944304526105059e-31
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.9721522630525295e-31 AND -9.860761315262648e-32
        ) THEN -1.9721522630525295e-31
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.860761315262648e-32 AND -4.930380657631324e-32
        ) THEN -9.860761315262648e-32
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.930380657631324e-32 AND -2.465190328815662e-32
        ) THEN -4.930380657631324e-32
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.465190328815662e-32 AND -1.232595164407831e-32
        ) THEN -2.465190328815662e-32
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.232595164407831e-32 AND -6.162975822039155e-33
        ) THEN -1.232595164407831e-32
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.162975822039155e-33 AND -3.0814879110195774e-33
        ) THEN -6.162975822039155e-33
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.0814879110195774e-33 AND -1.5407439555097887e-33
        ) THEN -3.0814879110195774e-33
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.5407439555097887e-33 AND -7.703719777548943e-34
        ) THEN -1.5407439555097887e-33
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.703719777548943e-34 AND -3.851859888774472e-34
        ) THEN -7.703719777548943e-34
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.851859888774472e-34 AND -1.925929944387236e-34
        ) THEN -3.851859888774472e-34
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.925929944387236e-34 AND -9.62964972193618e-35
        ) THEN -1.925929944387236e-34
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.62964972193618e-35 AND -4.81482486096809e-35
        ) THEN -9.62964972193618e-35
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.81482486096809e-35 AND -2.407412430484045e-35
        ) THEN -4.81482486096809e-35
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.407412430484045e-35 AND -1.2037062152420224e-35
        ) THEN -2.407412430484045e-35
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.2037062152420224e-35 AND -6.018531076210112e-36
        ) THEN -1.2037062152420224e-35
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.018531076210112e-36 AND -3.009265538105056e-36
        ) THEN -6.018531076210112e-36
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.009265538105056e-36 AND -1.504632769052528e-36
        ) THEN -3.009265538105056e-36
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.504632769052528e-36 AND -7.52316384526264e-37
        ) THEN -1.504632769052528e-36
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.52316384526264e-37 AND -3.76158192263132e-37
        ) THEN -7.52316384526264e-37
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.76158192263132e-37 AND -1.88079096131566e-37
        ) THEN -3.76158192263132e-37
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.88079096131566e-37 AND -9.4039548065783e-38
        ) THEN -1.88079096131566e-37
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.4039548065783e-38 AND -4.70197740328915e-38
        ) THEN -9.4039548065783e-38
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.70197740328915e-38 AND -2.350988701644575e-38
        ) THEN -4.70197740328915e-38
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.350988701644575e-38 AND -1.1754943508222875e-38
        ) THEN -2.350988701644575e-38
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.1754943508222875e-38 AND -5.877471754111438e-39
        ) THEN -1.1754943508222875e-38
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.877471754111438e-39 AND -2.938735877055719e-39
        ) THEN -5.877471754111438e-39
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.938735877055719e-39 AND -1.4693679385278594e-39
        ) THEN -2.938735877055719e-39
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.4693679385278594e-39 AND -7.346839692639297e-40
        ) THEN -1.4693679385278594e-39
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.346839692639297e-40 AND -3.6734198463196485e-40
        ) THEN -7.346839692639297e-40
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.6734198463196485e-40 AND -1.8367099231598242e-40
        ) THEN -3.6734198463196485e-40
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.8367099231598242e-40 AND -9.183549615799121e-41
        ) THEN -1.8367099231598242e-40
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.183549615799121e-41 AND -4.591774807899561e-41
        ) THEN -9.183549615799121e-41
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.591774807899561e-41 AND -2.2958874039497803e-41
        ) THEN -4.591774807899561e-41
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.2958874039497803e-41 AND -1.1479437019748901e-41
        ) THEN -2.2958874039497803e-41
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.1479437019748901e-41 AND -5.739718509874451e-42
        ) THEN -1.1479437019748901e-41
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.739718509874451e-42 AND -2.8698592549372254e-42
        ) THEN -5.739718509874451e-42
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.8698592549372254e-42 AND -1.4349296274686127e-42
        ) THEN -2.8698592549372254e-42
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.4349296274686127e-42 AND -7.174648137343064e-43
        ) THEN -1.4349296274686127e-42
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.174648137343064e-43 AND -3.587324068671532e-43
        ) THEN -7.174648137343064e-43
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.587324068671532e-43 AND -1.793662034335766e-43
        ) THEN -3.587324068671532e-43
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.793662034335766e-43 AND -8.96831017167883e-44
        ) THEN -1.793662034335766e-43
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.96831017167883e-44 AND -4.484155085839415e-44
        ) THEN -8.96831017167883e-44
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.484155085839415e-44 AND -2.2420775429197073e-44
        ) THEN -4.484155085839415e-44
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.2420775429197073e-44 AND -1.1210387714598537e-44
        ) THEN -2.2420775429197073e-44
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.1210387714598537e-44 AND -5.605193857299268e-45
        ) THEN -1.1210387714598537e-44
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.605193857299268e-45 AND -2.802596928649634e-45
        ) THEN -5.605193857299268e-45
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.802596928649634e-45 AND -1.401298464324817e-45
        ) THEN -2.802596928649634e-45
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.401298464324817e-45 AND -7.006492321624085e-46
        ) THEN -1.401298464324817e-45
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.006492321624085e-46 AND -3.503246160812043e-46
        ) THEN -7.006492321624085e-46
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.503246160812043e-46 AND -1.7516230804060213e-46
        ) THEN -3.503246160812043e-46
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.7516230804060213e-46 AND -8.758115402030107e-47
        ) THEN -1.7516230804060213e-46
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.758115402030107e-47 AND -4.3790577010150533e-47
        ) THEN -8.758115402030107e-47
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.3790577010150533e-47 AND -2.1895288505075267e-47
        ) THEN -4.3790577010150533e-47
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.1895288505075267e-47 AND -1.0947644252537633e-47
        ) THEN -2.1895288505075267e-47
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.0947644252537633e-47 AND -5.473822126268817e-48
        ) THEN -1.0947644252537633e-47
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.473822126268817e-48 AND -2.7369110631344083e-48
        ) THEN -5.473822126268817e-48
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.7369110631344083e-48 AND -1.3684555315672042e-48
        ) THEN -2.7369110631344083e-48
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.3684555315672042e-48 AND -6.842277657836021e-49
        ) THEN -1.3684555315672042e-48
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.842277657836021e-49 AND -3.4211388289180104e-49
        ) THEN -6.842277657836021e-49
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.4211388289180104e-49 AND -1.7105694144590052e-49
        ) THEN -3.4211388289180104e-49
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.7105694144590052e-49 AND -8.552847072295026e-50
        ) THEN -1.7105694144590052e-49
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.552847072295026e-50 AND -4.276423536147513e-50
        ) THEN -8.552847072295026e-50
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.276423536147513e-50 AND -2.1382117680737565e-50
        ) THEN -4.276423536147513e-50
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.1382117680737565e-50 AND -1.0691058840368783e-50
        ) THEN -2.1382117680737565e-50
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.0691058840368783e-50 AND -5.345529420184391e-51
        ) THEN -1.0691058840368783e-50
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.345529420184391e-51 AND -2.6727647100921956e-51
        ) THEN -5.345529420184391e-51
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.6727647100921956e-51 AND -1.3363823550460978e-51
        ) THEN -2.6727647100921956e-51
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.3363823550460978e-51 AND -6.681911775230489e-52
        ) THEN -1.3363823550460978e-51
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.681911775230489e-52 AND -3.3409558876152446e-52
        ) THEN -6.681911775230489e-52
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.3409558876152446e-52 AND -1.6704779438076223e-52
        ) THEN -3.3409558876152446e-52
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.6704779438076223e-52 AND -8.352389719038111e-53
        ) THEN -1.6704779438076223e-52
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.352389719038111e-53 AND -4.176194859519056e-53
        ) THEN -8.352389719038111e-53
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.176194859519056e-53 AND -2.088097429759528e-53
        ) THEN -4.176194859519056e-53
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.088097429759528e-53 AND -1.044048714879764e-53
        ) THEN -2.088097429759528e-53
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.044048714879764e-53 AND -5.22024357439882e-54
        ) THEN -1.044048714879764e-53
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.22024357439882e-54 AND -2.61012178719941e-54
        ) THEN -5.22024357439882e-54
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.61012178719941e-54 AND -1.305060893599705e-54
        ) THEN -2.61012178719941e-54
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.305060893599705e-54 AND -6.525304467998525e-55
        ) THEN -1.305060893599705e-54
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.525304467998525e-55 AND -3.2626522339992623e-55
        ) THEN -6.525304467998525e-55
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.2626522339992623e-55 AND -1.6313261169996311e-55
        ) THEN -3.2626522339992623e-55
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.6313261169996311e-55 AND -8.156630584998156e-56
        ) THEN -1.6313261169996311e-55
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.156630584998156e-56 AND -4.078315292499078e-56
        ) THEN -8.156630584998156e-56
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.078315292499078e-56 AND -2.039157646249539e-56
        ) THEN -4.078315292499078e-56
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.039157646249539e-56 AND -1.0195788231247695e-56
        ) THEN -2.039157646249539e-56
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.0195788231247695e-56 AND -5.0978941156238473e-57
        ) THEN -1.0195788231247695e-56
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.0978941156238473e-57 AND -2.5489470578119236e-57
        ) THEN -5.0978941156238473e-57
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.5489470578119236e-57 AND -1.2744735289059618e-57
        ) THEN -2.5489470578119236e-57
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.2744735289059618e-57 AND -6.372367644529809e-58
        ) THEN -1.2744735289059618e-57
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.372367644529809e-58 AND -3.1861838222649046e-58
        ) THEN -6.372367644529809e-58
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.1861838222649046e-58 AND -1.5930919111324523e-58
        ) THEN -3.1861838222649046e-58
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.5930919111324523e-58 AND -7.965459555662261e-59
        ) THEN -1.5930919111324523e-58
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.965459555662261e-59 AND -3.982729777831131e-59
        ) THEN -7.965459555662261e-59
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.982729777831131e-59 AND -1.9913648889155653e-59
        ) THEN -3.982729777831131e-59
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.9913648889155653e-59 AND -9.956824444577827e-60
        ) THEN -1.9913648889155653e-59
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.956824444577827e-60 AND -4.9784122222889134e-60
        ) THEN -9.956824444577827e-60
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.9784122222889134e-60 AND -2.4892061111444567e-60
        ) THEN -4.9784122222889134e-60
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.4892061111444567e-60 AND -1.2446030555722283e-60
        ) THEN -2.4892061111444567e-60
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.2446030555722283e-60 AND -6.223015277861142e-61
        ) THEN -1.2446030555722283e-60
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.223015277861142e-61 AND -3.111507638930571e-61
        ) THEN -6.223015277861142e-61
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.111507638930571e-61 AND -1.5557538194652854e-61
        ) THEN -3.111507638930571e-61
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.5557538194652854e-61 AND -7.778769097326427e-62
        ) THEN -1.5557538194652854e-61
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.778769097326427e-62 AND -3.8893845486632136e-62
        ) THEN -7.778769097326427e-62
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.8893845486632136e-62 AND -1.9446922743316068e-62
        ) THEN -3.8893845486632136e-62
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.9446922743316068e-62 AND -9.723461371658034e-63
        ) THEN -1.9446922743316068e-62
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.723461371658034e-63 AND -4.861730685829017e-63
        ) THEN -9.723461371658034e-63
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.861730685829017e-63 AND -2.4308653429145085e-63
        ) THEN -4.861730685829017e-63
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.4308653429145085e-63 AND -1.2154326714572542e-63
        ) THEN -2.4308653429145085e-63
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.2154326714572542e-63 AND -6.077163357286271e-64
        ) THEN -1.2154326714572542e-63
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.077163357286271e-64 AND -3.0385816786431356e-64
        ) THEN -6.077163357286271e-64
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.0385816786431356e-64 AND -1.5192908393215678e-64
        ) THEN -3.0385816786431356e-64
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.5192908393215678e-64 AND -7.596454196607839e-65
        ) THEN -1.5192908393215678e-64
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.596454196607839e-65 AND -3.7982270983039195e-65
        ) THEN -7.596454196607839e-65
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.7982270983039195e-65 AND -1.8991135491519597e-65
        ) THEN -3.7982270983039195e-65
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.8991135491519597e-65 AND -9.495567745759799e-66
        ) THEN -1.8991135491519597e-65
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.495567745759799e-66 AND -4.7477838728798994e-66
        ) THEN -9.495567745759799e-66
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.7477838728798994e-66 AND -2.3738919364399497e-66
        ) THEN -4.7477838728798994e-66
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.3738919364399497e-66 AND -1.1869459682199748e-66
        ) THEN -2.3738919364399497e-66
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.1869459682199748e-66 AND -5.934729841099874e-67
        ) THEN -1.1869459682199748e-66
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.934729841099874e-67 AND -2.967364920549937e-67
        ) THEN -5.934729841099874e-67
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.967364920549937e-67 AND -1.4836824602749686e-67
        ) THEN -2.967364920549937e-67
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.4836824602749686e-67 AND -7.418412301374843e-68
        ) THEN -1.4836824602749686e-67
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.418412301374843e-68 AND -3.7092061506874214e-68
        ) THEN -7.418412301374843e-68
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.7092061506874214e-68 AND -1.8546030753437107e-68
        ) THEN -3.7092061506874214e-68
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.8546030753437107e-68 AND -9.273015376718553e-69
        ) THEN -1.8546030753437107e-68
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.273015376718553e-69 AND -4.636507688359277e-69
        ) THEN -9.273015376718553e-69
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.636507688359277e-69 AND -2.3182538441796384e-69
        ) THEN -4.636507688359277e-69
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.3182538441796384e-69 AND -1.1591269220898192e-69
        ) THEN -2.3182538441796384e-69
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.1591269220898192e-69 AND -5.795634610449096e-70
        ) THEN -1.1591269220898192e-69
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.795634610449096e-70 AND -2.897817305224548e-70
        ) THEN -5.795634610449096e-70
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.897817305224548e-70 AND -1.448908652612274e-70
        ) THEN -2.897817305224548e-70
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.448908652612274e-70 AND -7.24454326306137e-71
        ) THEN -1.448908652612274e-70
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.24454326306137e-71 AND -3.622271631530685e-71
        ) THEN -7.24454326306137e-71
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.622271631530685e-71 AND -1.8111358157653425e-71
        ) THEN -3.622271631530685e-71
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.8111358157653425e-71 AND -9.055679078826712e-72
        ) THEN -1.8111358157653425e-71
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.055679078826712e-72 AND -4.527839539413356e-72
        ) THEN -9.055679078826712e-72
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.527839539413356e-72 AND -2.263919769706678e-72
        ) THEN -4.527839539413356e-72
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.263919769706678e-72 AND -1.131959884853339e-72
        ) THEN -2.263919769706678e-72
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.131959884853339e-72 AND -5.659799424266695e-73
        ) THEN -1.131959884853339e-72
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.659799424266695e-73 AND -2.8298997121333476e-73
        ) THEN -5.659799424266695e-73
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.8298997121333476e-73 AND -1.4149498560666738e-73
        ) THEN -2.8298997121333476e-73
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.4149498560666738e-73 AND -7.074749280333369e-74
        ) THEN -1.4149498560666738e-73
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.074749280333369e-74 AND -3.5373746401666845e-74
        ) THEN -7.074749280333369e-74
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.5373746401666845e-74 AND -1.7686873200833423e-74
        ) THEN -3.5373746401666845e-74
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.7686873200833423e-74 AND -8.843436600416711e-75
        ) THEN -1.7686873200833423e-74
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.843436600416711e-75 AND -4.421718300208356e-75
        ) THEN -8.843436600416711e-75
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.421718300208356e-75 AND -2.210859150104178e-75
        ) THEN -4.421718300208356e-75
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.210859150104178e-75 AND -1.105429575052089e-75
        ) THEN -2.210859150104178e-75
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.105429575052089e-75 AND -5.527147875260445e-76
        ) THEN -1.105429575052089e-75
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.527147875260445e-76 AND -2.7635739376302223e-76
        ) THEN -5.527147875260445e-76
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.7635739376302223e-76 AND -1.3817869688151111e-76
        ) THEN -2.7635739376302223e-76
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.3817869688151111e-76 AND -6.908934844075556e-77
        ) THEN -1.3817869688151111e-76
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.908934844075556e-77 AND -3.454467422037778e-77
        ) THEN -6.908934844075556e-77
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.454467422037778e-77 AND -1.727233711018889e-77
        ) THEN -3.454467422037778e-77
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.727233711018889e-77 AND -8.636168555094445e-78
        ) THEN -1.727233711018889e-77
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.636168555094445e-78 AND -4.3180842775472223e-78
        ) THEN -8.636168555094445e-78
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.3180842775472223e-78 AND -2.1590421387736112e-78
        ) THEN -4.3180842775472223e-78
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.1590421387736112e-78 AND -1.0795210693868056e-78
        ) THEN -2.1590421387736112e-78
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.0795210693868056e-78 AND -5.397605346934028e-79
        ) THEN -1.0795210693868056e-78
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.397605346934028e-79 AND -2.698802673467014e-79
        ) THEN -5.397605346934028e-79
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.698802673467014e-79 AND -1.349401336733507e-79
        ) THEN -2.698802673467014e-79
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.349401336733507e-79 AND -6.747006683667535e-80
        ) THEN -1.349401336733507e-79
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.747006683667535e-80 AND -3.3735033418337674e-80
        ) THEN -6.747006683667535e-80
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.3735033418337674e-80 AND -1.6867516709168837e-80
        ) THEN -3.3735033418337674e-80
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.6867516709168837e-80 AND -8.433758354584419e-81
        ) THEN -1.6867516709168837e-80
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.433758354584419e-81 AND -4.2168791772922093e-81
        ) THEN -8.433758354584419e-81
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.2168791772922093e-81 AND -2.1084395886461046e-81
        ) THEN -4.2168791772922093e-81
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.1084395886461046e-81 AND -1.0542197943230523e-81
        ) THEN -2.1084395886461046e-81
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.0542197943230523e-81 AND -5.271098971615262e-82
        ) THEN -1.0542197943230523e-81
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.271098971615262e-82 AND -2.635549485807631e-82
        ) THEN -5.271098971615262e-82
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.635549485807631e-82 AND -1.3177747429038154e-82
        ) THEN -2.635549485807631e-82
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.3177747429038154e-82 AND -6.588873714519077e-83
        ) THEN -1.3177747429038154e-82
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.588873714519077e-83 AND -3.2944368572595385e-83
        ) THEN -6.588873714519077e-83
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.2944368572595385e-83 AND -1.6472184286297693e-83
        ) THEN -3.2944368572595385e-83
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.6472184286297693e-83 AND -8.236092143148846e-84
        ) THEN -1.6472184286297693e-83
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.236092143148846e-84 AND -4.118046071574423e-84
        ) THEN -8.236092143148846e-84
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.118046071574423e-84 AND -2.0590230357872116e-84
        ) THEN -4.118046071574423e-84
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.0590230357872116e-84 AND -1.0295115178936058e-84
        ) THEN -2.0590230357872116e-84
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.0295115178936058e-84 AND -5.147557589468029e-85
        ) THEN -1.0295115178936058e-84
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.147557589468029e-85 AND -2.5737787947340145e-85
        ) THEN -5.147557589468029e-85
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.5737787947340145e-85 AND -1.2868893973670072e-85
        ) THEN -2.5737787947340145e-85
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.2868893973670072e-85 AND -6.434446986835036e-86
        ) THEN -1.2868893973670072e-85
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.434446986835036e-86 AND -3.217223493417518e-86
        ) THEN -6.434446986835036e-86
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.217223493417518e-86 AND -1.608611746708759e-86
        ) THEN -3.217223493417518e-86
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.608611746708759e-86 AND -8.043058733543795e-87
        ) THEN -1.608611746708759e-86
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.043058733543795e-87 AND -4.021529366771898e-87
        ) THEN -8.043058733543795e-87
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.021529366771898e-87 AND -2.010764683385949e-87
        ) THEN -4.021529366771898e-87
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.010764683385949e-87 AND -1.0053823416929744e-87
        ) THEN -2.010764683385949e-87
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.0053823416929744e-87 AND -5.026911708464872e-88
        ) THEN -1.0053823416929744e-87
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.026911708464872e-88 AND -2.513455854232436e-88
        ) THEN -5.026911708464872e-88
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.513455854232436e-88 AND -1.256727927116218e-88
        ) THEN -2.513455854232436e-88
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.256727927116218e-88 AND -6.28363963558109e-89
        ) THEN -1.256727927116218e-88
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.28363963558109e-89 AND -3.141819817790545e-89
        ) THEN -6.28363963558109e-89
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.141819817790545e-89 AND -1.5709099088952725e-89
        ) THEN -3.141819817790545e-89
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.5709099088952725e-89 AND -7.854549544476363e-90
        ) THEN -1.5709099088952725e-89
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.854549544476363e-90 AND -3.9272747722381812e-90
        ) THEN -7.854549544476363e-90
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.9272747722381812e-90 AND -1.9636373861190906e-90
        ) THEN -3.9272747722381812e-90
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.9636373861190906e-90 AND -9.818186930595453e-91
        ) THEN -1.9636373861190906e-90
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.818186930595453e-91 AND -4.909093465297727e-91
        ) THEN -9.818186930595453e-91
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.909093465297727e-91 AND -2.4545467326488633e-91
        ) THEN -4.909093465297727e-91
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.4545467326488633e-91 AND -1.2272733663244316e-91
        ) THEN -2.4545467326488633e-91
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.2272733663244316e-91 AND -6.136366831622158e-92
        ) THEN -1.2272733663244316e-91
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.136366831622158e-92 AND -3.068183415811079e-92
        ) THEN -6.136366831622158e-92
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.068183415811079e-92 AND -1.5340917079055395e-92
        ) THEN -3.068183415811079e-92
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.5340917079055395e-92 AND -7.670458539527698e-93
        ) THEN -1.5340917079055395e-92
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.670458539527698e-93 AND -3.835229269763849e-93
        ) THEN -7.670458539527698e-93
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.835229269763849e-93 AND -1.9176146348819244e-93
        ) THEN -3.835229269763849e-93
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.9176146348819244e-93 AND -9.588073174409622e-94
        ) THEN -1.9176146348819244e-93
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.588073174409622e-94 AND -4.794036587204811e-94
        ) THEN -9.588073174409622e-94
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.794036587204811e-94 AND -2.3970182936024055e-94
        ) THEN -4.794036587204811e-94
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.3970182936024055e-94 AND -1.1985091468012028e-94
        ) THEN -2.3970182936024055e-94
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.1985091468012028e-94 AND -5.992545734006014e-95
        ) THEN -1.1985091468012028e-94
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.992545734006014e-95 AND -2.996272867003007e-95
        ) THEN -5.992545734006014e-95
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.996272867003007e-95 AND -1.4981364335015035e-95
        ) THEN -2.996272867003007e-95
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.4981364335015035e-95 AND -7.490682167507517e-96
        ) THEN -1.4981364335015035e-95
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.490682167507517e-96 AND -3.745341083753759e-96
        ) THEN -7.490682167507517e-96
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.745341083753759e-96 AND -1.8726705418768793e-96
        ) THEN -3.745341083753759e-96
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.8726705418768793e-96 AND -9.363352709384397e-97
        ) THEN -1.8726705418768793e-96
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.363352709384397e-97 AND -4.6816763546921983e-97
        ) THEN -9.363352709384397e-97
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.6816763546921983e-97 AND -2.3408381773460992e-97
        ) THEN -4.6816763546921983e-97
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.3408381773460992e-97 AND -1.1704190886730496e-97
        ) THEN -2.3408381773460992e-97
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.1704190886730496e-97 AND -5.852095443365248e-98
        ) THEN -1.1704190886730496e-97
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.852095443365248e-98 AND -2.926047721682624e-98
        ) THEN -5.852095443365248e-98
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.926047721682624e-98 AND -1.463023860841312e-98
        ) THEN -2.926047721682624e-98
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.463023860841312e-98 AND -7.31511930420656e-99
        ) THEN -1.463023860841312e-98
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.31511930420656e-99 AND -3.65755965210328e-99
        ) THEN -7.31511930420656e-99
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.65755965210328e-99 AND -1.82877982605164e-99
        ) THEN -3.65755965210328e-99
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.82877982605164e-99 AND -9.1438991302582e-100
        ) THEN -1.82877982605164e-99
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.1438991302582e-100 AND -4.5719495651291e-100
        ) THEN -9.1438991302582e-100
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.5719495651291e-100 AND -2.28597478256455e-100
        ) THEN -4.5719495651291e-100
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.28597478256455e-100 AND -1.142987391282275e-100
        ) THEN -2.28597478256455e-100
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.142987391282275e-100 AND -5.714936956411375e-101
        ) THEN -1.142987391282275e-100
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.714936956411375e-101 AND -2.8574684782056875e-101
        ) THEN -5.714936956411375e-101
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.8574684782056875e-101 AND -1.4287342391028437e-101
        ) THEN -2.8574684782056875e-101
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.4287342391028437e-101 AND -7.143671195514219e-102
        ) THEN -1.4287342391028437e-101
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.143671195514219e-102 AND -3.5718355977571093e-102
        ) THEN -7.143671195514219e-102
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.5718355977571093e-102 AND -1.7859177988785547e-102
        ) THEN -3.5718355977571093e-102
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.7859177988785547e-102 AND -8.929588994392773e-103
        ) THEN -1.7859177988785547e-102
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.929588994392773e-103 AND -4.464794497196387e-103
        ) THEN -8.929588994392773e-103
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.464794497196387e-103 AND -2.2323972485981933e-103
        ) THEN -4.464794497196387e-103
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.2323972485981933e-103 AND -1.1161986242990967e-103
        ) THEN -2.2323972485981933e-103
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.1161986242990967e-103 AND -5.5809931214954833e-104
        ) THEN -1.1161986242990967e-103
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.5809931214954833e-104 AND -2.7904965607477417e-104
        ) THEN -5.5809931214954833e-104
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.7904965607477417e-104 AND -1.3952482803738708e-104
        ) THEN -2.7904965607477417e-104
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.3952482803738708e-104 AND -6.976241401869354e-105
        ) THEN -1.3952482803738708e-104
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.976241401869354e-105 AND -3.488120700934677e-105
        ) THEN -6.976241401869354e-105
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.488120700934677e-105 AND -1.7440603504673385e-105
        ) THEN -3.488120700934677e-105
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.7440603504673385e-105 AND -8.720301752336693e-106
        ) THEN -1.7440603504673385e-105
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.720301752336693e-106 AND -4.3601508761683463e-106
        ) THEN -8.720301752336693e-106
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.3601508761683463e-106 AND -2.1800754380841732e-106
        ) THEN -4.3601508761683463e-106
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.1800754380841732e-106 AND -1.0900377190420866e-106
        ) THEN -2.1800754380841732e-106
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.0900377190420866e-106 AND -5.450188595210433e-107
        ) THEN -1.0900377190420866e-106
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.450188595210433e-107 AND -2.7250942976052165e-107
        ) THEN -5.450188595210433e-107
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.7250942976052165e-107 AND -1.3625471488026082e-107
        ) THEN -2.7250942976052165e-107
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.3625471488026082e-107 AND -6.812735744013041e-108
        ) THEN -1.3625471488026082e-107
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.812735744013041e-108 AND -3.4063678720065206e-108
        ) THEN -6.812735744013041e-108
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.4063678720065206e-108 AND -1.7031839360032603e-108
        ) THEN -3.4063678720065206e-108
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.7031839360032603e-108 AND -8.515919680016301e-109
        ) THEN -1.7031839360032603e-108
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.515919680016301e-109 AND -4.257959840008151e-109
        ) THEN -8.515919680016301e-109
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.257959840008151e-109 AND -2.1289799200040754e-109
        ) THEN -4.257959840008151e-109
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.1289799200040754e-109 AND -1.0644899600020377e-109
        ) THEN -2.1289799200040754e-109
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.0644899600020377e-109 AND -5.3224498000101884e-110
        ) THEN -1.0644899600020377e-109
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.3224498000101884e-110 AND -2.6612249000050942e-110
        ) THEN -5.3224498000101884e-110
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.6612249000050942e-110 AND -1.3306124500025471e-110
        ) THEN -2.6612249000050942e-110
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.3306124500025471e-110 AND -6.653062250012736e-111
        ) THEN -1.3306124500025471e-110
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.653062250012736e-111 AND -3.326531125006368e-111
        ) THEN -6.653062250012736e-111
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.326531125006368e-111 AND -1.663265562503184e-111
        ) THEN -3.326531125006368e-111
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.663265562503184e-111 AND -8.31632781251592e-112
        ) THEN -1.663265562503184e-111
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.31632781251592e-112 AND -4.15816390625796e-112
        ) THEN -8.31632781251592e-112
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.15816390625796e-112 AND -2.07908195312898e-112
        ) THEN -4.15816390625796e-112
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.07908195312898e-112 AND -1.03954097656449e-112
        ) THEN -2.07908195312898e-112
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.03954097656449e-112 AND -5.19770488282245e-113
        ) THEN -1.03954097656449e-112
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.19770488282245e-113 AND -2.598852441411225e-113
        ) THEN -5.19770488282245e-113
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.598852441411225e-113 AND -1.2994262207056124e-113
        ) THEN -2.598852441411225e-113
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.2994262207056124e-113 AND -6.497131103528062e-114
        ) THEN -1.2994262207056124e-113
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.497131103528062e-114 AND -3.248565551764031e-114
        ) THEN -6.497131103528062e-114
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.248565551764031e-114 AND -1.6242827758820155e-114
        ) THEN -3.248565551764031e-114
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.6242827758820155e-114 AND -8.121413879410078e-115
        ) THEN -1.6242827758820155e-114
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.121413879410078e-115 AND -4.060706939705039e-115
        ) THEN -8.121413879410078e-115
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.060706939705039e-115 AND -2.0303534698525194e-115
        ) THEN -4.060706939705039e-115
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.0303534698525194e-115 AND -1.0151767349262597e-115
        ) THEN -2.0303534698525194e-115
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.0151767349262597e-115 AND -5.075883674631299e-116
        ) THEN -1.0151767349262597e-115
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.075883674631299e-116 AND -2.5379418373156492e-116
        ) THEN -5.075883674631299e-116
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.5379418373156492e-116 AND -1.2689709186578246e-116
        ) THEN -2.5379418373156492e-116
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.2689709186578246e-116 AND -6.344854593289123e-117
        ) THEN -1.2689709186578246e-116
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.344854593289123e-117 AND -3.1724272966445615e-117
        ) THEN -6.344854593289123e-117
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.1724272966445615e-117 AND -1.5862136483222808e-117
        ) THEN -3.1724272966445615e-117
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.5862136483222808e-117 AND -7.931068241611404e-118
        ) THEN -1.5862136483222808e-117
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.931068241611404e-118 AND -3.965534120805702e-118
        ) THEN -7.931068241611404e-118
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.965534120805702e-118 AND -1.982767060402851e-118
        ) THEN -3.965534120805702e-118
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.982767060402851e-118 AND -9.913835302014255e-119
        ) THEN -1.982767060402851e-118
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.913835302014255e-119 AND -4.9569176510071274e-119
        ) THEN -9.913835302014255e-119
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.9569176510071274e-119 AND -2.4784588255035637e-119
        ) THEN -4.9569176510071274e-119
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.4784588255035637e-119 AND -1.2392294127517818e-119
        ) THEN -2.4784588255035637e-119
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.2392294127517818e-119 AND -6.196147063758909e-120
        ) THEN -1.2392294127517818e-119
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.196147063758909e-120 AND -3.0980735318794546e-120
        ) THEN -6.196147063758909e-120
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.0980735318794546e-120 AND -1.5490367659397273e-120
        ) THEN -3.0980735318794546e-120
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.5490367659397273e-120 AND -7.745183829698637e-121
        ) THEN -1.5490367659397273e-120
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.745183829698637e-121 AND -3.8725919148493183e-121
        ) THEN -7.745183829698637e-121
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.8725919148493183e-121 AND -1.9362959574246591e-121
        ) THEN -3.8725919148493183e-121
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.9362959574246591e-121 AND -9.681479787123296e-122
        ) THEN -1.9362959574246591e-121
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.681479787123296e-122 AND -4.840739893561648e-122
        ) THEN -9.681479787123296e-122
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.840739893561648e-122 AND -2.420369946780824e-122
        ) THEN -4.840739893561648e-122
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.420369946780824e-122 AND -1.210184973390412e-122
        ) THEN -2.420369946780824e-122
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.210184973390412e-122 AND -6.05092486695206e-123
        ) THEN -1.210184973390412e-122
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.05092486695206e-123 AND -3.02546243347603e-123
        ) THEN -6.05092486695206e-123
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.02546243347603e-123 AND -1.512731216738015e-123
        ) THEN -3.02546243347603e-123
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.512731216738015e-123 AND -7.563656083690075e-124
        ) THEN -1.512731216738015e-123
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.563656083690075e-124 AND -3.7818280418450374e-124
        ) THEN -7.563656083690075e-124
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.7818280418450374e-124 AND -1.8909140209225187e-124
        ) THEN -3.7818280418450374e-124
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.8909140209225187e-124 AND -9.454570104612593e-125
        ) THEN -1.8909140209225187e-124
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.454570104612593e-125 AND -4.727285052306297e-125
        ) THEN -9.454570104612593e-125
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.727285052306297e-125 AND -2.3636425261531484e-125
        ) THEN -4.727285052306297e-125
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.3636425261531484e-125 AND -1.1818212630765742e-125
        ) THEN -2.3636425261531484e-125
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.1818212630765742e-125 AND -5.909106315382871e-126
        ) THEN -1.1818212630765742e-125
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.909106315382871e-126 AND -2.9545531576914354e-126
        ) THEN -5.909106315382871e-126
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.9545531576914354e-126 AND -1.4772765788457177e-126
        ) THEN -2.9545531576914354e-126
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.4772765788457177e-126 AND -7.386382894228589e-127
        ) THEN -1.4772765788457177e-126
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.386382894228589e-127 AND -3.6931914471142943e-127
        ) THEN -7.386382894228589e-127
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.6931914471142943e-127 AND -1.8465957235571472e-127
        ) THEN -3.6931914471142943e-127
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.8465957235571472e-127 AND -9.232978617785736e-128
        ) THEN -1.8465957235571472e-127
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.232978617785736e-128 AND -4.616489308892868e-128
        ) THEN -9.232978617785736e-128
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.616489308892868e-128 AND -2.308244654446434e-128
        ) THEN -4.616489308892868e-128
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.308244654446434e-128 AND -1.154122327223217e-128
        ) THEN -2.308244654446434e-128
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.154122327223217e-128 AND -5.770611636116085e-129
        ) THEN -1.154122327223217e-128
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.770611636116085e-129 AND -2.8853058180580424e-129
        ) THEN -5.770611636116085e-129
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.8853058180580424e-129 AND -1.4426529090290212e-129
        ) THEN -2.8853058180580424e-129
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.4426529090290212e-129 AND -7.213264545145106e-130
        ) THEN -1.4426529090290212e-129
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.213264545145106e-130 AND -3.606632272572553e-130
        ) THEN -7.213264545145106e-130
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.606632272572553e-130 AND -1.8033161362862765e-130
        ) THEN -3.606632272572553e-130
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.8033161362862765e-130 AND -9.016580681431383e-131
        ) THEN -1.8033161362862765e-130
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.016580681431383e-131 AND -4.5082903407156913e-131
        ) THEN -9.016580681431383e-131
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.5082903407156913e-131 AND -2.2541451703578456e-131
        ) THEN -4.5082903407156913e-131
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.2541451703578456e-131 AND -1.1270725851789228e-131
        ) THEN -2.2541451703578456e-131
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.1270725851789228e-131 AND -5.635362925894614e-132
        ) THEN -1.1270725851789228e-131
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.635362925894614e-132 AND -2.817681462947307e-132
        ) THEN -5.635362925894614e-132
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.817681462947307e-132 AND -1.4088407314736535e-132
        ) THEN -2.817681462947307e-132
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.4088407314736535e-132 AND -7.044203657368268e-133
        ) THEN -1.4088407314736535e-132
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.044203657368268e-133 AND -3.522101828684134e-133
        ) THEN -7.044203657368268e-133
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.522101828684134e-133 AND -1.761050914342067e-133
        ) THEN -3.522101828684134e-133
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.761050914342067e-133 AND -8.805254571710335e-134
        ) THEN -1.761050914342067e-133
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.805254571710335e-134 AND -4.4026272858551673e-134
        ) THEN -8.805254571710335e-134
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.4026272858551673e-134 AND -2.2013136429275836e-134
        ) THEN -4.4026272858551673e-134
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.2013136429275836e-134 AND -1.1006568214637918e-134
        ) THEN -2.2013136429275836e-134
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.1006568214637918e-134 AND -5.503284107318959e-135
        ) THEN -1.1006568214637918e-134
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.503284107318959e-135 AND -2.7516420536594796e-135
        ) THEN -5.503284107318959e-135
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.7516420536594796e-135 AND -1.3758210268297398e-135
        ) THEN -2.7516420536594796e-135
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.3758210268297398e-135 AND -6.879105134148699e-136
        ) THEN -1.3758210268297398e-135
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.879105134148699e-136 AND -3.4395525670743494e-136
        ) THEN -6.879105134148699e-136
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.4395525670743494e-136 AND -1.7197762835371747e-136
        ) THEN -3.4395525670743494e-136
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.7197762835371747e-136 AND -8.598881417685874e-137
        ) THEN -1.7197762835371747e-136
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.598881417685874e-137 AND -4.299440708842937e-137
        ) THEN -8.598881417685874e-137
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.299440708842937e-137 AND -2.1497203544214684e-137
        ) THEN -4.299440708842937e-137
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.1497203544214684e-137 AND -1.0748601772107342e-137
        ) THEN -2.1497203544214684e-137
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.0748601772107342e-137 AND -5.374300886053671e-138
        ) THEN -1.0748601772107342e-137
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.374300886053671e-138 AND -2.6871504430268355e-138
        ) THEN -5.374300886053671e-138
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.6871504430268355e-138 AND -1.3435752215134178e-138
        ) THEN -2.6871504430268355e-138
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.3435752215134178e-138 AND -6.717876107567089e-139
        ) THEN -1.3435752215134178e-138
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.717876107567089e-139 AND -3.3589380537835444e-139
        ) THEN -6.717876107567089e-139
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.3589380537835444e-139 AND -1.6794690268917722e-139
        ) THEN -3.3589380537835444e-139
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.6794690268917722e-139 AND -8.397345134458861e-140
        ) THEN -1.6794690268917722e-139
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.397345134458861e-140 AND -4.1986725672294305e-140
        ) THEN -8.397345134458861e-140
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.1986725672294305e-140 AND -2.0993362836147152e-140
        ) THEN -4.1986725672294305e-140
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.0993362836147152e-140 AND -1.0496681418073576e-140
        ) THEN -2.0993362836147152e-140
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.0496681418073576e-140 AND -5.248340709036788e-141
        ) THEN -1.0496681418073576e-140
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.248340709036788e-141 AND -2.624170354518394e-141
        ) THEN -5.248340709036788e-141
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.624170354518394e-141 AND -1.312085177259197e-141
        ) THEN -2.624170354518394e-141
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.312085177259197e-141 AND -6.560425886295985e-142
        ) THEN -1.312085177259197e-141
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.560425886295985e-142 AND -3.2802129431479926e-142
        ) THEN -6.560425886295985e-142
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.2802129431479926e-142 AND -1.6401064715739963e-142
        ) THEN -3.2802129431479926e-142
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.6401064715739963e-142 AND -8.200532357869981e-143
        ) THEN -1.6401064715739963e-142
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.200532357869981e-143 AND -4.100266178934991e-143
        ) THEN -8.200532357869981e-143
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.100266178934991e-143 AND -2.0501330894674953e-143
        ) THEN -4.100266178934991e-143
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.0501330894674953e-143 AND -1.0250665447337477e-143
        ) THEN -2.0501330894674953e-143
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.0250665447337477e-143 AND -5.1253327236687384e-144
        ) THEN -1.0250665447337477e-143
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.1253327236687384e-144 AND -2.5626663618343692e-144
        ) THEN -5.1253327236687384e-144
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.5626663618343692e-144 AND -1.2813331809171846e-144
        ) THEN -2.5626663618343692e-144
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.2813331809171846e-144 AND -6.406665904585923e-145
        ) THEN -1.2813331809171846e-144
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.406665904585923e-145 AND -3.2033329522929615e-145
        ) THEN -6.406665904585923e-145
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.2033329522929615e-145 AND -1.6016664761464807e-145
        ) THEN -3.2033329522929615e-145
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.6016664761464807e-145 AND -8.008332380732404e-146
        ) THEN -1.6016664761464807e-145
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.008332380732404e-146 AND -4.004166190366202e-146
        ) THEN -8.008332380732404e-146
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.004166190366202e-146 AND -2.002083095183101e-146
        ) THEN -4.004166190366202e-146
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.002083095183101e-146 AND -1.0010415475915505e-146
        ) THEN -2.002083095183101e-146
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.0010415475915505e-146 AND -5.0052077379577523e-147
        ) THEN -1.0010415475915505e-146
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.0052077379577523e-147 AND -2.5026038689788762e-147
        ) THEN -5.0052077379577523e-147
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.5026038689788762e-147 AND -1.2513019344894381e-147
        ) THEN -2.5026038689788762e-147
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.2513019344894381e-147 AND -6.256509672447191e-148
        ) THEN -1.2513019344894381e-147
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.256509672447191e-148 AND -3.1282548362235952e-148
        ) THEN -6.256509672447191e-148
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.1282548362235952e-148 AND -1.5641274181117976e-148
        ) THEN -3.1282548362235952e-148
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.5641274181117976e-148 AND -7.820637090558988e-149
        ) THEN -1.5641274181117976e-148
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.820637090558988e-149 AND -3.910318545279494e-149
        ) THEN -7.820637090558988e-149
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.910318545279494e-149 AND -1.955159272639747e-149
        ) THEN -3.910318545279494e-149
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.955159272639747e-149 AND -9.775796363198735e-150
        ) THEN -1.955159272639747e-149
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.775796363198735e-150 AND -4.887898181599368e-150
        ) THEN -9.775796363198735e-150
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.887898181599368e-150 AND -2.443949090799684e-150
        ) THEN -4.887898181599368e-150
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.443949090799684e-150 AND -1.221974545399842e-150
        ) THEN -2.443949090799684e-150
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.221974545399842e-150 AND -6.10987272699921e-151
        ) THEN -1.221974545399842e-150
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.10987272699921e-151 AND -3.054936363499605e-151
        ) THEN -6.10987272699921e-151
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.054936363499605e-151 AND -1.5274681817498023e-151
        ) THEN -3.054936363499605e-151
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.5274681817498023e-151 AND -7.637340908749012e-152
        ) THEN -1.5274681817498023e-151
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.637340908749012e-152 AND -3.818670454374506e-152
        ) THEN -7.637340908749012e-152
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.818670454374506e-152 AND -1.909335227187253e-152
        ) THEN -3.818670454374506e-152
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.909335227187253e-152 AND -9.546676135936265e-153
        ) THEN -1.909335227187253e-152
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.546676135936265e-153 AND -4.7733380679681323e-153
        ) THEN -9.546676135936265e-153
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.7733380679681323e-153 AND -2.3866690339840662e-153
        ) THEN -4.7733380679681323e-153
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.3866690339840662e-153 AND -1.1933345169920331e-153
        ) THEN -2.3866690339840662e-153
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.1933345169920331e-153 AND -5.966672584960166e-154
        ) THEN -1.1933345169920331e-153
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.966672584960166e-154 AND -2.983336292480083e-154
        ) THEN -5.966672584960166e-154
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.983336292480083e-154 AND -1.4916681462400413e-154
        ) THEN -2.983336292480083e-154
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.4916681462400413e-154 AND -7.458340731200207e-155
        ) THEN -1.4916681462400413e-154
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.458340731200207e-155 AND -3.7291703656001034e-155
        ) THEN -7.458340731200207e-155
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.7291703656001034e-155 AND -1.8645851828000517e-155
        ) THEN -3.7291703656001034e-155
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.8645851828000517e-155 AND -9.322925914000258e-156
        ) THEN -1.8645851828000517e-155
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.322925914000258e-156 AND -4.661462957000129e-156
        ) THEN -9.322925914000258e-156
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.661462957000129e-156 AND -2.3307314785000646e-156
        ) THEN -4.661462957000129e-156
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.3307314785000646e-156 AND -1.1653657392500323e-156
        ) THEN -2.3307314785000646e-156
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.1653657392500323e-156 AND -5.826828696250162e-157
        ) THEN -1.1653657392500323e-156
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.826828696250162e-157 AND -2.913414348125081e-157
        ) THEN -5.826828696250162e-157
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.913414348125081e-157 AND -1.4567071740625404e-157
        ) THEN -2.913414348125081e-157
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.4567071740625404e-157 AND -7.283535870312702e-158
        ) THEN -1.4567071740625404e-157
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.283535870312702e-158 AND -3.641767935156351e-158
        ) THEN -7.283535870312702e-158
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.641767935156351e-158 AND -1.8208839675781755e-158
        ) THEN -3.641767935156351e-158
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.8208839675781755e-158 AND -9.104419837890877e-159
        ) THEN -1.8208839675781755e-158
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.104419837890877e-159 AND -4.552209918945439e-159
        ) THEN -9.104419837890877e-159
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.552209918945439e-159 AND -2.2761049594727193e-159
        ) THEN -4.552209918945439e-159
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.2761049594727193e-159 AND -1.1380524797363597e-159
        ) THEN -2.2761049594727193e-159
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.1380524797363597e-159 AND -5.6902623986817984e-160
        ) THEN -1.1380524797363597e-159
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.6902623986817984e-160 AND -2.8451311993408992e-160
        ) THEN -5.6902623986817984e-160
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.8451311993408992e-160 AND -1.4225655996704496e-160
        ) THEN -2.8451311993408992e-160
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.4225655996704496e-160 AND -7.112827998352248e-161
        ) THEN -1.4225655996704496e-160
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.112827998352248e-161 AND -3.556413999176124e-161
        ) THEN -7.112827998352248e-161
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.556413999176124e-161 AND -1.778206999588062e-161
        ) THEN -3.556413999176124e-161
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.778206999588062e-161 AND -8.89103499794031e-162
        ) THEN -1.778206999588062e-161
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.89103499794031e-162 AND -4.445517498970155e-162
        ) THEN -8.89103499794031e-162
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.445517498970155e-162 AND -2.2227587494850775e-162
        ) THEN -4.445517498970155e-162
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.2227587494850775e-162 AND -1.1113793747425387e-162
        ) THEN -2.2227587494850775e-162
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.1113793747425387e-162 AND -5.556896873712694e-163
        ) THEN -1.1113793747425387e-162
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.556896873712694e-163 AND -2.778448436856347e-163
        ) THEN -5.556896873712694e-163
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.778448436856347e-163 AND -1.3892242184281734e-163
        ) THEN -2.778448436856347e-163
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.3892242184281734e-163 AND -6.946121092140867e-164
        ) THEN -1.3892242184281734e-163
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.946121092140867e-164 AND -3.4730605460704336e-164
        ) THEN -6.946121092140867e-164
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.4730605460704336e-164 AND -1.7365302730352168e-164
        ) THEN -3.4730605460704336e-164
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.7365302730352168e-164 AND -8.682651365176084e-165
        ) THEN -1.7365302730352168e-164
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.682651365176084e-165 AND -4.341325682588042e-165
        ) THEN -8.682651365176084e-165
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.341325682588042e-165 AND -2.170662841294021e-165
        ) THEN -4.341325682588042e-165
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.170662841294021e-165 AND -1.0853314206470105e-165
        ) THEN -2.170662841294021e-165
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.0853314206470105e-165 AND -5.426657103235053e-166
        ) THEN -1.0853314206470105e-165
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.426657103235053e-166 AND -2.7133285516175262e-166
        ) THEN -5.426657103235053e-166
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.7133285516175262e-166 AND -1.3566642758087631e-166
        ) THEN -2.7133285516175262e-166
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.3566642758087631e-166 AND -6.783321379043816e-167
        ) THEN -1.3566642758087631e-166
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.783321379043816e-167 AND -3.391660689521908e-167
        ) THEN -6.783321379043816e-167
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.391660689521908e-167 AND -1.695830344760954e-167
        ) THEN -3.391660689521908e-167
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.695830344760954e-167 AND -8.47915172380477e-168
        ) THEN -1.695830344760954e-167
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.47915172380477e-168 AND -4.239575861902385e-168
        ) THEN -8.47915172380477e-168
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.239575861902385e-168 AND -2.1197879309511924e-168
        ) THEN -4.239575861902385e-168
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.1197879309511924e-168 AND -1.0598939654755962e-168
        ) THEN -2.1197879309511924e-168
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.0598939654755962e-168 AND -5.299469827377981e-169
        ) THEN -1.0598939654755962e-168
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.299469827377981e-169 AND -2.6497349136889905e-169
        ) THEN -5.299469827377981e-169
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.6497349136889905e-169 AND -1.3248674568444952e-169
        ) THEN -2.6497349136889905e-169
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.3248674568444952e-169 AND -6.624337284222476e-170
        ) THEN -1.3248674568444952e-169
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.624337284222476e-170 AND -3.312168642111238e-170
        ) THEN -6.624337284222476e-170
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.312168642111238e-170 AND -1.656084321055619e-170
        ) THEN -3.312168642111238e-170
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.656084321055619e-170 AND -8.280421605278095e-171
        ) THEN -1.656084321055619e-170
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.280421605278095e-171 AND -4.140210802639048e-171
        ) THEN -8.280421605278095e-171
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.140210802639048e-171 AND -2.070105401319524e-171
        ) THEN -4.140210802639048e-171
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.070105401319524e-171 AND -1.035052700659762e-171
        ) THEN -2.070105401319524e-171
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.035052700659762e-171 AND -5.17526350329881e-172
        ) THEN -1.035052700659762e-171
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.17526350329881e-172 AND -2.587631751649405e-172
        ) THEN -5.17526350329881e-172
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.587631751649405e-172 AND -1.2938158758247024e-172
        ) THEN -2.587631751649405e-172
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.2938158758247024e-172 AND -6.469079379123512e-173
        ) THEN -1.2938158758247024e-172
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.469079379123512e-173 AND -3.234539689561756e-173
        ) THEN -6.469079379123512e-173
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.234539689561756e-173 AND -1.617269844780878e-173
        ) THEN -3.234539689561756e-173
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.617269844780878e-173 AND -8.08634922390439e-174
        ) THEN -1.617269844780878e-173
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.08634922390439e-174 AND -4.043174611952195e-174
        ) THEN -8.08634922390439e-174
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.043174611952195e-174 AND -2.0215873059760975e-174
        ) THEN -4.043174611952195e-174
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.0215873059760975e-174 AND -1.0107936529880487e-174
        ) THEN -2.0215873059760975e-174
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.0107936529880487e-174 AND -5.053968264940244e-175
        ) THEN -1.0107936529880487e-174
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.053968264940244e-175 AND -2.526984132470122e-175
        ) THEN -5.053968264940244e-175
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.526984132470122e-175 AND -1.263492066235061e-175
        ) THEN -2.526984132470122e-175
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.263492066235061e-175 AND -6.317460331175305e-176
        ) THEN -1.263492066235061e-175
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.317460331175305e-176 AND -3.1587301655876523e-176
        ) THEN -6.317460331175305e-176
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.1587301655876523e-176 AND -1.5793650827938261e-176
        ) THEN -3.1587301655876523e-176
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.5793650827938261e-176 AND -7.896825413969131e-177
        ) THEN -1.5793650827938261e-176
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.896825413969131e-177 AND -3.9484127069845653e-177
        ) THEN -7.896825413969131e-177
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.9484127069845653e-177 AND -1.9742063534922827e-177
        ) THEN -3.9484127069845653e-177
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.9742063534922827e-177 AND -9.871031767461413e-178
        ) THEN -1.9742063534922827e-177
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.871031767461413e-178 AND -4.935515883730707e-178
        ) THEN -9.871031767461413e-178
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.935515883730707e-178 AND -2.4677579418653533e-178
        ) THEN -4.935515883730707e-178
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.4677579418653533e-178 AND -1.2338789709326767e-178
        ) THEN -2.4677579418653533e-178
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.2338789709326767e-178 AND -6.169394854663383e-179
        ) THEN -1.2338789709326767e-178
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.169394854663383e-179 AND -3.084697427331692e-179
        ) THEN -6.169394854663383e-179
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.084697427331692e-179 AND -1.542348713665846e-179
        ) THEN -3.084697427331692e-179
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.542348713665846e-179 AND -7.71174356832923e-180
        ) THEN -1.542348713665846e-179
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.71174356832923e-180 AND -3.855871784164615e-180
        ) THEN -7.71174356832923e-180
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.855871784164615e-180 AND -1.9279358920823073e-180
        ) THEN -3.855871784164615e-180
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.9279358920823073e-180 AND -9.639679460411536e-181
        ) THEN -1.9279358920823073e-180
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.639679460411536e-181 AND -4.819839730205768e-181
        ) THEN -9.639679460411536e-181
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.819839730205768e-181 AND -2.409919865102884e-181
        ) THEN -4.819839730205768e-181
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.409919865102884e-181 AND -1.204959932551442e-181
        ) THEN -2.409919865102884e-181
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.204959932551442e-181 AND -6.02479966275721e-182
        ) THEN -1.204959932551442e-181
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.02479966275721e-182 AND -3.012399831378605e-182
        ) THEN -6.02479966275721e-182
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.012399831378605e-182 AND -1.5061999156893026e-182
        ) THEN -3.012399831378605e-182
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.5061999156893026e-182 AND -7.530999578446513e-183
        ) THEN -1.5061999156893026e-182
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.530999578446513e-183 AND -3.7654997892232564e-183
        ) THEN -7.530999578446513e-183
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.7654997892232564e-183 AND -1.8827498946116282e-183
        ) THEN -3.7654997892232564e-183
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.8827498946116282e-183 AND -9.413749473058141e-184
        ) THEN -1.8827498946116282e-183
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.413749473058141e-184 AND -4.706874736529071e-184
        ) THEN -9.413749473058141e-184
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.706874736529071e-184 AND -2.3534373682645353e-184
        ) THEN -4.706874736529071e-184
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.3534373682645353e-184 AND -1.1767186841322676e-184
        ) THEN -2.3534373682645353e-184
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.1767186841322676e-184 AND -5.883593420661338e-185
        ) THEN -1.1767186841322676e-184
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.883593420661338e-185 AND -2.941796710330669e-185
        ) THEN -5.883593420661338e-185
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.941796710330669e-185 AND -1.4708983551653345e-185
        ) THEN -2.941796710330669e-185
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.4708983551653345e-185 AND -7.354491775826673e-186
        ) THEN -1.4708983551653345e-185
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.354491775826673e-186 AND -3.6772458879133364e-186
        ) THEN -7.354491775826673e-186
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.6772458879133364e-186 AND -1.8386229439566682e-186
        ) THEN -3.6772458879133364e-186
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.8386229439566682e-186 AND -9.193114719783341e-187
        ) THEN -1.8386229439566682e-186
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.193114719783341e-187 AND -4.5965573598916705e-187
        ) THEN -9.193114719783341e-187
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.5965573598916705e-187 AND -2.2982786799458352e-187
        ) THEN -4.5965573598916705e-187
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.2982786799458352e-187 AND -1.1491393399729176e-187
        ) THEN -2.2982786799458352e-187
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.1491393399729176e-187 AND -5.745696699864588e-188
        ) THEN -1.1491393399729176e-187
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.745696699864588e-188 AND -2.872848349932294e-188
        ) THEN -5.745696699864588e-188
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.872848349932294e-188 AND -1.436424174966147e-188
        ) THEN -2.872848349932294e-188
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.436424174966147e-188 AND -7.182120874830735e-189
        ) THEN -1.436424174966147e-188
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.182120874830735e-189 AND -3.5910604374153675e-189
        ) THEN -7.182120874830735e-189
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.5910604374153675e-189 AND -1.7955302187076838e-189
        ) THEN -3.5910604374153675e-189
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.7955302187076838e-189 AND -8.977651093538419e-190
        ) THEN -1.7955302187076838e-189
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.977651093538419e-190 AND -4.4888255467692094e-190
        ) THEN -8.977651093538419e-190
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.4888255467692094e-190 AND -2.2444127733846047e-190
        ) THEN -4.4888255467692094e-190
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.2444127733846047e-190 AND -1.1222063866923024e-190
        ) THEN -2.2444127733846047e-190
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.1222063866923024e-190 AND -5.611031933461512e-191
        ) THEN -1.1222063866923024e-190
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.611031933461512e-191 AND -2.805515966730756e-191
        ) THEN -5.611031933461512e-191
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.805515966730756e-191 AND -1.402757983365378e-191
        ) THEN -2.805515966730756e-191
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.402757983365378e-191 AND -7.01378991682689e-192
        ) THEN -1.402757983365378e-191
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.01378991682689e-192 AND -3.506894958413445e-192
        ) THEN -7.01378991682689e-192
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.506894958413445e-192 AND -1.7534474792067224e-192
        ) THEN -3.506894958413445e-192
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.7534474792067224e-192 AND -8.767237396033612e-193
        ) THEN -1.7534474792067224e-192
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.767237396033612e-193 AND -4.383618698016806e-193
        ) THEN -8.767237396033612e-193
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.383618698016806e-193 AND -2.191809349008403e-193
        ) THEN -4.383618698016806e-193
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.191809349008403e-193 AND -1.0959046745042015e-193
        ) THEN -2.191809349008403e-193
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.0959046745042015e-193 AND -5.479523372521008e-194
        ) THEN -1.0959046745042015e-193
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.479523372521008e-194 AND -2.739761686260504e-194
        ) THEN -5.479523372521008e-194
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.739761686260504e-194 AND -1.369880843130252e-194
        ) THEN -2.739761686260504e-194
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.369880843130252e-194 AND -6.84940421565126e-195
        ) THEN -1.369880843130252e-194
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.84940421565126e-195 AND -3.42470210782563e-195
        ) THEN -6.84940421565126e-195
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.42470210782563e-195 AND -1.712351053912815e-195
        ) THEN -3.42470210782563e-195
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.712351053912815e-195 AND -8.561755269564074e-196
        ) THEN -1.712351053912815e-195
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.561755269564074e-196 AND -4.280877634782037e-196
        ) THEN -8.561755269564074e-196
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.280877634782037e-196 AND -2.1404388173910186e-196
        ) THEN -4.280877634782037e-196
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.1404388173910186e-196 AND -1.0702194086955093e-196
        ) THEN -2.1404388173910186e-196
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.0702194086955093e-196 AND -5.351097043477547e-197
        ) THEN -1.0702194086955093e-196
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.351097043477547e-197 AND -2.6755485217387732e-197
        ) THEN -5.351097043477547e-197
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.6755485217387732e-197 AND -1.3377742608693866e-197
        ) THEN -2.6755485217387732e-197
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.3377742608693866e-197 AND -6.688871304346933e-198
        ) THEN -1.3377742608693866e-197
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.688871304346933e-198 AND -3.3444356521734666e-198
        ) THEN -6.688871304346933e-198
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.3444356521734666e-198 AND -1.6722178260867333e-198
        ) THEN -3.3444356521734666e-198
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.6722178260867333e-198 AND -8.361089130433666e-199
        ) THEN -1.6722178260867333e-198
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.361089130433666e-199 AND -4.180544565216833e-199
        ) THEN -8.361089130433666e-199
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.180544565216833e-199 AND -2.0902722826084166e-199
        ) THEN -4.180544565216833e-199
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.0902722826084166e-199 AND -1.0451361413042083e-199
        ) THEN -2.0902722826084166e-199
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.0451361413042083e-199 AND -5.225680706521042e-200
        ) THEN -1.0451361413042083e-199
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.225680706521042e-200 AND -2.612840353260521e-200
        ) THEN -5.225680706521042e-200
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.612840353260521e-200 AND -1.3064201766302604e-200
        ) THEN -2.612840353260521e-200
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.3064201766302604e-200 AND -6.532100883151302e-201
        ) THEN -1.3064201766302604e-200
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.532100883151302e-201 AND -3.266050441575651e-201
        ) THEN -6.532100883151302e-201
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.266050441575651e-201 AND -1.6330252207878255e-201
        ) THEN -3.266050441575651e-201
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.6330252207878255e-201 AND -8.165126103939127e-202
        ) THEN -1.6330252207878255e-201
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.165126103939127e-202 AND -4.082563051969564e-202
        ) THEN -8.165126103939127e-202
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.082563051969564e-202 AND -2.041281525984782e-202
        ) THEN -4.082563051969564e-202
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.041281525984782e-202 AND -1.020640762992391e-202
        ) THEN -2.041281525984782e-202
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.020640762992391e-202 AND -5.103203814961955e-203
        ) THEN -1.020640762992391e-202
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.103203814961955e-203 AND -2.5516019074809773e-203
        ) THEN -5.103203814961955e-203
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.5516019074809773e-203 AND -1.2758009537404886e-203
        ) THEN -2.5516019074809773e-203
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.2758009537404886e-203 AND -6.379004768702443e-204
        ) THEN -1.2758009537404886e-203
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.379004768702443e-204 AND -3.1895023843512216e-204
        ) THEN -6.379004768702443e-204
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.1895023843512216e-204 AND -1.5947511921756108e-204
        ) THEN -3.1895023843512216e-204
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.5947511921756108e-204 AND -7.973755960878054e-205
        ) THEN -1.5947511921756108e-204
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.973755960878054e-205 AND -3.986877980439027e-205
        ) THEN -7.973755960878054e-205
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.986877980439027e-205 AND -1.9934389902195135e-205
        ) THEN -3.986877980439027e-205
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.9934389902195135e-205 AND -9.967194951097568e-206
        ) THEN -1.9934389902195135e-205
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.967194951097568e-206 AND -4.983597475548784e-206
        ) THEN -9.967194951097568e-206
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.983597475548784e-206 AND -2.491798737774392e-206
        ) THEN -4.983597475548784e-206
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.491798737774392e-206 AND -1.245899368887196e-206
        ) THEN -2.491798737774392e-206
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.245899368887196e-206 AND -6.22949684443598e-207
        ) THEN -1.245899368887196e-206
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.22949684443598e-207 AND -3.11474842221799e-207
        ) THEN -6.22949684443598e-207
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.11474842221799e-207 AND -1.557374211108995e-207
        ) THEN -3.11474842221799e-207
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.557374211108995e-207 AND -7.786871055544975e-208
        ) THEN -1.557374211108995e-207
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.786871055544975e-208 AND -3.8934355277724873e-208
        ) THEN -7.786871055544975e-208
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.8934355277724873e-208 AND -1.9467177638862437e-208
        ) THEN -3.8934355277724873e-208
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.9467177638862437e-208 AND -9.733588819431218e-209
        ) THEN -1.9467177638862437e-208
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.733588819431218e-209 AND -4.866794409715609e-209
        ) THEN -9.733588819431218e-209
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.866794409715609e-209 AND -2.4333972048578046e-209
        ) THEN -4.866794409715609e-209
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.4333972048578046e-209 AND -1.2166986024289023e-209
        ) THEN -2.4333972048578046e-209
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.2166986024289023e-209 AND -6.083493012144512e-210
        ) THEN -1.2166986024289023e-209
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.083493012144512e-210 AND -3.041746506072256e-210
        ) THEN -6.083493012144512e-210
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.041746506072256e-210 AND -1.520873253036128e-210
        ) THEN -3.041746506072256e-210
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.520873253036128e-210 AND -7.60436626518064e-211
        ) THEN -1.520873253036128e-210
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.60436626518064e-211 AND -3.80218313259032e-211
        ) THEN -7.60436626518064e-211
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.80218313259032e-211 AND -1.90109156629516e-211
        ) THEN -3.80218313259032e-211
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.90109156629516e-211 AND -9.5054578314758e-212
        ) THEN -1.90109156629516e-211
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.5054578314758e-212 AND -4.7527289157379e-212
        ) THEN -9.5054578314758e-212
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.7527289157379e-212 AND -2.37636445786895e-212
        ) THEN -4.7527289157379e-212
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.37636445786895e-212 AND -1.188182228934475e-212
        ) THEN -2.37636445786895e-212
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.188182228934475e-212 AND -5.940911144672375e-213
        ) THEN -1.188182228934475e-212
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.940911144672375e-213 AND -2.9704555723361872e-213
        ) THEN -5.940911144672375e-213
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.9704555723361872e-213 AND -1.4852277861680936e-213
        ) THEN -2.9704555723361872e-213
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.4852277861680936e-213 AND -7.426138930840468e-214
        ) THEN -1.4852277861680936e-213
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.426138930840468e-214 AND -3.713069465420234e-214
        ) THEN -7.426138930840468e-214
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.713069465420234e-214 AND -1.856534732710117e-214
        ) THEN -3.713069465420234e-214
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.856534732710117e-214 AND -9.282673663550585e-215
        ) THEN -1.856534732710117e-214
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.282673663550585e-215 AND -4.641336831775293e-215
        ) THEN -9.282673663550585e-215
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.641336831775293e-215 AND -2.3206684158876463e-215
        ) THEN -4.641336831775293e-215
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.3206684158876463e-215 AND -1.1603342079438231e-215
        ) THEN -2.3206684158876463e-215
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.1603342079438231e-215 AND -5.801671039719116e-216
        ) THEN -1.1603342079438231e-215
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.801671039719116e-216 AND -2.900835519859558e-216
        ) THEN -5.801671039719116e-216
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.900835519859558e-216 AND -1.450417759929779e-216
        ) THEN -2.900835519859558e-216
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.450417759929779e-216 AND -7.252088799648895e-217
        ) THEN -1.450417759929779e-216
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.252088799648895e-217 AND -3.6260443998244473e-217
        ) THEN -7.252088799648895e-217
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.6260443998244473e-217 AND -1.8130221999122236e-217
        ) THEN -3.6260443998244473e-217
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.8130221999122236e-217 AND -9.065110999561118e-218
        ) THEN -1.8130221999122236e-217
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.065110999561118e-218 AND -4.532555499780559e-218
        ) THEN -9.065110999561118e-218
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.532555499780559e-218 AND -2.2662777498902796e-218
        ) THEN -4.532555499780559e-218
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.2662777498902796e-218 AND -1.1331388749451398e-218
        ) THEN -2.2662777498902796e-218
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.1331388749451398e-218 AND -5.665694374725699e-219
        ) THEN -1.1331388749451398e-218
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.665694374725699e-219 AND -2.8328471873628494e-219
        ) THEN -5.665694374725699e-219
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.8328471873628494e-219 AND -1.4164235936814247e-219
        ) THEN -2.8328471873628494e-219
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.4164235936814247e-219 AND -7.082117968407124e-220
        ) THEN -1.4164235936814247e-219
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.082117968407124e-220 AND -3.541058984203562e-220
        ) THEN -7.082117968407124e-220
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.541058984203562e-220 AND -1.770529492101781e-220
        ) THEN -3.541058984203562e-220
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.770529492101781e-220 AND -8.852647460508905e-221
        ) THEN -1.770529492101781e-220
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.852647460508905e-221 AND -4.4263237302544523e-221
        ) THEN -8.852647460508905e-221
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.4263237302544523e-221 AND -2.2131618651272261e-221
        ) THEN -4.4263237302544523e-221
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.2131618651272261e-221 AND -1.1065809325636131e-221
        ) THEN -2.2131618651272261e-221
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.1065809325636131e-221 AND -5.5329046628180653e-222
        ) THEN -1.1065809325636131e-221
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.5329046628180653e-222 AND -2.7664523314090327e-222
        ) THEN -5.5329046628180653e-222
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.7664523314090327e-222 AND -1.3832261657045163e-222
        ) THEN -2.7664523314090327e-222
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.3832261657045163e-222 AND -6.916130828522582e-223
        ) THEN -1.3832261657045163e-222
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.916130828522582e-223 AND -3.458065414261291e-223
        ) THEN -6.916130828522582e-223
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.458065414261291e-223 AND -1.7290327071306454e-223
        ) THEN -3.458065414261291e-223
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.7290327071306454e-223 AND -8.645163535653227e-224
        ) THEN -1.7290327071306454e-223
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.645163535653227e-224 AND -4.322581767826614e-224
        ) THEN -8.645163535653227e-224
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.322581767826614e-224 AND -2.161290883913307e-224
        ) THEN -4.322581767826614e-224
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.161290883913307e-224 AND -1.0806454419566534e-224
        ) THEN -2.161290883913307e-224
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.0806454419566534e-224 AND -5.403227209783267e-225
        ) THEN -1.0806454419566534e-224
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.403227209783267e-225 AND -2.7016136048916335e-225
        ) THEN -5.403227209783267e-225
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.7016136048916335e-225 AND -1.3508068024458167e-225
        ) THEN -2.7016136048916335e-225
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.3508068024458167e-225 AND -6.754034012229084e-226
        ) THEN -1.3508068024458167e-225
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.754034012229084e-226 AND -3.377017006114542e-226
        ) THEN -6.754034012229084e-226
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.377017006114542e-226 AND -1.688508503057271e-226
        ) THEN -3.377017006114542e-226
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.688508503057271e-226 AND -8.442542515286355e-227
        ) THEN -1.688508503057271e-226
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.442542515286355e-227 AND -4.2212712576431773e-227
        ) THEN -8.442542515286355e-227
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.2212712576431773e-227 AND -2.1106356288215886e-227
        ) THEN -4.2212712576431773e-227
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.1106356288215886e-227 AND -1.0553178144107943e-227
        ) THEN -2.1106356288215886e-227
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.0553178144107943e-227 AND -5.276589072053972e-228
        ) THEN -1.0553178144107943e-227
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.276589072053972e-228 AND -2.638294536026986e-228
        ) THEN -5.276589072053972e-228
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.638294536026986e-228 AND -1.319147268013493e-228
        ) THEN -2.638294536026986e-228
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.319147268013493e-228 AND -6.595736340067465e-229
        ) THEN -1.319147268013493e-228
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.595736340067465e-229 AND -3.2978681700337323e-229
        ) THEN -6.595736340067465e-229
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.2978681700337323e-229 AND -1.6489340850168661e-229
        ) THEN -3.2978681700337323e-229
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.6489340850168661e-229 AND -8.244670425084331e-230
        ) THEN -1.6489340850168661e-229
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.244670425084331e-230 AND -4.1223352125421653e-230
        ) THEN -8.244670425084331e-230
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.1223352125421653e-230 AND -2.0611676062710827e-230
        ) THEN -4.1223352125421653e-230
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.0611676062710827e-230 AND -1.0305838031355413e-230
        ) THEN -2.0611676062710827e-230
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.0305838031355413e-230 AND -5.152919015677707e-231
        ) THEN -1.0305838031355413e-230
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.152919015677707e-231 AND -2.5764595078388533e-231
        ) THEN -5.152919015677707e-231
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.5764595078388533e-231 AND -1.2882297539194267e-231
        ) THEN -2.5764595078388533e-231
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.2882297539194267e-231 AND -6.441148769597133e-232
        ) THEN -1.2882297539194267e-231
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.441148769597133e-232 AND -3.220574384798567e-232
        ) THEN -6.441148769597133e-232
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.220574384798567e-232 AND -1.6102871923992833e-232
        ) THEN -3.220574384798567e-232
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.6102871923992833e-232 AND -8.051435961996417e-233
        ) THEN -1.6102871923992833e-232
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.051435961996417e-233 AND -4.0257179809982083e-233
        ) THEN -8.051435961996417e-233
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.0257179809982083e-233 AND -2.0128589904991042e-233
        ) THEN -4.0257179809982083e-233
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.0128589904991042e-233 AND -1.0064294952495521e-233
        ) THEN -2.0128589904991042e-233
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.0064294952495521e-233 AND -5.0321474762477604e-234
        ) THEN -1.0064294952495521e-233
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.0321474762477604e-234 AND -2.5160737381238802e-234
        ) THEN -5.0321474762477604e-234
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.5160737381238802e-234 AND -1.2580368690619401e-234
        ) THEN -2.5160737381238802e-234
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.2580368690619401e-234 AND -6.290184345309701e-235
        ) THEN -1.2580368690619401e-234
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.290184345309701e-235 AND -3.1450921726548502e-235
        ) THEN -6.290184345309701e-235
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.1450921726548502e-235 AND -1.5725460863274251e-235
        ) THEN -3.1450921726548502e-235
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.5725460863274251e-235 AND -7.862730431637126e-236
        ) THEN -1.5725460863274251e-235
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.862730431637126e-236 AND -3.931365215818563e-236
        ) THEN -7.862730431637126e-236
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.931365215818563e-236 AND -1.9656826079092814e-236
        ) THEN -3.931365215818563e-236
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.9656826079092814e-236 AND -9.828413039546407e-237
        ) THEN -1.9656826079092814e-236
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.828413039546407e-237 AND -4.914206519773204e-237
        ) THEN -9.828413039546407e-237
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.914206519773204e-237 AND -2.457103259886602e-237
        ) THEN -4.914206519773204e-237
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.457103259886602e-237 AND -1.228551629943301e-237
        ) THEN -2.457103259886602e-237
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.228551629943301e-237 AND -6.142758149716505e-238
        ) THEN -1.228551629943301e-237
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.142758149716505e-238 AND -3.0713790748582522e-238
        ) THEN -6.142758149716505e-238
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.0713790748582522e-238 AND -1.5356895374291261e-238
        ) THEN -3.0713790748582522e-238
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.5356895374291261e-238 AND -7.678447687145631e-239
        ) THEN -1.5356895374291261e-238
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.678447687145631e-239 AND -3.8392238435728152e-239
        ) THEN -7.678447687145631e-239
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.8392238435728152e-239 AND -1.9196119217864076e-239
        ) THEN -3.8392238435728152e-239
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.9196119217864076e-239 AND -9.598059608932038e-240
        ) THEN -1.9196119217864076e-239
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.598059608932038e-240 AND -4.799029804466019e-240
        ) THEN -9.598059608932038e-240
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.799029804466019e-240 AND -2.3995149022330095e-240
        ) THEN -4.799029804466019e-240
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.3995149022330095e-240 AND -1.1997574511165048e-240
        ) THEN -2.3995149022330095e-240
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.1997574511165048e-240 AND -5.998787255582524e-241
        ) THEN -1.1997574511165048e-240
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.998787255582524e-241 AND -2.999393627791262e-241
        ) THEN -5.998787255582524e-241
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.999393627791262e-241 AND -1.499696813895631e-241
        ) THEN -2.999393627791262e-241
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.499696813895631e-241 AND -7.498484069478155e-242
        ) THEN -1.499696813895631e-241
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.498484069478155e-242 AND -3.7492420347390774e-242
        ) THEN -7.498484069478155e-242
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.7492420347390774e-242 AND -1.8746210173695387e-242
        ) THEN -3.7492420347390774e-242
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.8746210173695387e-242 AND -9.373105086847693e-243
        ) THEN -1.8746210173695387e-242
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.373105086847693e-243 AND -4.686552543423847e-243
        ) THEN -9.373105086847693e-243
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.686552543423847e-243 AND -2.3432762717119234e-243
        ) THEN -4.686552543423847e-243
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.3432762717119234e-243 AND -1.1716381358559617e-243
        ) THEN -2.3432762717119234e-243
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.1716381358559617e-243 AND -5.858190679279809e-244
        ) THEN -1.1716381358559617e-243
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.858190679279809e-244 AND -2.9290953396399042e-244
        ) THEN -5.858190679279809e-244
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.9290953396399042e-244 AND -1.4645476698199521e-244
        ) THEN -2.9290953396399042e-244
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.4645476698199521e-244 AND -7.322738349099761e-245
        ) THEN -1.4645476698199521e-244
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.322738349099761e-245 AND -3.6613691745498803e-245
        ) THEN -7.322738349099761e-245
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.6613691745498803e-245 AND -1.8306845872749401e-245
        ) THEN -3.6613691745498803e-245
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.8306845872749401e-245 AND -9.153422936374701e-246
        ) THEN -1.8306845872749401e-245
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.153422936374701e-246 AND -4.5767114681873503e-246
        ) THEN -9.153422936374701e-246
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.5767114681873503e-246 AND -2.2883557340936752e-246
        ) THEN -4.5767114681873503e-246
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.2883557340936752e-246 AND -1.1441778670468376e-246
        ) THEN -2.2883557340936752e-246
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.1441778670468376e-246 AND -5.720889335234188e-247
        ) THEN -1.1441778670468376e-246
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.720889335234188e-247 AND -2.860444667617094e-247
        ) THEN -5.720889335234188e-247
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.860444667617094e-247 AND -1.430222333808547e-247
        ) THEN -2.860444667617094e-247
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.430222333808547e-247 AND -7.151111669042735e-248
        ) THEN -1.430222333808547e-247
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.151111669042735e-248 AND -3.5755558345213674e-248
        ) THEN -7.151111669042735e-248
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.5755558345213674e-248 AND -1.7877779172606837e-248
        ) THEN -3.5755558345213674e-248
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.7877779172606837e-248 AND -8.938889586303419e-249
        ) THEN -1.7877779172606837e-248
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.938889586303419e-249 AND -4.4694447931517093e-249
        ) THEN -8.938889586303419e-249
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.4694447931517093e-249 AND -2.2347223965758547e-249
        ) THEN -4.4694447931517093e-249
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.2347223965758547e-249 AND -1.1173611982879273e-249
        ) THEN -2.2347223965758547e-249
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.1173611982879273e-249 AND -5.586805991439637e-250
        ) THEN -1.1173611982879273e-249
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.586805991439637e-250 AND -2.7934029957198183e-250
        ) THEN -5.586805991439637e-250
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.7934029957198183e-250 AND -1.3967014978599092e-250
        ) THEN -2.7934029957198183e-250
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.3967014978599092e-250 AND -6.983507489299546e-251
        ) THEN -1.3967014978599092e-250
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.983507489299546e-251 AND -3.491753744649773e-251
        ) THEN -6.983507489299546e-251
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.491753744649773e-251 AND -1.7458768723248864e-251
        ) THEN -3.491753744649773e-251
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.7458768723248864e-251 AND -8.729384361624432e-252
        ) THEN -1.7458768723248864e-251
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.729384361624432e-252 AND -4.364692180812216e-252
        ) THEN -8.729384361624432e-252
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.364692180812216e-252 AND -2.182346090406108e-252
        ) THEN -4.364692180812216e-252
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.182346090406108e-252 AND -1.091173045203054e-252
        ) THEN -2.182346090406108e-252
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.091173045203054e-252 AND -5.45586522601527e-253
        ) THEN -1.091173045203054e-252
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.45586522601527e-253 AND -2.727932613007635e-253
        ) THEN -5.45586522601527e-253
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.727932613007635e-253 AND -1.3639663065038175e-253
        ) THEN -2.727932613007635e-253
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.3639663065038175e-253 AND -6.819831532519088e-254
        ) THEN -1.3639663065038175e-253
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.819831532519088e-254 AND -3.409915766259544e-254
        ) THEN -6.819831532519088e-254
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.409915766259544e-254 AND -1.704957883129772e-254
        ) THEN -3.409915766259544e-254
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.704957883129772e-254 AND -8.52478941564886e-255
        ) THEN -1.704957883129772e-254
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.52478941564886e-255 AND -4.26239470782443e-255
        ) THEN -8.52478941564886e-255
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.26239470782443e-255 AND -2.131197353912215e-255
        ) THEN -4.26239470782443e-255
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.131197353912215e-255 AND -1.0655986769561075e-255
        ) THEN -2.131197353912215e-255
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.0655986769561075e-255 AND -5.327993384780537e-256
        ) THEN -1.0655986769561075e-255
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.327993384780537e-256 AND -2.6639966923902686e-256
        ) THEN -5.327993384780537e-256
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.6639966923902686e-256 AND -1.3319983461951343e-256
        ) THEN -2.6639966923902686e-256
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.3319983461951343e-256 AND -6.659991730975672e-257
        ) THEN -1.3319983461951343e-256
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.659991730975672e-257 AND -3.329995865487836e-257
        ) THEN -6.659991730975672e-257
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.329995865487836e-257 AND -1.664997932743918e-257
        ) THEN -3.329995865487836e-257
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.664997932743918e-257 AND -8.32498966371959e-258
        ) THEN -1.664997932743918e-257
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.32498966371959e-258 AND -4.162494831859795e-258
        ) THEN -8.32498966371959e-258
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.162494831859795e-258 AND -2.0812474159298974e-258
        ) THEN -4.162494831859795e-258
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.0812474159298974e-258 AND -1.0406237079649487e-258
        ) THEN -2.0812474159298974e-258
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.0406237079649487e-258 AND -5.2031185398247434e-259
        ) THEN -1.0406237079649487e-258
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.2031185398247434e-259 AND -2.6015592699123717e-259
        ) THEN -5.2031185398247434e-259
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.6015592699123717e-259 AND -1.3007796349561859e-259
        ) THEN -2.6015592699123717e-259
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.3007796349561859e-259 AND -6.503898174780929e-260
        ) THEN -1.3007796349561859e-259
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.503898174780929e-260 AND -3.2519490873904646e-260
        ) THEN -6.503898174780929e-260
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.2519490873904646e-260 AND -1.6259745436952323e-260
        ) THEN -3.2519490873904646e-260
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.6259745436952323e-260 AND -8.129872718476162e-261
        ) THEN -1.6259745436952323e-260
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.129872718476162e-261 AND -4.064936359238081e-261
        ) THEN -8.129872718476162e-261
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.064936359238081e-261 AND -2.0324681796190404e-261
        ) THEN -4.064936359238081e-261
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.0324681796190404e-261 AND -1.0162340898095202e-261
        ) THEN -2.0324681796190404e-261
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.0162340898095202e-261 AND -5.081170449047601e-262
        ) THEN -1.0162340898095202e-261
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.081170449047601e-262 AND -2.5405852245238005e-262
        ) THEN -5.081170449047601e-262
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.5405852245238005e-262 AND -1.2702926122619002e-262
        ) THEN -2.5405852245238005e-262
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.2702926122619002e-262 AND -6.351463061309501e-263
        ) THEN -1.2702926122619002e-262
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.351463061309501e-263 AND -3.1757315306547506e-263
        ) THEN -6.351463061309501e-263
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.1757315306547506e-263 AND -1.5878657653273753e-263
        ) THEN -3.1757315306547506e-263
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.5878657653273753e-263 AND -7.939328826636877e-264
        ) THEN -1.5878657653273753e-263
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.939328826636877e-264 AND -3.9696644133184383e-264
        ) THEN -7.939328826636877e-264
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.9696644133184383e-264 AND -1.9848322066592191e-264
        ) THEN -3.9696644133184383e-264
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.9848322066592191e-264 AND -9.924161033296096e-265
        ) THEN -1.9848322066592191e-264
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.924161033296096e-265 AND -4.962080516648048e-265
        ) THEN -9.924161033296096e-265
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.962080516648048e-265 AND -2.481040258324024e-265
        ) THEN -4.962080516648048e-265
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.481040258324024e-265 AND -1.240520129162012e-265
        ) THEN -2.481040258324024e-265
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.240520129162012e-265 AND -6.20260064581006e-266
        ) THEN -1.240520129162012e-265
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.20260064581006e-266 AND -3.10130032290503e-266
        ) THEN -6.20260064581006e-266
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.10130032290503e-266 AND -1.550650161452515e-266
        ) THEN -3.10130032290503e-266
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.550650161452515e-266 AND -7.753250807262575e-267
        ) THEN -1.550650161452515e-266
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.753250807262575e-267 AND -3.8766254036312874e-267
        ) THEN -7.753250807262575e-267
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.8766254036312874e-267 AND -1.9383127018156437e-267
        ) THEN -3.8766254036312874e-267
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.9383127018156437e-267 AND -9.691563509078218e-268
        ) THEN -1.9383127018156437e-267
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.691563509078218e-268 AND -4.845781754539109e-268
        ) THEN -9.691563509078218e-268
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.845781754539109e-268 AND -2.4228908772695546e-268
        ) THEN -4.845781754539109e-268
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.4228908772695546e-268 AND -1.2114454386347773e-268
        ) THEN -2.4228908772695546e-268
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.2114454386347773e-268 AND -6.057227193173887e-269
        ) THEN -1.2114454386347773e-268
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.057227193173887e-269 AND -3.0286135965869433e-269
        ) THEN -6.057227193173887e-269
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.0286135965869433e-269 AND -1.5143067982934716e-269
        ) THEN -3.0286135965869433e-269
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.5143067982934716e-269 AND -7.571533991467358e-270
        ) THEN -1.5143067982934716e-269
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.571533991467358e-270 AND -3.785766995733679e-270
        ) THEN -7.571533991467358e-270
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.785766995733679e-270 AND -1.8928834978668395e-270
        ) THEN -3.785766995733679e-270
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.8928834978668395e-270 AND -9.464417489334198e-271
        ) THEN -1.8928834978668395e-270
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.464417489334198e-271 AND -4.732208744667099e-271
        ) THEN -9.464417489334198e-271
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.732208744667099e-271 AND -2.3661043723335494e-271
        ) THEN -4.732208744667099e-271
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.3661043723335494e-271 AND -1.1830521861667747e-271
        ) THEN -2.3661043723335494e-271
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.1830521861667747e-271 AND -5.915260930833874e-272
        ) THEN -1.1830521861667747e-271
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.915260930833874e-272 AND -2.957630465416937e-272
        ) THEN -5.915260930833874e-272
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.957630465416937e-272 AND -1.4788152327084684e-272
        ) THEN -2.957630465416937e-272
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.4788152327084684e-272 AND -7.394076163542342e-273
        ) THEN -1.4788152327084684e-272
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.394076163542342e-273 AND -3.697038081771171e-273
        ) THEN -7.394076163542342e-273
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.697038081771171e-273 AND -1.8485190408855855e-273
        ) THEN -3.697038081771171e-273
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.8485190408855855e-273 AND -9.242595204427927e-274
        ) THEN -1.8485190408855855e-273
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.242595204427927e-274 AND -4.621297602213964e-274
        ) THEN -9.242595204427927e-274
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.621297602213964e-274 AND -2.310648801106982e-274
        ) THEN -4.621297602213964e-274
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.310648801106982e-274 AND -1.155324400553491e-274
        ) THEN -2.310648801106982e-274
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.155324400553491e-274 AND -5.776622002767455e-275
        ) THEN -1.155324400553491e-274
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.776622002767455e-275 AND -2.8883110013837273e-275
        ) THEN -5.776622002767455e-275
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.8883110013837273e-275 AND -1.4441555006918637e-275
        ) THEN -2.8883110013837273e-275
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.4441555006918637e-275 AND -7.220777503459318e-276
        ) THEN -1.4441555006918637e-275
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.220777503459318e-276 AND -3.610388751729659e-276
        ) THEN -7.220777503459318e-276
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.610388751729659e-276 AND -1.8051943758648296e-276
        ) THEN -3.610388751729659e-276
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.8051943758648296e-276 AND -9.025971879324148e-277
        ) THEN -1.8051943758648296e-276
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.025971879324148e-277 AND -4.512985939662074e-277
        ) THEN -9.025971879324148e-277
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.512985939662074e-277 AND -2.256492969831037e-277
        ) THEN -4.512985939662074e-277
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.256492969831037e-277 AND -1.1282464849155185e-277
        ) THEN -2.256492969831037e-277
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.1282464849155185e-277 AND -5.641232424577593e-278
        ) THEN -1.1282464849155185e-277
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.641232424577593e-278 AND -2.8206162122887962e-278
        ) THEN -5.641232424577593e-278
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.8206162122887962e-278 AND -1.4103081061443981e-278
        ) THEN -2.8206162122887962e-278
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.4103081061443981e-278 AND -7.051540530721991e-279
        ) THEN -1.4103081061443981e-278
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.051540530721991e-279 AND -3.5257702653609953e-279
        ) THEN -7.051540530721991e-279
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.5257702653609953e-279 AND -1.7628851326804976e-279
        ) THEN -3.5257702653609953e-279
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.7628851326804976e-279 AND -8.814425663402488e-280
        ) THEN -1.7628851326804976e-279
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.814425663402488e-280 AND -4.407212831701244e-280
        ) THEN -8.814425663402488e-280
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.407212831701244e-280 AND -2.203606415850622e-280
        ) THEN -4.407212831701244e-280
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.203606415850622e-280 AND -1.101803207925311e-280
        ) THEN -2.203606415850622e-280
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.101803207925311e-280 AND -5.509016039626555e-281
        ) THEN -1.101803207925311e-280
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.509016039626555e-281 AND -2.7545080198132776e-281
        ) THEN -5.509016039626555e-281
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.7545080198132776e-281 AND -1.3772540099066388e-281
        ) THEN -2.7545080198132776e-281
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.3772540099066388e-281 AND -6.886270049533194e-282
        ) THEN -1.3772540099066388e-281
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.886270049533194e-282 AND -3.443135024766597e-282
        ) THEN -6.886270049533194e-282
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.443135024766597e-282 AND -1.7215675123832985e-282
        ) THEN -3.443135024766597e-282
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.7215675123832985e-282 AND -8.607837561916492e-283
        ) THEN -1.7215675123832985e-282
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.607837561916492e-283 AND -4.303918780958246e-283
        ) THEN -8.607837561916492e-283
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.303918780958246e-283 AND -2.151959390479123e-283
        ) THEN -4.303918780958246e-283
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.151959390479123e-283 AND -1.0759796952395615e-283
        ) THEN -2.151959390479123e-283
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.0759796952395615e-283 AND -5.379898476197808e-284
        ) THEN -1.0759796952395615e-283
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.379898476197808e-284 AND -2.689949238098904e-284
        ) THEN -5.379898476197808e-284
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.689949238098904e-284 AND -1.344974619049452e-284
        ) THEN -2.689949238098904e-284
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.344974619049452e-284 AND -6.72487309524726e-285
        ) THEN -1.344974619049452e-284
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.72487309524726e-285 AND -3.36243654762363e-285
        ) THEN -6.72487309524726e-285
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.36243654762363e-285 AND -1.681218273811815e-285
        ) THEN -3.36243654762363e-285
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.681218273811815e-285 AND -8.406091369059075e-286
        ) THEN -1.681218273811815e-285
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.406091369059075e-286 AND -4.2030456845295373e-286
        ) THEN -8.406091369059075e-286
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.2030456845295373e-286 AND -2.1015228422647686e-286
        ) THEN -4.2030456845295373e-286
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.1015228422647686e-286 AND -1.0507614211323843e-286
        ) THEN -2.1015228422647686e-286
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.0507614211323843e-286 AND -5.253807105661922e-287
        ) THEN -1.0507614211323843e-286
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.253807105661922e-287 AND -2.626903552830961e-287
        ) THEN -5.253807105661922e-287
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.626903552830961e-287 AND -1.3134517764154804e-287
        ) THEN -2.626903552830961e-287
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.3134517764154804e-287 AND -6.567258882077402e-288
        ) THEN -1.3134517764154804e-287
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.567258882077402e-288 AND -3.283629441038701e-288
        ) THEN -6.567258882077402e-288
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.283629441038701e-288 AND -1.6418147205193505e-288
        ) THEN -3.283629441038701e-288
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.6418147205193505e-288 AND -8.209073602596753e-289
        ) THEN -1.6418147205193505e-288
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.209073602596753e-289 AND -4.1045368012983762e-289
        ) THEN -8.209073602596753e-289
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.1045368012983762e-289 AND -2.0522684006491881e-289
        ) THEN -4.1045368012983762e-289
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.0522684006491881e-289 AND -1.0261342003245941e-289
        ) THEN -2.0522684006491881e-289
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.0261342003245941e-289 AND -5.1306710016229703e-290
        ) THEN -1.0261342003245941e-289
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.1306710016229703e-290 AND -2.5653355008114852e-290
        ) THEN -5.1306710016229703e-290
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.5653355008114852e-290 AND -1.2826677504057426e-290
        ) THEN -2.5653355008114852e-290
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.2826677504057426e-290 AND -6.413338752028713e-291
        ) THEN -1.2826677504057426e-290
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.413338752028713e-291 AND -3.2066693760143564e-291
        ) THEN -6.413338752028713e-291
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.2066693760143564e-291 AND -1.6033346880071782e-291
        ) THEN -3.2066693760143564e-291
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.6033346880071782e-291 AND -8.016673440035891e-292
        ) THEN -1.6033346880071782e-291
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.016673440035891e-292 AND -4.008336720017946e-292
        ) THEN -8.016673440035891e-292
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.008336720017946e-292 AND -2.004168360008973e-292
        ) THEN -4.008336720017946e-292
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.004168360008973e-292 AND -1.0020841800044864e-292
        ) THEN -2.004168360008973e-292
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.0020841800044864e-292 AND -5.010420900022432e-293
        ) THEN -1.0020841800044864e-292
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.010420900022432e-293 AND -2.505210450011216e-293
        ) THEN -5.010420900022432e-293
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.505210450011216e-293 AND -1.252605225005608e-293
        ) THEN -2.505210450011216e-293
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.252605225005608e-293 AND -6.26302612502804e-294
        ) THEN -1.252605225005608e-293
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.26302612502804e-294 AND -3.13151306251402e-294
        ) THEN -6.26302612502804e-294
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.13151306251402e-294 AND -1.56575653125701e-294
        ) THEN -3.13151306251402e-294
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.56575653125701e-294 AND -7.82878265628505e-295
        ) THEN -1.56575653125701e-294
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.82878265628505e-295 AND -3.914391328142525e-295
        ) THEN -7.82878265628505e-295
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.914391328142525e-295 AND -1.9571956640712625e-295
        ) THEN -3.914391328142525e-295
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.9571956640712625e-295 AND -9.785978320356312e-296
        ) THEN -1.9571956640712625e-295
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.785978320356312e-296 AND -4.892989160178156e-296
        ) THEN -9.785978320356312e-296
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.892989160178156e-296 AND -2.446494580089078e-296
        ) THEN -4.892989160178156e-296
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.446494580089078e-296 AND -1.223247290044539e-296
        ) THEN -2.446494580089078e-296
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.223247290044539e-296 AND -6.116236450222695e-297
        ) THEN -1.223247290044539e-296
        WHEN (
          anon_6."Birthdate_1" BETWEEN -6.116236450222695e-297 AND -3.0581182251113476e-297
        ) THEN -6.116236450222695e-297
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.0581182251113476e-297 AND -1.5290591125556738e-297
        ) THEN -3.0581182251113476e-297
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.5290591125556738e-297 AND -7.645295562778369e-298
        ) THEN -1.5290591125556738e-297
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.645295562778369e-298 AND -3.8226477813891845e-298
        ) THEN -7.645295562778369e-298
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.8226477813891845e-298 AND -1.9113238906945923e-298
        ) THEN -3.8226477813891845e-298
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.9113238906945923e-298 AND -9.556619453472961e-299
        ) THEN -1.9113238906945923e-298
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.556619453472961e-299 AND -4.778309726736481e-299
        ) THEN -9.556619453472961e-299
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.778309726736481e-299 AND -2.3891548633682403e-299
        ) THEN -4.778309726736481e-299
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.3891548633682403e-299 AND -1.1945774316841202e-299
        ) THEN -2.3891548633682403e-299
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.1945774316841202e-299 AND -5.972887158420601e-300
        ) THEN -1.1945774316841202e-299
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.972887158420601e-300 AND -2.9864435792103004e-300
        ) THEN -5.972887158420601e-300
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.9864435792103004e-300 AND -1.4932217896051502e-300
        ) THEN -2.9864435792103004e-300
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.4932217896051502e-300 AND -7.466108948025751e-301
        ) THEN -1.4932217896051502e-300
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.466108948025751e-301 AND -3.7330544740128755e-301
        ) THEN -7.466108948025751e-301
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.7330544740128755e-301 AND -1.8665272370064378e-301
        ) THEN -3.7330544740128755e-301
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.8665272370064378e-301 AND -9.332636185032189e-302
        ) THEN -1.8665272370064378e-301
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.332636185032189e-302 AND -4.6663180925160944e-302
        ) THEN -9.332636185032189e-302
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.6663180925160944e-302 AND -2.3331590462580472e-302
        ) THEN -4.6663180925160944e-302
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.3331590462580472e-302 AND -1.1665795231290236e-302
        ) THEN -2.3331590462580472e-302
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.1665795231290236e-302 AND -5.832897615645118e-303
        ) THEN -1.1665795231290236e-302
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.832897615645118e-303 AND -2.916448807822559e-303
        ) THEN -5.832897615645118e-303
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.916448807822559e-303 AND -1.4582244039112795e-303
        ) THEN -2.916448807822559e-303
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.4582244039112795e-303 AND -7.291122019556398e-304
        ) THEN -1.4582244039112795e-303
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.291122019556398e-304 AND -3.645561009778199e-304
        ) THEN -7.291122019556398e-304
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.645561009778199e-304 AND -1.8227805048890994e-304
        ) THEN -3.645561009778199e-304
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.8227805048890994e-304 AND -9.113902524445497e-305
        ) THEN -1.8227805048890994e-304
        WHEN (
          anon_6."Birthdate_1" BETWEEN -9.113902524445497e-305 AND -4.5569512622227484e-305
        ) THEN -9.113902524445497e-305
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.5569512622227484e-305 AND -2.2784756311113742e-305
        ) THEN -4.5569512622227484e-305
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.2784756311113742e-305 AND -1.1392378155556871e-305
        ) THEN -2.2784756311113742e-305
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.1392378155556871e-305 AND -5.696189077778436e-306
        ) THEN -1.1392378155556871e-305
        WHEN (
          anon_6."Birthdate_1" BETWEEN -5.696189077778436e-306 AND -2.848094538889218e-306
        ) THEN -5.696189077778436e-306
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.848094538889218e-306 AND -1.424047269444609e-306
        ) THEN -2.848094538889218e-306
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.424047269444609e-306 AND -7.120236347223045e-307
        ) THEN -1.424047269444609e-306
        WHEN (
          anon_6."Birthdate_1" BETWEEN -7.120236347223045e-307 AND -3.5601181736115222e-307
        ) THEN -7.120236347223045e-307
        WHEN (
          anon_6."Birthdate_1" BETWEEN -3.5601181736115222e-307 AND -1.7800590868057611e-307
        ) THEN -3.5601181736115222e-307
        WHEN (
          anon_6."Birthdate_1" BETWEEN -1.7800590868057611e-307 AND -8.900295434028806e-308
        ) THEN -1.7800590868057611e-307
        WHEN (
          anon_6."Birthdate_1" BETWEEN -8.900295434028806e-308 AND -4.450147717014403e-308
        ) THEN -8.900295434028806e-308
        WHEN (
          anon_6."Birthdate_1" BETWEEN -4.450147717014403e-308 AND -2.2250738585072014e-308
        ) THEN -4.450147717014403e-308
        WHEN (
          anon_6."Birthdate_1" BETWEEN -2.2250738585072014e-308 AND 0
        ) THEN -2.2250738585072014e-308
        WHEN (
          anon_6."Birthdate_1" BETWEEN 0 AND 1.1125369292536007e-308
        ) THEN 0
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.1125369292536007e-308 AND 2.2250738585072014e-308
        ) THEN 1.1125369292536007e-308
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.2250738585072014e-308 AND 4.450147717014403e-308
        ) THEN 2.2250738585072014e-308
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.450147717014403e-308 AND 8.900295434028806e-308
        ) THEN 4.450147717014403e-308
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.900295434028806e-308 AND 1.7800590868057611e-307
        ) THEN 8.900295434028806e-308
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.7800590868057611e-307 AND 3.5601181736115222e-307
        ) THEN 1.7800590868057611e-307
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.5601181736115222e-307 AND 7.120236347223045e-307
        ) THEN 3.5601181736115222e-307
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.120236347223045e-307 AND 1.424047269444609e-306
        ) THEN 7.120236347223045e-307
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.424047269444609e-306 AND 2.848094538889218e-306
        ) THEN 1.424047269444609e-306
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.848094538889218e-306 AND 5.696189077778436e-306
        ) THEN 2.848094538889218e-306
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.696189077778436e-306 AND 1.1392378155556871e-305
        ) THEN 5.696189077778436e-306
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.1392378155556871e-305 AND 2.2784756311113742e-305
        ) THEN 1.1392378155556871e-305
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.2784756311113742e-305 AND 4.5569512622227484e-305
        ) THEN 2.2784756311113742e-305
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.5569512622227484e-305 AND 9.113902524445497e-305
        ) THEN 4.5569512622227484e-305
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.113902524445497e-305 AND 1.8227805048890994e-304
        ) THEN 9.113902524445497e-305
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.8227805048890994e-304 AND 3.645561009778199e-304
        ) THEN 1.8227805048890994e-304
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.645561009778199e-304 AND 7.291122019556398e-304
        ) THEN 3.645561009778199e-304
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.291122019556398e-304 AND 1.4582244039112795e-303
        ) THEN 7.291122019556398e-304
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.4582244039112795e-303 AND 2.916448807822559e-303
        ) THEN 1.4582244039112795e-303
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.916448807822559e-303 AND 5.832897615645118e-303
        ) THEN 2.916448807822559e-303
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.832897615645118e-303 AND 1.1665795231290236e-302
        ) THEN 5.832897615645118e-303
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.1665795231290236e-302 AND 2.3331590462580472e-302
        ) THEN 1.1665795231290236e-302
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.3331590462580472e-302 AND 4.6663180925160944e-302
        ) THEN 2.3331590462580472e-302
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.6663180925160944e-302 AND 9.332636185032189e-302
        ) THEN 4.6663180925160944e-302
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.332636185032189e-302 AND 1.8665272370064378e-301
        ) THEN 9.332636185032189e-302
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.8665272370064378e-301 AND 3.7330544740128755e-301
        ) THEN 1.8665272370064378e-301
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.7330544740128755e-301 AND 7.466108948025751e-301
        ) THEN 3.7330544740128755e-301
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.466108948025751e-301 AND 1.4932217896051502e-300
        ) THEN 7.466108948025751e-301
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.4932217896051502e-300 AND 2.9864435792103004e-300
        ) THEN 1.4932217896051502e-300
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.9864435792103004e-300 AND 5.972887158420601e-300
        ) THEN 2.9864435792103004e-300
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.972887158420601e-300 AND 1.1945774316841202e-299
        ) THEN 5.972887158420601e-300
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.1945774316841202e-299 AND 2.3891548633682403e-299
        ) THEN 1.1945774316841202e-299
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.3891548633682403e-299 AND 4.778309726736481e-299
        ) THEN 2.3891548633682403e-299
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.778309726736481e-299 AND 9.556619453472961e-299
        ) THEN 4.778309726736481e-299
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.556619453472961e-299 AND 1.9113238906945923e-298
        ) THEN 9.556619453472961e-299
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.9113238906945923e-298 AND 3.8226477813891845e-298
        ) THEN 1.9113238906945923e-298
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.8226477813891845e-298 AND 7.645295562778369e-298
        ) THEN 3.8226477813891845e-298
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.645295562778369e-298 AND 1.5290591125556738e-297
        ) THEN 7.645295562778369e-298
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.5290591125556738e-297 AND 3.0581182251113476e-297
        ) THEN 1.5290591125556738e-297
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.0581182251113476e-297 AND 6.116236450222695e-297
        ) THEN 3.0581182251113476e-297
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.116236450222695e-297 AND 1.223247290044539e-296
        ) THEN 6.116236450222695e-297
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.223247290044539e-296 AND 2.446494580089078e-296
        ) THEN 1.223247290044539e-296
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.446494580089078e-296 AND 4.892989160178156e-296
        ) THEN 2.446494580089078e-296
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.892989160178156e-296 AND 9.785978320356312e-296
        ) THEN 4.892989160178156e-296
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.785978320356312e-296 AND 1.9571956640712625e-295
        ) THEN 9.785978320356312e-296
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.9571956640712625e-295 AND 3.914391328142525e-295
        ) THEN 1.9571956640712625e-295
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.914391328142525e-295 AND 7.82878265628505e-295
        ) THEN 3.914391328142525e-295
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.82878265628505e-295 AND 1.56575653125701e-294
        ) THEN 7.82878265628505e-295
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.56575653125701e-294 AND 3.13151306251402e-294
        ) THEN 1.56575653125701e-294
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.13151306251402e-294 AND 6.26302612502804e-294
        ) THEN 3.13151306251402e-294
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.26302612502804e-294 AND 1.252605225005608e-293
        ) THEN 6.26302612502804e-294
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.252605225005608e-293 AND 2.505210450011216e-293
        ) THEN 1.252605225005608e-293
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.505210450011216e-293 AND 5.010420900022432e-293
        ) THEN 2.505210450011216e-293
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.010420900022432e-293 AND 1.0020841800044864e-292
        ) THEN 5.010420900022432e-293
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.0020841800044864e-292 AND 2.004168360008973e-292
        ) THEN 1.0020841800044864e-292
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.004168360008973e-292 AND 4.008336720017946e-292
        ) THEN 2.004168360008973e-292
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.008336720017946e-292 AND 8.016673440035891e-292
        ) THEN 4.008336720017946e-292
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.016673440035891e-292 AND 1.6033346880071782e-291
        ) THEN 8.016673440035891e-292
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.6033346880071782e-291 AND 3.2066693760143564e-291
        ) THEN 1.6033346880071782e-291
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.2066693760143564e-291 AND 6.413338752028713e-291
        ) THEN 3.2066693760143564e-291
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.413338752028713e-291 AND 1.2826677504057426e-290
        ) THEN 6.413338752028713e-291
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.2826677504057426e-290 AND 2.5653355008114852e-290
        ) THEN 1.2826677504057426e-290
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.5653355008114852e-290 AND 5.1306710016229703e-290
        ) THEN 2.5653355008114852e-290
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.1306710016229703e-290 AND 1.0261342003245941e-289
        ) THEN 5.1306710016229703e-290
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.0261342003245941e-289 AND 2.0522684006491881e-289
        ) THEN 1.0261342003245941e-289
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.0522684006491881e-289 AND 4.1045368012983762e-289
        ) THEN 2.0522684006491881e-289
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.1045368012983762e-289 AND 8.209073602596753e-289
        ) THEN 4.1045368012983762e-289
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.209073602596753e-289 AND 1.6418147205193505e-288
        ) THEN 8.209073602596753e-289
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.6418147205193505e-288 AND 3.283629441038701e-288
        ) THEN 1.6418147205193505e-288
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.283629441038701e-288 AND 6.567258882077402e-288
        ) THEN 3.283629441038701e-288
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.567258882077402e-288 AND 1.3134517764154804e-287
        ) THEN 6.567258882077402e-288
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.3134517764154804e-287 AND 2.626903552830961e-287
        ) THEN 1.3134517764154804e-287
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.626903552830961e-287 AND 5.253807105661922e-287
        ) THEN 2.626903552830961e-287
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.253807105661922e-287 AND 1.0507614211323843e-286
        ) THEN 5.253807105661922e-287
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.0507614211323843e-286 AND 2.1015228422647686e-286
        ) THEN 1.0507614211323843e-286
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.1015228422647686e-286 AND 4.2030456845295373e-286
        ) THEN 2.1015228422647686e-286
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.2030456845295373e-286 AND 8.406091369059075e-286
        ) THEN 4.2030456845295373e-286
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.406091369059075e-286 AND 1.681218273811815e-285
        ) THEN 8.406091369059075e-286
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.681218273811815e-285 AND 3.36243654762363e-285
        ) THEN 1.681218273811815e-285
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.36243654762363e-285 AND 6.72487309524726e-285
        ) THEN 3.36243654762363e-285
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.72487309524726e-285 AND 1.344974619049452e-284
        ) THEN 6.72487309524726e-285
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.344974619049452e-284 AND 2.689949238098904e-284
        ) THEN 1.344974619049452e-284
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.689949238098904e-284 AND 5.379898476197808e-284
        ) THEN 2.689949238098904e-284
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.379898476197808e-284 AND 1.0759796952395615e-283
        ) THEN 5.379898476197808e-284
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.0759796952395615e-283 AND 2.151959390479123e-283
        ) THEN 1.0759796952395615e-283
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.151959390479123e-283 AND 4.303918780958246e-283
        ) THEN 2.151959390479123e-283
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.303918780958246e-283 AND 8.607837561916492e-283
        ) THEN 4.303918780958246e-283
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.607837561916492e-283 AND 1.7215675123832985e-282
        ) THEN 8.607837561916492e-283
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.7215675123832985e-282 AND 3.443135024766597e-282
        ) THEN 1.7215675123832985e-282
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.443135024766597e-282 AND 6.886270049533194e-282
        ) THEN 3.443135024766597e-282
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.886270049533194e-282 AND 1.3772540099066388e-281
        ) THEN 6.886270049533194e-282
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.3772540099066388e-281 AND 2.7545080198132776e-281
        ) THEN 1.3772540099066388e-281
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.7545080198132776e-281 AND 5.509016039626555e-281
        ) THEN 2.7545080198132776e-281
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.509016039626555e-281 AND 1.101803207925311e-280
        ) THEN 5.509016039626555e-281
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.101803207925311e-280 AND 2.203606415850622e-280
        ) THEN 1.101803207925311e-280
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.203606415850622e-280 AND 4.407212831701244e-280
        ) THEN 2.203606415850622e-280
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.407212831701244e-280 AND 8.814425663402488e-280
        ) THEN 4.407212831701244e-280
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.814425663402488e-280 AND 1.7628851326804976e-279
        ) THEN 8.814425663402488e-280
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.7628851326804976e-279 AND 3.5257702653609953e-279
        ) THEN 1.7628851326804976e-279
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.5257702653609953e-279 AND 7.051540530721991e-279
        ) THEN 3.5257702653609953e-279
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.051540530721991e-279 AND 1.4103081061443981e-278
        ) THEN 7.051540530721991e-279
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.4103081061443981e-278 AND 2.8206162122887962e-278
        ) THEN 1.4103081061443981e-278
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.8206162122887962e-278 AND 5.641232424577593e-278
        ) THEN 2.8206162122887962e-278
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.641232424577593e-278 AND 1.1282464849155185e-277
        ) THEN 5.641232424577593e-278
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.1282464849155185e-277 AND 2.256492969831037e-277
        ) THEN 1.1282464849155185e-277
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.256492969831037e-277 AND 4.512985939662074e-277
        ) THEN 2.256492969831037e-277
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.512985939662074e-277 AND 9.025971879324148e-277
        ) THEN 4.512985939662074e-277
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.025971879324148e-277 AND 1.8051943758648296e-276
        ) THEN 9.025971879324148e-277
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.8051943758648296e-276 AND 3.610388751729659e-276
        ) THEN 1.8051943758648296e-276
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.610388751729659e-276 AND 7.220777503459318e-276
        ) THEN 3.610388751729659e-276
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.220777503459318e-276 AND 1.4441555006918637e-275
        ) THEN 7.220777503459318e-276
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.4441555006918637e-275 AND 2.8883110013837273e-275
        ) THEN 1.4441555006918637e-275
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.8883110013837273e-275 AND 5.776622002767455e-275
        ) THEN 2.8883110013837273e-275
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.776622002767455e-275 AND 1.155324400553491e-274
        ) THEN 5.776622002767455e-275
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.155324400553491e-274 AND 2.310648801106982e-274
        ) THEN 1.155324400553491e-274
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.310648801106982e-274 AND 4.621297602213964e-274
        ) THEN 2.310648801106982e-274
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.621297602213964e-274 AND 9.242595204427927e-274
        ) THEN 4.621297602213964e-274
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.242595204427927e-274 AND 1.8485190408855855e-273
        ) THEN 9.242595204427927e-274
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.8485190408855855e-273 AND 3.697038081771171e-273
        ) THEN 1.8485190408855855e-273
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.697038081771171e-273 AND 7.394076163542342e-273
        ) THEN 3.697038081771171e-273
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.394076163542342e-273 AND 1.4788152327084684e-272
        ) THEN 7.394076163542342e-273
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.4788152327084684e-272 AND 2.957630465416937e-272
        ) THEN 1.4788152327084684e-272
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.957630465416937e-272 AND 5.915260930833874e-272
        ) THEN 2.957630465416937e-272
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.915260930833874e-272 AND 1.1830521861667747e-271
        ) THEN 5.915260930833874e-272
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.1830521861667747e-271 AND 2.3661043723335494e-271
        ) THEN 1.1830521861667747e-271
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.3661043723335494e-271 AND 4.732208744667099e-271
        ) THEN 2.3661043723335494e-271
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.732208744667099e-271 AND 9.464417489334198e-271
        ) THEN 4.732208744667099e-271
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.464417489334198e-271 AND 1.8928834978668395e-270
        ) THEN 9.464417489334198e-271
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.8928834978668395e-270 AND 3.785766995733679e-270
        ) THEN 1.8928834978668395e-270
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.785766995733679e-270 AND 7.571533991467358e-270
        ) THEN 3.785766995733679e-270
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.571533991467358e-270 AND 1.5143067982934716e-269
        ) THEN 7.571533991467358e-270
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.5143067982934716e-269 AND 3.0286135965869433e-269
        ) THEN 1.5143067982934716e-269
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.0286135965869433e-269 AND 6.057227193173887e-269
        ) THEN 3.0286135965869433e-269
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.057227193173887e-269 AND 1.2114454386347773e-268
        ) THEN 6.057227193173887e-269
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.2114454386347773e-268 AND 2.4228908772695546e-268
        ) THEN 1.2114454386347773e-268
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.4228908772695546e-268 AND 4.845781754539109e-268
        ) THEN 2.4228908772695546e-268
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.845781754539109e-268 AND 9.691563509078218e-268
        ) THEN 4.845781754539109e-268
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.691563509078218e-268 AND 1.9383127018156437e-267
        ) THEN 9.691563509078218e-268
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.9383127018156437e-267 AND 3.8766254036312874e-267
        ) THEN 1.9383127018156437e-267
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.8766254036312874e-267 AND 7.753250807262575e-267
        ) THEN 3.8766254036312874e-267
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.753250807262575e-267 AND 1.550650161452515e-266
        ) THEN 7.753250807262575e-267
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.550650161452515e-266 AND 3.10130032290503e-266
        ) THEN 1.550650161452515e-266
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.10130032290503e-266 AND 6.20260064581006e-266
        ) THEN 3.10130032290503e-266
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.20260064581006e-266 AND 1.240520129162012e-265
        ) THEN 6.20260064581006e-266
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.240520129162012e-265 AND 2.481040258324024e-265
        ) THEN 1.240520129162012e-265
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.481040258324024e-265 AND 4.962080516648048e-265
        ) THEN 2.481040258324024e-265
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.962080516648048e-265 AND 9.924161033296096e-265
        ) THEN 4.962080516648048e-265
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.924161033296096e-265 AND 1.9848322066592191e-264
        ) THEN 9.924161033296096e-265
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.9848322066592191e-264 AND 3.9696644133184383e-264
        ) THEN 1.9848322066592191e-264
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.9696644133184383e-264 AND 7.939328826636877e-264
        ) THEN 3.9696644133184383e-264
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.939328826636877e-264 AND 1.5878657653273753e-263
        ) THEN 7.939328826636877e-264
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.5878657653273753e-263 AND 3.1757315306547506e-263
        ) THEN 1.5878657653273753e-263
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.1757315306547506e-263 AND 6.351463061309501e-263
        ) THEN 3.1757315306547506e-263
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.351463061309501e-263 AND 1.2702926122619002e-262
        ) THEN 6.351463061309501e-263
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.2702926122619002e-262 AND 2.5405852245238005e-262
        ) THEN 1.2702926122619002e-262
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.5405852245238005e-262 AND 5.081170449047601e-262
        ) THEN 2.5405852245238005e-262
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.081170449047601e-262 AND 1.0162340898095202e-261
        ) THEN 5.081170449047601e-262
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.0162340898095202e-261 AND 2.0324681796190404e-261
        ) THEN 1.0162340898095202e-261
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.0324681796190404e-261 AND 4.064936359238081e-261
        ) THEN 2.0324681796190404e-261
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.064936359238081e-261 AND 8.129872718476162e-261
        ) THEN 4.064936359238081e-261
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.129872718476162e-261 AND 1.6259745436952323e-260
        ) THEN 8.129872718476162e-261
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.6259745436952323e-260 AND 3.2519490873904646e-260
        ) THEN 1.6259745436952323e-260
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.2519490873904646e-260 AND 6.503898174780929e-260
        ) THEN 3.2519490873904646e-260
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.503898174780929e-260 AND 1.3007796349561859e-259
        ) THEN 6.503898174780929e-260
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.3007796349561859e-259 AND 2.6015592699123717e-259
        ) THEN 1.3007796349561859e-259
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.6015592699123717e-259 AND 5.2031185398247434e-259
        ) THEN 2.6015592699123717e-259
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.2031185398247434e-259 AND 1.0406237079649487e-258
        ) THEN 5.2031185398247434e-259
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.0406237079649487e-258 AND 2.0812474159298974e-258
        ) THEN 1.0406237079649487e-258
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.0812474159298974e-258 AND 4.162494831859795e-258
        ) THEN 2.0812474159298974e-258
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.162494831859795e-258 AND 8.32498966371959e-258
        ) THEN 4.162494831859795e-258
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.32498966371959e-258 AND 1.664997932743918e-257
        ) THEN 8.32498966371959e-258
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.664997932743918e-257 AND 3.329995865487836e-257
        ) THEN 1.664997932743918e-257
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.329995865487836e-257 AND 6.659991730975672e-257
        ) THEN 3.329995865487836e-257
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.659991730975672e-257 AND 1.3319983461951343e-256
        ) THEN 6.659991730975672e-257
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.3319983461951343e-256 AND 2.6639966923902686e-256
        ) THEN 1.3319983461951343e-256
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.6639966923902686e-256 AND 5.327993384780537e-256
        ) THEN 2.6639966923902686e-256
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.327993384780537e-256 AND 1.0655986769561075e-255
        ) THEN 5.327993384780537e-256
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.0655986769561075e-255 AND 2.131197353912215e-255
        ) THEN 1.0655986769561075e-255
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.131197353912215e-255 AND 4.26239470782443e-255
        ) THEN 2.131197353912215e-255
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.26239470782443e-255 AND 8.52478941564886e-255
        ) THEN 4.26239470782443e-255
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.52478941564886e-255 AND 1.704957883129772e-254
        ) THEN 8.52478941564886e-255
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.704957883129772e-254 AND 3.409915766259544e-254
        ) THEN 1.704957883129772e-254
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.409915766259544e-254 AND 6.819831532519088e-254
        ) THEN 3.409915766259544e-254
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.819831532519088e-254 AND 1.3639663065038175e-253
        ) THEN 6.819831532519088e-254
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.3639663065038175e-253 AND 2.727932613007635e-253
        ) THEN 1.3639663065038175e-253
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.727932613007635e-253 AND 5.45586522601527e-253
        ) THEN 2.727932613007635e-253
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.45586522601527e-253 AND 1.091173045203054e-252
        ) THEN 5.45586522601527e-253
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.091173045203054e-252 AND 2.182346090406108e-252
        ) THEN 1.091173045203054e-252
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.182346090406108e-252 AND 4.364692180812216e-252
        ) THEN 2.182346090406108e-252
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.364692180812216e-252 AND 8.729384361624432e-252
        ) THEN 4.364692180812216e-252
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.729384361624432e-252 AND 1.7458768723248864e-251
        ) THEN 8.729384361624432e-252
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.7458768723248864e-251 AND 3.491753744649773e-251
        ) THEN 1.7458768723248864e-251
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.491753744649773e-251 AND 6.983507489299546e-251
        ) THEN 3.491753744649773e-251
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.983507489299546e-251 AND 1.3967014978599092e-250
        ) THEN 6.983507489299546e-251
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.3967014978599092e-250 AND 2.7934029957198183e-250
        ) THEN 1.3967014978599092e-250
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.7934029957198183e-250 AND 5.586805991439637e-250
        ) THEN 2.7934029957198183e-250
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.586805991439637e-250 AND 1.1173611982879273e-249
        ) THEN 5.586805991439637e-250
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.1173611982879273e-249 AND 2.2347223965758547e-249
        ) THEN 1.1173611982879273e-249
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.2347223965758547e-249 AND 4.4694447931517093e-249
        ) THEN 2.2347223965758547e-249
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.4694447931517093e-249 AND 8.938889586303419e-249
        ) THEN 4.4694447931517093e-249
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.938889586303419e-249 AND 1.7877779172606837e-248
        ) THEN 8.938889586303419e-249
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.7877779172606837e-248 AND 3.5755558345213674e-248
        ) THEN 1.7877779172606837e-248
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.5755558345213674e-248 AND 7.151111669042735e-248
        ) THEN 3.5755558345213674e-248
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.151111669042735e-248 AND 1.430222333808547e-247
        ) THEN 7.151111669042735e-248
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.430222333808547e-247 AND 2.860444667617094e-247
        ) THEN 1.430222333808547e-247
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.860444667617094e-247 AND 5.720889335234188e-247
        ) THEN 2.860444667617094e-247
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.720889335234188e-247 AND 1.1441778670468376e-246
        ) THEN 5.720889335234188e-247
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.1441778670468376e-246 AND 2.2883557340936752e-246
        ) THEN 1.1441778670468376e-246
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.2883557340936752e-246 AND 4.5767114681873503e-246
        ) THEN 2.2883557340936752e-246
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.5767114681873503e-246 AND 9.153422936374701e-246
        ) THEN 4.5767114681873503e-246
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.153422936374701e-246 AND 1.8306845872749401e-245
        ) THEN 9.153422936374701e-246
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.8306845872749401e-245 AND 3.6613691745498803e-245
        ) THEN 1.8306845872749401e-245
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.6613691745498803e-245 AND 7.322738349099761e-245
        ) THEN 3.6613691745498803e-245
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.322738349099761e-245 AND 1.4645476698199521e-244
        ) THEN 7.322738349099761e-245
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.4645476698199521e-244 AND 2.9290953396399042e-244
        ) THEN 1.4645476698199521e-244
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.9290953396399042e-244 AND 5.858190679279809e-244
        ) THEN 2.9290953396399042e-244
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.858190679279809e-244 AND 1.1716381358559617e-243
        ) THEN 5.858190679279809e-244
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.1716381358559617e-243 AND 2.3432762717119234e-243
        ) THEN 1.1716381358559617e-243
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.3432762717119234e-243 AND 4.686552543423847e-243
        ) THEN 2.3432762717119234e-243
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.686552543423847e-243 AND 9.373105086847693e-243
        ) THEN 4.686552543423847e-243
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.373105086847693e-243 AND 1.8746210173695387e-242
        ) THEN 9.373105086847693e-243
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.8746210173695387e-242 AND 3.7492420347390774e-242
        ) THEN 1.8746210173695387e-242
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.7492420347390774e-242 AND 7.498484069478155e-242
        ) THEN 3.7492420347390774e-242
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.498484069478155e-242 AND 1.499696813895631e-241
        ) THEN 7.498484069478155e-242
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.499696813895631e-241 AND 2.999393627791262e-241
        ) THEN 1.499696813895631e-241
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.999393627791262e-241 AND 5.998787255582524e-241
        ) THEN 2.999393627791262e-241
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.998787255582524e-241 AND 1.1997574511165048e-240
        ) THEN 5.998787255582524e-241
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.1997574511165048e-240 AND 2.3995149022330095e-240
        ) THEN 1.1997574511165048e-240
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.3995149022330095e-240 AND 4.799029804466019e-240
        ) THEN 2.3995149022330095e-240
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.799029804466019e-240 AND 9.598059608932038e-240
        ) THEN 4.799029804466019e-240
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.598059608932038e-240 AND 1.9196119217864076e-239
        ) THEN 9.598059608932038e-240
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.9196119217864076e-239 AND 3.8392238435728152e-239
        ) THEN 1.9196119217864076e-239
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.8392238435728152e-239 AND 7.678447687145631e-239
        ) THEN 3.8392238435728152e-239
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.678447687145631e-239 AND 1.5356895374291261e-238
        ) THEN 7.678447687145631e-239
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.5356895374291261e-238 AND 3.0713790748582522e-238
        ) THEN 1.5356895374291261e-238
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.0713790748582522e-238 AND 6.142758149716505e-238
        ) THEN 3.0713790748582522e-238
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.142758149716505e-238 AND 1.228551629943301e-237
        ) THEN 6.142758149716505e-238
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.228551629943301e-237 AND 2.457103259886602e-237
        ) THEN 1.228551629943301e-237
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.457103259886602e-237 AND 4.914206519773204e-237
        ) THEN 2.457103259886602e-237
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.914206519773204e-237 AND 9.828413039546407e-237
        ) THEN 4.914206519773204e-237
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.828413039546407e-237 AND 1.9656826079092814e-236
        ) THEN 9.828413039546407e-237
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.9656826079092814e-236 AND 3.931365215818563e-236
        ) THEN 1.9656826079092814e-236
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.931365215818563e-236 AND 7.862730431637126e-236
        ) THEN 3.931365215818563e-236
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.862730431637126e-236 AND 1.5725460863274251e-235
        ) THEN 7.862730431637126e-236
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.5725460863274251e-235 AND 3.1450921726548502e-235
        ) THEN 1.5725460863274251e-235
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.1450921726548502e-235 AND 6.290184345309701e-235
        ) THEN 3.1450921726548502e-235
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.290184345309701e-235 AND 1.2580368690619401e-234
        ) THEN 6.290184345309701e-235
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.2580368690619401e-234 AND 2.5160737381238802e-234
        ) THEN 1.2580368690619401e-234
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.5160737381238802e-234 AND 5.0321474762477604e-234
        ) THEN 2.5160737381238802e-234
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.0321474762477604e-234 AND 1.0064294952495521e-233
        ) THEN 5.0321474762477604e-234
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.0064294952495521e-233 AND 2.0128589904991042e-233
        ) THEN 1.0064294952495521e-233
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.0128589904991042e-233 AND 4.0257179809982083e-233
        ) THEN 2.0128589904991042e-233
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.0257179809982083e-233 AND 8.051435961996417e-233
        ) THEN 4.0257179809982083e-233
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.051435961996417e-233 AND 1.6102871923992833e-232
        ) THEN 8.051435961996417e-233
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.6102871923992833e-232 AND 3.220574384798567e-232
        ) THEN 1.6102871923992833e-232
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.220574384798567e-232 AND 6.441148769597133e-232
        ) THEN 3.220574384798567e-232
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.441148769597133e-232 AND 1.2882297539194267e-231
        ) THEN 6.441148769597133e-232
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.2882297539194267e-231 AND 2.5764595078388533e-231
        ) THEN 1.2882297539194267e-231
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.5764595078388533e-231 AND 5.152919015677707e-231
        ) THEN 2.5764595078388533e-231
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.152919015677707e-231 AND 1.0305838031355413e-230
        ) THEN 5.152919015677707e-231
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.0305838031355413e-230 AND 2.0611676062710827e-230
        ) THEN 1.0305838031355413e-230
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.0611676062710827e-230 AND 4.1223352125421653e-230
        ) THEN 2.0611676062710827e-230
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.1223352125421653e-230 AND 8.244670425084331e-230
        ) THEN 4.1223352125421653e-230
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.244670425084331e-230 AND 1.6489340850168661e-229
        ) THEN 8.244670425084331e-230
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.6489340850168661e-229 AND 3.2978681700337323e-229
        ) THEN 1.6489340850168661e-229
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.2978681700337323e-229 AND 6.595736340067465e-229
        ) THEN 3.2978681700337323e-229
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.595736340067465e-229 AND 1.319147268013493e-228
        ) THEN 6.595736340067465e-229
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.319147268013493e-228 AND 2.638294536026986e-228
        ) THEN 1.319147268013493e-228
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.638294536026986e-228 AND 5.276589072053972e-228
        ) THEN 2.638294536026986e-228
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.276589072053972e-228 AND 1.0553178144107943e-227
        ) THEN 5.276589072053972e-228
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.0553178144107943e-227 AND 2.1106356288215886e-227
        ) THEN 1.0553178144107943e-227
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.1106356288215886e-227 AND 4.2212712576431773e-227
        ) THEN 2.1106356288215886e-227
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.2212712576431773e-227 AND 8.442542515286355e-227
        ) THEN 4.2212712576431773e-227
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.442542515286355e-227 AND 1.688508503057271e-226
        ) THEN 8.442542515286355e-227
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.688508503057271e-226 AND 3.377017006114542e-226
        ) THEN 1.688508503057271e-226
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.377017006114542e-226 AND 6.754034012229084e-226
        ) THEN 3.377017006114542e-226
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.754034012229084e-226 AND 1.3508068024458167e-225
        ) THEN 6.754034012229084e-226
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.3508068024458167e-225 AND 2.7016136048916335e-225
        ) THEN 1.3508068024458167e-225
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.7016136048916335e-225 AND 5.403227209783267e-225
        ) THEN 2.7016136048916335e-225
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.403227209783267e-225 AND 1.0806454419566534e-224
        ) THEN 5.403227209783267e-225
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.0806454419566534e-224 AND 2.161290883913307e-224
        ) THEN 1.0806454419566534e-224
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.161290883913307e-224 AND 4.322581767826614e-224
        ) THEN 2.161290883913307e-224
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.322581767826614e-224 AND 8.645163535653227e-224
        ) THEN 4.322581767826614e-224
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.645163535653227e-224 AND 1.7290327071306454e-223
        ) THEN 8.645163535653227e-224
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.7290327071306454e-223 AND 3.458065414261291e-223
        ) THEN 1.7290327071306454e-223
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.458065414261291e-223 AND 6.916130828522582e-223
        ) THEN 3.458065414261291e-223
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.916130828522582e-223 AND 1.3832261657045163e-222
        ) THEN 6.916130828522582e-223
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.3832261657045163e-222 AND 2.7664523314090327e-222
        ) THEN 1.3832261657045163e-222
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.7664523314090327e-222 AND 5.5329046628180653e-222
        ) THEN 2.7664523314090327e-222
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.5329046628180653e-222 AND 1.1065809325636131e-221
        ) THEN 5.5329046628180653e-222
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.1065809325636131e-221 AND 2.2131618651272261e-221
        ) THEN 1.1065809325636131e-221
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.2131618651272261e-221 AND 4.4263237302544523e-221
        ) THEN 2.2131618651272261e-221
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.4263237302544523e-221 AND 8.852647460508905e-221
        ) THEN 4.4263237302544523e-221
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.852647460508905e-221 AND 1.770529492101781e-220
        ) THEN 8.852647460508905e-221
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.770529492101781e-220 AND 3.541058984203562e-220
        ) THEN 1.770529492101781e-220
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.541058984203562e-220 AND 7.082117968407124e-220
        ) THEN 3.541058984203562e-220
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.082117968407124e-220 AND 1.4164235936814247e-219
        ) THEN 7.082117968407124e-220
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.4164235936814247e-219 AND 2.8328471873628494e-219
        ) THEN 1.4164235936814247e-219
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.8328471873628494e-219 AND 5.665694374725699e-219
        ) THEN 2.8328471873628494e-219
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.665694374725699e-219 AND 1.1331388749451398e-218
        ) THEN 5.665694374725699e-219
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.1331388749451398e-218 AND 2.2662777498902796e-218
        ) THEN 1.1331388749451398e-218
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.2662777498902796e-218 AND 4.532555499780559e-218
        ) THEN 2.2662777498902796e-218
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.532555499780559e-218 AND 9.065110999561118e-218
        ) THEN 4.532555499780559e-218
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.065110999561118e-218 AND 1.8130221999122236e-217
        ) THEN 9.065110999561118e-218
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.8130221999122236e-217 AND 3.6260443998244473e-217
        ) THEN 1.8130221999122236e-217
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.6260443998244473e-217 AND 7.252088799648895e-217
        ) THEN 3.6260443998244473e-217
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.252088799648895e-217 AND 1.450417759929779e-216
        ) THEN 7.252088799648895e-217
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.450417759929779e-216 AND 2.900835519859558e-216
        ) THEN 1.450417759929779e-216
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.900835519859558e-216 AND 5.801671039719116e-216
        ) THEN 2.900835519859558e-216
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.801671039719116e-216 AND 1.1603342079438231e-215
        ) THEN 5.801671039719116e-216
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.1603342079438231e-215 AND 2.3206684158876463e-215
        ) THEN 1.1603342079438231e-215
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.3206684158876463e-215 AND 4.641336831775293e-215
        ) THEN 2.3206684158876463e-215
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.641336831775293e-215 AND 9.282673663550585e-215
        ) THEN 4.641336831775293e-215
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.282673663550585e-215 AND 1.856534732710117e-214
        ) THEN 9.282673663550585e-215
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.856534732710117e-214 AND 3.713069465420234e-214
        ) THEN 1.856534732710117e-214
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.713069465420234e-214 AND 7.426138930840468e-214
        ) THEN 3.713069465420234e-214
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.426138930840468e-214 AND 1.4852277861680936e-213
        ) THEN 7.426138930840468e-214
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.4852277861680936e-213 AND 2.9704555723361872e-213
        ) THEN 1.4852277861680936e-213
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.9704555723361872e-213 AND 5.940911144672375e-213
        ) THEN 2.9704555723361872e-213
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.940911144672375e-213 AND 1.188182228934475e-212
        ) THEN 5.940911144672375e-213
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.188182228934475e-212 AND 2.37636445786895e-212
        ) THEN 1.188182228934475e-212
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.37636445786895e-212 AND 4.7527289157379e-212
        ) THEN 2.37636445786895e-212
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.7527289157379e-212 AND 9.5054578314758e-212
        ) THEN 4.7527289157379e-212
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.5054578314758e-212 AND 1.90109156629516e-211
        ) THEN 9.5054578314758e-212
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.90109156629516e-211 AND 3.80218313259032e-211
        ) THEN 1.90109156629516e-211
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.80218313259032e-211 AND 7.60436626518064e-211
        ) THEN 3.80218313259032e-211
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.60436626518064e-211 AND 1.520873253036128e-210
        ) THEN 7.60436626518064e-211
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.520873253036128e-210 AND 3.041746506072256e-210
        ) THEN 1.520873253036128e-210
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.041746506072256e-210 AND 6.083493012144512e-210
        ) THEN 3.041746506072256e-210
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.083493012144512e-210 AND 1.2166986024289023e-209
        ) THEN 6.083493012144512e-210
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.2166986024289023e-209 AND 2.4333972048578046e-209
        ) THEN 1.2166986024289023e-209
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.4333972048578046e-209 AND 4.866794409715609e-209
        ) THEN 2.4333972048578046e-209
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.866794409715609e-209 AND 9.733588819431218e-209
        ) THEN 4.866794409715609e-209
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.733588819431218e-209 AND 1.9467177638862437e-208
        ) THEN 9.733588819431218e-209
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.9467177638862437e-208 AND 3.8934355277724873e-208
        ) THEN 1.9467177638862437e-208
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.8934355277724873e-208 AND 7.786871055544975e-208
        ) THEN 3.8934355277724873e-208
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.786871055544975e-208 AND 1.557374211108995e-207
        ) THEN 7.786871055544975e-208
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.557374211108995e-207 AND 3.11474842221799e-207
        ) THEN 1.557374211108995e-207
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.11474842221799e-207 AND 6.22949684443598e-207
        ) THEN 3.11474842221799e-207
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.22949684443598e-207 AND 1.245899368887196e-206
        ) THEN 6.22949684443598e-207
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.245899368887196e-206 AND 2.491798737774392e-206
        ) THEN 1.245899368887196e-206
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.491798737774392e-206 AND 4.983597475548784e-206
        ) THEN 2.491798737774392e-206
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.983597475548784e-206 AND 9.967194951097568e-206
        ) THEN 4.983597475548784e-206
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.967194951097568e-206 AND 1.9934389902195135e-205
        ) THEN 9.967194951097568e-206
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.9934389902195135e-205 AND 3.986877980439027e-205
        ) THEN 1.9934389902195135e-205
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.986877980439027e-205 AND 7.973755960878054e-205
        ) THEN 3.986877980439027e-205
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.973755960878054e-205 AND 1.5947511921756108e-204
        ) THEN 7.973755960878054e-205
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.5947511921756108e-204 AND 3.1895023843512216e-204
        ) THEN 1.5947511921756108e-204
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.1895023843512216e-204 AND 6.379004768702443e-204
        ) THEN 3.1895023843512216e-204
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.379004768702443e-204 AND 1.2758009537404886e-203
        ) THEN 6.379004768702443e-204
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.2758009537404886e-203 AND 2.5516019074809773e-203
        ) THEN 1.2758009537404886e-203
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.5516019074809773e-203 AND 5.103203814961955e-203
        ) THEN 2.5516019074809773e-203
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.103203814961955e-203 AND 1.020640762992391e-202
        ) THEN 5.103203814961955e-203
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.020640762992391e-202 AND 2.041281525984782e-202
        ) THEN 1.020640762992391e-202
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.041281525984782e-202 AND 4.082563051969564e-202
        ) THEN 2.041281525984782e-202
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.082563051969564e-202 AND 8.165126103939127e-202
        ) THEN 4.082563051969564e-202
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.165126103939127e-202 AND 1.6330252207878255e-201
        ) THEN 8.165126103939127e-202
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.6330252207878255e-201 AND 3.266050441575651e-201
        ) THEN 1.6330252207878255e-201
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.266050441575651e-201 AND 6.532100883151302e-201
        ) THEN 3.266050441575651e-201
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.532100883151302e-201 AND 1.3064201766302604e-200
        ) THEN 6.532100883151302e-201
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.3064201766302604e-200 AND 2.612840353260521e-200
        ) THEN 1.3064201766302604e-200
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.612840353260521e-200 AND 5.225680706521042e-200
        ) THEN 2.612840353260521e-200
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.225680706521042e-200 AND 1.0451361413042083e-199
        ) THEN 5.225680706521042e-200
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.0451361413042083e-199 AND 2.0902722826084166e-199
        ) THEN 1.0451361413042083e-199
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.0902722826084166e-199 AND 4.180544565216833e-199
        ) THEN 2.0902722826084166e-199
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.180544565216833e-199 AND 8.361089130433666e-199
        ) THEN 4.180544565216833e-199
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.361089130433666e-199 AND 1.6722178260867333e-198
        ) THEN 8.361089130433666e-199
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.6722178260867333e-198 AND 3.3444356521734666e-198
        ) THEN 1.6722178260867333e-198
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.3444356521734666e-198 AND 6.688871304346933e-198
        ) THEN 3.3444356521734666e-198
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.688871304346933e-198 AND 1.3377742608693866e-197
        ) THEN 6.688871304346933e-198
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.3377742608693866e-197 AND 2.6755485217387732e-197
        ) THEN 1.3377742608693866e-197
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.6755485217387732e-197 AND 5.351097043477547e-197
        ) THEN 2.6755485217387732e-197
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.351097043477547e-197 AND 1.0702194086955093e-196
        ) THEN 5.351097043477547e-197
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.0702194086955093e-196 AND 2.1404388173910186e-196
        ) THEN 1.0702194086955093e-196
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.1404388173910186e-196 AND 4.280877634782037e-196
        ) THEN 2.1404388173910186e-196
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.280877634782037e-196 AND 8.561755269564074e-196
        ) THEN 4.280877634782037e-196
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.561755269564074e-196 AND 1.712351053912815e-195
        ) THEN 8.561755269564074e-196
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.712351053912815e-195 AND 3.42470210782563e-195
        ) THEN 1.712351053912815e-195
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.42470210782563e-195 AND 6.84940421565126e-195
        ) THEN 3.42470210782563e-195
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.84940421565126e-195 AND 1.369880843130252e-194
        ) THEN 6.84940421565126e-195
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.369880843130252e-194 AND 2.739761686260504e-194
        ) THEN 1.369880843130252e-194
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.739761686260504e-194 AND 5.479523372521008e-194
        ) THEN 2.739761686260504e-194
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.479523372521008e-194 AND 1.0959046745042015e-193
        ) THEN 5.479523372521008e-194
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.0959046745042015e-193 AND 2.191809349008403e-193
        ) THEN 1.0959046745042015e-193
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.191809349008403e-193 AND 4.383618698016806e-193
        ) THEN 2.191809349008403e-193
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.383618698016806e-193 AND 8.767237396033612e-193
        ) THEN 4.383618698016806e-193
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.767237396033612e-193 AND 1.7534474792067224e-192
        ) THEN 8.767237396033612e-193
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.7534474792067224e-192 AND 3.506894958413445e-192
        ) THEN 1.7534474792067224e-192
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.506894958413445e-192 AND 7.01378991682689e-192
        ) THEN 3.506894958413445e-192
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.01378991682689e-192 AND 1.402757983365378e-191
        ) THEN 7.01378991682689e-192
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.402757983365378e-191 AND 2.805515966730756e-191
        ) THEN 1.402757983365378e-191
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.805515966730756e-191 AND 5.611031933461512e-191
        ) THEN 2.805515966730756e-191
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.611031933461512e-191 AND 1.1222063866923024e-190
        ) THEN 5.611031933461512e-191
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.1222063866923024e-190 AND 2.2444127733846047e-190
        ) THEN 1.1222063866923024e-190
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.2444127733846047e-190 AND 4.4888255467692094e-190
        ) THEN 2.2444127733846047e-190
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.4888255467692094e-190 AND 8.977651093538419e-190
        ) THEN 4.4888255467692094e-190
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.977651093538419e-190 AND 1.7955302187076838e-189
        ) THEN 8.977651093538419e-190
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.7955302187076838e-189 AND 3.5910604374153675e-189
        ) THEN 1.7955302187076838e-189
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.5910604374153675e-189 AND 7.182120874830735e-189
        ) THEN 3.5910604374153675e-189
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.182120874830735e-189 AND 1.436424174966147e-188
        ) THEN 7.182120874830735e-189
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.436424174966147e-188 AND 2.872848349932294e-188
        ) THEN 1.436424174966147e-188
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.872848349932294e-188 AND 5.745696699864588e-188
        ) THEN 2.872848349932294e-188
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.745696699864588e-188 AND 1.1491393399729176e-187
        ) THEN 5.745696699864588e-188
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.1491393399729176e-187 AND 2.2982786799458352e-187
        ) THEN 1.1491393399729176e-187
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.2982786799458352e-187 AND 4.5965573598916705e-187
        ) THEN 2.2982786799458352e-187
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.5965573598916705e-187 AND 9.193114719783341e-187
        ) THEN 4.5965573598916705e-187
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.193114719783341e-187 AND 1.8386229439566682e-186
        ) THEN 9.193114719783341e-187
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.8386229439566682e-186 AND 3.6772458879133364e-186
        ) THEN 1.8386229439566682e-186
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.6772458879133364e-186 AND 7.354491775826673e-186
        ) THEN 3.6772458879133364e-186
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.354491775826673e-186 AND 1.4708983551653345e-185
        ) THEN 7.354491775826673e-186
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.4708983551653345e-185 AND 2.941796710330669e-185
        ) THEN 1.4708983551653345e-185
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.941796710330669e-185 AND 5.883593420661338e-185
        ) THEN 2.941796710330669e-185
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.883593420661338e-185 AND 1.1767186841322676e-184
        ) THEN 5.883593420661338e-185
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.1767186841322676e-184 AND 2.3534373682645353e-184
        ) THEN 1.1767186841322676e-184
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.3534373682645353e-184 AND 4.706874736529071e-184
        ) THEN 2.3534373682645353e-184
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.706874736529071e-184 AND 9.413749473058141e-184
        ) THEN 4.706874736529071e-184
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.413749473058141e-184 AND 1.8827498946116282e-183
        ) THEN 9.413749473058141e-184
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.8827498946116282e-183 AND 3.7654997892232564e-183
        ) THEN 1.8827498946116282e-183
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.7654997892232564e-183 AND 7.530999578446513e-183
        ) THEN 3.7654997892232564e-183
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.530999578446513e-183 AND 1.5061999156893026e-182
        ) THEN 7.530999578446513e-183
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.5061999156893026e-182 AND 3.012399831378605e-182
        ) THEN 1.5061999156893026e-182
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.012399831378605e-182 AND 6.02479966275721e-182
        ) THEN 3.012399831378605e-182
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.02479966275721e-182 AND 1.204959932551442e-181
        ) THEN 6.02479966275721e-182
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.204959932551442e-181 AND 2.409919865102884e-181
        ) THEN 1.204959932551442e-181
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.409919865102884e-181 AND 4.819839730205768e-181
        ) THEN 2.409919865102884e-181
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.819839730205768e-181 AND 9.639679460411536e-181
        ) THEN 4.819839730205768e-181
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.639679460411536e-181 AND 1.9279358920823073e-180
        ) THEN 9.639679460411536e-181
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.9279358920823073e-180 AND 3.855871784164615e-180
        ) THEN 1.9279358920823073e-180
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.855871784164615e-180 AND 7.71174356832923e-180
        ) THEN 3.855871784164615e-180
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.71174356832923e-180 AND 1.542348713665846e-179
        ) THEN 7.71174356832923e-180
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.542348713665846e-179 AND 3.084697427331692e-179
        ) THEN 1.542348713665846e-179
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.084697427331692e-179 AND 6.169394854663383e-179
        ) THEN 3.084697427331692e-179
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.169394854663383e-179 AND 1.2338789709326767e-178
        ) THEN 6.169394854663383e-179
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.2338789709326767e-178 AND 2.4677579418653533e-178
        ) THEN 1.2338789709326767e-178
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.4677579418653533e-178 AND 4.935515883730707e-178
        ) THEN 2.4677579418653533e-178
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.935515883730707e-178 AND 9.871031767461413e-178
        ) THEN 4.935515883730707e-178
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.871031767461413e-178 AND 1.9742063534922827e-177
        ) THEN 9.871031767461413e-178
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.9742063534922827e-177 AND 3.9484127069845653e-177
        ) THEN 1.9742063534922827e-177
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.9484127069845653e-177 AND 7.896825413969131e-177
        ) THEN 3.9484127069845653e-177
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.896825413969131e-177 AND 1.5793650827938261e-176
        ) THEN 7.896825413969131e-177
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.5793650827938261e-176 AND 3.1587301655876523e-176
        ) THEN 1.5793650827938261e-176
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.1587301655876523e-176 AND 6.317460331175305e-176
        ) THEN 3.1587301655876523e-176
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.317460331175305e-176 AND 1.263492066235061e-175
        ) THEN 6.317460331175305e-176
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.263492066235061e-175 AND 2.526984132470122e-175
        ) THEN 1.263492066235061e-175
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.526984132470122e-175 AND 5.053968264940244e-175
        ) THEN 2.526984132470122e-175
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.053968264940244e-175 AND 1.0107936529880487e-174
        ) THEN 5.053968264940244e-175
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.0107936529880487e-174 AND 2.0215873059760975e-174
        ) THEN 1.0107936529880487e-174
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.0215873059760975e-174 AND 4.043174611952195e-174
        ) THEN 2.0215873059760975e-174
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.043174611952195e-174 AND 8.08634922390439e-174
        ) THEN 4.043174611952195e-174
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.08634922390439e-174 AND 1.617269844780878e-173
        ) THEN 8.08634922390439e-174
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.617269844780878e-173 AND 3.234539689561756e-173
        ) THEN 1.617269844780878e-173
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.234539689561756e-173 AND 6.469079379123512e-173
        ) THEN 3.234539689561756e-173
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.469079379123512e-173 AND 1.2938158758247024e-172
        ) THEN 6.469079379123512e-173
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.2938158758247024e-172 AND 2.587631751649405e-172
        ) THEN 1.2938158758247024e-172
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.587631751649405e-172 AND 5.17526350329881e-172
        ) THEN 2.587631751649405e-172
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.17526350329881e-172 AND 1.035052700659762e-171
        ) THEN 5.17526350329881e-172
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.035052700659762e-171 AND 2.070105401319524e-171
        ) THEN 1.035052700659762e-171
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.070105401319524e-171 AND 4.140210802639048e-171
        ) THEN 2.070105401319524e-171
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.140210802639048e-171 AND 8.280421605278095e-171
        ) THEN 4.140210802639048e-171
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.280421605278095e-171 AND 1.656084321055619e-170
        ) THEN 8.280421605278095e-171
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.656084321055619e-170 AND 3.312168642111238e-170
        ) THEN 1.656084321055619e-170
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.312168642111238e-170 AND 6.624337284222476e-170
        ) THEN 3.312168642111238e-170
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.624337284222476e-170 AND 1.3248674568444952e-169
        ) THEN 6.624337284222476e-170
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.3248674568444952e-169 AND 2.6497349136889905e-169
        ) THEN 1.3248674568444952e-169
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.6497349136889905e-169 AND 5.299469827377981e-169
        ) THEN 2.6497349136889905e-169
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.299469827377981e-169 AND 1.0598939654755962e-168
        ) THEN 5.299469827377981e-169
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.0598939654755962e-168 AND 2.1197879309511924e-168
        ) THEN 1.0598939654755962e-168
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.1197879309511924e-168 AND 4.239575861902385e-168
        ) THEN 2.1197879309511924e-168
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.239575861902385e-168 AND 8.47915172380477e-168
        ) THEN 4.239575861902385e-168
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.47915172380477e-168 AND 1.695830344760954e-167
        ) THEN 8.47915172380477e-168
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.695830344760954e-167 AND 3.391660689521908e-167
        ) THEN 1.695830344760954e-167
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.391660689521908e-167 AND 6.783321379043816e-167
        ) THEN 3.391660689521908e-167
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.783321379043816e-167 AND 1.3566642758087631e-166
        ) THEN 6.783321379043816e-167
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.3566642758087631e-166 AND 2.7133285516175262e-166
        ) THEN 1.3566642758087631e-166
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.7133285516175262e-166 AND 5.426657103235053e-166
        ) THEN 2.7133285516175262e-166
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.426657103235053e-166 AND 1.0853314206470105e-165
        ) THEN 5.426657103235053e-166
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.0853314206470105e-165 AND 2.170662841294021e-165
        ) THEN 1.0853314206470105e-165
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.170662841294021e-165 AND 4.341325682588042e-165
        ) THEN 2.170662841294021e-165
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.341325682588042e-165 AND 8.682651365176084e-165
        ) THEN 4.341325682588042e-165
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.682651365176084e-165 AND 1.7365302730352168e-164
        ) THEN 8.682651365176084e-165
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.7365302730352168e-164 AND 3.4730605460704336e-164
        ) THEN 1.7365302730352168e-164
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.4730605460704336e-164 AND 6.946121092140867e-164
        ) THEN 3.4730605460704336e-164
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.946121092140867e-164 AND 1.3892242184281734e-163
        ) THEN 6.946121092140867e-164
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.3892242184281734e-163 AND 2.778448436856347e-163
        ) THEN 1.3892242184281734e-163
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.778448436856347e-163 AND 5.556896873712694e-163
        ) THEN 2.778448436856347e-163
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.556896873712694e-163 AND 1.1113793747425387e-162
        ) THEN 5.556896873712694e-163
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.1113793747425387e-162 AND 2.2227587494850775e-162
        ) THEN 1.1113793747425387e-162
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.2227587494850775e-162 AND 4.445517498970155e-162
        ) THEN 2.2227587494850775e-162
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.445517498970155e-162 AND 8.89103499794031e-162
        ) THEN 4.445517498970155e-162
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.89103499794031e-162 AND 1.778206999588062e-161
        ) THEN 8.89103499794031e-162
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.778206999588062e-161 AND 3.556413999176124e-161
        ) THEN 1.778206999588062e-161
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.556413999176124e-161 AND 7.112827998352248e-161
        ) THEN 3.556413999176124e-161
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.112827998352248e-161 AND 1.4225655996704496e-160
        ) THEN 7.112827998352248e-161
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.4225655996704496e-160 AND 2.8451311993408992e-160
        ) THEN 1.4225655996704496e-160
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.8451311993408992e-160 AND 5.6902623986817984e-160
        ) THEN 2.8451311993408992e-160
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.6902623986817984e-160 AND 1.1380524797363597e-159
        ) THEN 5.6902623986817984e-160
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.1380524797363597e-159 AND 2.2761049594727193e-159
        ) THEN 1.1380524797363597e-159
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.2761049594727193e-159 AND 4.552209918945439e-159
        ) THEN 2.2761049594727193e-159
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.552209918945439e-159 AND 9.104419837890877e-159
        ) THEN 4.552209918945439e-159
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.104419837890877e-159 AND 1.8208839675781755e-158
        ) THEN 9.104419837890877e-159
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.8208839675781755e-158 AND 3.641767935156351e-158
        ) THEN 1.8208839675781755e-158
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.641767935156351e-158 AND 7.283535870312702e-158
        ) THEN 3.641767935156351e-158
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.283535870312702e-158 AND 1.4567071740625404e-157
        ) THEN 7.283535870312702e-158
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.4567071740625404e-157 AND 2.913414348125081e-157
        ) THEN 1.4567071740625404e-157
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.913414348125081e-157 AND 5.826828696250162e-157
        ) THEN 2.913414348125081e-157
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.826828696250162e-157 AND 1.1653657392500323e-156
        ) THEN 5.826828696250162e-157
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.1653657392500323e-156 AND 2.3307314785000646e-156
        ) THEN 1.1653657392500323e-156
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.3307314785000646e-156 AND 4.661462957000129e-156
        ) THEN 2.3307314785000646e-156
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.661462957000129e-156 AND 9.322925914000258e-156
        ) THEN 4.661462957000129e-156
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.322925914000258e-156 AND 1.8645851828000517e-155
        ) THEN 9.322925914000258e-156
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.8645851828000517e-155 AND 3.7291703656001034e-155
        ) THEN 1.8645851828000517e-155
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.7291703656001034e-155 AND 7.458340731200207e-155
        ) THEN 3.7291703656001034e-155
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.458340731200207e-155 AND 1.4916681462400413e-154
        ) THEN 7.458340731200207e-155
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.4916681462400413e-154 AND 2.983336292480083e-154
        ) THEN 1.4916681462400413e-154
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.983336292480083e-154 AND 5.966672584960166e-154
        ) THEN 2.983336292480083e-154
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.966672584960166e-154 AND 1.1933345169920331e-153
        ) THEN 5.966672584960166e-154
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.1933345169920331e-153 AND 2.3866690339840662e-153
        ) THEN 1.1933345169920331e-153
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.3866690339840662e-153 AND 4.7733380679681323e-153
        ) THEN 2.3866690339840662e-153
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.7733380679681323e-153 AND 9.546676135936265e-153
        ) THEN 4.7733380679681323e-153
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.546676135936265e-153 AND 1.909335227187253e-152
        ) THEN 9.546676135936265e-153
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.909335227187253e-152 AND 3.818670454374506e-152
        ) THEN 1.909335227187253e-152
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.818670454374506e-152 AND 7.637340908749012e-152
        ) THEN 3.818670454374506e-152
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.637340908749012e-152 AND 1.5274681817498023e-151
        ) THEN 7.637340908749012e-152
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.5274681817498023e-151 AND 3.054936363499605e-151
        ) THEN 1.5274681817498023e-151
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.054936363499605e-151 AND 6.10987272699921e-151
        ) THEN 3.054936363499605e-151
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.10987272699921e-151 AND 1.221974545399842e-150
        ) THEN 6.10987272699921e-151
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.221974545399842e-150 AND 2.443949090799684e-150
        ) THEN 1.221974545399842e-150
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.443949090799684e-150 AND 4.887898181599368e-150
        ) THEN 2.443949090799684e-150
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.887898181599368e-150 AND 9.775796363198735e-150
        ) THEN 4.887898181599368e-150
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.775796363198735e-150 AND 1.955159272639747e-149
        ) THEN 9.775796363198735e-150
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.955159272639747e-149 AND 3.910318545279494e-149
        ) THEN 1.955159272639747e-149
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.910318545279494e-149 AND 7.820637090558988e-149
        ) THEN 3.910318545279494e-149
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.820637090558988e-149 AND 1.5641274181117976e-148
        ) THEN 7.820637090558988e-149
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.5641274181117976e-148 AND 3.1282548362235952e-148
        ) THEN 1.5641274181117976e-148
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.1282548362235952e-148 AND 6.256509672447191e-148
        ) THEN 3.1282548362235952e-148
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.256509672447191e-148 AND 1.2513019344894381e-147
        ) THEN 6.256509672447191e-148
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.2513019344894381e-147 AND 2.5026038689788762e-147
        ) THEN 1.2513019344894381e-147
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.5026038689788762e-147 AND 5.0052077379577523e-147
        ) THEN 2.5026038689788762e-147
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.0052077379577523e-147 AND 1.0010415475915505e-146
        ) THEN 5.0052077379577523e-147
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.0010415475915505e-146 AND 2.002083095183101e-146
        ) THEN 1.0010415475915505e-146
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.002083095183101e-146 AND 4.004166190366202e-146
        ) THEN 2.002083095183101e-146
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.004166190366202e-146 AND 8.008332380732404e-146
        ) THEN 4.004166190366202e-146
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.008332380732404e-146 AND 1.6016664761464807e-145
        ) THEN 8.008332380732404e-146
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.6016664761464807e-145 AND 3.2033329522929615e-145
        ) THEN 1.6016664761464807e-145
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.2033329522929615e-145 AND 6.406665904585923e-145
        ) THEN 3.2033329522929615e-145
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.406665904585923e-145 AND 1.2813331809171846e-144
        ) THEN 6.406665904585923e-145
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.2813331809171846e-144 AND 2.5626663618343692e-144
        ) THEN 1.2813331809171846e-144
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.5626663618343692e-144 AND 5.1253327236687384e-144
        ) THEN 2.5626663618343692e-144
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.1253327236687384e-144 AND 1.0250665447337477e-143
        ) THEN 5.1253327236687384e-144
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.0250665447337477e-143 AND 2.0501330894674953e-143
        ) THEN 1.0250665447337477e-143
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.0501330894674953e-143 AND 4.100266178934991e-143
        ) THEN 2.0501330894674953e-143
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.100266178934991e-143 AND 8.200532357869981e-143
        ) THEN 4.100266178934991e-143
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.200532357869981e-143 AND 1.6401064715739963e-142
        ) THEN 8.200532357869981e-143
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.6401064715739963e-142 AND 3.2802129431479926e-142
        ) THEN 1.6401064715739963e-142
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.2802129431479926e-142 AND 6.560425886295985e-142
        ) THEN 3.2802129431479926e-142
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.560425886295985e-142 AND 1.312085177259197e-141
        ) THEN 6.560425886295985e-142
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.312085177259197e-141 AND 2.624170354518394e-141
        ) THEN 1.312085177259197e-141
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.624170354518394e-141 AND 5.248340709036788e-141
        ) THEN 2.624170354518394e-141
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.248340709036788e-141 AND 1.0496681418073576e-140
        ) THEN 5.248340709036788e-141
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.0496681418073576e-140 AND 2.0993362836147152e-140
        ) THEN 1.0496681418073576e-140
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.0993362836147152e-140 AND 4.1986725672294305e-140
        ) THEN 2.0993362836147152e-140
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.1986725672294305e-140 AND 8.397345134458861e-140
        ) THEN 4.1986725672294305e-140
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.397345134458861e-140 AND 1.6794690268917722e-139
        ) THEN 8.397345134458861e-140
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.6794690268917722e-139 AND 3.3589380537835444e-139
        ) THEN 1.6794690268917722e-139
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.3589380537835444e-139 AND 6.717876107567089e-139
        ) THEN 3.3589380537835444e-139
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.717876107567089e-139 AND 1.3435752215134178e-138
        ) THEN 6.717876107567089e-139
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.3435752215134178e-138 AND 2.6871504430268355e-138
        ) THEN 1.3435752215134178e-138
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.6871504430268355e-138 AND 5.374300886053671e-138
        ) THEN 2.6871504430268355e-138
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.374300886053671e-138 AND 1.0748601772107342e-137
        ) THEN 5.374300886053671e-138
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.0748601772107342e-137 AND 2.1497203544214684e-137
        ) THEN 1.0748601772107342e-137
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.1497203544214684e-137 AND 4.299440708842937e-137
        ) THEN 2.1497203544214684e-137
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.299440708842937e-137 AND 8.598881417685874e-137
        ) THEN 4.299440708842937e-137
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.598881417685874e-137 AND 1.7197762835371747e-136
        ) THEN 8.598881417685874e-137
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.7197762835371747e-136 AND 3.4395525670743494e-136
        ) THEN 1.7197762835371747e-136
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.4395525670743494e-136 AND 6.879105134148699e-136
        ) THEN 3.4395525670743494e-136
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.879105134148699e-136 AND 1.3758210268297398e-135
        ) THEN 6.879105134148699e-136
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.3758210268297398e-135 AND 2.7516420536594796e-135
        ) THEN 1.3758210268297398e-135
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.7516420536594796e-135 AND 5.503284107318959e-135
        ) THEN 2.7516420536594796e-135
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.503284107318959e-135 AND 1.1006568214637918e-134
        ) THEN 5.503284107318959e-135
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.1006568214637918e-134 AND 2.2013136429275836e-134
        ) THEN 1.1006568214637918e-134
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.2013136429275836e-134 AND 4.4026272858551673e-134
        ) THEN 2.2013136429275836e-134
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.4026272858551673e-134 AND 8.805254571710335e-134
        ) THEN 4.4026272858551673e-134
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.805254571710335e-134 AND 1.761050914342067e-133
        ) THEN 8.805254571710335e-134
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.761050914342067e-133 AND 3.522101828684134e-133
        ) THEN 1.761050914342067e-133
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.522101828684134e-133 AND 7.044203657368268e-133
        ) THEN 3.522101828684134e-133
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.044203657368268e-133 AND 1.4088407314736535e-132
        ) THEN 7.044203657368268e-133
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.4088407314736535e-132 AND 2.817681462947307e-132
        ) THEN 1.4088407314736535e-132
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.817681462947307e-132 AND 5.635362925894614e-132
        ) THEN 2.817681462947307e-132
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.635362925894614e-132 AND 1.1270725851789228e-131
        ) THEN 5.635362925894614e-132
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.1270725851789228e-131 AND 2.2541451703578456e-131
        ) THEN 1.1270725851789228e-131
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.2541451703578456e-131 AND 4.5082903407156913e-131
        ) THEN 2.2541451703578456e-131
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.5082903407156913e-131 AND 9.016580681431383e-131
        ) THEN 4.5082903407156913e-131
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.016580681431383e-131 AND 1.8033161362862765e-130
        ) THEN 9.016580681431383e-131
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.8033161362862765e-130 AND 3.606632272572553e-130
        ) THEN 1.8033161362862765e-130
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.606632272572553e-130 AND 7.213264545145106e-130
        ) THEN 3.606632272572553e-130
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.213264545145106e-130 AND 1.4426529090290212e-129
        ) THEN 7.213264545145106e-130
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.4426529090290212e-129 AND 2.8853058180580424e-129
        ) THEN 1.4426529090290212e-129
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.8853058180580424e-129 AND 5.770611636116085e-129
        ) THEN 2.8853058180580424e-129
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.770611636116085e-129 AND 1.154122327223217e-128
        ) THEN 5.770611636116085e-129
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.154122327223217e-128 AND 2.308244654446434e-128
        ) THEN 1.154122327223217e-128
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.308244654446434e-128 AND 4.616489308892868e-128
        ) THEN 2.308244654446434e-128
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.616489308892868e-128 AND 9.232978617785736e-128
        ) THEN 4.616489308892868e-128
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.232978617785736e-128 AND 1.8465957235571472e-127
        ) THEN 9.232978617785736e-128
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.8465957235571472e-127 AND 3.6931914471142943e-127
        ) THEN 1.8465957235571472e-127
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.6931914471142943e-127 AND 7.386382894228589e-127
        ) THEN 3.6931914471142943e-127
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.386382894228589e-127 AND 1.4772765788457177e-126
        ) THEN 7.386382894228589e-127
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.4772765788457177e-126 AND 2.9545531576914354e-126
        ) THEN 1.4772765788457177e-126
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.9545531576914354e-126 AND 5.909106315382871e-126
        ) THEN 2.9545531576914354e-126
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.909106315382871e-126 AND 1.1818212630765742e-125
        ) THEN 5.909106315382871e-126
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.1818212630765742e-125 AND 2.3636425261531484e-125
        ) THEN 1.1818212630765742e-125
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.3636425261531484e-125 AND 4.727285052306297e-125
        ) THEN 2.3636425261531484e-125
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.727285052306297e-125 AND 9.454570104612593e-125
        ) THEN 4.727285052306297e-125
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.454570104612593e-125 AND 1.8909140209225187e-124
        ) THEN 9.454570104612593e-125
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.8909140209225187e-124 AND 3.7818280418450374e-124
        ) THEN 1.8909140209225187e-124
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.7818280418450374e-124 AND 7.563656083690075e-124
        ) THEN 3.7818280418450374e-124
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.563656083690075e-124 AND 1.512731216738015e-123
        ) THEN 7.563656083690075e-124
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.512731216738015e-123 AND 3.02546243347603e-123
        ) THEN 1.512731216738015e-123
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.02546243347603e-123 AND 6.05092486695206e-123
        ) THEN 3.02546243347603e-123
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.05092486695206e-123 AND 1.210184973390412e-122
        ) THEN 6.05092486695206e-123
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.210184973390412e-122 AND 2.420369946780824e-122
        ) THEN 1.210184973390412e-122
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.420369946780824e-122 AND 4.840739893561648e-122
        ) THEN 2.420369946780824e-122
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.840739893561648e-122 AND 9.681479787123296e-122
        ) THEN 4.840739893561648e-122
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.681479787123296e-122 AND 1.9362959574246591e-121
        ) THEN 9.681479787123296e-122
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.9362959574246591e-121 AND 3.8725919148493183e-121
        ) THEN 1.9362959574246591e-121
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.8725919148493183e-121 AND 7.745183829698637e-121
        ) THEN 3.8725919148493183e-121
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.745183829698637e-121 AND 1.5490367659397273e-120
        ) THEN 7.745183829698637e-121
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.5490367659397273e-120 AND 3.0980735318794546e-120
        ) THEN 1.5490367659397273e-120
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.0980735318794546e-120 AND 6.196147063758909e-120
        ) THEN 3.0980735318794546e-120
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.196147063758909e-120 AND 1.2392294127517818e-119
        ) THEN 6.196147063758909e-120
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.2392294127517818e-119 AND 2.4784588255035637e-119
        ) THEN 1.2392294127517818e-119
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.4784588255035637e-119 AND 4.9569176510071274e-119
        ) THEN 2.4784588255035637e-119
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.9569176510071274e-119 AND 9.913835302014255e-119
        ) THEN 4.9569176510071274e-119
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.913835302014255e-119 AND 1.982767060402851e-118
        ) THEN 9.913835302014255e-119
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.982767060402851e-118 AND 3.965534120805702e-118
        ) THEN 1.982767060402851e-118
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.965534120805702e-118 AND 7.931068241611404e-118
        ) THEN 3.965534120805702e-118
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.931068241611404e-118 AND 1.5862136483222808e-117
        ) THEN 7.931068241611404e-118
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.5862136483222808e-117 AND 3.1724272966445615e-117
        ) THEN 1.5862136483222808e-117
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.1724272966445615e-117 AND 6.344854593289123e-117
        ) THEN 3.1724272966445615e-117
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.344854593289123e-117 AND 1.2689709186578246e-116
        ) THEN 6.344854593289123e-117
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.2689709186578246e-116 AND 2.5379418373156492e-116
        ) THEN 1.2689709186578246e-116
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.5379418373156492e-116 AND 5.075883674631299e-116
        ) THEN 2.5379418373156492e-116
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.075883674631299e-116 AND 1.0151767349262597e-115
        ) THEN 5.075883674631299e-116
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.0151767349262597e-115 AND 2.0303534698525194e-115
        ) THEN 1.0151767349262597e-115
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.0303534698525194e-115 AND 4.060706939705039e-115
        ) THEN 2.0303534698525194e-115
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.060706939705039e-115 AND 8.121413879410078e-115
        ) THEN 4.060706939705039e-115
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.121413879410078e-115 AND 1.6242827758820155e-114
        ) THEN 8.121413879410078e-115
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.6242827758820155e-114 AND 3.248565551764031e-114
        ) THEN 1.6242827758820155e-114
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.248565551764031e-114 AND 6.497131103528062e-114
        ) THEN 3.248565551764031e-114
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.497131103528062e-114 AND 1.2994262207056124e-113
        ) THEN 6.497131103528062e-114
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.2994262207056124e-113 AND 2.598852441411225e-113
        ) THEN 1.2994262207056124e-113
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.598852441411225e-113 AND 5.19770488282245e-113
        ) THEN 2.598852441411225e-113
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.19770488282245e-113 AND 1.03954097656449e-112
        ) THEN 5.19770488282245e-113
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.03954097656449e-112 AND 2.07908195312898e-112
        ) THEN 1.03954097656449e-112
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.07908195312898e-112 AND 4.15816390625796e-112
        ) THEN 2.07908195312898e-112
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.15816390625796e-112 AND 8.31632781251592e-112
        ) THEN 4.15816390625796e-112
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.31632781251592e-112 AND 1.663265562503184e-111
        ) THEN 8.31632781251592e-112
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.663265562503184e-111 AND 3.326531125006368e-111
        ) THEN 1.663265562503184e-111
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.326531125006368e-111 AND 6.653062250012736e-111
        ) THEN 3.326531125006368e-111
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.653062250012736e-111 AND 1.3306124500025471e-110
        ) THEN 6.653062250012736e-111
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.3306124500025471e-110 AND 2.6612249000050942e-110
        ) THEN 1.3306124500025471e-110
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.6612249000050942e-110 AND 5.3224498000101884e-110
        ) THEN 2.6612249000050942e-110
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.3224498000101884e-110 AND 1.0644899600020377e-109
        ) THEN 5.3224498000101884e-110
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.0644899600020377e-109 AND 2.1289799200040754e-109
        ) THEN 1.0644899600020377e-109
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.1289799200040754e-109 AND 4.257959840008151e-109
        ) THEN 2.1289799200040754e-109
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.257959840008151e-109 AND 8.515919680016301e-109
        ) THEN 4.257959840008151e-109
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.515919680016301e-109 AND 1.7031839360032603e-108
        ) THEN 8.515919680016301e-109
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.7031839360032603e-108 AND 3.4063678720065206e-108
        ) THEN 1.7031839360032603e-108
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.4063678720065206e-108 AND 6.812735744013041e-108
        ) THEN 3.4063678720065206e-108
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.812735744013041e-108 AND 1.3625471488026082e-107
        ) THEN 6.812735744013041e-108
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.3625471488026082e-107 AND 2.7250942976052165e-107
        ) THEN 1.3625471488026082e-107
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.7250942976052165e-107 AND 5.450188595210433e-107
        ) THEN 2.7250942976052165e-107
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.450188595210433e-107 AND 1.0900377190420866e-106
        ) THEN 5.450188595210433e-107
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.0900377190420866e-106 AND 2.1800754380841732e-106
        ) THEN 1.0900377190420866e-106
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.1800754380841732e-106 AND 4.3601508761683463e-106
        ) THEN 2.1800754380841732e-106
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.3601508761683463e-106 AND 8.720301752336693e-106
        ) THEN 4.3601508761683463e-106
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.720301752336693e-106 AND 1.7440603504673385e-105
        ) THEN 8.720301752336693e-106
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.7440603504673385e-105 AND 3.488120700934677e-105
        ) THEN 1.7440603504673385e-105
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.488120700934677e-105 AND 6.976241401869354e-105
        ) THEN 3.488120700934677e-105
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.976241401869354e-105 AND 1.3952482803738708e-104
        ) THEN 6.976241401869354e-105
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.3952482803738708e-104 AND 2.7904965607477417e-104
        ) THEN 1.3952482803738708e-104
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.7904965607477417e-104 AND 5.5809931214954833e-104
        ) THEN 2.7904965607477417e-104
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.5809931214954833e-104 AND 1.1161986242990967e-103
        ) THEN 5.5809931214954833e-104
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.1161986242990967e-103 AND 2.2323972485981933e-103
        ) THEN 1.1161986242990967e-103
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.2323972485981933e-103 AND 4.464794497196387e-103
        ) THEN 2.2323972485981933e-103
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.464794497196387e-103 AND 8.929588994392773e-103
        ) THEN 4.464794497196387e-103
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.929588994392773e-103 AND 1.7859177988785547e-102
        ) THEN 8.929588994392773e-103
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.7859177988785547e-102 AND 3.5718355977571093e-102
        ) THEN 1.7859177988785547e-102
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.5718355977571093e-102 AND 7.143671195514219e-102
        ) THEN 3.5718355977571093e-102
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.143671195514219e-102 AND 1.4287342391028437e-101
        ) THEN 7.143671195514219e-102
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.4287342391028437e-101 AND 2.8574684782056875e-101
        ) THEN 1.4287342391028437e-101
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.8574684782056875e-101 AND 5.714936956411375e-101
        ) THEN 2.8574684782056875e-101
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.714936956411375e-101 AND 1.142987391282275e-100
        ) THEN 5.714936956411375e-101
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.142987391282275e-100 AND 2.28597478256455e-100
        ) THEN 1.142987391282275e-100
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.28597478256455e-100 AND 4.5719495651291e-100
        ) THEN 2.28597478256455e-100
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.5719495651291e-100 AND 9.1438991302582e-100
        ) THEN 4.5719495651291e-100
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.1438991302582e-100 AND 1.82877982605164e-99
        ) THEN 9.1438991302582e-100
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.82877982605164e-99 AND 3.65755965210328e-99
        ) THEN 1.82877982605164e-99
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.65755965210328e-99 AND 7.31511930420656e-99
        ) THEN 3.65755965210328e-99
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.31511930420656e-99 AND 1.463023860841312e-98
        ) THEN 7.31511930420656e-99
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.463023860841312e-98 AND 2.926047721682624e-98
        ) THEN 1.463023860841312e-98
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.926047721682624e-98 AND 5.852095443365248e-98
        ) THEN 2.926047721682624e-98
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.852095443365248e-98 AND 1.1704190886730496e-97
        ) THEN 5.852095443365248e-98
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.1704190886730496e-97 AND 2.3408381773460992e-97
        ) THEN 1.1704190886730496e-97
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.3408381773460992e-97 AND 4.6816763546921983e-97
        ) THEN 2.3408381773460992e-97
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.6816763546921983e-97 AND 9.363352709384397e-97
        ) THEN 4.6816763546921983e-97
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.363352709384397e-97 AND 1.8726705418768793e-96
        ) THEN 9.363352709384397e-97
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.8726705418768793e-96 AND 3.745341083753759e-96
        ) THEN 1.8726705418768793e-96
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.745341083753759e-96 AND 7.490682167507517e-96
        ) THEN 3.745341083753759e-96
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.490682167507517e-96 AND 1.4981364335015035e-95
        ) THEN 7.490682167507517e-96
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.4981364335015035e-95 AND 2.996272867003007e-95
        ) THEN 1.4981364335015035e-95
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.996272867003007e-95 AND 5.992545734006014e-95
        ) THEN 2.996272867003007e-95
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.992545734006014e-95 AND 1.1985091468012028e-94
        ) THEN 5.992545734006014e-95
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.1985091468012028e-94 AND 2.3970182936024055e-94
        ) THEN 1.1985091468012028e-94
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.3970182936024055e-94 AND 4.794036587204811e-94
        ) THEN 2.3970182936024055e-94
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.794036587204811e-94 AND 9.588073174409622e-94
        ) THEN 4.794036587204811e-94
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.588073174409622e-94 AND 1.9176146348819244e-93
        ) THEN 9.588073174409622e-94
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.9176146348819244e-93 AND 3.835229269763849e-93
        ) THEN 1.9176146348819244e-93
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.835229269763849e-93 AND 7.670458539527698e-93
        ) THEN 3.835229269763849e-93
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.670458539527698e-93 AND 1.5340917079055395e-92
        ) THEN 7.670458539527698e-93
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.5340917079055395e-92 AND 3.068183415811079e-92
        ) THEN 1.5340917079055395e-92
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.068183415811079e-92 AND 6.136366831622158e-92
        ) THEN 3.068183415811079e-92
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.136366831622158e-92 AND 1.2272733663244316e-91
        ) THEN 6.136366831622158e-92
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.2272733663244316e-91 AND 2.4545467326488633e-91
        ) THEN 1.2272733663244316e-91
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.4545467326488633e-91 AND 4.909093465297727e-91
        ) THEN 2.4545467326488633e-91
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.909093465297727e-91 AND 9.818186930595453e-91
        ) THEN 4.909093465297727e-91
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.818186930595453e-91 AND 1.9636373861190906e-90
        ) THEN 9.818186930595453e-91
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.9636373861190906e-90 AND 3.9272747722381812e-90
        ) THEN 1.9636373861190906e-90
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.9272747722381812e-90 AND 7.854549544476363e-90
        ) THEN 3.9272747722381812e-90
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.854549544476363e-90 AND 1.5709099088952725e-89
        ) THEN 7.854549544476363e-90
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.5709099088952725e-89 AND 3.141819817790545e-89
        ) THEN 1.5709099088952725e-89
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.141819817790545e-89 AND 6.28363963558109e-89
        ) THEN 3.141819817790545e-89
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.28363963558109e-89 AND 1.256727927116218e-88
        ) THEN 6.28363963558109e-89
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.256727927116218e-88 AND 2.513455854232436e-88
        ) THEN 1.256727927116218e-88
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.513455854232436e-88 AND 5.026911708464872e-88
        ) THEN 2.513455854232436e-88
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.026911708464872e-88 AND 1.0053823416929744e-87
        ) THEN 5.026911708464872e-88
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.0053823416929744e-87 AND 2.010764683385949e-87
        ) THEN 1.0053823416929744e-87
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.010764683385949e-87 AND 4.021529366771898e-87
        ) THEN 2.010764683385949e-87
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.021529366771898e-87 AND 8.043058733543795e-87
        ) THEN 4.021529366771898e-87
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.043058733543795e-87 AND 1.608611746708759e-86
        ) THEN 8.043058733543795e-87
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.608611746708759e-86 AND 3.217223493417518e-86
        ) THEN 1.608611746708759e-86
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.217223493417518e-86 AND 6.434446986835036e-86
        ) THEN 3.217223493417518e-86
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.434446986835036e-86 AND 1.2868893973670072e-85
        ) THEN 6.434446986835036e-86
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.2868893973670072e-85 AND 2.5737787947340145e-85
        ) THEN 1.2868893973670072e-85
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.5737787947340145e-85 AND 5.147557589468029e-85
        ) THEN 2.5737787947340145e-85
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.147557589468029e-85 AND 1.0295115178936058e-84
        ) THEN 5.147557589468029e-85
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.0295115178936058e-84 AND 2.0590230357872116e-84
        ) THEN 1.0295115178936058e-84
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.0590230357872116e-84 AND 4.118046071574423e-84
        ) THEN 2.0590230357872116e-84
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.118046071574423e-84 AND 8.236092143148846e-84
        ) THEN 4.118046071574423e-84
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.236092143148846e-84 AND 1.6472184286297693e-83
        ) THEN 8.236092143148846e-84
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.6472184286297693e-83 AND 3.2944368572595385e-83
        ) THEN 1.6472184286297693e-83
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.2944368572595385e-83 AND 6.588873714519077e-83
        ) THEN 3.2944368572595385e-83
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.588873714519077e-83 AND 1.3177747429038154e-82
        ) THEN 6.588873714519077e-83
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.3177747429038154e-82 AND 2.635549485807631e-82
        ) THEN 1.3177747429038154e-82
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.635549485807631e-82 AND 5.271098971615262e-82
        ) THEN 2.635549485807631e-82
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.271098971615262e-82 AND 1.0542197943230523e-81
        ) THEN 5.271098971615262e-82
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.0542197943230523e-81 AND 2.1084395886461046e-81
        ) THEN 1.0542197943230523e-81
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.1084395886461046e-81 AND 4.2168791772922093e-81
        ) THEN 2.1084395886461046e-81
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.2168791772922093e-81 AND 8.433758354584419e-81
        ) THEN 4.2168791772922093e-81
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.433758354584419e-81 AND 1.6867516709168837e-80
        ) THEN 8.433758354584419e-81
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.6867516709168837e-80 AND 3.3735033418337674e-80
        ) THEN 1.6867516709168837e-80
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.3735033418337674e-80 AND 6.747006683667535e-80
        ) THEN 3.3735033418337674e-80
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.747006683667535e-80 AND 1.349401336733507e-79
        ) THEN 6.747006683667535e-80
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.349401336733507e-79 AND 2.698802673467014e-79
        ) THEN 1.349401336733507e-79
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.698802673467014e-79 AND 5.397605346934028e-79
        ) THEN 2.698802673467014e-79
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.397605346934028e-79 AND 1.0795210693868056e-78
        ) THEN 5.397605346934028e-79
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.0795210693868056e-78 AND 2.1590421387736112e-78
        ) THEN 1.0795210693868056e-78
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.1590421387736112e-78 AND 4.3180842775472223e-78
        ) THEN 2.1590421387736112e-78
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.3180842775472223e-78 AND 8.636168555094445e-78
        ) THEN 4.3180842775472223e-78
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.636168555094445e-78 AND 1.727233711018889e-77
        ) THEN 8.636168555094445e-78
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.727233711018889e-77 AND 3.454467422037778e-77
        ) THEN 1.727233711018889e-77
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.454467422037778e-77 AND 6.908934844075556e-77
        ) THEN 3.454467422037778e-77
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.908934844075556e-77 AND 1.3817869688151111e-76
        ) THEN 6.908934844075556e-77
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.3817869688151111e-76 AND 2.7635739376302223e-76
        ) THEN 1.3817869688151111e-76
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.7635739376302223e-76 AND 5.527147875260445e-76
        ) THEN 2.7635739376302223e-76
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.527147875260445e-76 AND 1.105429575052089e-75
        ) THEN 5.527147875260445e-76
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.105429575052089e-75 AND 2.210859150104178e-75
        ) THEN 1.105429575052089e-75
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.210859150104178e-75 AND 4.421718300208356e-75
        ) THEN 2.210859150104178e-75
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.421718300208356e-75 AND 8.843436600416711e-75
        ) THEN 4.421718300208356e-75
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.843436600416711e-75 AND 1.7686873200833423e-74
        ) THEN 8.843436600416711e-75
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.7686873200833423e-74 AND 3.5373746401666845e-74
        ) THEN 1.7686873200833423e-74
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.5373746401666845e-74 AND 7.074749280333369e-74
        ) THEN 3.5373746401666845e-74
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.074749280333369e-74 AND 1.4149498560666738e-73
        ) THEN 7.074749280333369e-74
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.4149498560666738e-73 AND 2.8298997121333476e-73
        ) THEN 1.4149498560666738e-73
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.8298997121333476e-73 AND 5.659799424266695e-73
        ) THEN 2.8298997121333476e-73
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.659799424266695e-73 AND 1.131959884853339e-72
        ) THEN 5.659799424266695e-73
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.131959884853339e-72 AND 2.263919769706678e-72
        ) THEN 1.131959884853339e-72
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.263919769706678e-72 AND 4.527839539413356e-72
        ) THEN 2.263919769706678e-72
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.527839539413356e-72 AND 9.055679078826712e-72
        ) THEN 4.527839539413356e-72
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.055679078826712e-72 AND 1.8111358157653425e-71
        ) THEN 9.055679078826712e-72
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.8111358157653425e-71 AND 3.622271631530685e-71
        ) THEN 1.8111358157653425e-71
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.622271631530685e-71 AND 7.24454326306137e-71
        ) THEN 3.622271631530685e-71
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.24454326306137e-71 AND 1.448908652612274e-70
        ) THEN 7.24454326306137e-71
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.448908652612274e-70 AND 2.897817305224548e-70
        ) THEN 1.448908652612274e-70
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.897817305224548e-70 AND 5.795634610449096e-70
        ) THEN 2.897817305224548e-70
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.795634610449096e-70 AND 1.1591269220898192e-69
        ) THEN 5.795634610449096e-70
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.1591269220898192e-69 AND 2.3182538441796384e-69
        ) THEN 1.1591269220898192e-69
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.3182538441796384e-69 AND 4.636507688359277e-69
        ) THEN 2.3182538441796384e-69
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.636507688359277e-69 AND 9.273015376718553e-69
        ) THEN 4.636507688359277e-69
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.273015376718553e-69 AND 1.8546030753437107e-68
        ) THEN 9.273015376718553e-69
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.8546030753437107e-68 AND 3.7092061506874214e-68
        ) THEN 1.8546030753437107e-68
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.7092061506874214e-68 AND 7.418412301374843e-68
        ) THEN 3.7092061506874214e-68
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.418412301374843e-68 AND 1.4836824602749686e-67
        ) THEN 7.418412301374843e-68
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.4836824602749686e-67 AND 2.967364920549937e-67
        ) THEN 1.4836824602749686e-67
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.967364920549937e-67 AND 5.934729841099874e-67
        ) THEN 2.967364920549937e-67
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.934729841099874e-67 AND 1.1869459682199748e-66
        ) THEN 5.934729841099874e-67
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.1869459682199748e-66 AND 2.3738919364399497e-66
        ) THEN 1.1869459682199748e-66
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.3738919364399497e-66 AND 4.7477838728798994e-66
        ) THEN 2.3738919364399497e-66
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.7477838728798994e-66 AND 9.495567745759799e-66
        ) THEN 4.7477838728798994e-66
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.495567745759799e-66 AND 1.8991135491519597e-65
        ) THEN 9.495567745759799e-66
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.8991135491519597e-65 AND 3.7982270983039195e-65
        ) THEN 1.8991135491519597e-65
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.7982270983039195e-65 AND 7.596454196607839e-65
        ) THEN 3.7982270983039195e-65
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.596454196607839e-65 AND 1.5192908393215678e-64
        ) THEN 7.596454196607839e-65
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.5192908393215678e-64 AND 3.0385816786431356e-64
        ) THEN 1.5192908393215678e-64
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.0385816786431356e-64 AND 6.077163357286271e-64
        ) THEN 3.0385816786431356e-64
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.077163357286271e-64 AND 1.2154326714572542e-63
        ) THEN 6.077163357286271e-64
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.2154326714572542e-63 AND 2.4308653429145085e-63
        ) THEN 1.2154326714572542e-63
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.4308653429145085e-63 AND 4.861730685829017e-63
        ) THEN 2.4308653429145085e-63
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.861730685829017e-63 AND 9.723461371658034e-63
        ) THEN 4.861730685829017e-63
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.723461371658034e-63 AND 1.9446922743316068e-62
        ) THEN 9.723461371658034e-63
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.9446922743316068e-62 AND 3.8893845486632136e-62
        ) THEN 1.9446922743316068e-62
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.8893845486632136e-62 AND 7.778769097326427e-62
        ) THEN 3.8893845486632136e-62
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.778769097326427e-62 AND 1.5557538194652854e-61
        ) THEN 7.778769097326427e-62
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.5557538194652854e-61 AND 3.111507638930571e-61
        ) THEN 1.5557538194652854e-61
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.111507638930571e-61 AND 6.223015277861142e-61
        ) THEN 3.111507638930571e-61
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.223015277861142e-61 AND 1.2446030555722283e-60
        ) THEN 6.223015277861142e-61
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.2446030555722283e-60 AND 2.4892061111444567e-60
        ) THEN 1.2446030555722283e-60
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.4892061111444567e-60 AND 4.9784122222889134e-60
        ) THEN 2.4892061111444567e-60
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.9784122222889134e-60 AND 9.956824444577827e-60
        ) THEN 4.9784122222889134e-60
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.956824444577827e-60 AND 1.9913648889155653e-59
        ) THEN 9.956824444577827e-60
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.9913648889155653e-59 AND 3.982729777831131e-59
        ) THEN 1.9913648889155653e-59
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.982729777831131e-59 AND 7.965459555662261e-59
        ) THEN 3.982729777831131e-59
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.965459555662261e-59 AND 1.5930919111324523e-58
        ) THEN 7.965459555662261e-59
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.5930919111324523e-58 AND 3.1861838222649046e-58
        ) THEN 1.5930919111324523e-58
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.1861838222649046e-58 AND 6.372367644529809e-58
        ) THEN 3.1861838222649046e-58
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.372367644529809e-58 AND 1.2744735289059618e-57
        ) THEN 6.372367644529809e-58
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.2744735289059618e-57 AND 2.5489470578119236e-57
        ) THEN 1.2744735289059618e-57
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.5489470578119236e-57 AND 5.0978941156238473e-57
        ) THEN 2.5489470578119236e-57
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.0978941156238473e-57 AND 1.0195788231247695e-56
        ) THEN 5.0978941156238473e-57
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.0195788231247695e-56 AND 2.039157646249539e-56
        ) THEN 1.0195788231247695e-56
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.039157646249539e-56 AND 4.078315292499078e-56
        ) THEN 2.039157646249539e-56
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.078315292499078e-56 AND 8.156630584998156e-56
        ) THEN 4.078315292499078e-56
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.156630584998156e-56 AND 1.6313261169996311e-55
        ) THEN 8.156630584998156e-56
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.6313261169996311e-55 AND 3.2626522339992623e-55
        ) THEN 1.6313261169996311e-55
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.2626522339992623e-55 AND 6.525304467998525e-55
        ) THEN 3.2626522339992623e-55
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.525304467998525e-55 AND 1.305060893599705e-54
        ) THEN 6.525304467998525e-55
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.305060893599705e-54 AND 2.61012178719941e-54
        ) THEN 1.305060893599705e-54
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.61012178719941e-54 AND 5.22024357439882e-54
        ) THEN 2.61012178719941e-54
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.22024357439882e-54 AND 1.044048714879764e-53
        ) THEN 5.22024357439882e-54
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.044048714879764e-53 AND 2.088097429759528e-53
        ) THEN 1.044048714879764e-53
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.088097429759528e-53 AND 4.176194859519056e-53
        ) THEN 2.088097429759528e-53
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.176194859519056e-53 AND 8.352389719038111e-53
        ) THEN 4.176194859519056e-53
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.352389719038111e-53 AND 1.6704779438076223e-52
        ) THEN 8.352389719038111e-53
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.6704779438076223e-52 AND 3.3409558876152446e-52
        ) THEN 1.6704779438076223e-52
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.3409558876152446e-52 AND 6.681911775230489e-52
        ) THEN 3.3409558876152446e-52
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.681911775230489e-52 AND 1.3363823550460978e-51
        ) THEN 6.681911775230489e-52
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.3363823550460978e-51 AND 2.6727647100921956e-51
        ) THEN 1.3363823550460978e-51
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.6727647100921956e-51 AND 5.345529420184391e-51
        ) THEN 2.6727647100921956e-51
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.345529420184391e-51 AND 1.0691058840368783e-50
        ) THEN 5.345529420184391e-51
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.0691058840368783e-50 AND 2.1382117680737565e-50
        ) THEN 1.0691058840368783e-50
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.1382117680737565e-50 AND 4.276423536147513e-50
        ) THEN 2.1382117680737565e-50
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.276423536147513e-50 AND 8.552847072295026e-50
        ) THEN 4.276423536147513e-50
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.552847072295026e-50 AND 1.7105694144590052e-49
        ) THEN 8.552847072295026e-50
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.7105694144590052e-49 AND 3.4211388289180104e-49
        ) THEN 1.7105694144590052e-49
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.4211388289180104e-49 AND 6.842277657836021e-49
        ) THEN 3.4211388289180104e-49
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.842277657836021e-49 AND 1.3684555315672042e-48
        ) THEN 6.842277657836021e-49
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.3684555315672042e-48 AND 2.7369110631344083e-48
        ) THEN 1.3684555315672042e-48
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.7369110631344083e-48 AND 5.473822126268817e-48
        ) THEN 2.7369110631344083e-48
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.473822126268817e-48 AND 1.0947644252537633e-47
        ) THEN 5.473822126268817e-48
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.0947644252537633e-47 AND 2.1895288505075267e-47
        ) THEN 1.0947644252537633e-47
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.1895288505075267e-47 AND 4.3790577010150533e-47
        ) THEN 2.1895288505075267e-47
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.3790577010150533e-47 AND 8.758115402030107e-47
        ) THEN 4.3790577010150533e-47
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.758115402030107e-47 AND 1.7516230804060213e-46
        ) THEN 8.758115402030107e-47
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.7516230804060213e-46 AND 3.503246160812043e-46
        ) THEN 1.7516230804060213e-46
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.503246160812043e-46 AND 7.006492321624085e-46
        ) THEN 3.503246160812043e-46
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.006492321624085e-46 AND 1.401298464324817e-45
        ) THEN 7.006492321624085e-46
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.401298464324817e-45 AND 2.802596928649634e-45
        ) THEN 1.401298464324817e-45
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.802596928649634e-45 AND 5.605193857299268e-45
        ) THEN 2.802596928649634e-45
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.605193857299268e-45 AND 1.1210387714598537e-44
        ) THEN 5.605193857299268e-45
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.1210387714598537e-44 AND 2.2420775429197073e-44
        ) THEN 1.1210387714598537e-44
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.2420775429197073e-44 AND 4.484155085839415e-44
        ) THEN 2.2420775429197073e-44
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.484155085839415e-44 AND 8.96831017167883e-44
        ) THEN 4.484155085839415e-44
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.96831017167883e-44 AND 1.793662034335766e-43
        ) THEN 8.96831017167883e-44
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.793662034335766e-43 AND 3.587324068671532e-43
        ) THEN 1.793662034335766e-43
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.587324068671532e-43 AND 7.174648137343064e-43
        ) THEN 3.587324068671532e-43
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.174648137343064e-43 AND 1.4349296274686127e-42
        ) THEN 7.174648137343064e-43
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.4349296274686127e-42 AND 2.8698592549372254e-42
        ) THEN 1.4349296274686127e-42
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.8698592549372254e-42 AND 5.739718509874451e-42
        ) THEN 2.8698592549372254e-42
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.739718509874451e-42 AND 1.1479437019748901e-41
        ) THEN 5.739718509874451e-42
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.1479437019748901e-41 AND 2.2958874039497803e-41
        ) THEN 1.1479437019748901e-41
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.2958874039497803e-41 AND 4.591774807899561e-41
        ) THEN 2.2958874039497803e-41
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.591774807899561e-41 AND 9.183549615799121e-41
        ) THEN 4.591774807899561e-41
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.183549615799121e-41 AND 1.8367099231598242e-40
        ) THEN 9.183549615799121e-41
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.8367099231598242e-40 AND 3.6734198463196485e-40
        ) THEN 1.8367099231598242e-40
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.6734198463196485e-40 AND 7.346839692639297e-40
        ) THEN 3.6734198463196485e-40
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.346839692639297e-40 AND 1.4693679385278594e-39
        ) THEN 7.346839692639297e-40
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.4693679385278594e-39 AND 2.938735877055719e-39
        ) THEN 1.4693679385278594e-39
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.938735877055719e-39 AND 5.877471754111438e-39
        ) THEN 2.938735877055719e-39
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.877471754111438e-39 AND 1.1754943508222875e-38
        ) THEN 5.877471754111438e-39
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.1754943508222875e-38 AND 2.350988701644575e-38
        ) THEN 1.1754943508222875e-38
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.350988701644575e-38 AND 4.70197740328915e-38
        ) THEN 2.350988701644575e-38
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.70197740328915e-38 AND 9.4039548065783e-38
        ) THEN 4.70197740328915e-38
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.4039548065783e-38 AND 1.88079096131566e-37
        ) THEN 9.4039548065783e-38
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.88079096131566e-37 AND 3.76158192263132e-37
        ) THEN 1.88079096131566e-37
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.76158192263132e-37 AND 7.52316384526264e-37
        ) THEN 3.76158192263132e-37
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.52316384526264e-37 AND 1.504632769052528e-36
        ) THEN 7.52316384526264e-37
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.504632769052528e-36 AND 3.009265538105056e-36
        ) THEN 1.504632769052528e-36
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.009265538105056e-36 AND 6.018531076210112e-36
        ) THEN 3.009265538105056e-36
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.018531076210112e-36 AND 1.2037062152420224e-35
        ) THEN 6.018531076210112e-36
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.2037062152420224e-35 AND 2.407412430484045e-35
        ) THEN 1.2037062152420224e-35
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.407412430484045e-35 AND 4.81482486096809e-35
        ) THEN 2.407412430484045e-35
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.81482486096809e-35 AND 9.62964972193618e-35
        ) THEN 4.81482486096809e-35
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.62964972193618e-35 AND 1.925929944387236e-34
        ) THEN 9.62964972193618e-35
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.925929944387236e-34 AND 3.851859888774472e-34
        ) THEN 1.925929944387236e-34
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.851859888774472e-34 AND 7.703719777548943e-34
        ) THEN 3.851859888774472e-34
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.703719777548943e-34 AND 1.5407439555097887e-33
        ) THEN 7.703719777548943e-34
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.5407439555097887e-33 AND 3.0814879110195774e-33
        ) THEN 1.5407439555097887e-33
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.0814879110195774e-33 AND 6.162975822039155e-33
        ) THEN 3.0814879110195774e-33
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.162975822039155e-33 AND 1.232595164407831e-32
        ) THEN 6.162975822039155e-33
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.232595164407831e-32 AND 2.465190328815662e-32
        ) THEN 1.232595164407831e-32
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.465190328815662e-32 AND 4.930380657631324e-32
        ) THEN 2.465190328815662e-32
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.930380657631324e-32 AND 9.860761315262648e-32
        ) THEN 4.930380657631324e-32
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.860761315262648e-32 AND 1.9721522630525295e-31
        ) THEN 9.860761315262648e-32
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.9721522630525295e-31 AND 3.944304526105059e-31
        ) THEN 1.9721522630525295e-31
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.944304526105059e-31 AND 7.888609052210118e-31
        ) THEN 3.944304526105059e-31
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.888609052210118e-31 AND 1.5777218104420236e-30
        ) THEN 7.888609052210118e-31
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.5777218104420236e-30 AND 3.1554436208840472e-30
        ) THEN 1.5777218104420236e-30
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.1554436208840472e-30 AND 6.310887241768095e-30
        ) THEN 3.1554436208840472e-30
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.310887241768095e-30 AND 1.262177448353619e-29
        ) THEN 6.310887241768095e-30
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.262177448353619e-29 AND 2.524354896707238e-29
        ) THEN 1.262177448353619e-29
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.524354896707238e-29 AND 5.048709793414476e-29
        ) THEN 2.524354896707238e-29
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.048709793414476e-29 AND 1.0097419586828951e-28
        ) THEN 5.048709793414476e-29
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.0097419586828951e-28 AND 2.0194839173657902e-28
        ) THEN 1.0097419586828951e-28
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.0194839173657902e-28 AND 4.0389678347315804e-28
        ) THEN 2.0194839173657902e-28
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.0389678347315804e-28 AND 8.077935669463161e-28
        ) THEN 4.0389678347315804e-28
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.077935669463161e-28 AND 1.6155871338926322e-27
        ) THEN 8.077935669463161e-28
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.6155871338926322e-27 AND 3.2311742677852644e-27
        ) THEN 1.6155871338926322e-27
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.2311742677852644e-27 AND 6.462348535570529e-27
        ) THEN 3.2311742677852644e-27
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.462348535570529e-27 AND 1.2924697071141057e-26
        ) THEN 6.462348535570529e-27
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.2924697071141057e-26 AND 2.5849394142282115e-26
        ) THEN 1.2924697071141057e-26
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.5849394142282115e-26 AND 5.169878828456423e-26
        ) THEN 2.5849394142282115e-26
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.169878828456423e-26 AND 1.0339757656912846e-25
        ) THEN 5.169878828456423e-26
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.0339757656912846e-25 AND 2.0679515313825692e-25
        ) THEN 1.0339757656912846e-25
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.0679515313825692e-25 AND 4.1359030627651384e-25
        ) THEN 2.0679515313825692e-25
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.1359030627651384e-25 AND 8.271806125530277e-25
        ) THEN 4.1359030627651384e-25
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.271806125530277e-25 AND 1.6543612251060553e-24
        ) THEN 8.271806125530277e-25
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.6543612251060553e-24 AND 3.308722450212111e-24
        ) THEN 1.6543612251060553e-24
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.308722450212111e-24 AND 6.617444900424222e-24
        ) THEN 3.308722450212111e-24
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.617444900424222e-24 AND 1.3234889800848443e-23
        ) THEN 6.617444900424222e-24
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.3234889800848443e-23 AND 2.6469779601696886e-23
        ) THEN 1.3234889800848443e-23
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.6469779601696886e-23 AND 5.293955920339377e-23
        ) THEN 2.6469779601696886e-23
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.293955920339377e-23 AND 1.0587911840678754e-22
        ) THEN 5.293955920339377e-23
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.0587911840678754e-22 AND 2.117582368135751e-22
        ) THEN 1.0587911840678754e-22
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.117582368135751e-22 AND 4.235164736271502e-22
        ) THEN 2.117582368135751e-22
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.235164736271502e-22 AND 8.470329472543003e-22
        ) THEN 4.235164736271502e-22
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.470329472543003e-22 AND 1.6940658945086007e-21
        ) THEN 8.470329472543003e-22
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.6940658945086007e-21 AND 3.3881317890172014e-21
        ) THEN 1.6940658945086007e-21
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.3881317890172014e-21 AND 6.776263578034403e-21
        ) THEN 3.3881317890172014e-21
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.776263578034403e-21 AND 1.3552527156068805e-20
        ) THEN 6.776263578034403e-21
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.3552527156068805e-20 AND 2.710505431213761e-20
        ) THEN 1.3552527156068805e-20
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.710505431213761e-20 AND 5.421010862427522e-20
        ) THEN 2.710505431213761e-20
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.421010862427522e-20 AND 1.0842021724855044e-19
        ) THEN 5.421010862427522e-20
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.0842021724855044e-19 AND 2.168404344971009e-19
        ) THEN 1.0842021724855044e-19
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.168404344971009e-19 AND 4.336808689942018e-19
        ) THEN 2.168404344971009e-19
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.336808689942018e-19 AND 8.673617379884035e-19
        ) THEN 4.336808689942018e-19
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.673617379884035e-19 AND 1.734723475976807e-18
        ) THEN 8.673617379884035e-19
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.734723475976807e-18 AND 3.469446951953614e-18
        ) THEN 1.734723475976807e-18
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.469446951953614e-18 AND 6.938893903907228e-18
        ) THEN 3.469446951953614e-18
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.938893903907228e-18 AND 1.3877787807814457e-17
        ) THEN 6.938893903907228e-18
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.3877787807814457e-17 AND 2.7755575615628914e-17
        ) THEN 1.3877787807814457e-17
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.7755575615628914e-17 AND 5.551115123125783e-17
        ) THEN 2.7755575615628914e-17
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.551115123125783e-17 AND 1.1102230246251565e-16
        ) THEN 5.551115123125783e-17
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.1102230246251565e-16 AND 2.220446049250313e-16
        ) THEN 1.1102230246251565e-16
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.220446049250313e-16 AND 4.440892098500626e-16
        ) THEN 2.220446049250313e-16
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.440892098500626e-16 AND 8.881784197001252e-16
        ) THEN 4.440892098500626e-16
        WHEN (
          anon_6."Birthdate_1" BETWEEN 8.881784197001252e-16 AND 1.7763568394002505e-15
        ) THEN 8.881784197001252e-16
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.7763568394002505e-15 AND 3.552713678800501e-15
        ) THEN 1.7763568394002505e-15
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.552713678800501e-15 AND 7.105427357601002e-15
        ) THEN 3.552713678800501e-15
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.105427357601002e-15 AND 1.4210854715202004e-14
        ) THEN 7.105427357601002e-15
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.4210854715202004e-14 AND 2.842170943040401e-14
        ) THEN 1.4210854715202004e-14
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.842170943040401e-14 AND 5.684341886080802e-14
        ) THEN 2.842170943040401e-14
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.684341886080802e-14 AND 1.1368683772161603e-13
        ) THEN 5.684341886080802e-14
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.1368683772161603e-13 AND 2.2737367544323206e-13
        ) THEN 1.1368683772161603e-13
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.2737367544323206e-13 AND 4.547473508864641e-13
        ) THEN 2.2737367544323206e-13
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.547473508864641e-13 AND 9.094947017729282e-13
        ) THEN 4.547473508864641e-13
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.094947017729282e-13 AND 1.8189894035458565e-12
        ) THEN 9.094947017729282e-13
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.8189894035458565e-12 AND 3.637978807091713e-12
        ) THEN 1.8189894035458565e-12
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.637978807091713e-12 AND 7.275957614183426e-12
        ) THEN 3.637978807091713e-12
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.275957614183426e-12 AND 1.4551915228366852e-11
        ) THEN 7.275957614183426e-12
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.4551915228366852e-11 AND 2.9103830456733704e-11
        ) THEN 1.4551915228366852e-11
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.9103830456733704e-11 AND 5.820766091346741e-11
        ) THEN 2.9103830456733704e-11
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.820766091346741e-11 AND 1.1641532182693481e-10
        ) THEN 5.820766091346741e-11
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.1641532182693481e-10 AND 2.3283064365386963e-10
        ) THEN 1.1641532182693481e-10
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.3283064365386963e-10 AND 4.656612873077393e-10
        ) THEN 2.3283064365386963e-10
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.656612873077393e-10 AND 9.313225746154785e-10
        ) THEN 4.656612873077393e-10
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.313225746154785e-10 AND 1.862645149230957e-09
        ) THEN 9.313225746154785e-10
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.862645149230957e-09 AND 3.725290298461914e-09
        ) THEN 1.862645149230957e-09
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.725290298461914e-09 AND 7.450580596923828e-09
        ) THEN 3.725290298461914e-09
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.450580596923828e-09 AND 1.4901161193847656e-08
        ) THEN 7.450580596923828e-09
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.4901161193847656e-08 AND 2.9802322387695312e-08
        ) THEN 1.4901161193847656e-08
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.9802322387695312e-08 AND 5.960464477539063e-08
        ) THEN 2.9802322387695312e-08
        WHEN (
          anon_6."Birthdate_1" BETWEEN 5.960464477539063e-08 AND 1.1920928955078125e-07
        ) THEN 5.960464477539063e-08
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.1920928955078125e-07 AND 2.384185791015625e-07
        ) THEN 1.1920928955078125e-07
        WHEN (
          anon_6."Birthdate_1" BETWEEN 2.384185791015625e-07 AND 4.76837158203125e-07
        ) THEN 2.384185791015625e-07
        WHEN (
          anon_6."Birthdate_1" BETWEEN 4.76837158203125e-07 AND 9.5367431640625e-07
        ) THEN 4.76837158203125e-07
        WHEN (
          anon_6."Birthdate_1" BETWEEN 9.5367431640625e-07 AND 1.9073486328125e-06
        ) THEN 9.5367431640625e-07
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.9073486328125e-06 AND 3.814697265625e-06
        ) THEN 1.9073486328125e-06
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.814697265625e-06 AND 7.62939453125e-06
        ) THEN 3.814697265625e-06
        WHEN (
          anon_6."Birthdate_1" BETWEEN 7.62939453125e-06 AND 1.52587890625e-05
        ) THEN 7.62939453125e-06
        WHEN (
          anon_6."Birthdate_1" BETWEEN 1.52587890625e-05 AND 3.0517578125e-05
        ) THEN 1.52587890625e-05
        WHEN (
          anon_6."Birthdate_1" BETWEEN 3.0517578125e-05 AND 6.103515625e-05
        ) THEN 3.0517578125e-05
        WHEN (
          anon_6."Birthdate_1" BETWEEN 6.103515625e-05 AND 0.0001220703125
        ) THEN 6.103515625e-05
        WHEN (
          anon_6."Birthdate_1" BETWEEN 0.0001220703125 AND 0.000244140625
        ) THEN 0.0001220703125
        WHEN (
          anon_6."Birthdate_1" BETWEEN 0.000244140625 AND 0.00048828125
        ) THEN 0.000244140625
        WHEN (
          anon_6."Birthdate_1" BETWEEN 0.00048828125 AND 0.0009765625
        ) THEN 0.00048828125
        WHEN (
          anon_6."Birthdate_1" BETWEEN 0.0009765625 AND 0.001953125
        ) THEN 0.0009765625
        WHEN (
          anon_6."Birthdate_1" BETWEEN 0.001953125 AND 0.00390625
        ) THEN 0.001953125
        WHEN (
          anon_6."Birthdate_1" BETWEEN 0.00390625 AND 0.0078125
        ) THEN 0.00390625
        WHEN (
          anon_6."Birthdate_1" BETWEEN 0.0078125 AND 0.015625
        ) THEN 0.0078125
        WHEN (anon_6."Birthdate_1" BETWEEN 0.015625 AND 0.03125) THEN 0.015625
        WHEN (anon_6."Birthdate_1" BETWEEN 0.03125 AND 0.0625) THEN 0.03125
        WHEN (anon_6."Birthdate_1" BETWEEN 0.0625 AND 0.125) THEN 0.0625
        WHEN (anon_6."Birthdate_1" BETWEEN 0.125 AND 0.25) THEN 0.125
        WHEN (anon_6."Birthdate_1" BETWEEN 0.25 AND 0.5) THEN 0.25
        WHEN (anon_6."Birthdate_1" BETWEEN 0.5 AND 1) THEN 0.5
        WHEN (anon_6."Birthdate_1" BETWEEN 1 AND 2) THEN 1
        WHEN (anon_6."Birthdate_1" BETWEEN 2 AND 4) THEN 2
        WHEN (anon_6."Birthdate_1" BETWEEN 4 AND 8) THEN 4
        WHEN (anon_6."Birthdate_1" BETWEEN 8 AND 16) THEN 8
        WHEN (anon_6."Birthdate_1" BETWEEN 16 AND 32) THEN 16
        WHEN (anon_6."Birthdate_1" BETWEEN 32 AND 64) THEN 32
        WHEN (anon_6."Birthdate_1" BETWEEN 64 AND 128) THEN 64
        WHEN (anon_6."Birthdate_1" BETWEEN 128 AND 256) THEN 128
        WHEN (anon_6."Birthdate_1" BETWEEN 256 AND 512) THEN 256
        WHEN (anon_6."Birthdate_1" BETWEEN 512 AND 1024) THEN 512
        WHEN (anon_6."Birthdate_1" BETWEEN 1024 AND 2048) THEN 1024
        WHEN (anon_6."Birthdate_1" BETWEEN 2048 AND 4096) THEN 2048
        WHEN (anon_6."Birthdate_1" BETWEEN 4096 AND 8192) THEN 4096
        WHEN (anon_6."Birthdate_1" BETWEEN 8192 AND 13186) THEN 8192
      END AS binned
    FROM
      (
        SELECT
          anon_7."Birthdate_1" AS "Birthdate_1"
        FROM
          anon_7
      ) AS anon_6
  ),
  anon_4 AS (
    SELECT
      anon_5.binned AS binned
    FROM
      anon_5
  ),
  anon_3 AS (
    SELECT
      anon_4.binned AS binned,
      count(anon_4.binned) - 5.608151912689209 * log(1 - 2 * abs(random() - 0.5)) * sign(random() - 0.5) AS count_
    FROM
      anon_4
    GROUP BY
      anon_4.binned
  ),
  anon_2 AS (
    SELECT
      min(anon_3.binned) AS min_value
    FROM
      anon_3
    WHERE
      anon_3.count_ >= 159.05246301569716
  ),
  anon_1 AS (
    SELECT
      greatest(anon_2.min_value, -13423) AS min_value
    FROM
      anon_2
  )
SELECT
  '"private"."demographics_demo"."Birthdate"',
  'min',
  anon_1.min_value
FROM
  anon_1
            "#,
            // "SELECT CASE WHEN (CAST(basket_id AS FLOAT) BETWEEN 31198459904 AND 31950110720) THEN 31198459904 END AS binned FROM retail_transactions"
            // r#"
            // WITH anon_7 AS 
            // (SELECT "retail_transactions"."basket_id" AS basket_id_1 
            // FROM "retail_transactions"), 
            // anon_5 AS 
            // (SELECT CASE WHEN (anon_6.basket_id_1 BETWEEN 31198459904 AND 31950110720) THEN 31198459904 END AS binned 
            // FROM (SELECT anon_7.basket_id_1 AS basket_id_1 
            // FROM anon_7) AS anon_6), 
            // anon_4 AS 
            // (SELECT anon_5.binned AS binned 
            // FROM anon_5), 
            // anon_3 AS 
            // (SELECT anon_4.binned AS binned, count(anon_4.binned) - 196.21693235849753 * log(1 - 2 * abs(random() - 0.5)) * sign(random() - 0.5) AS count_ 
            // FROM anon_4 GROUP BY anon_4.binned), 
            // anon_2 AS 
            // (SELECT min(anon_3.binned) AS min_value 
            // FROM anon_3 
            // WHERE anon_3.count_ >= 4066.2556565246705), 
            // anon_1 AS 
            // (SELECT greatest(anon_2.min_value, 31198459904) AS min_value 
            // FROM anon_2)
            // SELECT '"retail_transactions"."basket_id"', 'min', anon_1.min_value 
            // FROM anon_1
            // "#,
            //r#"WITH "join_2fwl" ("field_ap9y", "field_fx4w", "field_gn0a", "field_kmqc", "field_dpbc", "field_hiol", "field_ut0n", "field_qcec", "field_z5g4", "field_2618", "field_n60z", "field_nx5j", "field_x7is", "field_t3fd", "field_15_v", "field_yrwo", "field_5yhm", "field_ggew", "field_ls8b") AS (SELECT * FROM "retail_transactions" AS "_LEFT_" JOIN "retail_demographics" AS "_RIGHT_" ON ("_LEFT_"."household_id") = ("_RIGHT_"."household_id")), "map_veta" ("field_ap9y", "store_id", "basket_id", "product_id", "quantity", "sales_value", "retail_disc", "coupon_disc", "coupon_match_disc", "week", "transaction_timestamp", "field_nx5j", "age", "income", "home_ownership", "marital_status", "household_size", "household_comp", "kids_count") AS (SELECT "field_ap9y" AS "field_ap9y", "field_fx4w" AS "store_id", "field_gn0a" AS "basket_id", "field_kmqc" AS "product_id", "field_dpbc" AS "quantity", "field_hiol" AS "sales_value", "field_ut0n" AS "retail_disc", "field_qcec" AS "coupon_disc", "field_z5g4" AS "coupon_match_disc", "field_2618" AS "week", "field_n60z" AS "transaction_timestamp", "field_nx5j" AS "field_nx5j", "field_x7is" AS "age", "field_t3fd" AS "income", "field_15_v" AS "home_ownership", "field_yrwo" AS "marital_status", "field_5yhm" AS "household_size", "field_ggew" AS "household_comp", "field_ls8b" AS "kids_count" FROM "join_2fwl"), "map_0q_a" ("household_id", "store_id", "basket_id", "product_id", "quantity", "sales_value", "retail_disc", "coupon_disc", "coupon_match_disc", "week", "transaction_timestamp", "household_id_1") AS (SELECT "field_ap9y" AS "household_id", "store_id" AS "store_id", "basket_id" AS "basket_id", "product_id" AS "product_id", "quantity" AS "quantity", "sales_value" AS "sales_value", "retail_disc" AS "retail_disc", "coupon_disc" AS "coupon_disc", "coupon_match_disc" AS "coupon_match_disc", "week" AS "week", "transaction_timestamp" AS "transaction_timestamp", "field_nx5j" AS "household_id_1" FROM "map_veta"), "map_ujyl" ("household_id", "store_id", "basket_id", "product_id", "quantity", "sales_value", "retail_disc", "coupon_disc", "coupon_match_disc", "week", "transaction_timestamp", "sarus_is_public", "sarus_privacy_unit", "sarus_weights") AS (SELECT "household_id" AS "household_id", "store_id" AS "store_id", "basket_id" AS "basket_id", "product_id" AS "product_id", "quantity" AS "quantity", "sales_value" AS "sales_value", "retail_disc" AS "retail_disc", "coupon_disc" AS "coupon_disc", "coupon_match_disc" AS "coupon_match_disc", "week" AS "week", "transaction_timestamp" AS "transaction_timestamp", 0 AS "sarus_is_public", CASE WHEN "household_id" IS NULL THEN NULL ELSE MD5(CONCAT(MD5(CONCAT('4e71a9747b840a5fc1476b8a63ef4530f631e3a8c861565b92a7338b94f236bc', MD5(CAST("household_id" AS TEXT)))))) END AS "sarus_privacy_unit", 1 AS "sarus_weights" FROM "map_0q_a"), "map_ri9m" ("store_id_1") AS (SELECT "store_id" AS "store_id_1" FROM "map_ujyl"), "map_tkb3" ("store_id_1") AS (SELECT "store_id_1" AS "store_id_1" FROM "map_ri9m"), "map_jydp" ("binned") AS (SELECT CASE WHEN (("store_id_1") >= (27)) AND (("store_id_1") <= (32)) THEN 27 WHEN (("store_id_1") >= (32)) AND (("store_id_1") <= (64)) THEN 32 WHEN (("store_id_1") >= (64)) AND (("store_id_1") <= (128)) THEN 64 WHEN (("store_id_1") >= (128)) AND (("store_id_1") <= (256)) THEN 128 WHEN (("store_id_1") >= (256)) AND (("store_id_1") <= (512)) THEN 256 WHEN (("store_id_1") >= (512)) AND (("store_id_1") <= (1024)) THEN 512 WHEN (("store_id_1") >= (1024)) AND (("store_id_1") <= (2048)) THEN 1024 WHEN (("store_id_1") >= (2048)) AND (("store_id_1") <= (4096)) THEN 2048 WHEN (("store_id_1") >= (4096)) AND (("store_id_1") <= (8192)) THEN 4096 WHEN (("store_id_1") >= (8192)) AND (("store_id_1") <= (16384)) THEN 8192 WHEN (("store_id_1") >= (16384)) AND (("store_id_1") <= (32768)) THEN 16384 WHEN (("store_id_1") >= (32768)) AND (("store_id_1") <= (33923)) THEN 32768 ELSE NULL END AS "binned" FROM "map_tkb3"), "map_t55y" ("binned") AS (SELECT "binned" AS "binned" FROM "map_jydp"), "map_aji4" ("field_epaa") AS (SELECT "binned" AS "field_epaa" FROM "map_t55y"), "map_bruo" ("field_s6a9") AS (SELECT "field_epaa" AS "field_s6a9" FROM "map_aji4"), "reduce_ccnf" ("field_s6a9", "field_kw5m") AS (SELECT "field_s6a9" AS "field_s6a9", COUNT("field_s6a9") AS "field_kw5m" FROM "map_bruo" GROUP BY "field_s6a9"), "map_9rjv" ("field_epaa", "field_15zm") AS (SELECT "field_s6a9" AS "field_epaa", "field_kw5m" AS "field_15zm" FROM "reduce_ccnf"), "map_1k0m" ("binned", "count_") AS (SELECT "field_epaa" AS "binned", ("field_15zm") - (((196.21693235849753) * (LOG((1) - ((2) * (ABS((RANDOM()) - (0.5))))))) * (SIGN((RANDOM()) - (0.5)))) AS "count_" FROM "map_9rjv"), "map_58ls" ("field_epaa") AS (SELECT "binned" AS "field_epaa" FROM "map_1k0m" WHERE ("count_") >= (4553.836394758849)), "map_mfax" ("field_s6a9") AS (SELECT "field_epaa" AS "field_s6a9" FROM "map_58ls"), "reduce_8uaa" ("field_ojnh") AS (SELECT MAX("field_s6a9") AS "field_ojnh" FROM "map_mfax"), "map_up98" ("max_value") AS (SELECT (2) * ("field_ojnh") AS "max_value" FROM "reduce_8uaa"), "map_pndb" ("max_value") AS (SELECT LEAST("max_value", 33923) AS "max_value" FROM "map_up98"), "map_naat" ("field_k6ag", "field_r_rl", "max_value") AS (SELECT '"retail_transactions"."store_id"' AS "field_k6ag", 'max' AS "field_r_rl", "max_value" AS "max_value" FROM "map_pndb"), "map_j5bj" ("field_k6ag", "field_r_rl", "max_value") AS (SELECT "field_k6ag" AS "field_k6ag", "field_r_rl" AS "field_r_rl", "max_value" AS "max_value" FROM "map_naat") SELECT * FROM "map_j5bj" "#,
            // "SELECT COUNT(DISTINCT household_id) AS unique_customers FROM retail_transactions",
            // "SELECT * FROM retail_transactions t1 INNER JOIN retail_transactions t2 ON t1.product_id = t2.product_id",
            // "SELECT COUNT(*) FROM retail_transactions t INNER JOIN retail_products p ON t.product_id = p.product_id",
            // "SELECT * FROM retail_transactions t INNER JOIN retail_products p ON t.product_id = p.product_id",
            // "SELECT department, AVG(sales_value) AS average_sales FROM retail_transactions INNER JOIN retail_products ON retail_transactions.product_id = retail_products.product_id GROUP BY department",
            // "SELECT * FROM retail_transactions INNER JOIN retail_products ON retail_transactions.product_id = retail_products.product_id",
            // "WITH ranked_products AS (SELECT product_id, COUNT(*) AS my_count FROM retail_transactions GROUP BY product_id) SELECT product_id FROM ranked_products ORDER BY my_count",
            // //"SELECT t.product_id, p.product_category, COUNT(*) AS purchase_count FROM retail_transactions t INNER JOIN retail_products p ON t.product_id = p.product_id WHERE t.transaction_timestamp < CAST('2023-02-01' AS date) GROUP BY t.product_id, p.product_category", // cast date from string does not work in where
            // "SELECT t.product_id, p.product_category, COUNT(*) AS purchase_count FROM retail_transactions t INNER JOIN retail_products p ON t.product_id = p.product_id WHERE t.transaction_timestamp > '2023-01-01' AND t.transaction_timestamp < '2023-02-01' GROUP BY t.product_id, p.product_category",
            // "SELECT DISTINCT age, income FROM retail_demographics",
            // "SELECT age, income FROM retail_demographics GROUP BY age, income",
            // "SELECT quantity, AVG(sales_value) FROM retail_demographics AS d JOIN retail_transactions AS t ON d.household_id = t.household_id WHERE quantity > 0 AND sales_value > 0 AND sales_value < 100 GROUP BY quantity",
            // "WITH stats_stores AS (SELECT store_id, SUM(sales_value) AS sum_sales_value, AVG(retail_disc) FROM retail_transactions WHERE sales_value > 0 AND sales_value < 100 AND retail_disc > 0 AND retail_disc < 10 GROUP BY store_id) SELECT * FROM stats_stores WHERE sum_sales_value != 1",
            // "SELECT p.product_id, p.brand, COUNT(*) FROM retail_products p INNER JOIN retail_transactions t ON p.product_id = t.product_id GROUP BY p.product_id, p.brand",
            // "SELECT t.household_id, store_id, AVG(sales_value) FROM retail_demographics AS d JOIN retail_transactions AS t ON d.household_id = t.household_id WHERE sales_value > 0 AND sales_value < 100 GROUP BY t.household_id, store_id",
            // "SELECT * FROM retail_transactions AS t INNER JOIN retail_demographics p ON t.household_id = p.household_id",
        ];
        for query_str in queries {
            println!("\n{query_str}");
            let query = parse(query_str).unwrap();
            let relation = Relation::try_from(query.with(&relations)).unwrap();
            relation.display_dot().unwrap();
            let qq = ast::Query::from(&relation);
            println!("QUERY: \n{}\n",qq);
            let dp_relation = relation
                .rewrite_with_differential_privacy(
                    &relations,
                    synthetic_data.clone(),
                    privacy_unit.clone(),
                    dp_parameters.clone(),
                )
                .unwrap();
            dp_relation.relation().display_dot().unwrap();
        }
    }

    #[test]
    fn test_retail_with_pu_and_weight_column_in_tables() {
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
                .collect::<Schema>(),
            )
            .size(1000)
            .build();
        let retail_demographics: Relation = Relation::table()
            .name("retail_demographics")
            .schema(
                vec![
                    (
                        "household_id",
                        DataType::integer(),
                        Some(Constraint::Unique),
                    ),
                    ("age", DataType::integer(), None),
                    ("income", DataType::float(), None),
                    ("home_ownership", DataType::text(), None),
                    ("marital_status", DataType::text(), None),
                    ("household_size", DataType::integer(), None),
                    ("household_comp", DataType::text(), None),
                    ("kids_count", DataType::integer(), None),
                    (
                        "my_privacy_unit",
                        DataType::optional(DataType::id()),
                        Some(Constraint::Unique),
                    ),
                    ("my_weight", DataType::float_min(0.0), None),
                ]
                .into_iter()
                .collect::<Schema>(),
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
                    ("my_privacy_unit", DataType::optional(DataType::id()), None),
                    ("my_weight", DataType::float_min(0.0), None),
                ]
                .into_iter()
                .collect::<Schema>(),
            )
            .size(10000)
            .build();
        let relations: Hierarchy<Arc<Relation>> =
            vec![retail_transactions, retail_demographics, retail_products]
                .iter()
                .map(|t| (Identifier::from(t.name()), Arc::new(t.clone().into())))
                .collect();
        let synthetic_data = Some(SyntheticData::new(Hierarchy::from([
            (
                vec!["retail_transactions"],
                Identifier::from("synthetic_retail_transactions"),
            ),
            (
                vec!["retail_demographics"],
                Identifier::from("synthetic_retail_demographics"),
            ),
            (
                vec!["retail_products"],
                Identifier::from("synthetic_retail_products"),
            ),
        ])));

        let privacy_unit_paths = vec![
            (
                "retail_demographics",
                vec![],
                "my_privacy_unit",
                "my_weight",
            ),
            (
                "retail_transactions",
                vec![("household_id", "retail_demographics", "household_id")],
                "my_privacy_unit",
                "my_weight",
            ),
        ];

        let privacy_unit = PrivacyUnit::from((privacy_unit_paths, false));

        let dp_parameters = DpParameters::from_epsilon_delta(1., 1e-3);

        let queries = [
            "SELECT COUNT(DISTINCT household_id) AS unique_customers FROM retail_transactions",
            "SELECT * FROM retail_transactions t1 INNER JOIN retail_transactions t2 ON t1.product_id = t2.product_id",
            "SELECT COUNT(*) FROM retail_transactions t INNER JOIN retail_products p ON t.product_id = p.product_id",
            "SELECT * FROM retail_transactions t INNER JOIN retail_products p ON t.product_id = p.product_id",
            "SELECT department, AVG(sales_value) AS average_sales FROM retail_transactions INNER JOIN retail_products ON retail_transactions.product_id = retail_products.product_id GROUP BY department",
            "SELECT * FROM retail_transactions INNER JOIN retail_products ON retail_transactions.product_id = retail_products.product_id",
            "WITH ranked_products AS (SELECT product_id, COUNT(*) AS my_count FROM retail_transactions GROUP BY product_id) SELECT product_id FROM ranked_products ORDER BY my_count",
            "SELECT t.product_id, p.product_category, COUNT(*) AS purchase_count FROM retail_transactions t INNER JOIN retail_products p ON t.product_id = p.product_id WHERE t.transaction_timestamp > '2023-01-01' AND t.transaction_timestamp < '2023-02-01' GROUP BY t.product_id, p.product_category",
            "SELECT DISTINCT age, income FROM retail_demographics",
            "SELECT age, income FROM retail_demographics GROUP BY age, income",
            "SELECT quantity, AVG(sales_value) FROM retail_demographics AS d JOIN retail_transactions AS t ON d.household_id = t.household_id WHERE quantity > 0 AND sales_value > 0 AND sales_value < 100 GROUP BY quantity",
            "WITH stats_stores AS (SELECT store_id, SUM(sales_value) AS sum_sales_value, AVG(retail_disc) FROM retail_transactions WHERE sales_value > 0 AND sales_value < 100 AND retail_disc > 0 AND retail_disc < 10 GROUP BY store_id) SELECT * FROM stats_stores WHERE sum_sales_value != 1",
            "SELECT p.product_id, p.brand, COUNT(*) FROM retail_products p INNER JOIN retail_transactions t ON p.product_id = t.product_id GROUP BY p.product_id, p.brand",
            "SELECT t.household_id, store_id, AVG(sales_value) FROM retail_demographics AS d JOIN retail_transactions AS t ON d.household_id = t.household_id WHERE sales_value > 0 AND sales_value < 100 GROUP BY t.household_id, store_id",
            "SELECT * FROM retail_transactions AS t INNER JOIN retail_products p ON t.product_id = p.product_id",
            "SELECT * FROM retail_transactions AS t INNER JOIN retail_demographics p ON t.household_id = p.household_id",
        ];
        for query_str in queries {
            println!("\n{query_str}");
            let query = parse(query_str).unwrap();
            let relation = Relation::try_from(query.with(&relations)).unwrap();
            relation.display_dot().unwrap();
            let dp_relation = relation
                .rewrite_with_differential_privacy(
                    &relations,
                    synthetic_data.clone(),
                    privacy_unit.clone(),
                    dp_parameters.clone(),
                )
                .unwrap();
            dp_relation.relation().display_dot().unwrap();
        }
    }

    #[test]
    fn test_retail_with_pu_and_weight_column_in_tables_rewrite_as_pup() {
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
                    ("my_privacy_unit", DataType::optional(DataType::id())),
                    ("my_weight", DataType::float_min(0.0)),
                ]
                .into_iter()
                .collect::<Schema>(),
            )
            .size(1000)
            .build();
        let retail_demographics: Relation = Relation::table()
            .name("retail_demographics")
            .schema(
                vec![
                    (
                        "household_id",
                        DataType::integer(),
                        Some(Constraint::Unique),
                    ),
                    ("age", DataType::integer(), None),
                    ("income", DataType::float(), None),
                    ("home_ownership", DataType::text(), None),
                    ("marital_status", DataType::text(), None),
                    ("household_size", DataType::integer(), None),
                    ("household_comp", DataType::text(), None),
                    ("kids_count", DataType::integer(), None),
                    (
                        "my_privacy_unit",
                        DataType::optional(DataType::id()),
                        Some(Constraint::Unique),
                    ),
                    ("my_weight", DataType::float_min(0.0), None),
                ]
                .into_iter()
                .collect::<Schema>(),
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
                .collect::<Schema>(),
            )
            .size(10000)
            .build();
        let relations: Hierarchy<Arc<Relation>> =
            vec![retail_transactions, retail_demographics, retail_products]
                .iter()
                .map(|t| (Identifier::from(t.name()), Arc::new(t.clone().into())))
                .collect();
        let synthetic_data = Some(SyntheticData::new(Hierarchy::from([
            (
                vec!["retail_transactions"],
                Identifier::from("synthetic_retail_transactions"),
            ),
            (
                vec!["retail_demographics"],
                Identifier::from("synthetic_retail_demographics"),
            ),
            (
                vec!["retail_products"],
                Identifier::from("synthetic_retail_products"),
            ),
        ])));

        let privacy_unit_paths = vec![
            (
                "retail_demographics",
                vec![],
                "my_privacy_unit",
                "my_weight",
            ),
            (
                "retail_transactions",
                vec![],
                "my_privacy_unit",
                "my_weight",
            ),
        ];

        let privacy_unit = PrivacyUnit::from((privacy_unit_paths, false));

        let dp_parameters = DpParameters::from_epsilon_delta(1., 1e-3);

        let hard_pup_queries = [
            "SELECT household_id, COUNT(*) FROM retail_demographics GROUP BY household_id",
            "SELECT d.household_id FROM retail_demographics as d JOIN retail_transactions AS t ON (d.household_id=t.household_id)",
            "SELECT * FROM retail_transactions AS t INNER JOIN retail_demographics p ON t.household_id = p.household_id",
            "SELECT COUNT(*) FROM retail_transactions AS t INNER JOIN retail_demographics p USING (household_id)",
            "WITH my_tab AS (SELECT COUNT(*) FROM retail_transactions AS t INNER JOIN retail_demographics p USING (household_id) GROUP BY household_id) SELECT * FROM my_tab",
            "SELECT COUNT(DISTINCT household_id) AS unique_customers FROM retail_transactions",
            "SELECT * FROM retail_transactions t1 INNER JOIN retail_transactions t2 ON t1.product_id = t2.product_id",
            "SELECT COUNT(*) FROM retail_transactions t INNER JOIN retail_products p ON t.product_id = p.product_id",
            "SELECT department, AVG(sales_value) AS average_sales FROM retail_transactions INNER JOIN retail_products ON retail_transactions.product_id = retail_products.product_id GROUP BY department",
            "WITH ranked_products AS (SELECT product_id, COUNT(*) AS my_count FROM retail_transactions GROUP BY product_id) SELECT product_id FROM ranked_products ORDER BY my_count",
            "SELECT t.product_id, p.product_category, COUNT(*) AS purchase_count FROM retail_transactions t INNER JOIN retail_products p ON t.product_id = p.product_id WHERE t.transaction_timestamp > '2023-01-01' AND t.transaction_timestamp < '2023-02-01' GROUP BY t.product_id, p.product_category",
            "SELECT DISTINCT age, income FROM retail_demographics",
            "SELECT age, income FROM retail_demographics GROUP BY age, income",
            "SELECT quantity, AVG(sales_value) FROM retail_demographics AS d JOIN retail_transactions AS t ON d.household_id = t.household_id WHERE quantity > 0 AND sales_value > 0 AND sales_value < 100 GROUP BY quantity",
            "WITH stats_stores AS (SELECT store_id, SUM(sales_value) AS sum_sales_value, AVG(retail_disc) FROM retail_transactions WHERE sales_value > 0 AND sales_value < 100 AND retail_disc > 0 AND retail_disc < 10 GROUP BY store_id) SELECT * FROM stats_stores WHERE sum_sales_value != 1",
            "SELECT p.product_id, p.brand, COUNT(*) FROM retail_products p INNER JOIN retail_transactions t ON p.product_id = t.product_id GROUP BY p.product_id, p.brand",
            "SELECT t.household_id, store_id, AVG(sales_value) FROM retail_demographics AS d JOIN retail_transactions AS t ON d.household_id = t.household_id WHERE sales_value > 0 AND sales_value < 100 GROUP BY t.household_id, store_id",
        ];

        let soft_pup_queries = [
            "WITH my_tab AS (SELECT * FROM retail_demographics) SELECT * FROM my_tab",
            "SELECT * FROM retail_transactions t INNER JOIN retail_products p ON t.product_id = p.product_id",
            "SELECT * FROM retail_products INNER JOIN retail_transactions USING(product_id)",
            "SELECT product_category, COUNT(*) FROM retail_products GROUP BY product_category",
        ];
        for query_str in soft_pup_queries {
            println!("\n{query_str}");
            let query = parse(query_str).unwrap();
            let relation = Relation::try_from(query.with(&relations)).unwrap();
            relation.display_dot().unwrap();
            let pup_relation = relation
                .rewrite_as_privacy_unit_preserving(
                    &relations,
                    synthetic_data.clone(),
                    privacy_unit.clone(),
                    dp_parameters.clone(),
                    Some(Strategy::Soft),
                )
                .unwrap();
            pup_relation.relation().display_dot().unwrap();
        }

        for query_str in hard_pup_queries {
            println!("\n{query_str}");
            let query = parse(query_str).unwrap();
            let relation = Relation::try_from(query.with(&relations)).unwrap();
            relation.display_dot().unwrap();
            let pup_relation = relation
                .rewrite_as_privacy_unit_preserving(
                    &relations,
                    synthetic_data.clone(),
                    privacy_unit.clone(),
                    dp_parameters.clone(),
                    Some(Strategy::Hard),
                )
                .unwrap();
            let non_pup_queries = relation.rewrite_as_privacy_unit_preserving(
                &relations,
                synthetic_data.clone(),
                privacy_unit.clone(),
                dp_parameters.clone(),
                Some(Strategy::Soft),
            );
            assert_eq!(
                non_pup_queries.unwrap_err().to_string(),
                Error::unreachable_property("privacy_unit_preserving").to_string()
            );
            pup_relation.relation().display_dot().unwrap();
        }
    }

    #[test]
    fn test_census() {
        let census: Relation = Relation::table()
            .name("census")
            .schema(
                vec![
                    ("capital_loss", DataType::integer()),
                    ("age", DataType::integer()),
                ]
                .into_iter()
                .collect::<Schema>(),
            )
            .size(1000)
            .build();
        let relations: Hierarchy<Arc<Relation>> = vec![census]
            .iter()
            .map(|t| (Identifier::from(t.name()), Arc::new(t.clone().into())))
            .collect();
        let synthetic_data = Some(SyntheticData::new(Hierarchy::from([(
            vec!["census"],
            Identifier::from("SYNTHETIC_census"),
        )])));
        let privacy_unit = PrivacyUnit::from(vec![("census", vec![], "_PRIVACY_UNIT_ROW_")]);
        let dp_parameters = DpParameters::from_epsilon_delta(1., 1e-3);

        let queries = [
            "SELECT SUM(CAST(capital_loss AS float) / 100000.) AS my_sum FROM census WHERE capital_loss > 2231. AND capital_loss < 4356.;",
            "SELECT SUM(capital_loss) AS my_sum FROM census WHERE capital_loss > 2231. AND capital_loss < 4356.;",
            "SELECT SUM(capital_loss / 100) AS my_sum FROM census WHERE capital_loss > 2231. AND capital_loss < 4356.;",
            "SELECT SUM(CASE WHEN age > 70 THEN 1 ELSE 0 END) AS s1 FROM census WHERE age > 20 AND age < 90;"
        ];
        for query_str in queries {
            println!("\n{query_str}");
            let query = parse(query_str).unwrap();
            let relation = Relation::try_from(query.with(&relations)).unwrap();
            relation.display_dot().unwrap();
            let dp_relation = relation
                .rewrite_with_differential_privacy(
                    &relations,
                    synthetic_data.clone(),
                    privacy_unit.clone(),
                    dp_parameters.clone(),
                )
                .unwrap();
            dp_relation.relation().display_dot().unwrap();
            println!("dp_event = {}", dp_relation.dp_event());
            assert!(!dp_relation.dp_event().is_no_op());
        }
    }

    #[test]
    fn test_patients() {
        let axa_patients: Relation = Relation::table()
            .name("axa_patients")
            .schema(
                vec![
                    ("Id", DataType::text()),
                    ("BIRTHDATE", DataType::text()),
                    ("GENDER", DataType::text()),
                    ("ZIP", DataType::integer()),
                ]
                .into_iter()
                .collect::<Schema>(),
            )
            .size(10901)
            .build();
        let axa_encounters: Relation = Relation::table()
            .name("axa_encounters")
            .schema(
                vec![
                    ("Id", DataType::text()),
                    ("START", DataType::text()),
                    ("STOP", DataType::text()),
                    ("PATIENT", DataType::text()),
                    ("ORGANIZATION", DataType::text()),
                    ("PROVIDER", DataType::text()),
                    ("PAYER", DataType::text()),
                    ("ENCOUNTERCLASS", DataType::text()),
                    ("CODE", DataType::integer()),
                    ("DESCRIPTION", DataType::text()),
                    ("BASE_ENCOUNTER_COST", DataType::float()),
                    ("TOTAL_CLAIM_COST", DataType::float_min(-1.)),
                    ("PAYER_COVERAGE", DataType::float()),
                    ("REASON_CODE", DataType::integer()),
                    ("REASONDESCRIPTION", DataType::integer()),
                ]
                .into_iter()
                .collect::<Schema>(),
            )
            .size(77727)
            .build();
        let relations: Hierarchy<Arc<Relation>> = vec![axa_patients, axa_encounters]
            .iter()
            .map(|t| (Identifier::from(t.name()), Arc::new(t.clone().into())))
            .collect();
        let synthetic_data = Some(SyntheticData::new(Hierarchy::from([
            (
                vec!["axa_patients"],
                Identifier::from("synthetic_axa_patients"),
            ),
            (
                vec!["axa_encounters"],
                Identifier::from("synthetic_axa_encounters"),
            ),
        ])));
        let privacy_unit = PrivacyUnit::from(vec![
            ("axa_patients", vec![], "Id"),
            (
                "axa_encounters",
                vec![("PATIENT", "axa_patients", "Id")],
                "Id",
            ),
        ]);
        let dp_parameters = DpParameters::from_epsilon_delta(1., 1e-3);

        let queries = [r#"
            SELECT
                "ENCOUNTERCLASS",
                COUNT(p."Id") as patient_count,
                SUM("TOTAL_CLAIM_COST") as sum_cost,
                AVG("TOTAL_CLAIM_COST") as avg_cost
            FROM  axa_patients p
            JOIN axa_encounters e
            ON p."Id" = e."PATIENT"
            GROUP BY "ENCOUNTERCLASS"
            "#];
        for query_str in queries {
            println!("\n{query_str}");
            let query = parse(query_str).unwrap();
            let relation = Relation::try_from(query.with(&relations)).unwrap();
            relation.display_dot().unwrap();
            let dp_relation = relation
                .rewrite_with_differential_privacy(
                    &relations,
                    synthetic_data.clone(),
                    privacy_unit.clone(),
                    dp_parameters.clone(),
                )
                .unwrap();
            dp_relation.relation().display_dot().unwrap();
        }
    }
}
