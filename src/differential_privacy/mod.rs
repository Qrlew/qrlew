//! # Methods to transform `Relation`s into differentially private ones
//!
//! This is experimental and little tested yet.
//!

pub mod aggregates;
pub mod budget;
pub mod group_by;
pub mod private_query;

use crate::{
    builder::With,
    differential_privacy::private_query::PrivateQuery,
    expr, protection,
    relation::{rewriting, Reduce, Relation},
    Ready,
};
use std::{error, fmt, ops::Deref, result};

#[derive(Debug, PartialEq, Clone)]
pub enum Error {
    InvalidRelation(String),
    DPCompilationError(String),
    GroupingKeysError(String),
    BudgetError(String),
    Other(String),
}

impl Error {
    pub fn invalid_relation(relation: impl fmt::Display) -> Error {
        Error::InvalidRelation(format!("{relation} is invalid"))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidRelation(relation) => writeln!(f, "{relation} invalid."),
            Error::DPCompilationError(desc) => writeln!(f, "DPCompilationError: {}", desc),
            Error::GroupingKeysError(desc) => writeln!(f, "GroupingKeysError: {}", desc),
            Error::BudgetError(desc) => writeln!(f, "BudgetError: {}", desc),
            Error::Other(err) => writeln!(f, "{}", err),
        }
    }
}

impl From<expr::Error> for Error {
    fn from(err: expr::Error) -> Self {
        Error::Other(err.to_string())
    }
}
impl From<rewriting::Error> for Error {
    fn from(err: rewriting::Error) -> Self {
        Error::Other(err.to_string())
    }
}
impl From<protection::Error> for Error {
    fn from(err: protection::Error) -> Self {
        Error::Other(err.to_string())
    }
}

impl error::Error for Error {}
pub type Result<T> = result::Result<T, Error>;

/// A DP Relation
#[derive(Clone, Debug)]
pub struct DPRelation {
    relation: Relation,
    private_query: PrivateQuery,
}

impl From<DPRelation> for Relation {
    fn from(value: DPRelation) -> Self {
        value.relation
    }
}

impl DPRelation {
    pub fn new(relation: Relation, private_query: PrivateQuery) -> Self {
        DPRelation {
            relation,
            private_query,
        }
    }

    pub fn relation(&self) -> &Relation {
        &self.relation
    }

    pub fn private_query(&self) -> &PrivateQuery {
        &self.private_query
    }
}

impl Deref for DPRelation {
    type Target = Relation;

    fn deref(&self) -> &Self::Target {
        &self.relation
    }
}

impl From<DPRelation> for (Relation, PrivateQuery) {
    fn from(value: DPRelation) -> Self {
        (value.relation, value.private_query)
    }
}

impl From<(Relation, PrivateQuery)> for DPRelation {
    fn from(value: (Relation, PrivateQuery)) -> Self {
        DPRelation::new(value.0, value.1)
    }
}

impl Reduce {
    /// Rewrite a `Reduce` into DP:
    ///     - Protect the grouping keys
    ///     - Add noise on the aggregations
    pub fn differentially_private(
        self,
        epsilon: f64,
        delta: f64,
        epsilon_tau_thresholding: f64,
        delta_tau_thresholding: f64,
    ) -> Result<DPRelation> {
        let mut private_query = PrivateQuery::null();

        // DP rewrite group by
        let reduce_with_dp_group_by = if self.group_by().is_empty() {
            self
        } else {
            let (dp_grouping_values, private_query_group_by) = self
                .differentially_private_group_by(epsilon_tau_thresholding, delta_tau_thresholding)?
                .into();
            let input_relation_with_protected_group_by = self
                .input()
                .clone()
                .join_with_grouping_values(dp_grouping_values)?;
            let reduce: Reduce = Reduce::builder()
                .with(self)
                .input(input_relation_with_protected_group_by)
                .build();
            private_query = private_query.compose(private_query_group_by);
            reduce
        };

        // if the (epsilon_tau_thresholding, delta_tau_thresholding) budget has
        // not been spent, allocate it to the aggregations.
        let (epsilon, delta) = if private_query.is_null() {
            (epsilon + epsilon_tau_thresholding, delta + delta_tau_thresholding)
        } else {
            (epsilon, delta)
        };

        // DP rewrite aggregates
        let (dp_relation, private_query_agg) = reduce_with_dp_group_by
            .differentially_private_aggregates(epsilon, delta)?
            .into();
        private_query = private_query.compose(private_query_agg);
        Ok((dp_relation, private_query).into())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use super::*;
    use crate::{
        ast,
        builder::With,
        data_type::{DataType, DataTyped, Variant as _},
        display::Dot,
        expr::{AggregateColumn, Expr},
        io::{postgresql, Database},
        protection::{Protection, Strategy},
        relation::{Map, Relation, Variant, Schema, Field},
        protection::ProtectedEntity,
    };

    #[test]
    fn test_dp_rewrite_reduce_without_group_by() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        let table = relations
            .get(&["item_table".to_string()])
            .unwrap()
            .deref()
            .clone();
        let (epsilon, delta) = (1., 1e-3);
        let (epsilon_tau_thresholding, delta_tau_thresholding) = (0.5, 2e-3);

        // protect the inputs
        let protection = Protection::from((
            &relations,
            vec![
                (
                    "item_table",
                    vec![("order_id", "order_table", "id")],
                    "date",
                ),
                ("order_table", vec![], "date"),
            ],
            Strategy::Hard,
        ));
        let pep_table = protection.table(&table.try_into().unwrap()).unwrap();
        let reduce = Reduce::new(
            "my_reduce".to_string(),
            vec![("sum_price".to_string(), AggregateColumn::sum("price"))],
            vec![],
            pep_table.deref().clone().into(),
        );
        let relation = Relation::from(reduce.clone());
        relation.display_dot().unwrap();

        let (dp_relation, private_query) = reduce
            .differentially_private(
                epsilon,
                delta,
                epsilon_tau_thresholding,
                delta_tau_thresholding,
            )
            .unwrap()
            .into();
        dp_relation.display_dot().unwrap();
        assert_eq!(
            private_query,
            PrivateQuery::gaussian_from_epsilon_delta_sensitivity(epsilon + epsilon_tau_thresholding, delta + delta_tau_thresholding, 50.)
        );
        assert!(dp_relation
            .data_type()
            .is_subset_of(&DataType::structured([("sum_price", DataType::float())])));

        let query: &str = &ast::Query::from(&dp_relation).to_string();
        println!("{query}");
        _ = database.query(query).unwrap();
    }

    #[test]
    fn test_dp_rewrite_reduce_group_by_possible_values() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        let table = relations
            .get(&["item_table".to_string()])
            .unwrap()
            .deref()
            .clone();
        let (epsilon, delta) = (1., 1e-3);
        let (epsilon_tau_thresholding, delta_tau_thresholding) = (0.5, 2e-3);

        // protect the inputs
        let protection = Protection::from((
            &relations,
            vec![
                (
                    "item_table",
                    vec![("order_id", "order_table", "id")],
                    "date",
                ),
                ("order_table", vec![], "date"),
            ],
            Strategy::Hard,
        ));
        let pep_table = protection.table(&table.try_into().unwrap()).unwrap();
        let map: Map = Relation::map()
            .with(("order_id", expr!(order_id)))
            .with(("price", expr!(price)))
            .filter(Expr::in_list(
                Expr::col("order_id"),
                Expr::list(vec![1, 2, 3, 4, 5]),
            ))
            .input(pep_table.deref().clone())
            .build();
        let pep_map = protection.map(&map.try_into().unwrap(), pep_table).unwrap();

        let reduce = Reduce::new(
            "my_reduce".to_string(),
            vec![("sum_price".to_string(), AggregateColumn::sum("price"))],
            vec![expr!(order_id)],
            pep_map.deref().clone().into(),
        );
        let relation = Relation::from(reduce.clone());
        relation.display_dot().unwrap();

        let (dp_relation, private_query) = reduce
            .differentially_private(
                epsilon,
                delta,
                epsilon_tau_thresholding,
                delta_tau_thresholding,
            )
            .unwrap()
            .into();
        dp_relation.display_dot().unwrap();
        assert_eq!(
            private_query,
            PrivateQuery::gaussian_from_epsilon_delta_sensitivity(
                epsilon + epsilon_tau_thresholding,
                delta + delta_tau_thresholding,
                50.
            )
        );
        assert!(dp_relation
            .data_type()
            .is_subset_of(&DataType::structured([("sum_price", DataType::float())])));

        let query: &str = &ast::Query::from(&dp_relation).to_string();
        println!("{query}");
        _ = database.query(query).unwrap();
    }

    #[test]
    fn test_dp_rewrite_reduce_group_by_tau_thresholding() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        let table = relations
            .get(&["item_table".to_string()])
            .unwrap()
            .deref()
            .clone();
        let (epsilon, delta) = (1., 1e-3);
        let (epsilon_tau_thresholding, delta_tau_thresholding) = (0.5, 2e-3);

        // protect the inputs
        let protection = Protection::from((
            &relations,
            vec![
                (
                    "item_table",
                    vec![("order_id", "order_table", "id")],
                    "date",
                ),
                ("order_table", vec![], "date"),
            ],
            Strategy::Hard,
        ));
        let pep_table = protection.table(&table.try_into().unwrap()).unwrap();
        let map: Map = Relation::map()
            .with(("order_id", expr!(order_id)))
            .with(("price", expr!(price)))
            .input(pep_table.deref().clone())
            .build();
        let pep_map = protection.map(&map.try_into().unwrap(), pep_table).unwrap();

        let reduce = Reduce::new(
            "my_reduce".to_string(),
            vec![("sum_price".to_string(), AggregateColumn::sum("price"))],
            vec![expr!(order_id)],
            pep_map.deref().clone().into(),
        );
        let relation = Relation::from(reduce.clone());
        relation.display_dot().unwrap();

        let (dp_relation, private_query) = reduce
            .differentially_private(
                epsilon,
                delta,
                epsilon_tau_thresholding,
                delta_tau_thresholding,
            )
            .unwrap()
            .into();
        dp_relation.display_dot().unwrap();
        assert_eq!(
            private_query,
            vec![
                PrivateQuery::EpsilonDelta(epsilon_tau_thresholding, delta_tau_thresholding),
                PrivateQuery::gaussian_from_epsilon_delta_sensitivity(epsilon, delta, 50.)
            ]
            .into()
        );
        assert!(dp_relation
            .data_type()
            .is_subset_of(&DataType::structured([("sum_price", DataType::float())])));

        let query: &str = &ast::Query::from(&dp_relation).to_string();
        println!("{query}");
        _ = database.query(query).unwrap();
    }

    #[test]
    fn test_dp_rewrite_reduce_group_by_possible_both() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        let table = relations
            .get(&["item_table".to_string()])
            .unwrap()
            .deref()
            .clone();
        let (epsilon, delta) = (1., 1e-3);
        let (epsilon_tau_thresholding, delta_tau_thresholding) = (0.5, 2e-3);

        // protect the inputs
        let protection = Protection::from((
            &relations,
            vec![
                (
                    "item_table",
                    vec![("order_id", "order_table", "id")],
                    "date",
                ),
                ("order_table", vec![], "date"),
            ],
            Strategy::Hard,
        ));
        let pep_table = protection.table(&table.try_into().unwrap()).unwrap();
        let map: Map = Relation::map()
            .with(("order_id", expr!(order_id)))
            .with(("item", expr!(item)))
            .with(("price", expr!(price)))
            .filter(Expr::in_list(
                Expr::col("order_id"),
                Expr::list(vec![1, 2, 3, 4, 5]),
            ))
            .input(pep_table.deref().clone())
            .build();
        let pep_map = protection.map(&map.try_into().unwrap(), pep_table).unwrap();

        let reduce = Reduce::new(
            "my_reduce".to_string(),
            vec![
                ("item".to_string(), AggregateColumn::first("item")),
                ("order_id".to_string(), AggregateColumn::first("order_id")),
                ("sum_price".to_string(), AggregateColumn::sum("price")),
            ],
            vec![expr!(order_id), expr!(item)],
            pep_map.deref().clone().into(),
        );
        let relation = Relation::from(reduce.clone());
        relation.display_dot().unwrap();

        let (dp_relation, private_query) = reduce
            .differentially_private(
                epsilon,
                delta,
                epsilon_tau_thresholding,
                delta_tau_thresholding,
            )
            .unwrap()
            .into();
        dp_relation.display_dot().unwrap();
        assert_eq!(
            private_query,
            vec![
                PrivateQuery::EpsilonDelta(epsilon_tau_thresholding, delta_tau_thresholding),
                PrivateQuery::gaussian_from_epsilon_delta_sensitivity(epsilon, delta, 50.)
            ]
            .into()
        );
        assert!(dp_relation.schema()[0]
            .data_type()
            .is_subset_of(&DataType::text()));
        assert_eq!(
            dp_relation.schema()[1].data_type(),
            DataType::integer_values(vec![1, 2, 3, 4, 5])
        );
        assert!(dp_relation.schema()[2]
            .data_type()
            .is_subset_of(&DataType::float()));

        let query: &str = &ast::Query::from(&dp_relation).to_string();
        println!("{query}");
        _ = database.query(query).unwrap();
    }

    #[test]
    fn test_differentially_private_output_all_grouping_keys_simple() {
        // test the results contains all the possible keys
        let mut database = postgresql::test_database();
        let relations = database.relations();
        let table = relations.get(&["large_user_table".into()]).unwrap().as_ref().clone();
        let new_schema: Schema = table.schema()
            .iter()
            .map(|f| if f.name() == "city" {
                Field::from_name_data_type("city", DataType::text())
            } else {
                f.clone()
            })
            .collect();
        let table:Relation = Relation::table()
            .path(["large_user_table"])
            .name("more_users")
            .size(100000)
            .schema(new_schema)
            .build();
        let input: Relation = Relation::map()
            .name("map_relation")
            .with(("income", expr!(income)))
            .with(("city", expr!(city)))
            .with((
                ProtectedEntity::protected_entity_id(),
                expr!(id),
            ))
            .with((
                ProtectedEntity::protected_entity_weight(),
                expr!(id),
            ))
            .filter(
                Expr::in_list(
                    Expr::col("city"),
                    Expr::list(vec!["Paris".to_string(), "London".to_string()]),
                )
            )
            .input(table.clone())
            .build();
        let reduce: Reduce = Relation::reduce()
            .name("reduce_relation")
            .with(("city".to_string(), AggregateColumn::first("city")))
            .with(("count_income".to_string(), AggregateColumn::count("income")))
            .group_by(expr!(city))
            .input(input)
            .build();
        let (dp_relation, private_query) = reduce
        .differentially_private(
            10.,
            1e-5,
            1.,
            1e-2,
        )
            .unwrap()
            .into();
        println!("{}", private_query);
        dp_relation.display_dot().unwrap();
        let query: &str = &ast::Query::from(&dp_relation).to_string();
        let results = database
            .query(query)
            .unwrap();
        println!("results = {:?}", results);
        let city_keys: HashSet<_> = results.iter()
            .map(|row| row.to_vec().clone()[0].clone().to_string())
            .collect();
        let correct_keys: HashSet<_> = vec!["London".to_string(), "Paris".to_string()].into_iter().collect();
        assert_eq!(city_keys, correct_keys);
    }

    #[test]
    fn test_differentially_private_output_all_grouping_keys() {
        // test the results contains all the possible keys
        let mut database = postgresql::test_database();
        let relations = database.relations();
        let table = relations.get(&["large_user_table".into()]).unwrap().as_ref().clone();
        let new_schema: Schema = table.schema()
            .iter()
            .map(|f| if f.name() == "city" {
                Field::from_name_data_type("city", DataType::text())
            } else {
                f.clone()
            })
            .collect();
        let table:Relation = Relation::table()
            .path(["large_user_table"])
            .name("more_users")
            .size(100000)
            .schema(new_schema)
            .build();
        let input: Relation = Relation::map()
            .name("map_relation")
            .with(("income", expr!(income)))
            .with(("city", expr!(city)))
            .with(("age", expr!(age)))
            .with((
                ProtectedEntity::protected_entity_id(),
                expr!(id),
            ))
            .with((
                ProtectedEntity::protected_entity_weight(),
                expr!(id),
            ))
            .filter(
                Expr::in_list(
                    Expr::col("city"),
                    Expr::list(vec!["Paris".to_string(), "London".to_string()]),
                )
            )
            .input(table.clone())
            .build();
        let reduce: Reduce = Relation::reduce()
            .name("reduce_relation")
            .with(("city".to_string(), AggregateColumn::first("city")))
            .with(("age".to_string(), AggregateColumn::first("age")))
            .with(("sum_income".to_string(), AggregateColumn::sum("income")))
            .group_by(expr!(city))
            .group_by(expr!(age))
            .input(input)
            .build();
        let (dp_relation, private_query) = reduce
        .differentially_private(
            10.,
            1e-5,
            1.,
            1e-2,
        )
            .unwrap()
            .into();
        println!("{}", private_query);
        dp_relation.display_dot().unwrap();
        let query: &str = &ast::Query::from(&dp_relation).to_string();
        let results = database
            .query(query)
            .unwrap();
        println!("{:?}", results);
        let city_keys: HashSet<_> = results.iter()
            .map(|row| row.to_vec().clone()[0].clone().to_string())
            .collect();
        let correct_keys: HashSet<_> = vec!["London".to_string(), "Paris".to_string()].into_iter().collect();
        assert_eq!(city_keys, correct_keys);
    }
}
