//! # Methods to transform `Relation`s into differentially private ones
//!
//! This is experimental and little tested yet.
//!

pub mod aggregates;
pub mod dp_parameters;
pub mod group_by;
pub mod dp_event;

use crate::{
    builder::With,
    expr, privacy_unit_tracking::{self, privacy_unit, PupRelation},
    relation::{rewriting, Constraint, Reduce, Relation, Variant},
    Ready,
};
use std::{error, fmt, ops::Deref, result};

/// Some exports
pub use dp_parameters::DpParameters;
pub use dp_event::DpEvent;

use self::aggregates::DpAggregatesParameters;

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
impl From<privacy_unit_tracking::Error> for Error {
    fn from(err: privacy_unit_tracking::Error) -> Self {
        Error::Other(err.to_string())
    }
}

impl error::Error for Error {}
pub type Result<T> = result::Result<T, Error>;

/// A DP Relation
#[derive(Clone, Debug)]
pub struct DpRelation {
    relation: Relation,
    dp_event: DpEvent,
}

impl From<DpRelation> for Relation {
    fn from(value: DpRelation) -> Self {
        value.relation
    }
}

impl DpRelation {
    pub fn new(relation: Relation, dp_event: DpEvent) -> Self {
        DpRelation {
            relation,
            dp_event,
        }
    }

    pub fn relation(&self) -> &Relation {
        &self.relation
    }

    pub fn dp_event(&self) -> &DpEvent {
        &self.dp_event
    }
}

impl Deref for DpRelation {
    type Target = Relation;

    fn deref(&self) -> &Self::Target {
        &self.relation
    }
}

impl From<DpRelation> for (Relation, DpEvent) {
    fn from(value: DpRelation) -> Self {
        (value.relation, value.dp_event)
    }
}

impl From<(Relation, DpEvent)> for DpRelation {
    fn from(value: (Relation, DpEvent)) -> Self {
        DpRelation::new(value.0, value.1)
    }
}



impl Reduce {
    /// Rewrite a `Reduce` into DP:
    ///     - Protect the grouping keys
    ///     - Add noise on the aggregations
    pub fn differentially_private(
        self,
        parameters: &DpParameters,
    ) -> Result<DpRelation> {
        let mut dp_event = DpEvent::no_op();
        let max_size = self.size().max().unwrap().clone();
        let pup_input = PupRelation::try_from(self.input().clone())?;
        let privacy_unit_unique = pup_input.schema()[pup_input.privacy_unit()].has_unique_or_primary_key_constraint();

        // DP rewrite group by
        let reduce_with_dp_group_by = if self.group_by().is_empty() {
            self
        } else {
            let (dp_grouping_values, dp_event_group_by) = self
                .differentially_private_group_by(parameters.epsilon*parameters.tau_thresholding_share, parameters.delta*parameters.tau_thresholding_share)?
                .into();
            let input_relation_with_privacy_tracked_group_by = self
                .input()
                .clone()
                .join_with_grouping_values(dp_grouping_values)?;
            let reduce: Reduce = Reduce::builder()
                .with(self)
                .input(input_relation_with_privacy_tracked_group_by)
                .build();
            dp_event = dp_event.compose(dp_event_group_by);
            reduce
        };

        // if the (epsilon_tau_thresholding, delta_tau_thresholding) budget has
        // not been spent, allocate it to the aggregations.
        let aggregation_share = if dp_event.is_no_op() {1.} else {1.-parameters.tau_thresholding_share};
        let aggregation_parameters = DpAggregatesParameters::from_dp_parameters(parameters.clone(), aggregation_share)
            .with_size(usize::try_from(max_size).unwrap())
            .with_privacy_unit_unique(privacy_unit_unique);
        
        // DP rewrite aggregates
        let (dp_relation, dp_event_agg) = reduce_with_dp_group_by
            .differentially_private_aggregates(aggregation_parameters)?
            .into();
        dp_event = dp_event.compose(dp_event_agg);
        Ok((dp_relation, dp_event).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ast,
        builder::With,
        data_type::{DataType, DataTyped, Variant as _},
        display::Dot,
        expr::{AggregateColumn, Expr},
        io::{postgresql, Database},
        privacy_unit_tracking::{PrivacyUnit,PrivacyUnitTracking, Strategy, PupRelation},
        relation::{Field, Map, Relation, Schema, Variant, Constraint},
    };
    use std::{collections::HashSet, sync::Arc};

    #[test]
    fn test_dp_rewrite_reduce_without_group_by() {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        let parameters = DpParameters::from_epsilon_delta(1., 1e-3);

        // privacy track the inputs
        let table = relations
            .get(&["item_table".to_string()])
            .unwrap()
            .deref()
            .clone();
        let privacy_unit_tracking = PrivacyUnitTracking::from((
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
        let pup_table = privacy_unit_tracking
            .table(&table.clone().try_into().unwrap())
            .unwrap();
        let reduce = Reduce::new(
            "my_reduce".to_string(),
            vec![("sum_price".to_string(), AggregateColumn::sum("price"))],
            vec![],
            pup_table.deref().clone().into(),
        );
        let relation = Relation::from(reduce.clone());
        relation.display_dot().unwrap();

        let (dp_relation, dp_event) = reduce
            .differentially_private(&parameters)
            .unwrap()
            .into();
        dp_relation.display_dot().unwrap();
        let mult: f64 = 2000.*DpAggregatesParameters::from_dp_parameters(parameters.clone(), 1.).privacy_unit_multiplicity();
        assert!(matches!(dp_event, DpEvent::Gaussian { noise_multiplier: _ }));
        assert!(dp_relation
            .data_type()
            .is_subset_of(&DataType::structured([("sum_price", DataType::float())])));

        let query: &str = &ast::Query::from(&dp_relation).to_string();
        _ = database.query(query).unwrap();

        // input a map
        let table = relations
            .get(&["table_1".to_string()])
            .unwrap()
            .deref()
            .clone();
        let privacy_unit_tracking = PrivacyUnitTracking::from((
            &relations,
            vec![
                (
                    "table_1",
                    vec![],
                    PrivacyUnit::privacy_unit_row(),
                ),
            ],
            Strategy::Hard,
        ));
        let pup_table = privacy_unit_tracking
            .table(&table.clone().try_into().unwrap())
            .unwrap();
        let map = Map::new(
            "my_map".to_string(),
            vec![("my_d".to_string(), expr!(d/100))],
            None,
            vec![],
            None,
            None,
            Arc::new(table.into()),
        );
        let pup_map = privacy_unit_tracking
            .map(
                &map.clone().try_into().unwrap(),
                PupRelation(Relation::from(pup_table))
            )
            .unwrap();
        let reduce = Reduce::new(
            "my_reduce".to_string(),
            vec![("sum_d".to_string(), AggregateColumn::sum("my_d"))],
            vec![],
            pup_map.deref().clone().into(),
        );
        let relation = Relation::from(reduce.clone());
        relation.display_dot().unwrap();

        let (dp_relation, dp_event) = reduce
            .differentially_private(&parameters)
            .unwrap()
            .into();
        dp_relation.display_dot().unwrap();
        assert!(dp_event.is_no_op()); // private query is null beacause we have computed the sum of zeros
        assert_eq!(dp_relation.data_type(), DataType::structured([("sum_d", DataType::float_value(0.))]));

        let query: &str = &ast::Query::from(&dp_relation).to_string();
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
        let parameters = DpParameters::from_epsilon_delta(1., 1e-3);

        // privacy track the inputs
        let privacy_unit_tracking = PrivacyUnitTracking::from((
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
        let pup_table = privacy_unit_tracking
            .table(&table.try_into().unwrap())
            .unwrap();
        let map: Map = Relation::map()
            .with(("order_id", expr!(order_id)))
            .with(("price", expr!(price)))
            .filter(Expr::in_list(
                Expr::col("order_id"),
                Expr::list(vec![1, 2, 3, 4, 5]),
            ))
            .input(pup_table.deref().clone())
            .build();
        let pup_map = privacy_unit_tracking
            .map(&map.try_into().unwrap(), pup_table)
            .unwrap();

        let reduce = Reduce::new(
            "my_reduce".to_string(),
            vec![("sum_price".to_string(), AggregateColumn::sum("price"))],
            vec!["order_id".into()],
            pup_map.deref().clone().into(),
        );
        let relation = Relation::from(reduce.clone());
        relation.display_dot().unwrap();

        let (dp_relation, dp_event) = reduce
            .differentially_private(&parameters)
            .unwrap()
            .into();
        dp_relation.display_dot().unwrap();
        assert!(matches!(dp_event, DpEvent::Gaussian { noise_multiplier: _ }));
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
        let parameters = DpParameters::from_epsilon_delta(1., 1e-3);

        // privacy track the inputs
        let privacy_unit_tracking = PrivacyUnitTracking::from((
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
        let pup_table = privacy_unit_tracking
            .table(&table.try_into().unwrap())
            .unwrap();
        let map: Map = Relation::map()
            .with(("order_id", expr!(order_id)))
            .with(("price", expr!(price)))
            .input(pup_table.deref().clone())
            .build();
        let pup_map = privacy_unit_tracking
            .map(&map.try_into().unwrap(), pup_table)
            .unwrap();

        let reduce = Reduce::new(
            "my_reduce".to_string(),
            vec![("sum_price".to_string(), AggregateColumn::sum("price"))],
            vec!["order_id".into()],
            pup_map.deref().clone().into(),
        );
        let relation = Relation::from(reduce.clone());
        relation.display_dot().unwrap();

        let (dp_relation, dp_event) = reduce
            .differentially_private(&parameters)
            .unwrap()
            .into();
        dp_relation.display_dot().unwrap();
        assert!(matches!(dp_event, DpEvent::Composed { events: _ }));
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
        let parameters = DpParameters::from_epsilon_delta(1., 1e-3);

        // privacy track the inputs
        let privacy_unit_tracking = PrivacyUnitTracking::from((
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
        let pup_table = privacy_unit_tracking
            .table(&table.try_into().unwrap())
            .unwrap();
        let map: Map = Relation::map()
            .with(("order_id", expr!(order_id)))
            .with(("item", expr!(item)))
            .with(("price", expr!(price)))
            .filter(Expr::in_list(
                Expr::col("order_id"),
                Expr::list(vec![1, 2, 3, 4, 5]),
            ))
            .input(pup_table.deref().clone())
            .build();
        let pup_map = privacy_unit_tracking
            .map(&map.try_into().unwrap(), pup_table)
            .unwrap();

        let reduce = Reduce::new(
            "my_reduce".to_string(),
            vec![
                ("item".to_string(), AggregateColumn::first("item")),
                ("order_id".to_string(), AggregateColumn::first("order_id")),
                ("sum_price".to_string(), AggregateColumn::sum("price")),
            ],
            vec!["order_id".into(), "item".into()],
            pup_map.deref().clone().into(),
        );
        let relation = Relation::from(reduce.clone());
        relation.display_dot().unwrap();

        let (dp_relation, dp_event) = reduce
            .differentially_private(&parameters)
            .unwrap()
            .into();
        dp_relation.display_dot().unwrap();
        assert!(matches!(dp_event, DpEvent::Composed { events: _ }));
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
        let table = relations
            .get(&["large_user_table".into()])
            .unwrap()
            .as_ref()
            .clone();
        let new_schema: Schema = table
            .schema()
            .iter()
            .map(|f| {
                if f.name() == "city" {
                    Field::from_name_data_type("city", DataType::text())
                } else {
                    f.clone()
                }
            })
            .collect();
        let table: Relation = Relation::table()
            .path(["large_user_table"])
            .name("more_users")
            .size(100000)
            .schema(new_schema)
            .build();
        let input: Relation = Relation::map()
            .name("map_relation")
            .with(("income", expr!(income)))
            .with(("city", expr!(city)))
            .with((PrivacyUnit::privacy_unit(), expr!(id)))
            .with((PrivacyUnit::privacy_unit_weight(), expr!(id)))
            .filter(Expr::in_list(
                Expr::col("city"),
                Expr::list(vec!["Paris".to_string(), "London".to_string()]),
            ))
            .input(table.clone())
            .build();
        let reduce: Reduce = Relation::reduce()
            .name("reduce_relation")
            .with(("city".to_string(), AggregateColumn::first("city")))
            .with(("count_income".to_string(), AggregateColumn::count("income")))
            .group_by(expr!(city))
            .input(input)
            .build();
        let (dp_relation, dp_event) = reduce
            .differentially_private(&DpParameters::from_epsilon_delta(10., 1e-5))
            .unwrap()
            .into();
        println!("{}", dp_event);
        dp_relation.display_dot().unwrap();
        let query: &str = &ast::Query::from(&dp_relation).to_string();
        let results = database.query(query).unwrap();
        println!("results = {:?}", results);
        let city_keys: HashSet<_> = results
            .iter()
            .map(|row| row.to_vec().clone()[0].clone().to_string())
            .collect();
        let correct_keys: HashSet<_> = vec!["London".to_string(), "Paris".to_string()]
            .into_iter()
            .collect();
        assert_eq!(city_keys, correct_keys);
    }

    #[test]
    fn test_differentially_private_output_all_grouping_keys() {
        // test the results contains all the possible keys
        let mut database = postgresql::test_database();
        let relations = database.relations();
        let table = relations
            .get(&["large_user_table".into()])
            .unwrap()
            .as_ref()
            .clone();
        let new_schema: Schema = table
            .schema()
            .iter()
            .map(|f| {
                if f.name() == "city" {
                    Field::from_name_data_type("city", DataType::text())
                } else {
                    f.clone()
                }
            })
            .collect();
        let table: Relation = Relation::table()
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
            .with((PrivacyUnit::privacy_unit(), expr!(id)))
            .with((PrivacyUnit::privacy_unit_weight(), expr!(id)))
            .filter(Expr::in_list(
                Expr::col("city"),
                Expr::list(vec!["Paris".to_string(), "London".to_string()]),
            ))
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
        let (dp_relation, dp_event) = reduce
            .differentially_private(&DpParameters::from_epsilon_delta(10., 1e-5))
            .unwrap()
            .into();
        println!("{}", dp_event);
        dp_relation.display_dot().unwrap();
        let query: &str = &ast::Query::from(&dp_relation).to_string();
        let results = database.query(query).unwrap();
        println!("{:?}", results);
        let city_keys: HashSet<_> = results
            .iter()
            .map(|row| row.to_vec().clone()[0].clone().to_string())
            .collect();
        let correct_keys: HashSet<_> = vec!["London".to_string(), "Paris".to_string()]
            .into_iter()
            .collect();
        assert_eq!(city_keys, correct_keys);
    }

    #[test]
    fn test_dp_rewrite_reduce() {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        let table = relations
            .get(&["table_1".to_string()])
            .unwrap()
            .deref()
            .clone();
        let parameters = DpParameters::from_epsilon_delta(1., 1e-3);

        // privacy track the inputs
        let privacy_unit_tracking = PrivacyUnitTracking::from((
            &relations,
            vec![("table_1", vec![], PrivacyUnit::privacy_unit_row())],
            Strategy::Hard,
        ));
        let pup_table = privacy_unit_tracking
            .table(&table.try_into().unwrap())
            .unwrap();
        let reduce = Reduce::new(
            "my_reduce".to_string(),
            vec![
                ("sum_a".to_string(), AggregateColumn::sum("a")),
                ("d".to_string(), AggregateColumn::first("d")),
                ("max_d".to_string(), AggregateColumn::max("d")),
            ],
            vec!["d".into()],
            pup_table.deref().clone().into(),
        );
        let relation = Relation::from(reduce.clone());
        relation.display_dot().unwrap();

        let (dp_relation, dp_event) = reduce
            .differentially_private(&parameters)
            .unwrap()
            .into();
        dp_relation.display_dot().unwrap();
        assert_eq!(
            dp_event,
            DpEvent::epsilon_delta(parameters.epsilon*parameters.tau_thresholding_share, parameters.delta*parameters.tau_thresholding_share)
                .compose(DpEvent::gaussian_from_epsilon_delta_sensitivity(parameters.epsilon*(1.-parameters.tau_thresholding_share), parameters.delta*(1.-parameters.tau_thresholding_share), 10.))
        );
        let correct_schema: Schema = vec![
            ("sum_a", DataType::float_interval(0., 100.), None),
            ("d", DataType::integer_interval(0, 10), Some(Constraint::Unique)),
            ("max_d", DataType::integer_interval(0, 10), Some(Constraint::Unique)),
        ].into_iter()
        .collect();
        assert_eq!(dp_relation.schema(), &correct_schema);

        let query: &str = &ast::Query::from(&dp_relation).to_string();
        println!("{query}");
        _ = database.query(query).unwrap();
    }
}
