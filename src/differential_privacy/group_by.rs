use super::Error;
use crate::{
    builder::{Ready, With, WithIterator},
    differential_privacy::{dp_event, DpEvent, DpRelation, Result},
    expr::Expr,
    namer::{self, name_from_content},
    privacy_unit_tracking::{PrivacyUnit, PupRelation},
    relation::{Join, Reduce, Relation, Variant as _},
};

pub const COUNT_DISTINCT_PID: &str = "_COUNT_DISTINCT_PID_";

impl Reduce {
    /// Returns a `DPRelation` whose:
    ///     - `relation` outputs all the DP values of the `self` grouping keys
    ///     - `dp_event` stores the invoked DP mechanisms
    pub fn differentially_private_group_by(
        &self,
        epsilon: f64,
        delta: f64,
        max_privacy_unit_groups: u64,
    ) -> Result<DpRelation> {
        if self.group_by().is_empty() {
            Err(Error::GroupingKeysError(format!("No grouping keys")))
        } else {
            let relation: Relation = Relation::map()
                .with_iter(
                    self.group_by()
                        .into_iter()
                        .map(|col| (col.to_string(), Expr::Column(col.clone())))
                        .collect::<Vec<_>>(),
                )
                .with((
                    PrivacyUnit::privacy_unit(),
                    Expr::col(PrivacyUnit::privacy_unit()),
                ))
                .with((
                    PrivacyUnit::privacy_unit_weight(),
                    Expr::col(PrivacyUnit::privacy_unit_weight()),
                ))
                .input(self.input().clone())
                .build();
            PupRelation::try_from(relation)?.dp_values(epsilon, delta, max_privacy_unit_groups)
        }
    }
}

impl PupRelation {
    /// Returns a `DPRelation` whose:
    ///     - `relation` outputs the (epsilon, delta)-DP values
    /// (found by tau-thresholding) of the fields of the current `Relation`
    ///     - `dp_event` stores the invoked DP mechanisms
    fn tau_thresholding_values(
        self,
        epsilon: f64,
        delta: f64,
        max_privacy_unit_groups: u64,
    ) -> Result<DpRelation> {
        // It limits the PU contribution to at most max_privacy_unit_groups random groups
        // It counts distinct PUs
        // It applies tau-thresholding
        if epsilon == 0. || delta == 0. {
            return Err(Error::BudgetError(format!(
                "Not enough budget for tau-thresholding. Got: (espilon, delta) = ({epsilon}, {delta})"
            )));
        }
        // Build a reduce grouping by columns and the PU
        let columns: Vec<&str> = self
            .schema()
            .iter()
            .filter_map(|f| {
                if f.name() == self.privacy_unit() || f.name() == self.privacy_unit_weight() {
                    None
                } else {
                    Some(f.name())
                }
            })
            .collect();
        let columns_and_pu: Vec<_> = columns
            .iter()
            .cloned()
            .chain(std::iter::once(self.privacy_unit()))
            .collect();
        let red = Relation::from(self.clone()).unique(&columns_and_pu);

        let rel_with_limited_pu_contributions =
            red.limit_col_contributions(self.privacy_unit(), max_privacy_unit_groups);

        let mut columns_aggs: Vec<(&str, Expr)> = vec![(
            COUNT_DISTINCT_PID,
            Expr::count(Expr::col(self.privacy_unit())),
        )];
        let mut columns_groups: Vec<Expr> = vec![];
        columns.into_iter().for_each(|c| {
            let col = Expr::col(c);
            columns_aggs.push((c, Expr::first(col.clone())));
            columns_groups.push(col);
        });

        // Count distinct PUs.
        let rel: Relation = Relation::reduce()
            .with_iter(columns_aggs)
            .group_by_iter(columns_groups)
            .input(rel_with_limited_pu_contributions)
            .build();

        // Apply noise
        let name_sigmas = vec![(
            COUNT_DISTINCT_PID,
            dp_event::gaussian_noise(epsilon, delta, (max_privacy_unit_groups as f64).sqrt()),
        )];
        let rel = rel.add_gaussian_noise(&name_sigmas);

        // Returns a `Relation::Map` with the right field names and with `COUNT(DISTINCT PID) > tau`
        let tau = dp_event::gaussian_tau(epsilon, delta, max_privacy_unit_groups as f64);
        let filter_column = [(COUNT_DISTINCT_PID, (Some(tau.into()), None, vec![]))]
            .into_iter()
            .collect();
        let relation = rel
            .filter_columns(filter_column)
            .filter_fields(|f| columns_and_pu.contains(&f));
        Ok(DpRelation::new(
            relation,
            DpEvent::epsilon_delta(epsilon, delta),
        ))
    }

    /// Returns a DPRelation whose:
    ///     - first field is a Relation whose outputs are
    /// (epsilon, delta)-DP values of grouping keys of the current PUPRelation,
    ///     - second field is a PrivateQuery corresponding the used mechanisms
    /// The (epsilon, delta)-DP values are found by:
    ///     - Using the propagated public values of the grouping columns when they exist
    ///     - Applying tau-thresholding mechanism with the (epsilon, delta) privacy parameters for t
    /// he columns that do not have public values
    pub fn dp_values(
        self,
        epsilon: f64,
        delta: f64,
        max_privacy_unit_groups: u64,
    ) -> Result<DpRelation> {
        // TODO this code is super-ugly rewrite it
        let public_columns: Vec<String> = self
            .schema()
            .iter()
            .filter_map(|f| {
                (f.name() != self.privacy_unit()
                    && f.name() != self.privacy_unit_weight()
                    && f.all_values())
                .then_some(f.name().to_string())
            })
            .collect();
        let all_columns_are_public = public_columns.len() == self.schema().len() - 2;

        if public_columns.is_empty() {
            let name = namer::name_from_content("FILTER_", &self.name());
            self.with_name(name)?
                .tau_thresholding_values(epsilon, delta, max_privacy_unit_groups)
        } else if all_columns_are_public {
            Ok(DpRelation::new(
                self.with_public_values(&public_columns)?,
                DpEvent::no_op(),
            ))
        } else {
            let (relation, dp_event) = self
                .clone()
                .with_name(namer::name_from_content("FILTER_", &self.name()))?
                .filter_fields(|f| !public_columns.contains(&f.to_string()))?
                .tau_thresholding_values(epsilon, delta, max_privacy_unit_groups)?
                .into();
            let relation = self
                .with_public_values(&public_columns)?
                .cross_join(relation)?;
            Ok(DpRelation::new(relation, dp_event))
        }
    }
}

impl Relation {
    fn with_public_values(&self, public_columns: &Vec<String>) -> Result<Relation> {
        let relation_with_private_values = self
            .clone()
            .filter_fields(|f| public_columns.contains(&f.to_string()));
        Ok(relation_with_private_values.public_values()?)
    }

    /// We join the `self` `Relation` with the `grouping_values Relation`;
    /// We use a `LEFT OUTER` join for guaranteeing that all the possible grouping keys are released
    pub fn join_with_grouping_values(self, grouping_values: Relation) -> Result<Relation> {
        let left = grouping_values;
        let right = self;

        let on: Vec<Expr> = left
            .schema()
            .iter()
            .map(|f| {
                Expr::eq(
                    Expr::qcol(Join::left_name(), f.name()),
                    Expr::qcol(Join::right_name(), f.name()),
                )
            })
            .collect();

        let names = right
            .schema()
            .iter()
            .map(|f| f.name().to_string())
            .collect::<Vec<_>>();
        let left_names = left
            .schema()
            .iter()
            .map(|f| f.name().to_string())
            .collect::<Vec<_>>();
        let right_names = right
            .schema()
            .iter()
            .map(|f| {
                let name = f.name().to_string();
                if left_names.contains(&name) {
                    name_from_content("left_".to_string(), f)
                } else {
                    name
                }
            })
            .collect::<Vec<_>>();

        let join_rel: Relation = Relation::join()
            .right(right)
            .right_names(right_names.clone())
            .left(left)
            .left_names(left_names.clone())
            .left_outer(Expr::val(true))
            .on_iter(on)
            .build();

        Ok(join_rel.filter_fields(|f| names.contains(&f.to_string())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ast,
        builder::With,
        data_type::{DataType, DataTyped, Variant},
        display::Dot,
        expr::AggregateColumn,
        io::{postgresql, Database},
        privacy_unit_tracking::{PrivacyUnit, PrivacyUnitTracking, Strategy},
        relation::{Field, Join, Schema},
    };
    use std::{collections::HashSet, ops::Deref};

    #[test]
    fn test_tau_thresholding_values() {
        let table: Relation = Relation::table()
            .name("table")
            .schema(
                Schema::builder()
                    .with(("a", DataType::integer_range(1..=10)))
                    .with(("b", DataType::integer_values([1, 2, 5, 6, 7, 8])))
                    .with(("c", DataType::integer_range(5..=20)))
                    .with((
                        PrivacyUnit::privacy_unit(),
                        DataType::integer_range(1..=100),
                    ))
                    .with((
                        PrivacyUnit::privacy_unit_weight(),
                        DataType::float_interval(0., 1.),
                    ))
                    .build(),
            )
            .build();
        let pup_table = PupRelation(table);

        let (rel, pq) = pup_table
            .clone()
            .tau_thresholding_values(1., 0.003, 1)
            .unwrap()
            .into();
        rel.display_dot().unwrap();
        assert_eq!(
            rel.data_type(),
            DataType::structured([
                ("a", DataType::integer_range(1..=10)),
                ("b", DataType::integer_values([1, 2, 5, 6, 7, 8])),
                ("c", DataType::integer_range(5..=20))
            ])
        );
        assert_eq!(pq, DpEvent::epsilon_delta(1., 0.003))
    }

    #[test]
    fn test_dp_values() {
        // Only possible values
        let table: Relation = Relation::table()
            .name("table")
            .schema(
                Schema::builder()
                    .with(("a", DataType::integer_values([1, 2, 4, 6])))
                    .with(("b", DataType::float_values([1.2, 4.6, 7.8])))
                    .with((
                        PrivacyUnit::privacy_unit(),
                        DataType::integer_range(1..=100),
                    ))
                    .with((
                        PrivacyUnit::privacy_unit_weight(),
                        DataType::float_interval(0., 1.),
                    ))
                    .build(),
            )
            .build();
        let pup_table = PupRelation(table);
        let (rel, pq) = pup_table.dp_values(1., 0.003, 1).unwrap().into();
        matches!(rel, Relation::Join(_));
        assert_eq!(
            rel.data_type(),
            DataType::structured([
                ("a", DataType::integer_values([1, 2, 4, 6])),
                ("b", DataType::float_values([1.2, 4.6, 7.8]))
            ])
        );
        assert!(matches!(rel.inputs()[0], &Relation::Values(_)));
        assert!(matches!(rel.inputs()[1], &Relation::Values(_)));
        assert_eq!(pq, DpEvent::no_op());

        // Only tau-thresholding values
        let table: Relation = Relation::table()
            .name("table")
            .schema(
                Schema::builder()
                    .with(("a", DataType::integer_range(1..=10)))
                    .with(("b", DataType::float_range(5.4..=20.)))
                    .with((
                        PrivacyUnit::privacy_unit(),
                        DataType::integer_range(1..=100),
                    ))
                    .with((
                        PrivacyUnit::privacy_unit_weight(),
                        DataType::float_interval(0., 1.),
                    ))
                    .build(),
            )
            .build();
        let pup_table = PupRelation(table);
        let (rel, pq) = pup_table.dp_values(1., 0.003, 1).unwrap().into();
        assert!(matches!(rel, Relation::Map(_)));
        assert_eq!(
            rel.data_type(),
            DataType::structured([
                ("a", DataType::integer_range(1..=10)),
                ("b", DataType::float_range(5.4..=20.))
            ])
        );
        assert_eq!(pq, DpEvent::epsilon_delta(1., 0.003));

        // Both possible and tau-thresholding values
        let table: Relation = Relation::table()
            .name("table")
            .schema(
                Schema::builder()
                    .with(("a", DataType::integer_range(1..=5)))
                    .with(("b", DataType::integer_values([1, 2, 5, 6, 7, 8])))
                    .with(("c", DataType::integer_range(5..=20)))
                    .with((
                        PrivacyUnit::privacy_unit(),
                        DataType::integer_range(1..=100),
                    ))
                    .with((
                        PrivacyUnit::privacy_unit_weight(),
                        DataType::float_interval(0., 1.),
                    ))
                    .build(),
            )
            .build();
        let pup_table = PupRelation(table);
        let (rel, pq) = pup_table.dp_values(1., 0.003, 1).unwrap().into();
        assert!(matches!(rel, Relation::Join(_)));
        assert!(matches!(rel.inputs()[0], &Relation::Values(_)));
        assert!(matches!(rel.inputs()[1], &Relation::Map(_)));
        assert_eq!(
            rel.data_type(),
            DataType::structured([
                ("b", DataType::integer_values([1, 2, 5, 6, 7, 8])),
                ("a", DataType::integer_range(1..=5)),
                ("c", DataType::integer_range(5..=20))
            ])
        );
        assert_eq!(pq, DpEvent::epsilon_delta(1., 0.003));
    }

    #[test]
    fn test_differentially_private_group_by_simple() {
        let table: Relation = Relation::table()
            .name("table")
            .schema(
                Schema::builder()
                    .with(("a", DataType::integer_range(1..=10)))
                    .with(("b", DataType::integer_values([1, 2, 5, 6, 7, 8])))
                    .with(("c", DataType::integer_range(5..=20)))
                    .with((
                        PrivacyUnit::privacy_unit(),
                        DataType::integer_range(1..=100),
                    ))
                    .with((
                        PrivacyUnit::privacy_unit_weight(),
                        DataType::float_interval(0., 1.),
                    ))
                    .build(),
            )
            .build();
        let (epsilon, delta) = (1., 1e-3);

        // Without GROUPBY: Error
        let reduce: Reduce = Relation::reduce()
            .name("reduce_relation")
            .with(("sum_a".to_string(), AggregateColumn::sum("a")))
            .input(table.clone())
            .build();
        let dp_reduce = reduce.differentially_private_group_by(epsilon, delta, 1);
        assert!(dp_reduce.is_err());

        // With GROUPBY. Only one column with possible values
        let reduce: Reduce = Relation::reduce()
            .name("reduce_relation")
            .with(("sum_a".to_string(), AggregateColumn::sum("a")))
            .group_by(expr!(b))
            .input(table.clone())
            .build();
        let (dp_relation, dp_event) = reduce
            .differentially_private_group_by(epsilon, delta, 1)
            .unwrap()
            .into();
        dp_relation.display_dot().unwrap();
        assert_eq!(dp_event, DpEvent::no_op());
        assert_eq!(
            dp_relation.data_type(),
            DataType::structured([("b", DataType::integer_values([1, 2, 5, 6, 7, 8]))])
        );

        // With GROUPBY. Columns with tau-thresholding values
        let reduce: Reduce = Relation::reduce()
            .name("reduce_relation")
            .with(("sum_a".to_string(), AggregateColumn::sum("a")))
            .group_by_iter(vec![expr!(a), expr!(c)])
            .input(table.clone())
            .build();
        let (dp_relation, dp_event) = reduce
            .differentially_private_group_by(epsilon, delta, 1)
            .unwrap()
            .into();
        assert_eq!(dp_event, DpEvent::epsilon_delta(epsilon, delta));
        assert_eq!(
            dp_relation.data_type(),
            DataType::structured([
                ("a", DataType::integer_range(1..=10)),
                ("c", DataType::integer_range(5..=20))
            ])
        );

        // With GROUPBY. Both tau-thresholding and possible values
        let reduce: Reduce = Relation::reduce()
            .name("reduce_relation")
            .with(("sum_a".to_string(), AggregateColumn::sum("a")))
            .group_by(expr!(c))
            .group_by(expr!(b))
            .input(table.clone())
            .build();
        let (dp_relation, dp_event) = reduce
            .differentially_private_group_by(epsilon, delta, 1)
            .unwrap()
            .into();
        assert_eq!(dp_event, DpEvent::epsilon_delta(epsilon, delta));
        assert_eq!(
            dp_relation.data_type(),
            DataType::structured([
                ("c", DataType::integer_range(5..=20)),
                ("b", DataType::integer_values([1, 2, 5, 6, 7, 8]))
            ])
        );
    }

    #[test]
    fn test_diffrential_privacy_group_by_input_map() {
        let table: Relation = Relation::table()
            .name("table")
            .schema(
                Schema::builder()
                    .with(("a", DataType::integer_range(1..=10)))
                    .with(("b", DataType::integer_values([1, 2, 5, 6, 7, 8])))
                    .with(("c", DataType::float_interval(1., 2.)))
                    .with((
                        PrivacyUnit::privacy_unit(),
                        DataType::integer_range(1..=100),
                    ))
                    .with((
                        PrivacyUnit::privacy_unit_weight(),
                        DataType::float_interval(0., 1.),
                    ))
                    .build(),
            )
            .build();
        let (epsilon, delta) = (1., 1e-3);

        let input: Relation = Relation::map()
            .name("map_relation")
            .with(("a", expr!(a)))
            .with(("twice_b", expr!(2 * b)))
            .with(("c", expr!(3 * c)))
            .with((
                PrivacyUnit::privacy_unit(),
                Expr::col(PrivacyUnit::privacy_unit()),
            ))
            .with((
                PrivacyUnit::privacy_unit_weight(),
                Expr::col(PrivacyUnit::privacy_unit_weight()),
            ))
            .input(table.clone())
            .build();
        let reduce: Reduce = Relation::reduce()
            .name("reduce_relation")
            .with(("sum_a".to_string(), AggregateColumn::sum("a")))
            .group_by(expr!(c))
            .group_by(expr!(twice_b))
            .input(input)
            .build();

        let (dp_relation, dp_event) = reduce
            .differentially_private_group_by(epsilon, delta, 1)
            .unwrap()
            .into();
        dp_relation.display_dot().unwrap();
        assert!(matches!(dp_relation, Relation::Join(_)));
        assert!(matches!(dp_relation.inputs()[0], &Relation::Values(_)));
        assert!(matches!(dp_relation.inputs()[1], &Relation::Map(_)));
        assert_eq!(dp_event, DpEvent::epsilon_delta(epsilon, delta));
        assert_eq!(
            dp_relation.data_type(),
            DataType::structured([
                ("twice_b", DataType::integer_values([2, 4, 10, 12, 14, 16])),
                ("c", DataType::float_interval(3., 6.0)),
            ])
        );

        // WHERE IN LIST
        let input: Relation = Relation::map()
            .name("map_relation")
            .with(("a", expr!(a)))
            .with(("twice_b", expr!(2 * b)))
            .with(("c", expr!(3 * c)))
            .filter(Expr::in_list(Expr::col("c"), Expr::list(vec![1., 1.5])))
            .with((
                PrivacyUnit::privacy_unit(),
                Expr::col(PrivacyUnit::privacy_unit()),
            ))
            .with((
                PrivacyUnit::privacy_unit_weight(),
                Expr::col(PrivacyUnit::privacy_unit_weight()),
            ))
            .input(table.clone())
            .build();
        let reduce: Reduce = Relation::reduce()
            .name("reduce_relation")
            .with(("sum_a".to_string(), AggregateColumn::sum("a")))
            .group_by(expr!(c))
            .group_by(expr!(twice_b))
            .input(input)
            .build();
        let (dp_relation, dp_event) = reduce
            .differentially_private_group_by(epsilon, delta, 1)
            .unwrap()
            .into();
        dp_relation.display_dot().unwrap();
        assert_eq!(dp_event, DpEvent::no_op());
        assert!(matches!(dp_relation.inputs()[0], &Relation::Values(_)));
        assert!(matches!(dp_relation.inputs()[1], &Relation::Values(_)));
        assert_eq!(
            dp_relation.data_type(),
            DataType::structured([
                ("twice_b", DataType::integer_values([2, 4, 10, 12, 14, 16])),
                ("c", DataType::float_values([3., 4.5])),
            ])
        );
    }

    #[test]
    fn test_differentially_private_complex() {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        let (epsilon, delta) = (1., 1e-3);

        let left = relations
            .get(&["item_table".to_string()])
            .unwrap()
            .deref()
            .clone();
        let right = relations
            .get(&["order_table".to_string()])
            .unwrap()
            .deref()
            .clone();
        let join: Join = Join::builder()
            .inner(Expr::val(true))
            .on_eq("order_id", "id")
            .left(left.clone())
            .right(right.clone())
            .left_names(vec!["order_id", "items", "price"])
            .right_names(vec!["id", "user_id", "description", "date"])
            .build();
        Relation::from(join.clone()).display_dot().unwrap();
        let privacy_unit_tracking = PrivacyUnitTracking::from((
            &relations,
            vec![
                ("item_table", vec![("order_id", "order_table", "id")], "id"),
                ("order_table", vec![], "id"),
            ],
            Strategy::Hard,
        ));
        let pup_left = privacy_unit_tracking
            .table(&left.try_into().unwrap())
            .unwrap();
        let pup_right = privacy_unit_tracking
            .table(&right.try_into().unwrap())
            .unwrap();
        let pup_join = privacy_unit_tracking
            .join(&join, pup_left, pup_right)
            .unwrap();

        let map: Relation = Relation::map()
            .name("map_relation")
            .with(("items", expr!(items)))
            .with(("twice_price", expr!(2 * price)))
            .with(("date", expr!(date)))
            .with((
                PrivacyUnit::privacy_unit(),
                Expr::col(PrivacyUnit::privacy_unit()),
            ))
            .with((
                PrivacyUnit::privacy_unit_weight(),
                Expr::col(PrivacyUnit::privacy_unit_weight()),
            ))
            .input(pup_join.deref().clone())
            .build();

        let reduce: Reduce = Relation::reduce()
            .name("reduce_relation")
            .with(("sum_price".to_string(), AggregateColumn::sum("twice_price")))
            .group_by(expr!(date))
            .group_by(expr!(items))
            .input(map)
            .build();

        let (dp_relation, dp_event) = reduce
            .differentially_private_group_by(epsilon, delta, 1)
            .unwrap()
            .into();
        dp_relation.display_dot().unwrap();
        assert_eq!(dp_event, DpEvent::epsilon_delta(epsilon, delta));
        matches!(dp_relation, Relation::Map(_));
        assert!(dp_relation.data_type().is_subset_of(&DataType::structured([
            ("date", DataType::date()),
            ("items", DataType::text()),
        ])));
        let dp_query = ast::Query::from(&dp_relation);
        _ = database.query(&dp_query.to_string()).unwrap();
    }

    #[test]
    fn test_differentially_private_output_all_grouping_keys() {
        // test the results contains all the keys asked by the user (i.e. in the WHERE )
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
            .with(("sum_income".to_string(), AggregateColumn::sum("income")))
            .group_by(expr!(city))
            .group_by(expr!(age))
            .input(input)
            .build();
        let (dp_relation, _) = reduce
            .differentially_private_group_by(1., 1e-2, 1)
            .unwrap()
            .into();
        dp_relation.display_dot().unwrap();
        let query: &str = &ast::Query::from(&dp_relation).to_string();
        let results = database.query(query).unwrap();
        let city_keys: HashSet<_> = results
            .iter()
            .map(|row| row.to_vec().clone()[0].clone().to_string())
            .collect();
        let correct_keys: HashSet<_> = vec!["London".to_string(), "Paris".to_string()]
            .into_iter()
            .collect();
        assert_eq!(city_keys, correct_keys);

        let input_relation_with_privacy_tracked_group_by = reduce
            .input()
            .clone()
            .join_with_grouping_values(dp_relation)
            .unwrap();
        input_relation_with_privacy_tracked_group_by
            .display_dot()
            .unwrap();
        let query: &str =
            &ast::Query::from(&input_relation_with_privacy_tracked_group_by).to_string();
        let results = database.query(query).unwrap();
        let city_keys: HashSet<_> = results
            .iter()
            .map(|row| row.to_vec().clone()[0].clone().to_string())
            .collect();
        println!("{:?}", city_keys);
        assert_eq!(city_keys, correct_keys);
    }

    #[test]
    fn test_differentially_private_group_by_spefic_aggregate() {
        let table: Relation = Relation::table()
            .name("table")
            .schema(
                Schema::builder()
                    .with(("a", DataType::integer_range(1..=10)))
                    .with(("b", DataType::integer_values([1, 2, 5, 6, 7, 8])))
                    .with(("c", DataType::integer_range(5..=20)))
                    .with((
                        PrivacyUnit::privacy_unit(),
                        DataType::integer_range(1..=100),
                    ))
                    .with((
                        PrivacyUnit::privacy_unit_weight(),
                        DataType::float_interval(0., 1.),
                    ))
                    .build(),
            )
            .size(100)
            .build();
        let (epsilon, delta) = (1., 1e-3);

        // GROUP BY and the aggregate input the same column
        let reduce: Reduce = Relation::reduce()
            .name("reduce_relation")
            .with(("sum_a".to_string(), AggregateColumn::sum("a")))
            .group_by(expr!(a))
            .input(table.clone())
            .build();
        let (dp_relation, dp_event) = reduce
            .differentially_private_group_by(epsilon, delta, 1)
            .unwrap()
            .into();
        dp_relation.display_dot().unwrap();
        assert_eq!(dp_event, DpEvent::epsilon_delta(epsilon, delta));
        assert_eq!(
            dp_relation.data_type(),
            DataType::structured([("a", DataType::integer_range(1..=10))])
        );

        let reduce: Reduce = Relation::reduce()
            .name("reduce_relation")
            .with(("sum_a".to_string(), AggregateColumn::sum("a")))
            .with_group_by_column("a")
            .input(table.clone())
            .build();
        let (dp_relation, dp_event) = reduce
            .differentially_private_group_by(epsilon, delta, 1)
            .unwrap()
            .into();
        dp_relation.display_dot().unwrap();
        assert_eq!(dp_event, DpEvent::epsilon_delta(epsilon, delta));
        assert_eq!(
            dp_relation.data_type(),
            DataType::structured([("a", DataType::integer_range(1..=10))])
        );
    }
}
