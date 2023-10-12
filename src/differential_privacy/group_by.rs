use crate::{
    builder::Ready,
    differential_privacy::{private_query, DPRelation, Error, PrivateQuery, Result},
    expr::{aggregate, Expr},
    namer,
    protection::{PEPReduce, PEPRelation},
    relation::{Relation, Variant as _},
};

pub const COUNT_DISTINCT_PE_ID: &str = "_COUNT_DISTINCT_PE_ID_";

impl PEPReduce {
    /// Returns a `DPRelation` whose:
    ///     - `relation` outputs all the DP values of the `self` grouping keys
    ///     - `private_query` stores the invoked DP mechanisms
    pub fn dp_compile_group_by(&self, epsilon: f64, delta: f64) -> Result<DPRelation> {
        let protected_entity_id = self.protected_entity_id();
        let protected_entity_weight = self.protected_entity_weight();

        let grouping_cols = self.group_by_names();
        let input_relation = self.inputs()[0].clone();
        if grouping_cols.len() > 1 {
            PEPRelation::try_from(input_relation.filter_fields(|f| {
                grouping_cols.contains(&f)
                    || f == protected_entity_id
                    || f == protected_entity_weight
            }))?
            .dp_values(epsilon, delta)
        } else {
            Err(Error::GroupingKeysError(format!(
                "Cannot group by {protected_entity_id}"
            )))
        }
    }
}

impl PEPRelation {
    /// Returns a `DPRelation` whose:
    ///     - `relation` outputs the (epsilon, delta)-DP values
    /// (found by tau-thresholding) of the fields of the current `Relation`
    ///     - `private_query` stores the invoked DP mechanisms
    fn tau_thresholding_values(self, epsilon: f64, delta: f64) -> Result<DPRelation> {
        // compute COUNT (DISTINCT PE_ID) GROUP BY columns
        let columns: Vec<String> = self
            .schema()
            .iter()
            .cloned()
            .filter_map(|f| {
                if f.name() == self.protected_entity_id()
                    || f.name() == self.protected_entity_weight()
                {
                    None
                } else {
                    Some(f.name().to_string())
                }
            })
            .collect();
        let columns: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();
        let aggregates = vec![(COUNT_DISTINCT_PE_ID, aggregate::Aggregate::Count)];
        let peid = self.protected_entity_id().to_string();
        let rel =
            Relation::from(self).distinct_aggregates(peid.as_ref(), columns.clone(), aggregates);

        // Apply noise
        let name_sigmas = vec![(
            COUNT_DISTINCT_PE_ID,
            private_query::gaussian_noise(epsilon, delta, 1.),
        )];
        let rel = rel.add_gaussian_noise(name_sigmas);

        // Returns a `Relation::Map` with the right field names and with `COUNT(DISTINCT PE_ID) > tau`
        let tau = private_query::gaussian_tau(epsilon, delta, 1.0);
        let filter_column = [(COUNT_DISTINCT_PE_ID, (Some(tau.into()), None, vec![]))]
            .into_iter()
            .collect();
        let relation = rel
            .filter_columns(filter_column)
            .filter_fields(|f| columns.contains(&f));
        Ok(DPRelation::new(
            relation,
            PrivateQuery::EpsilonDelta(epsilon, delta),
        ))
    }

    /// Returns a DPRelation whose:
    ///     - first field is a Relation whose outputs are
    /// (epsilon, delta)-DP values of grouping keys of the current PEPRelation,
    ///     - second field is a PrivateQuery corresponding the used mechanisms
    /// The (epsilon, delta)-DP values are found by:
    ///     - Using the propagated public values of the grouping columns when they exist
    ///     - Applying tau-thresholding mechanism with the (epsilon, delta) privacy parameters for t
    /// he columns that do not have public values
    pub fn dp_values(self, epsilon: f64, delta: f64) -> Result<DPRelation> {
        let public_columns: Vec<String> = self
            .schema()
            .iter()
            .filter_map(|f| {
                (f.name() != self.protected_entity_id()
                    && f.name() != self.protected_entity_weight()
                    && f.all_values())
                .then_some(f.name().to_string())
            })
            .collect();
        let all_columns_are_public = public_columns.len() == self.schema().len() - 2;

        if public_columns.is_empty() {
            let name = namer::name_from_content("FILTER_", &self.name());
            self.with_name(name)?
                .tau_thresholding_values(epsilon, delta)
        } else if all_columns_are_public {
            Ok(DPRelation::new(
                self.with_public_values(&public_columns)?,
                PrivateQuery::null(),
            ))
        } else {
            let (relation, private_query) = self
                .clone()
                .with_name(namer::name_from_content("FILTER_", &self.name()))?
                .filter_fields(|f| !public_columns.contains(&f.to_string()))?
                .tau_thresholding_values(epsilon, delta)?
                .into();
            let relation = self
                .with_public_values(&public_columns)?
                .cross_join(relation)?;
            Ok(DPRelation::new(relation, private_query))
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

    pub fn join_with_grouping_values(self, grouping_values: Relation) -> Result<Relation> {
        let on: Vec<Expr> = grouping_values
            .schema()
            .iter()
            .map(|f| {
                Expr::eq(
                    Expr::qcol(self.name().to_string(), f.name().to_string()),
                    Expr::qcol(grouping_values.name().to_string(), f.name().to_string()),
                )
            })
            .collect();

        let names = self
            .schema()
            .iter()
            .map(|f| f.name().to_string())
            .collect::<Vec<_>>();

        let join_rel: Relation = Relation::join()
            .right(self)
            .right_names(names.clone())
            .left(grouping_values)
            .inner()
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
        data_type::{DataType, DataTyped},
        display::Dot,
        expr::AggregateColumn,
        hierarchy::Hierarchy,
        io::{postgresql, Database},
        protection::{PE_ID, PE_WEIGHT},
        relation::Schema,
    };
    use std::sync::Arc;

    #[test]
    fn test_tau_thresholding_values() {
        let table: Relation = Relation::table()
            .name("table")
            .schema(
                Schema::builder()
                    .with(("a", DataType::integer_range(1..=10)))
                    .with(("b", DataType::integer_values([1, 2, 5, 6, 7, 8])))
                    .with(("c", DataType::integer_range(5..=20)))
                    .with((PE_ID, DataType::integer_range(1..=100)))
                    .with((PE_WEIGHT, DataType::float_interval(0., 1.)))
                    .build(),
            )
            .build();
        let protected_table = PEPRelation(table);

        let (rel, pq) = protected_table
            .clone()
            .tau_thresholding_values(1., 0.003)
            .unwrap()
            .into();
        //rel.display_dot();
        assert_eq!(
            rel.data_type(),
            DataType::structured([
                ("a", DataType::integer_range(1..=10)),
                ("b", DataType::integer_values([1, 2, 5, 6, 7, 8])),
                ("c", DataType::integer_range(5..=20))
            ])
        );
        assert_eq!(pq, PrivateQuery::EpsilonDelta(1., 0.003))
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
                    .with((PE_ID, DataType::integer_range(1..=100)))
                    .with((PE_WEIGHT, DataType::float_interval(0., 1.)))
                    .build(),
            )
            .build();
        let protected_table = PEPRelation(table);
        let (rel, pq) = protected_table.dp_values(1., 0.003).unwrap().into();
        matches!(rel, Relation::Join(_));
        //rel.display_dot();
        assert_eq!(
            rel.data_type(),
            DataType::structured([
                ("a", DataType::integer_values([1, 2, 4, 6])),
                ("b", DataType::float_values([1.2, 4.6, 7.8]))
            ])
        );
        assert_eq!(pq, PrivateQuery::null());

        // Only tau-thresholding values
        let table: Relation = Relation::table()
            .name("table")
            .schema(
                Schema::builder()
                    .with(("a", DataType::integer_range(1..=10)))
                    .with(("b", DataType::float_range(5.4..=20.)))
                    .with((PE_ID, DataType::integer_range(1..=100)))
                    .with((PE_WEIGHT, DataType::float_interval(0., 1.)))
                    .build(),
            )
            .build();
        let protected_table = PEPRelation(table);
        let (rel, pq) = protected_table.dp_values(1., 0.003).unwrap().into();
        matches!(rel, Relation::Map(_));
        //rel.display_dot();
        assert_eq!(
            rel.data_type(),
            DataType::structured([
                ("a", DataType::integer_range(1..=10)),
                ("b", DataType::float_range(5.4..=20.))
            ])
        );
        assert_eq!(pq, PrivateQuery::EpsilonDelta(1., 0.003));

        // Both possible and tau-thresholding values
        let table: Relation = Relation::table()
            .name("table")
            .schema(
                Schema::builder()
                    .with(("a", DataType::integer_range(1..=5)))
                    .with(("b", DataType::integer_values([1, 2, 5, 6, 7, 8])))
                    .with(("c", DataType::integer_range(5..=20)))
                    .with((PE_ID, DataType::integer_range(1..=100)))
                    .with((PE_WEIGHT, DataType::float_interval(0., 1.)))
                    .build(),
            )
            .build();
        let protected_table = PEPRelation(table);
        let (rel, pq) = protected_table.dp_values(1., 0.003).unwrap().into();
        matches!(rel, Relation::Join(_));
        rel.display_dot();
        assert_eq!(
            rel.data_type(),
            DataType::structured([
                ("b", DataType::integer_values([1, 2, 5, 6, 7, 8])),
                ("a", DataType::integer_range(1..=5)),
                ("c", DataType::integer_range(5..=20))
            ])
        );
        assert_eq!(pq, PrivateQuery::EpsilonDelta(1., 0.003));
    }

    // #[test]
    // fn test_dp_compile_group_by_simple() {
    //     let table: Relation = Relation::table()
    //         .name("table")
    //         .schema(
    //             Schema::builder()
    //                 .with(("a", DataType::integer_range(1..=10)))
    //                 .with(("b", DataType::integer_values([1, 2, 5, 6, 7, 8])))
    //                 .with(("c", DataType::integer_range(5..=20)))
    //                 .with(("id", DataType::integer_range(1..=100)))
    //                 .build(),
    //         )
    //         .build();
    //     let relations: Hierarchy<Arc<Relation>> = vec![("table", Arc::new(table.clone()))]
    //         .into_iter()
    //         .collect();
    //     let (epsilon, delta) = (1., 1e-3);

    //     // Without GROUPBY: Error
    //     let relation: Relation = Relation::reduce()
    //         .name("reduce_relation")
    //         .with(("sum_a".to_string(), AggregateColumn::sum("a")))
    //         .input(table.clone())
    //         .build();
    //     //relation.display_dot().unwrap();
    //     let pep_relation = Relation::from(
    //         relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]),
    //     );
    //     //pep_relation.display_dot().unwrap();
    //     if let Relation::Reduce(reduce) = pep_relation {
    //         let pep_reduce = PEPReduce::try_from(reduce).unwrap();
    //         assert!(pep_reduce.dp_compile_group_by(epsilon, delta).is_err());
    //     } else {
    //         panic!()
    //     }

    //     // With GROUPBY. Only one column with possible values
    //     let relation: Relation = Relation::reduce()
    //         .name("reduce_relation")
    //         .with(("sum_a".to_string(), AggregateColumn::sum("a")))
    //         .group_by(expr!(b))
    //         .input(table.clone())
    //         .build();
    //     let pep_relation = Relation::from(
    //         relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]),
    //     );
    //     //pep_relation.display_dot().unwrap();
    //     if let Relation::Reduce(reduce) = pep_relation {
    //         let pep_reduce = PEPReduce::try_from(reduce).unwrap();
    //         let (dp_relation, private_query) = pep_reduce
    //             .dp_compile_group_by(epsilon, delta)
    //             .unwrap()
    //             .into();
    //         assert_eq!(private_query, PrivateQuery::null());
    //         assert_eq!(
    //             dp_relation.data_type(),
    //             DataType::structured([("b", DataType::integer_values([1, 2, 5, 6, 7, 8]))])
    //         );
    //     } else {
    //         panic!()
    //     }

    //     // With GROUPBY. Only one column with tau-thresholding values
    //     let relation: Relation = Relation::reduce()
    //         .name("reduce_relation")
    //         .with(("sum_a".to_string(), AggregateColumn::sum("a")))
    //         .group_by(expr!(c))
    //         .input(table.clone())
    //         .build();
    //     let pep_relation = Relation::from(
    //         relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]),
    //     );
    //     //pep_relation.display_dot().unwrap();
    //     if let Relation::Reduce(reduce) = pep_relation {
    //         let pep_reduce = PEPReduce::try_from(reduce).unwrap();
    //         let (dp_relation, private_query) = pep_reduce
    //             .dp_compile_group_by(epsilon, delta)
    //             .unwrap()
    //             .into();
    //         assert_eq!(private_query, PrivateQuery::EpsilonDelta(epsilon, delta));
    //         assert_eq!(
    //             dp_relation.data_type(),
    //             DataType::structured([("c", DataType::integer_range(5..=20)),])
    //         );
    //     } else {
    //         panic!()
    //     }

    //     // With GROUPBY. Both tau-thresholding and possible values
    //     let relation: Relation = Relation::reduce()
    //         .name("reduce_relation")
    //         .with(("sum_a".to_string(), AggregateColumn::sum("a")))
    //         .group_by(expr!(c))
    //         .group_by(expr!(b))
    //         .input(table.clone())
    //         .build();
    //     let pep_relation = Relation::from(
    //         relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]),
    //     );
    //     //pep_relation.display_dot().unwrap();
    //     if let Relation::Reduce(reduce) = pep_relation {
    //         let pep_reduce = PEPReduce::try_from(reduce).unwrap();
    //         let (dp_relation, private_query) = pep_reduce
    //             .dp_compile_group_by(epsilon, delta)
    //             .unwrap()
    //             .into();
    //         assert_eq!(private_query, PrivateQuery::EpsilonDelta(epsilon, delta));
    //         assert_eq!(
    //             dp_relation.data_type(),
    //             DataType::structured([
    //                 ("c", DataType::integer_range(5..=20)),
    //                 ("b", DataType::integer_values([1, 2, 5, 6, 7, 8]))
    //             ])
    //         );
    //     } else {
    //         panic!()
    //     }
    // }

    // #[test]
    // fn test_dp_compile_group_by_input_map() {
    //     let table: Relation = Relation::table()
    //         .name("table")
    //         .schema(
    //             Schema::builder()
    //                 .with(("a", DataType::integer_range(1..=10)))
    //                 .with(("b", DataType::integer_values([1, 2, 5, 6, 7, 8])))
    //                 .with(("c", DataType::float_interval(1., 2.)))
    //                 .with(("id", DataType::integer_range(1..=100)))
    //                 .build(),
    //         )
    //         .build();
    //     let relations: Hierarchy<Arc<Relation>> = vec![("table", Arc::new(table.clone()))]
    //         .into_iter()
    //         .collect();
    //     let (epsilon, delta) = (1., 1e-3);

    //     let input: Relation = Relation::map()
    //         .name("map_relation")
    //         .with(("a", expr!(a)))
    //         .with(("twice_b", expr!(2 * b)))
    //         .with(("c", expr!(3 * c)))
    //         .input(table.clone())
    //         .build();
    //     let relation: Relation = Relation::reduce()
    //         .name("reduce_relation")
    //         .with(("sum_a".to_string(), AggregateColumn::sum("a")))
    //         .group_by(expr!(c))
    //         .group_by(expr!(twice_b))
    //         .input(input)
    //         .build();
    //     let pep_relation = Relation::from(
    //         relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]),
    //     );
    //     pep_relation.display_dot().unwrap();
    //     if let Relation::Reduce(reduce) = pep_relation {
    //         let pep_reduce = PEPReduce::try_from(reduce).unwrap();
    //         let (dp_relation, private_query) = pep_reduce
    //             .dp_compile_group_by(epsilon, delta)
    //             .unwrap()
    //             .into();
    //         dp_relation.display_dot().unwrap();
    //         assert_eq!(private_query, PrivateQuery::EpsilonDelta(epsilon, delta));
    //         assert_eq!(
    //             dp_relation.data_type(),
    //             DataType::structured([
    //                 ("twice_b", DataType::integer_values([2, 4, 10, 12, 14, 16])),
    //                 ("c", DataType::float_interval(3., 6.0)),
    //             ])
    //         );
    //     } else {
    //         panic!()
    //     }

    //     // WHERE IN LIST
    //     let input: Relation = Relation::map()
    //         .name("map_relation")
    //         .with(("a", expr!(a)))
    //         .with(("twice_b", expr!(2 * b)))
    //         .with(("c", expr!(3 * c)))
    //         .filter(Expr::in_list(Expr::col("c"), Expr::list(vec![1., 1.5])))
    //         .input(table.clone())
    //         .build();
    //     let relation: Relation = Relation::reduce()
    //         .name("reduce_relation")
    //         .with(("sum_a".to_string(), AggregateColumn::sum("a")))
    //         .group_by(expr!(c))
    //         .group_by(expr!(twice_b))
    //         .input(input)
    //         .build();
    //     let pep_relation = Relation::from(
    //         relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]),
    //     );
    //     pep_relation.display_dot().unwrap();
    //     if let Relation::Reduce(reduce) = pep_relation {
    //         let pep_reduce = PEPReduce::try_from(reduce).unwrap();
    //         let (dp_relation, private_query) = pep_reduce
    //             .dp_compile_group_by(epsilon, delta)
    //             .unwrap()
    //             .into();
    //         dp_relation.display_dot().unwrap();
    //         assert_eq!(private_query, PrivateQuery::null());
    //         assert_eq!(
    //             dp_relation.data_type(),
    //             DataType::structured([
    //                 ("twice_b", DataType::integer_values([2, 4, 10, 12, 14, 16])),
    //                 ("c", DataType::float_values([3., 4.5])),
    //             ])
    //         );
    //     } else {
    //         panic!()
    //     }
    // }

    // #[test]
    // fn test_dp_compile_complex() {
    //     let mut database = postgresql::test_database();
    //     let relations = database.relations();
    //     let (epsilon, delta) = (1., 1e-3);

    //     let join: Relation = Relation::join()
    //         .name("join_relation")
    //         .left(relations["table_1"].clone())
    //         .right(relations["table_2"].clone())
    //         .inner()
    //         .on(Expr::eq(Expr::col("a"), Expr::col("x")))
    //         .left_names(vec!["a", "b", "c", "d"])
    //         .right_names(vec!["x", "y", "z"])
    //         .build();

    //     let map: Relation = Relation::map()
    //         .name("map_relation")
    //         .with(("b", expr!(b)))
    //         .with(("d", expr!(2 * d)))
    //         .with(("my_z", expr!(z)))
    //         .input(join)
    //         .build();

    //     let relation: Relation = Relation::reduce()
    //         .name("reduce_relation")
    //         .with(("sum_b".to_string(), AggregateColumn::sum("b")))
    //         .group_by(expr!(d))
    //         .group_by(expr!(my_z))
    //         .input(map)
    //         .build();

    //     let pep_relation = Relation::from(relation.force_protect_from_field_paths(
    //         &relations,
    //         vec![
    //             ("table_1", vec![], "c"),
    //             ("table_2", vec![("x", "table_1", "a")], "c"),
    //         ],
    //     ));
    //     pep_relation.display_dot().unwrap();
    //     if let Relation::Reduce(reduce) = pep_relation {
    //         let pep_reduce = PEPReduce::try_from(reduce).unwrap();
    //         let (dp_relation, private_query) = pep_reduce
    //             .dp_compile_group_by(epsilon, delta)
    //             .unwrap()
    //             .into();
    //         dp_relation.display_dot().unwrap();
    //         assert_eq!(private_query, PrivateQuery::EpsilonDelta(epsilon, delta));
    //         matches!(dp_relation, Relation::Join(_));
    //         matches!(dp_relation.inputs()[0], Relation::Values(_));
    //         matches!(dp_relation.inputs()[1], Relation::Map(_));
    //         assert_eq!(
    //             dp_relation.inputs()[0].data_type()["my_z"],
    //             DataType::text_values(["Foo".into(), "Bar".into()])
    //         );
    //         assert_eq!(
    //             dp_relation.inputs()[1].data_type()["d"],
    //             DataType::integer_interval(0, 20)
    //         );
    //         let dp_query = ast::Query::from(&dp_relation);
    //         database.query(&dp_query.to_string()).unwrap();
    //     } else {
    //         panic!()
    //     }
    // }
}
