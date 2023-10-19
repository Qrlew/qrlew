use crate::{
    builder::With,
    data_type::DataTyped,
    differential_privacy::private_query::PrivateQuery,
    differential_privacy::{private_query, DPRelation, Result},
    expr::{aggregate, AggregateColumn, Expr},
    relation::{field::Field, Map, Reduce, Relation, Variant as _},
    DataType, Ready, protection::PEPRelation, display::Dot,
};
use std::{
    cmp,
    collections::{HashMap, HashSet},
};

impl Field {
    pub fn clipping_value(self, multiplicity: i64) -> f64 {
        match self.data_type() {
            DataType::Float(f) => {
                let min = f.min().unwrap().abs();
                let max = f.max().unwrap().abs();
                (min + max + (min - max).abs()) / 2. * multiplicity as f64
            }
            DataType::Integer(i) => {
                let min = i.min().unwrap().abs();
                let max = i.max().unwrap().abs();
                (cmp::max(min, max) * multiplicity) as f64
            }
            _ => todo!(),
        }
    }
}

impl Relation {
    fn gaussian_mechanism(self, epsilon: f64, delta: f64, bounds: Vec<(&str, f64)>) -> DPRelation {
        let noise_multipliers = bounds
            .into_iter()
            .map(|(name, bound)| (name, private_query::gaussian_noise(epsilon, delta, bound)))
            .collect::<Vec<_>>();
        let private_query = noise_multipliers
            .iter()
            .map(|(_, n)| PrivateQuery::Gaussian(*n))
            .collect::<Vec<_>>()
            .into();
        DPRelation::new(self.add_gaussian_noise(noise_multipliers), private_query)
    }
}

impl Reduce {
    /// DP compile the sums
    fn differential_privacy_sums(
        self,
        epsilon: f64,
        delta: f64,
    ) -> Result<DPRelation> {
        // Collect groups
        let mut input_values_bound: Vec<(&str, f64)> = vec![];
        let mut names: HashMap<&str, &str> = HashMap::new();
        // Collect names, sums and bounds
        for (name, aggregate) in self.named_aggregates() {
            // Get value name
            let input_name = aggregate.column_name()?;
            names.insert(input_name, name);
            if aggregate.aggregate() == &aggregate::Aggregate::Sum {
                // add aggregate
                let input_data_type = self.input().schema()[input_name].data_type();
                let absolute_bound = input_data_type.absolute_upper_bound().unwrap_or(1.0);
                input_values_bound.push((input_name, absolute_bound));
            }
        }

        // Clip the relation
        let protected_input = PEPRelation::try_from(self.input().clone())?;
        let clipped_relation = self.input().clone().l2_clipped_sums(
            protected_input.protected_entity_id(),
            self.group_by_names(),
            input_values_bound.iter().cloned().collect(),
        );
        //clipped_relation.display_dot().unwrap();
        println!("SCHEMA = {}", clipped_relation.schema());

        let (dp_clipped_relation, private_query) = clipped_relation
            .gaussian_mechanism(epsilon, delta, input_values_bound)
            .into();
        println!("SCHEMA = {}", self.schema());
        println!("SCHEMA = {}", dp_clipped_relation.schema());
        let dp_clipped_relation =
            dp_clipped_relation
            .filter_fields(|n| names.get(n).is_some())
            .rename_fields(|n, _| names.get(n).unwrap_or(&n).to_string())
            //.filter_fields(|n| self.schema().field(n).is_ok());
            ;

        Ok(DPRelation::new(dp_clipped_relation, private_query))
    }
}

impl Reduce {
    /// Rewrite aggregations as sums and compile sums
    pub fn differential_privacy_aggregates(self, epsilon: f64, delta: f64) -> Result<DPRelation> {
        let mut output = Map::builder();
        let mut sums = Reduce::builder();
        // Add aggregate colums
        Relation::from(self.clone()).display_dot();
        for (name, aggregate) in self.named_aggregates().into_iter() {
            match aggregate.aggregate() {
                aggregate::Aggregate::First => {
                    sums = sums.with((
                        aggregate.column_name()?,
                        AggregateColumn::col(aggregate.column_name()?),
                    ));
                    output = output.with((name, Expr::col(aggregate.column_name()?)));
                }
                aggregate::Aggregate::Mean => {
                    let sum_col = &format!("_SUM_{}", aggregate.column_name()?);
                    let count_col = &format!("_COUNT_{}", aggregate.column_name()?);
                    sums = sums
                        .with((count_col, Expr::sum(Expr::val(1.))))
                        .with((sum_col, Expr::sum(Expr::col(aggregate.column_name()?))));
                    output = output.with((
                        name,
                        Expr::divide(
                            Expr::col(sum_col),
                            Expr::greatest(Expr::val(1.), Expr::col(count_col)),
                        ),
                    ))
                }
                aggregate::Aggregate::Count => {
                    let count_col = &format!("_COUNT_{}", aggregate.column_name()?);
                    sums = sums.with((count_col, Expr::sum(Expr::val(1.))));
                    output = output.with((name, Expr::col(count_col)));
                }
                aggregate::Aggregate::Sum => {
                    let sum_col = &format!("_SUM_{}", aggregate.column_name()?);
                    sums = sums.with((sum_col, Expr::sum(Expr::col(aggregate.column_name()?))));
                    output = output.with((name, Expr::col(sum_col)));
                }
                aggregate::Aggregate::Std => todo!(),
                aggregate::Aggregate::Var => todo!(),
                _ => (),
            }
        }
        sums = sums.group_by_iter(self.group_by().iter().cloned());

        let sums: Reduce = sums.input(self.input().clone()).build();
        Relation::from(sums.clone()).display_dot();
        let dp_sums =
            sums.differential_privacy_sums(epsilon, delta)?;
        Ok(DPRelation::new(
            output.input(dp_sums.relation().clone()).build(),
            dp_sums.private_query().clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ast,
        builder::With,
        data_type::Variant,
        display::Dot,
        hierarchy::Hierarchy,
        io::{postgresql, Database},
        relation::Schema,
        sql::parse,
        Relation,
        protection::{Protection, Strategy}
    };
    use std::ops::Deref;

    #[test]
    fn test_table_with_noise() {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        // CReate a relation to add noise to
        let relation = Relation::try_from(
            parse("SELECT sum(price) FROM item_table GROUP BY order_id")
                .unwrap()
                .with(&relations),
        )
        .unwrap();
        println!("Schema = {}", relation.schema());
        relation.display_dot().unwrap();
        // Add noise directly
        for row in database
            .query("SELECT random(), sum(price) FROM item_table GROUP BY order_id")
            .unwrap()
        {
            println!("Row = {row}");
        }
    }

    #[test]
    fn test_differential_privacy_sums_no_group_by() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        let table = relations
            .get(&["item_table".to_string()])
            .unwrap()
            .deref()
            .clone();
        let (epsilon, delta) = (1., 1e-3);

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
            pep_table.deref().clone().into()
        );
        let relation = Relation::from(reduce.clone());
        relation.display_dot().unwrap();

        let dp_relation = reduce
            .differential_privacy_sums(
                epsilon,
                delta,
            ).unwrap();
        dp_relation.display_dot().unwrap();
        matches!(dp_relation.data_type()["sum_price"], DataType::Float(_));

        let query: &str = &ast::Query::from(&relation).to_string();
        println!("{query}");
        _ = database
            .query(query)
            .unwrap()
            .iter()
            .map(ToString::to_string);
    }

    #[test]
    fn test_differential_privacy_sums_with_group_by() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        let table = relations
            .get(&["item_table".to_string()])
            .unwrap()
            .deref()
            .clone();
        let (epsilon, delta) = (1., 1e-3);

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
            vec![("my_sum_price".to_string(), AggregateColumn::sum("price"))],
            vec![Expr::col("item")],
            pep_table.deref().clone().into()
        );
        let relation = Relation::from(reduce.clone());
        relation.display_dot().unwrap();

        let dp_relation = reduce
            .differential_privacy_sums(
                epsilon,
                delta,
            ).unwrap();
        dp_relation.display_dot().unwrap();
        assert_eq!(dp_relation.schema().len(), 1);
        matches!(dp_relation.data_type()["my_sum_price"], DataType::Float(_));

        let query: &str = &ast::Query::from(&relation).to_string();
        println!("{query}");
        _ = database
            .query(query)
            .unwrap()
            .iter()
            .map(ToString::to_string);
    }

    #[test]
    fn test_differential_privacy_aggregates() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        let table = relations
            .get(&["item_table".to_string()])
            .unwrap()
            .deref()
            .clone();
        let (epsilon, delta) = (1., 1e-3);

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
            vec![
                // ("count_price".to_string(), AggregateColumn::count("price")),
                ("sum_price".to_string(), AggregateColumn::sum("price")),
                // ("avg_price".to_string(), AggregateColumn::mean("price")),
                // ("avg_order_id".to_string(), AggregateColumn::mean("order_id")),
            ],
            vec![],
            pep_table.deref().clone().into()
        );
        let relation = Relation::from(reduce.clone());
        relation.display_dot().unwrap();

        let dp_relation = reduce
            .differential_privacy_aggregates(
                epsilon,
                delta,
            ).unwrap();
        dp_relation.display_dot().unwrap();
        //matches!(dp_relation.data_type()["sum_price"], DataType::Float(_));

        let query: &str = &ast::Query::from(&relation).to_string();
        println!("{query}");
        _ = database
            .query(query)
            .unwrap()
            .iter()
            .map(ToString::to_string);
    }

    // #[test]
    // fn test_dp_compile_aggregates_with_avg() {
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

    //     let relation: Relation = Relation::reduce()
    //         .name("reduce_relation")
    //         .with(("avg_a".to_string(), AggregateColumn::mean("a")))
    //         .input(table.clone())
    //         .build();
    //     //relation.display_dot().unwrap();
    //     let pep_relation = Relation::from(
    //         relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]),
    //     );
    //     //pep_relation.display_dot().unwrap();
    //     if let Relation::Reduce(reduce) = pep_relation {
    //         let pep_reduce = PEPReduce::try_from(reduce).unwrap();
    //         let (dp_relation, private_query) = pep_reduce
    //             .dp_compile_aggregates(epsilon, delta)
    //             .unwrap()
    //             .into();
    //         dp_relation.display_dot().unwrap();
    //         assert_eq!(dp_relation.schema().len(), 1);
    //         matches!(dp_relation.data_type()["avg_a"], DataType::Float(_));
    //         assert_eq!(
    //             private_query,
    //             vec![
    //                 PrivateQuery::gaussian_privacy_pars(epsilon, delta, 1.),
    //                 PrivateQuery::gaussian_privacy_pars(epsilon, delta, 10.)
    //             ]
    //             .into()
    //         );
    //     } else {
    //         panic!()
    //     }
    // }

    // #[test]
    // fn test_dp_compile_aggregates_multi() {
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

    //     let relation: Relation = Relation::reduce()
    //         .name("reduce_relation")
    //         .with(("avg_a".to_string(), AggregateColumn::mean("a")))
    //         .with(("sum_a".to_string(), AggregateColumn::sum("a")))
    //         .with(("count_a".to_string(), AggregateColumn::count("a")))
    //         .with(("sum_b".to_string(), AggregateColumn::sum("b")))
    //         .input(table.clone())
    //         .build();
    //     //relation.display_dot().unwrap();
    //     let pep_relation = Relation::from(
    //         relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]),
    //     );
    //     //pep_relation.display_dot().unwrap();
    //     if let Relation::Reduce(reduce) = pep_relation {
    //         let pep_reduce = PEPReduce::try_from(reduce).unwrap();
    //         let (dp_relation, private_query) = pep_reduce
    //             .dp_compile_aggregates(epsilon, delta)
    //             .unwrap()
    //             .into();
    //         dp_relation.display_dot().unwrap();
    //         assert_eq!(dp_relation.schema().len(), 4);
    //         matches!(dp_relation.data_type()["avg_a"], DataType::Float(_));
    //         matches!(dp_relation.data_type()["sum_a"], DataType::Float(_));
    //         matches!(dp_relation.data_type()["count_a"], DataType::Float(_));
    //         matches!(dp_relation.data_type()["sum_b"], DataType::Float(_));
    //         assert_eq!(
    //             private_query,
    //             vec![
    //                 PrivateQuery::gaussian_privacy_pars(epsilon, delta, 1.),
    //                 PrivateQuery::gaussian_privacy_pars(epsilon, delta, 10.),
    //                 PrivateQuery::gaussian_privacy_pars(epsilon, delta, 8.),
    //             ]
    //             .into()
    //         );
    //     } else {
    //         panic!()
    //     }
    // }

    // #[test]
    // fn test_dp_compile_aggregates_map_input() {
    //     let table: Relation = Relation::table()
    //         .name("table")
    //         .schema(
    //             Schema::builder()
    //                 .with(("a", DataType::integer_range(1..=10)))
    //                 .with(("b", DataType::integer_values([1, 2, 5, 6, 7, 8])))
    //                 .with(("c", DataType::float_interval(-10., 2.)))
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
    //         .with(("sum_twice_b".to_string(), AggregateColumn::sum("twice_b")))
    //         .with(("sum_c".to_string(), AggregateColumn::sum("c")))
    //         .input(input)
    //         .build();
    //     let pep_relation = Relation::from(
    //         relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]),
    //     );
    //     pep_relation.display_dot().unwrap();
    //     if let Relation::Reduce(reduce) = pep_relation {
    //         let pep_reduce = PEPReduce::try_from(reduce).unwrap();
    //         let (dp_relation, private_query) = pep_reduce
    //             .dp_compile_aggregates(epsilon, delta)
    //             .unwrap()
    //             .into();
    //         dp_relation.display_dot().unwrap();
    //         assert_eq!(dp_relation.schema().len(), 3);
    //         matches!(dp_relation.data_type()["sum_a"], DataType::Float(_));
    //         matches!(dp_relation.data_type()["sum_twice_b"], DataType::Float(_));
    //         matches!(dp_relation.data_type()["sum_c"], DataType::Float(_));
    //         assert_eq!(
    //             private_query,
    //             vec![
    //                 PrivateQuery::gaussian_privacy_pars(epsilon, delta, 10.),
    //                 PrivateQuery::gaussian_privacy_pars(epsilon, delta, 16.),
    //                 PrivateQuery::gaussian_privacy_pars(epsilon, delta, 30.),
    //             ]
    //             .into()
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
    //         .with(("d", expr!(d)))
    //         .with(("twice_a", expr!(2 * a)))
    //         .with(("z", expr!(z)))
    //         .input(join)
    //         .build();

    //     let relation: Relation = Relation::reduce()
    //         .name("reduce_relation")
    //         .with(("sum_a".to_string(), AggregateColumn::sum("twice_a")))
    //         .with(("my_d", AggregateColumn::first("d")))
    //         .with(("avg_a".to_string(), AggregateColumn::mean("twice_a")))
    //         .with(("count_a".to_string(), AggregateColumn::count("twice_a")))
    //         .group_by(expr!(d))
    //         .group_by(expr!(z))
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
    //             .dp_compile_aggregates(epsilon, delta)
    //             .unwrap()
    //             .into();
    //         dp_relation.display_dot().unwrap();
    //         assert_eq!(
    //             private_query,
    //             vec![
    //                 PrivateQuery::gaussian_privacy_pars(epsilon, delta, 20.),
    //                 PrivateQuery::gaussian_privacy_pars(epsilon, delta, 1.),
    //             ]
    //             .into()
    //         );
    //         assert!(dp_relation.data_type().is_subset_of(&DataType::structured([
    //             ("sum_a", DataType::float()),
    //             ("my_d", DataType::integer_interval(0, 10)),
    //             ("avg_a", DataType::float()),
    //             ("count_a", DataType::float())
    //         ])));
    //         let dp_query = ast::Query::from(&dp_relation);
    //         database.query(&dp_query.to_string()).unwrap();
    //     } else {
    //         panic!()
    //     }
    // }
}
