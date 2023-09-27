use crate::{
    builder::With,
    data_type::DataTyped,
    differential_privacy::private_query::PrivateQuery,
    expr::{aggregate, AggregateColumn, Expr},
    protection::PEPReduce,
    relation::{field::Field, Map, Reduce, Relation, Variant as _},
    DataType, Ready,
    differential_privacy::{private_query, DPRelation, Result},

};
use std::{cmp, collections::{HashMap, HashSet}};


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
    fn dp_compile_sums(
        self,
        protected_entity_id: &str,
        protected_entity_weight: &str,
        epsilon: f64,
        delta: f64,
    ) -> Result<DPRelation> {
        // Collect groups
        let mut input_entities: Option<&str> = None;
        let mut input_groups: HashSet<&str> = self.group_by_names().into_iter().collect();
        let mut input_values_bound: Vec<(&str, f64)> = vec![];
        let mut names: HashMap<&str, &str> = HashMap::new();
        // Collect names, sums and bounds
        for (name, aggregate) in self.named_aggregates() {
            // Get value name
            let input_name = aggregate.column_name()?;
            names.insert(input_name, name);
            if name == protected_entity_id {
                // remove pe group
                input_groups.remove(&input_name);
                input_entities = Some(input_name);
            } else if aggregate.aggregate() == &aggregate::Aggregate::Sum
                && name != protected_entity_weight
            {
                // add aggregate
                let input_data_type = self.input().schema()[input_name].data_type();
                let absolute_bound = input_data_type.absolute_upper_bound().unwrap_or(1.0);
                input_values_bound.push((input_name, absolute_bound));
            }
        }

        // Clip the relation
        let clipped_relation = self.input().clone().l2_clipped_sums(
            input_entities.unwrap(),
            input_groups.into_iter().collect(),
            input_values_bound.iter().cloned().collect(),
        );

        let (dp_clipped_relation, private_query) = clipped_relation
            .gaussian_mechanism(epsilon, delta, input_values_bound)
            .into();
        let renamed_dp_clipped_relation =
            dp_clipped_relation.rename_fields(|n, _| names.get(n).unwrap_or(&n).to_string());
        Ok(DPRelation::new(renamed_dp_clipped_relation, private_query))
    }
}

impl PEPReduce {
    /// Rewrite aggregations as sums and compile sums
    pub fn dp_compile_aggregates(
        self,
        epsilon: f64,
        delta: f64,
    ) -> Result<DPRelation> {
        let protected_entity_id = self.protected_entity_id();
        let protected_entity_weight = self.protected_entity_weight();

        let mut output = Map::builder();
        let mut sums = Reduce::builder();
        // Add aggregate colums
        for (name, aggregate) in self.named_aggregates().into_iter() {
            match aggregate.aggregate() {
                aggregate::Aggregate::First => {
                    sums = sums.with((
                        aggregate.column_name()?,
                        AggregateColumn::col(aggregate.column_name()?),
                    ));
                    if name != protected_entity_id {
                        output = output.with((name, Expr::col(aggregate.column_name()?)));
                    }
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
                aggregate::Aggregate::Sum
                    if aggregate.column_name()? != protected_entity_weight =>
                {
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
        let dp_sums =
            sums.dp_compile_sums(protected_entity_id, protected_entity_weight, epsilon, delta)?;
        Ok(DPRelation::new(
            output.input(dp_sums.relation().clone()).build(),
            dp_sums.private_query().clone(),
        ))
    }
}



// impl PEPRelation {

//     /// Protects the results of the aggregations of the outest Reduce by adding gaussian noise scaled by
//     /// their sensitivity and the privacy parameters `epsilon`and `delta`
//     pub fn dp_compile_aggregates(self, epsilon: f64, delta: f64) -> Result<DPRelation> {
//         let protected_entity_id = self.protected_entity_id().to_string();
//         let protected_entity_weight = self.protected_entity_weight().to_string();

//         // Return a DP relation
//         let (dp_relation, private_query) = match Relation::from(self) {
//             Relation::Map(map) => {
//                 let dp_input = PEPRelation::try_from(map.input().clone())?
//                     .dp_compile_aggregates(epsilon, delta)?;
//                 let relation = Map::builder()
//                     .filter_fields_with(map, |f| {
//                         f != protected_entity_id.as_str() && f != protected_entity_weight.as_str()
//                     })
//                     .input(dp_input.relation().clone())
//                     .build();
//                 Ok(DPRelation::new(relation, dp_input.private_query().clone()))
//             }
//             Relation::Reduce(reduce) => reduce.dp_compile_aggregates(
//                 &protected_entity_id,
//                 &protected_entity_weight,
//                 epsilon,
//                 delta,
//             ),
//             Relation::Table(_) => todo!(),
//             Relation::Join(j) => {
//                 let (left_dp_relation, left_private_query) =
//                     PEPRelation::try_from(j.inputs()[0].clone())?
//                         .dp_compile_aggregates(epsilon, delta)?
//                         .into();
//                 let (right_dp_relation, right_private_query) =
//                     PEPRelation::try_from(j.inputs()[1].clone())?
//                         .dp_compile_aggregates(epsilon, delta)?
//                         .into();
//                 let relation: Relation = Join::builder()
//                     .left(left_dp_relation)
//                     .right(right_dp_relation)
//                     .with(j.clone())
//                     .build();
//                 Ok(DPRelation::new(
//                     relation,
//                     vec![left_private_query, right_private_query].into(),
//                 ))
//             }
//             Relation::Set(_) => todo!(),
//             Relation::Values(_) => todo!(),
//         }?
//         .into();

//         Ok(DPRelation::new(dp_relation, private_query))
//     }

// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        builder::With,
        display::Dot,
        io::{postgresql, Database},
        sql::parse,
        Relation,
        hierarchy::Hierarchy,
        relation::Schema,
    };
    use std::rc::Rc;

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
    fn test_dp_compile_aggregates_with_sum() {
        let table: Relation = Relation::table()
            .name("table")
            .schema(
                Schema::builder()
                    .with(("a", DataType::integer_range(1..=10)))
                    .with(("b", DataType::integer_values([1, 2, 5, 6, 7, 8])))
                    .with(("c", DataType::integer_range(5..=20)))
                    .with(("id", DataType::integer_range(1..=100)))
                    .build(),
            )
            .build();
        let relations: Hierarchy<Rc<Relation>> = vec![("table", Rc::new(table.clone()))]
            .into_iter()
            .collect();
        let (epsilon, delta) = (1., 1e-3);

        let relation: Relation = Relation::reduce()
            .name("reduce_relation")
            .group_by(expr!(b))
            .with(("sum_a".to_string(), AggregateColumn::sum("a")))
            .with(("b".to_string(), AggregateColumn::first("b")))
            .input(table.clone())
            .build();
        //relation.display_dot().unwrap();
        let pep_relation = Relation::from(relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]));
        //pep_relation.display_dot().unwrap();
        if let Relation::Reduce(reduce) = pep_relation {
            let pep_reduce = PEPReduce::try_from(reduce).unwrap();
            let (dp_relation, private_query) = pep_reduce.dp_compile_aggregates(epsilon, delta).unwrap().into();
            dp_relation.display_dot().unwrap();
            assert_eq!(dp_relation.schema().len(), 2);
            matches!(dp_relation.data_type()["sum_a"], DataType::Float(_));
            assert_eq!(dp_relation.data_type()["b"], DataType::integer_values([1, 2, 5, 6, 7, 8]));
            assert_eq!(private_query, PrivateQuery::gaussian_privacy_pars(epsilon, delta, 10.));
        } else {
            panic!()
        }
    }

    #[test]
    fn test_dp_compile_aggregates_with_count() {
        let table: Relation = Relation::table()
            .name("table")
            .schema(
                Schema::builder()
                    .with(("a", DataType::integer_range(1..=10)))
                    .with(("b", DataType::integer_values([1, 2, 5, 6, 7, 8])))
                    .with(("c", DataType::integer_range(5..=20)))
                    .with(("id", DataType::integer_range(1..=100)))
                    .build(),
            )
            .build();
        let relations: Hierarchy<Rc<Relation>> = vec![("table", Rc::new(table.clone()))]
            .into_iter()
            .collect();
        let (epsilon, delta) = (1., 1e-3);

        let relation: Relation = Relation::reduce()
            .name("reduce_relation")
            .with(("count_a".to_string(), AggregateColumn::count("a")))
            .group_by(expr!(b))
            .input(table.clone())
            .build();
        //relation.display_dot().unwrap();
        let pep_relation = Relation::from(relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]));
        //pep_relation.display_dot().unwrap();
        if let Relation::Reduce(reduce) = pep_relation {
            let pep_reduce = PEPReduce::try_from(reduce).unwrap();
            let (dp_relation, private_query) = pep_reduce.dp_compile_aggregates(epsilon, delta).unwrap().into();
            dp_relation.display_dot().unwrap();
            assert_eq!(dp_relation.schema().len(), 1);
            matches!(dp_relation.data_type()["count_a"], DataType::Float(_));
            assert_eq!(private_query, PrivateQuery::gaussian_privacy_pars(epsilon, delta, 1.));
        } else {
            panic!()
        }
    }

    #[test]
    fn test_dp_compile_aggregates_with_avg() {
        let table: Relation = Relation::table()
            .name("table")
            .schema(
                Schema::builder()
                    .with(("a", DataType::integer_range(1..=10)))
                    .with(("b", DataType::integer_values([1, 2, 5, 6, 7, 8])))
                    .with(("c", DataType::integer_range(5..=20)))
                    .with(("id", DataType::integer_range(1..=100)))
                    .build(),
            )
            .build();
        let relations: Hierarchy<Rc<Relation>> = vec![("table", Rc::new(table.clone()))]
            .into_iter()
            .collect();
        let (epsilon, delta) = (1., 1e-3);

        let relation: Relation = Relation::reduce()
            .name("reduce_relation")
            .with(("avg_a".to_string(), AggregateColumn::mean("a")))
            .input(table.clone())
            .build();
        //relation.display_dot().unwrap();
        let pep_relation = Relation::from(relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]));
        //pep_relation.display_dot().unwrap();
        if let Relation::Reduce(reduce) = pep_relation {
            let pep_reduce = PEPReduce::try_from(reduce).unwrap();
            let (dp_relation, private_query) = pep_reduce.dp_compile_aggregates(epsilon, delta).unwrap().into();
            dp_relation.display_dot().unwrap();
            assert_eq!(dp_relation.schema().len(), 1);
            matches!(dp_relation.data_type()["avg_a"], DataType::Float(_));
            assert_eq!(
                private_query,
                vec![PrivateQuery::gaussian_privacy_pars(epsilon, delta, 1.), PrivateQuery::gaussian_privacy_pars(epsilon, delta, 10.)].into()
            );
        } else {
            panic!()
        }
    }

    #[test]
    fn test_dp_compile_aggregates_multi() {
        let table: Relation = Relation::table()
            .name("table")
            .schema(
                Schema::builder()
                    .with(("a", DataType::integer_range(1..=10)))
                    .with(("b", DataType::integer_values([1, 2, 5, 6, 7, 8])))
                    .with(("c", DataType::integer_range(5..=20)))
                    .with(("id", DataType::integer_range(1..=100)))
                    .build(),
            )
            .build();
        let relations: Hierarchy<Rc<Relation>> = vec![("table", Rc::new(table.clone()))]
            .into_iter()
            .collect();
        let (epsilon, delta) = (1., 1e-3);

        let relation: Relation = Relation::reduce()
            .name("reduce_relation")
            .with(("avg_a".to_string(), AggregateColumn::mean("a")))
            .with(("sum_a".to_string(), AggregateColumn::sum("a")))
            .with(("count_a".to_string(), AggregateColumn::count("a")))
            .with(("sum_b".to_string(), AggregateColumn::sum("b")))
            .input(table.clone())
            .build();
        //relation.display_dot().unwrap();
        let pep_relation = Relation::from(relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]));
        //pep_relation.display_dot().unwrap();
        if let Relation::Reduce(reduce) = pep_relation {
            let pep_reduce = PEPReduce::try_from(reduce).unwrap();
            let (dp_relation, private_query) = pep_reduce.dp_compile_aggregates(epsilon, delta).unwrap().into();
            dp_relation.display_dot().unwrap();
            assert_eq!(dp_relation.schema().len(), 4);
            matches!(dp_relation.data_type()["avg_a"], DataType::Float(_));
            matches!(dp_relation.data_type()["sum_a"], DataType::Float(_));
            matches!(dp_relation.data_type()["count_a"], DataType::Float(_));
            matches!(dp_relation.data_type()["sum_b"], DataType::Float(_));
            assert_eq!(
                private_query,
                vec![
                    PrivateQuery::gaussian_privacy_pars(epsilon, delta, 1.),
                    PrivateQuery::gaussian_privacy_pars(epsilon, delta, 10.),
                    PrivateQuery::gaussian_privacy_pars(epsilon, delta, 8.),
                ].into()
            );
        } else {
            panic!()
        }
    }


    #[test]
    fn test_dp_compile_aggregates_map_input() {
        let table: Relation = Relation::table()
            .name("table")
            .schema(
                Schema::builder()
                    .with(("a", DataType::integer_range(1..=10)))
                    .with(("b", DataType::integer_values([1, 2, 5, 6, 7, 8])))
                    .with(("c", DataType::float_interval(-10., 2.)))
                    .with(("id", DataType::integer_range(1..=100)))
                    .build(),
            )
            .build();
        let relations: Hierarchy<Rc<Relation>> = vec![("table", Rc::new(table.clone()))]
            .into_iter()
            .collect();
        let (epsilon, delta) = (1., 1e-3);

        let input: Relation = Relation::map()
            .name("map_relation")
            .with(("a", expr!(a)))
            .with(("twice_b", expr!(2*b)))
            .with(("c", expr!(3*c)))
            .input(table.clone())
            .build();
        let relation: Relation = Relation::reduce()
            .name("reduce_relation")
            .with(("sum_a".to_string(), AggregateColumn::sum("a")))
            .with(("sum_twice_b".to_string(), AggregateColumn::sum("twice_b")))
            .with(("sum_c".to_string(), AggregateColumn::sum("c")))
            .input(input)
            .build();
        let pep_relation = Relation::from(relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]));
        pep_relation.display_dot().unwrap();
        if let Relation::Reduce(reduce) = pep_relation {
            let pep_reduce = PEPReduce::try_from(reduce).unwrap();
            let (dp_relation, private_query) = pep_reduce.dp_compile_aggregates(epsilon, delta).unwrap().into();
            dp_relation.display_dot().unwrap();
            assert_eq!(dp_relation.schema().len(), 3);
            matches!(dp_relation.data_type()["sum_a"], DataType::Float(_));
            matches!(dp_relation.data_type()["sum_twice_b"], DataType::Float(_));
            matches!(dp_relation.data_type()["sum_c"], DataType::Float(_));
            assert_eq!(
                private_query,
                vec![
                    PrivateQuery::gaussian_privacy_pars(epsilon, delta, 10.),
                    PrivateQuery::gaussian_privacy_pars(epsilon, delta, 16.),
                    PrivateQuery::gaussian_privacy_pars(epsilon, delta, 30.),
                ].into()
            );
        } else {
            panic!()
        }
    }

    #[test]
    fn test_dp_compile_aggregates_valid_queries() {
        todo!()
    }
}
