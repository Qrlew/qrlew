use crate::{
    builder::{With, WithIterator},
    data_type::DataTyped,
    differential_privacy::private_query::PrivateQuery,
    differential_privacy::{private_query, DPRelation, Error, Result},
    expr::{aggregate, AggregateColumn, Expr, Column, Identifier},
    privacy_unit_tracking::PUPRelation,
    relation::{field::Field, Map, Reduce, Relation, Variant},
    DataType, Ready, display::Dot,

};
use core::num;
use std::{cmp, collections::HashMap, ops::Deref};

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
    fn gaussian_mechanisms(self, epsilon: f64, delta: f64, bounds: Vec<(&str, f64)>) -> DPRelation {
        if epsilon>1. {
            // Cf. Theorem A.1. in (Dwork, Roth et al. 2014)
            log::warn!("Warning, epsilon>1 the gaussian mechanism applied will not be exactly epsilon,delta-DP!")
        }

        let number_of_agg = bounds.len() as f64;
        let (dp_relation, private_query) = if number_of_agg > 0. {
            let noise_multipliers = bounds
                .into_iter()
                .map(|(name, bound)| {
                    (
                        name,
                        private_query::gaussian_noise(
                            epsilon / number_of_agg,
                            delta / number_of_agg,
                            bound,
                        ),
                    )
                })
                .collect::<Vec<_>>();
            let private_query = noise_multipliers
                .iter()
                .map(|(_, n)| PrivateQuery::Gaussian(*n))
                .collect::<Vec<_>>()
                .into();
            (
                self.add_clipped_gaussian_noise(&noise_multipliers),
                private_query
            )
        } else {
            (self, PrivateQuery::null())
        };
        DPRelation::new(
            dp_relation,
            private_query,
        )
    }
}

impl PUPRelation {
    /// Builds a DPRelation wrapping a Relation::Reduce
    /// whose `aggregates` are the noisy sums of each column in `sums`
    /// and the group by columns are defined by `group_by_names`
    /// The budget is equally splitted among the sums.
    fn differentially_private_sums(
        self,
        sums: Vec<(&str, &str)>,
        group_by_names: Vec<&str>,
        epsilon: f64,
        delta: f64,
    ) -> Result<DPRelation> {
        if (epsilon == 0. || delta == 0.) && !sums.is_empty() {
            return Err(Error::BudgetError(format!(
                "Not enough budget for the aggregations. Got: (espilon, delta) = ({epsilon}, {delta})"
            )));
        }

        let input_values_bound = sums
            .iter()
            .map(|(s, c)| {
                (
                    *s,
                    *c,
                    self.schema()[*c]
                        .data_type()
                        .absolute_upper_bound()
                        .unwrap_or(1.0),
                )
            })
            .collect::<Vec<_>>();
        // Clip the relation
        let clipped_relation = self.deref().clone().l2_clipped_sums(
            self.privacy_unit(),
            &group_by_names,
            &input_values_bound,
        );
        let input_values_bound = input_values_bound
            .iter()
            .map(|(s, _, f)| (*s, *f))
            .collect::<Vec<_>>();
        let (dp_clipped_relation, private_query) = clipped_relation
            .gaussian_mechanisms(epsilon, delta, input_values_bound)
            .into();
        Ok(DPRelation::new(dp_clipped_relation, private_query))
    }

    /// Rewrite aggregations as sums and add noise to that sums.
    /// The budget is equally splitted among the sums.
    pub fn differentially_private_aggregates(
        self,
        named_aggregates: Vec<(&str, AggregateColumn)>,
        group_by: &[Column],
        epsilon: f64,
        delta: f64,
    ) -> Result<DPRelation> {
        let mut output_builder = Map::builder();
        let mut named_sums = vec![];
        let mut input_builder = Map::builder()
            .with((
                self.privacy_unit(),
                Expr::coalesce(
                    Expr::cast_as_text(Expr::col(self.privacy_unit())),
                    Expr::val(self.privacy_unit_default().to_string()),
                ),
            ))
            .with((
                self.privacy_unit_weight(),
                Expr::coalesce(Expr::col(self.privacy_unit_weight()), Expr::val(0.)),
            ));

        let mut group_by_names = vec![];
        (input_builder, group_by_names) =
            group_by
                .into_iter()
                .fold((input_builder, group_by_names), |(mut b, mut v), c| {
                    b = b.with((c.last().unwrap(), Expr::Column(c.clone())));
                    v.push(c.last().unwrap());
                    (b, v)
                });

        (input_builder, named_sums, output_builder) = named_aggregates.into_iter().fold(
            (input_builder, named_sums, output_builder),
            |(mut input_b, mut sums, mut output_b), (name, aggregate)| {
                let col_name = aggregate.column_name().unwrap().to_string();
                let one_col = format!("_ONE_{}", col_name);
                let sum_col = format!("_SUM_{}", col_name);
                let count_col = format!("_COUNT_{}", col_name);
                let square_col = format!("_SQUARE_{}", col_name);
                let sum_square_col = format!("_SUM_{}", square_col);
                match aggregate.aggregate() {
                    aggregate::Aggregate::First => {
                        assert!(group_by_names.contains(&col_name.as_str()));
                        output_b = output_b.with((name, Expr::col(col_name.as_str())))
                    }
                    aggregate::Aggregate::Mean => {
                        input_b = input_b
                            .with((col_name.as_str(), Expr::col(col_name.as_str())))
                            .with((one_col.as_str(), Expr::val(1.)));
                        sums.push((count_col.clone(), one_col));
                        sums.push((sum_col.clone(), col_name));
                        output_b = output_b.with((
                            name,
                            Expr::divide(
                                Expr::col(sum_col),
                                Expr::greatest(Expr::val(1.), Expr::col(count_col)),
                            ),
                        ))
                    }
                    aggregate::Aggregate::Count => {
                        input_b = input_b.with((one_col.as_str(), Expr::val(1.)));
                        sums.push((count_col.clone(), one_col));
                        output_b = output_b.with((name, Expr::cast_as_integer(Expr::col(count_col))));
                    }
                    aggregate::Aggregate::Sum => {
                        input_b = input_b.with((col_name.as_str(), Expr::col(col_name.as_str())));
                        sums.push((sum_col.clone(), col_name));
                        output_b = output_b.with((name, Expr::col(sum_col)));
                    }
                    aggregate::Aggregate::Std => {
                        input_b = input_b
                            .with((col_name.as_str(), Expr::col(col_name.as_str())))
                            .with((square_col.as_str(), Expr::pow(Expr::col(col_name.as_str()), Expr::val(2))))
                            .with((one_col.as_str(), Expr::val(1.)));
                        sums.push((count_col.clone(), one_col));
                        sums.push((sum_col.clone(), col_name));
                        sums.push((sum_square_col.clone(), square_col));
                        output_b = output_b.with((
                            name,
                            Expr::sqrt(Expr::greatest(
                                Expr::val(0.),
                                Expr::minus(
                                    Expr::divide(
                                        Expr::col(sum_square_col),
                                        Expr::greatest(Expr::val(1.), Expr::col(count_col.clone())),
                                    ),
                                    Expr::divide(
                                        Expr::col(sum_col),
                                        Expr::greatest(Expr::val(1.), Expr::col(count_col)),
                                    ),
                                )
                            ))
                        ))
                    }
                    aggregate::Aggregate::Var => {
                        input_b = input_b
                            .with((col_name.as_str(), Expr::col(col_name.as_str())))
                            .with((square_col.as_str(), Expr::pow(Expr::col(col_name.as_str()), Expr::val(2))))
                            .with((one_col.as_str(), Expr::val(1.)));
                        sums.push((count_col.clone(), one_col));
                        sums.push((sum_col.clone(), col_name));
                        sums.push((sum_square_col.clone(), square_col));
                        output_b = output_b.with((
                            name,
                            Expr::greatest(
                                Expr::val(0.),
                                Expr::minus(
                                    Expr::divide(
                                        Expr::col(sum_square_col),
                                        Expr::greatest(Expr::val(1.), Expr::col(count_col.clone())),
                                    ),
                                    Expr::divide(
                                        Expr::col(sum_col),
                                        Expr::greatest(Expr::val(1.), Expr::col(count_col)),
                                    ),
                                )
                            )
                        ))
                    }
                    _ => (),
                }
                (input_b, sums, output_b)
            },
        );

        let input: Relation = input_builder.input(self.deref().clone()).build();
        let pup_input = PUPRelation::try_from(input)?;
        let (dp_relation, private_query) = pup_input
            .differentially_private_sums(
                named_sums
                    .iter() // Convert &str to String
                    .map(|(s1, s2)| (s1.as_str(), s2.as_str()))
                    .collect::<Vec<_>>(),
                group_by_names,
                epsilon,
                delta,
            )?
            .into();
        let dp_relation = output_builder
            .input(dp_relation)
            .build();
        Ok(DPRelation::new(dp_relation, private_query))
    }
}

impl Reduce {
    /// Rewrite into DP the aggregations.
    pub fn differentially_private_aggregates(
        &self,
        epsilon: f64,
        delta: f64,
    ) -> Result<DPRelation> {
        let pup_input = PUPRelation::try_from(self.input().clone())?;

        // Split the aggregations with different DISTINCT clauses
        let reduces = self.split_distinct_aggregates();
        let epsilon = epsilon / (cmp::max(reduces.len(), 1) as f64);
        let delta = delta / (cmp::max(reduces.len(), 1) as f64);

        // Rewritte into differential privacy each `Reduce` then join them.
        let (relation, private_query) = reduces.iter()
            .map(|r| pup_input.clone().differentially_private_aggregates(
                r.named_aggregates()
                    .into_iter()
                    .map(|(n, agg)| (n, agg.clone()))
                    .collect(),
                self.group_by(),
                epsilon,
                delta,
            ))
            .reduce(|acc, dp_rel| {
                let acc = acc?;
                let dp_rel = dp_rel?;
                Ok(DPRelation::new(
                    acc.relation().clone().natural_inner_join(dp_rel.relation().clone()),
            acc.private_query().clone().compose(dp_rel.private_query().clone())
                ))
            })
            .unwrap()?
            .into();

        let relation: Relation = Relation::map()
            .input(relation)
            .with_iter(self.fields().into_iter().map(|f| (f.name(), Expr::col(f.name()))))
            .build();
        Ok((relation, private_query).into())
    }


    /// Returns a Vec of rewritten `Reduce` whose each item corresponds to a specific `DISTINCT` clause
    /// (e.g.: SUM(DISTINCT a) or COUNT(DISTINCT a) have the same `DISTINCT` clause). The original `Reduce``
    /// has been rewritten with `GROUP BY`s for each `DISTINCT` clause.
    fn split_distinct_aggregates(&self) -> Vec<Reduce> {
        let mut distinct_map: HashMap<Option<Column>, Vec<(String, AggregateColumn)>> = HashMap::new();
        let mut first_aggs: Vec<(String, AggregateColumn)> = vec![];
        for (agg, f) in self.aggregate().iter().zip(self.fields()) {
            match agg.aggregate() {
                aggregate::Aggregate::CountDistinct
                | aggregate::Aggregate::SumDistinct
                | aggregate::Aggregate::MeanDistinct
                | aggregate::Aggregate::VarDistinct
                | aggregate::Aggregate::StdDistinct => distinct_map.entry(Some(agg.column().clone())).or_insert(Vec::new()).push((f.name().to_string(), agg.clone())),
                aggregate::Aggregate::First => first_aggs.push((f.name().to_string(), agg.clone())),
                _ => distinct_map.entry(None).or_insert(Vec::new()).push((f.name().to_string(), agg.clone())),
            }
        }

        if distinct_map.len() == 0 {
            vec![self.clone()]
        } else {
            first_aggs.extend(
                self.group_by()
                .into_iter()
                .map(|x| (x.to_string(), AggregateColumn::new(aggregate::Aggregate::First, x.clone())))
                .collect::<Vec<_>>()
            );
            distinct_map.into_iter()
                .map(|(identifier, mut aggs)| {
                    aggs.extend(first_aggs.clone());
                    self.rewrite_distinct(identifier, aggs)
                })
                .collect()
        }
    }

    /// Rewrite the `DISTINCT` aggregate with a `GROUP BY`
    ///
    /// # Arguments
    /// - `self`: we reuse the `input` and `group_by` fields of the current `Reduce
    /// - `identifier`: The optionnal column `Identifier` associated with the `DISTINCT`, if `None` then the aggregates
    /// contain no `DISTINCT`.
    /// - `aggs` the vector of the `AggregateColumn` with their names
    ///
    /// Example 1 :
    /// (SELECT sum(DISTINCT col1), count(*) FROM table GROUP BY a, Some(col1), ("my_sum", sum(col1)))
    /// --> SELECT a AS a, sum(col1) AS my_sum FROM (SELECT a AS a, sum(col1) AS col1 FROM table GROUP BY a, col1) GROUP BY a
    ///
    /// Example 2 :
    /// (SELECT sum(DISTINCT col1), count(*) FROM table GROUP BY a, None, ("my_count", count(*)))
    /// --> SELECT a AS a, count(*) AS my_count FROM table GROUP BY a
    fn rewrite_distinct(&self, identifier: Option<Identifier>, aggs: Vec<(String, AggregateColumn)>) -> Reduce {
        let builder = Relation::reduce()
            .input(self.input().clone());
        if let Some(identifier) = identifier {
            let mut group_by = self.group_by()
                .into_iter()
                .map(|c| c.clone())
                .collect::<Vec<_>>();
            group_by.push(identifier);

            let first_aggs = group_by.clone()
                .into_iter()
                .map(|c| (c.to_string(), AggregateColumn::new(aggregate::Aggregate::First, c)));

            let group_by = group_by.into_iter()
                .map(|c| Expr::from(c.clone()))
                .collect::<Vec<_>>();

            let reduce: Relation = builder
                .group_by_iter(group_by)
                .with_iter(first_aggs)
                .build();

            let aggs = aggs.into_iter()
                .map(|(s, agg)| {
                    let new_agg = match agg.aggregate() {
                        aggregate::Aggregate::MeanDistinct => aggregate::Aggregate::Mean,
                        aggregate::Aggregate::CountDistinct => aggregate::Aggregate::Count,
                        aggregate::Aggregate::SumDistinct => aggregate::Aggregate::Sum,
                        aggregate::Aggregate::StdDistinct => aggregate::Aggregate::Std,
                        aggregate::Aggregate::VarDistinct => aggregate::Aggregate::Var,
                        aggregate::Aggregate::First => aggregate::Aggregate::First,
                        _ => todo!(),
                    };
                    (s, AggregateColumn::new(new_agg, agg.column().clone()))
                });
            Relation::reduce()
                .input(reduce)
                .group_by_iter(self.group_by().to_vec())
                .with_iter(aggs)
                .build()
        } else {
            builder
            .group_by_iter(self.group_by().clone().to_vec())
            .with_iter(aggs)
            .build()
        }
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
        io::{postgresql, Database},
        privacy_unit_tracking::{PrivacyUnitTracking, Strategy},
        sql::parse,
        Relation,
        relation::{Schema, Variant as _},
        privacy_unit_tracking::PrivacyUnit
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
    fn test_differentially_private_sums_no_group_by() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        let table = relations
            .get(&["item_table".to_string()])
            .unwrap()
            .deref()
            .clone();
        let (epsilon, delta) = (1., 1e-3);

        // privacy tracking of the inputs
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

        let reduce = Reduce::new(
            "my_reduce".to_string(),
            vec![("sum_price".to_string(), AggregateColumn::sum("price"))],
            vec![],
            pup_table.deref().clone().into(),
        );
        let relation = Relation::from(reduce.clone());
        relation.display_dot().unwrap();

        let dp_relation = PUPRelation::try_from(reduce.input().clone())
            .unwrap()
            .differentially_private_sums(vec![("sum_price", "price")], vec![], epsilon, delta)
            .unwrap();
        dp_relation.display_dot().unwrap();
        matches!(dp_relation.schema()[0].data_type(), DataType::Float(_));

        let query: &str = &ast::Query::from(&relation).to_string();
        println!("{query}");
        _ = database.query(query).unwrap();
    }

    #[test]
    fn test_differentially_private_sums_with_group_by() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        let table = relations
            .get(&["item_table".to_string()])
            .unwrap()
            .deref()
            .clone();
        let (epsilon, delta) = (1., 1e-3);

        // privacy tracking of the inputs
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

        let reduce = Reduce::new(
            "my_reduce".to_string(),
            vec![("my_sum_price".to_string(), AggregateColumn::sum("price"))],
            vec!["item".into()],
            pup_table.deref().clone().into(),
        );
        let relation = Relation::from(reduce.clone());
        //relation.display_dot().unwrap();

        let dp_relation = PUPRelation::try_from(reduce.input().clone())
            .unwrap()
            .differentially_private_sums(vec![("sum_price", "price")], vec!["item"], epsilon, delta)
            .unwrap();
        dp_relation.display_dot().unwrap();
        assert_eq!(dp_relation.schema().len(), 2);
        assert_eq!(dp_relation.schema()[0].data_type(), DataType::text());
        assert!(dp_relation.schema()[1]
            .data_type()
            .is_subset_of(&DataType::float()));

        let query: &str = &ast::Query::from(&relation).to_string();
        println!("{query}");
        _ = database.query(query).unwrap();
    }

    #[test]
    fn test_differentially_private_sums_group_by_aggregate() {
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
        let dp_relation = PUPRelation::try_from(reduce.input().clone())
            .unwrap()
            .differentially_private_sums(vec![("sum_a", "a")], vec!["a"], epsilon, delta)
            .unwrap();
        dp_relation.display_dot().unwrap();

        let reduce: Reduce = Relation::reduce()
            .name("reduce_relation")
            .with(("sum_a".to_string(), AggregateColumn::sum("a")))
            .with_group_by_column("a")
            .input(table.clone())
            .build();
        let dp_relation = PUPRelation::try_from(reduce.input().clone())
            .unwrap()
            .differentially_private_sums(vec![("sum_a", "a")], vec!["a"], epsilon, delta)
            .unwrap();
        dp_relation.display_dot().unwrap();
    }

    #[test]
    fn test_differentially_private_aggregates() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        let table = relations
            .get(&["item_table".to_string()])
            .unwrap()
            .deref()
            .clone();
        let (epsilon, delta) = (1., 1e-3);

        // privacy tracking of the inputs
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
        let reduce = Reduce::new(
            "my_reduce".to_string(),
            vec![
                ("count_price".to_string(), AggregateColumn::count("price")),
                ("sum_price".to_string(), AggregateColumn::sum("price")),
                ("avg_price".to_string(), AggregateColumn::mean("price")),
                ("var_price".to_string(), AggregateColumn::var("price")),
                ("std_price".to_string(), AggregateColumn::std("price")),
            ],
            vec![],
            pup_table.deref().clone().into(),
        );
        let relation = Relation::from(reduce.clone());
        relation.display_dot().unwrap();

        let dp_relation = reduce
            .differentially_private_aggregates(epsilon, delta)
            .unwrap();
        dp_relation.display_dot().unwrap();
        assert_eq!(dp_relation.schema().len(), 5);
        assert!(dp_relation
            .data_type()
            .is_subset_of(&DataType::structured(vec![
                ("count_price", DataType::float()),
                ("sum_price", DataType::float()),
                ("avg_price", DataType::float()),
                ("var_price", DataType::float_min(0.)),
                ("std_price", DataType::float_min(0.)),
            ])));
        let query: &str = &ast::Query::from(&relation).to_string();
        println!("\n{query}");
        _ = database.query(query).unwrap();
    }

    #[test]
    fn test_differentially_private_aggregates_with_group_by() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        let table = relations
            .get(&["item_table".to_string()])
            .unwrap()
            .deref()
            .clone();
        let (epsilon, delta) = (1., 1e-3);

        // privacy tracking of the inputs
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
        let reduce = Reduce::new(
            "my_reduce".to_string(),
            vec![
                ("count_price".to_string(), AggregateColumn::count("price")),
                ("sum_price".to_string(), AggregateColumn::sum("price")),
                ("avg_price".to_string(), AggregateColumn::mean("price")),
                ("var_price".to_string(), AggregateColumn::var("price")),
                ("std_price".to_string(), AggregateColumn::std("price")),
            ],
            vec!["item".into()],
            pup_table.deref().clone().into(),
        );
        let relation = Relation::from(reduce.clone());
        relation.display_dot().unwrap();

        let dp_relation = reduce
            .differentially_private_aggregates(epsilon, delta)
            .unwrap();
        dp_relation.display_dot().unwrap();
        assert_eq!(dp_relation.schema().len(), 5);
        assert!(dp_relation
            .data_type()
            .is_subset_of(&DataType::structured(vec![
                ("count_price", DataType::float()),
                ("sum_price", DataType::float()),
                ("avg_price", DataType::float()),
                ("var_price", DataType::float_min(0.)),
                ("std_price", DataType::float_min(0.)),
            ])));

        let query: &str = &ast::Query::from(&relation).to_string();
        println!("{query}");
        _ = database
            .query(query)
            .unwrap()
            .iter()
            .map(ToString::to_string);

        // the group by column is in the SELECT
        let reduce = Reduce::new(
            "my_reduce".to_string(),
            vec![
                ("count_price".to_string(), AggregateColumn::count("price")),
                ("sum_price".to_string(), AggregateColumn::sum("price")),
                ("my_item".to_string(), AggregateColumn::first("item")),
                ("avg_price".to_string(), AggregateColumn::mean("price")),
            ],
            vec!["item".into()],
            pup_table.deref().clone().into(),
        );
        let relation = Relation::from(reduce.clone());
        relation.display_dot().unwrap();

        let dp_relation = reduce
            .differentially_private_aggregates(epsilon, delta)
            .unwrap();
        dp_relation.display_dot().unwrap();
        assert_eq!(dp_relation.schema().len(), 4);
        assert!(dp_relation
            .data_type()
            .is_subset_of(&DataType::structured(vec![
                ("count_price", DataType::float()),
                ("sum_price", DataType::float()),
                ("my_item", DataType::text()),
                ("avg_price", DataType::float()),
            ])));

        let query: &str = &ast::Query::from(&relation).to_string();
        println!("{query}");
        _ = database.query(query).unwrap();
    }

    #[test]
    fn test_differentially_private_aggregates_with_distinct_aggregates() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        let table = relations
            .get(&["item_table".to_string()])
            .unwrap()
            .deref()
            .clone();
        let (epsilon, delta) = (1., 1e-3);

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

        // with group by
        let reduce = Reduce::new(
            "my_reduce".to_string(),
            vec![
                ("count_price".to_string(), AggregateColumn::count("price")),
                ("count_distinct_price".to_string(), AggregateColumn::count_distinct("price")),
                ("sum_price".to_string(), AggregateColumn::sum("price")),
                ("sum_distinct_price".to_string(), AggregateColumn::sum_distinct("price")),
                ("item".to_string(), AggregateColumn::first("item")),
            ],
            vec!["item".into()],
            pup_table.deref().clone().into(),
        );
        let relation = Relation::from(reduce.clone());

        let dp_relation = reduce
            .differentially_private_aggregates(epsilon, delta)
            .unwrap();
        dp_relation.display_dot().unwrap();
        assert_eq!(dp_relation.schema().len(), 5);
        assert!(dp_relation
            .data_type()
            .is_subset_of(&DataType::structured(vec![
                ("count_price", DataType::float()),
                ("count_distinct_price", DataType::float()),
                ("sum_price", DataType::float()),
                ("sum_distinct_price", DataType::float()),
                ("item", DataType::text()),
            ])));

        let query: &str = &ast::Query::from(&relation).to_string();
        println!("{query}");
        _ = database
            .query(query)
            .unwrap()
            .iter()
            .map(ToString::to_string);

        // no group by
        let reduce = Reduce::new(
            "my_reduce".to_string(),
            vec![
                ("count_price".to_string(), AggregateColumn::count("price")),
                ("count_distinct_price".to_string(), AggregateColumn::count_distinct("price")),
                ("sum_price".to_string(), AggregateColumn::sum("price")),
                ("sum_distinct_price".to_string(), AggregateColumn::sum_distinct("price")),
            ],
            vec![],
            pup_table.deref().clone().into(),
        );
        let relation = Relation::from(reduce.clone());

        let dp_relation = reduce
            .differentially_private_aggregates(epsilon, delta)
            .unwrap();
        dp_relation.display_dot().unwrap();
        assert_eq!(dp_relation.schema().len(), 4);
        assert!(dp_relation
            .data_type()
            .is_subset_of(&DataType::structured(vec![
                ("count_price", DataType::float()),
                ("count_distinct_price", DataType::float()),
                ("sum_price", DataType::float()),
                ("sum_distinct_price", DataType::float()),
            ])));

        let query: &str = &ast::Query::from(&relation).to_string();
        println!("{query}");
        _ = database
            .query(query)
            .unwrap()
            .iter()
            .map(ToString::to_string);
    }

    #[test]
    fn test_split_distinct_aggregates() {
        let schema: Schema = vec![
            ("a", DataType::float_interval(-2., 2.)),
            ("b", DataType::integer_interval(0, 10)),
            ("c", DataType::float_interval(0., 20.)),
            ("d", DataType::float_interval(0., 1.)),
        ]
        .into_iter()
        .collect();
        let table: Relation = Relation::table()
            .name("table")
            .schema(schema.clone())
            .size(1000)
            .build();

        // No distinct + no group by
        let reduce: Reduce = Relation::reduce()
        .input(table.clone())
        .with(("sum_a", AggregateColumn::sum("a")))
        .build();
        let reduces = reduce.split_distinct_aggregates();
        assert_eq!(reduces.len(), 1);
        assert_eq!(
            reduces[0].data_type(),
            DataType::structured([
                ("sum_a", DataType::float_interval(-2000., 2000.))
            ])
        );

        // No distinct + group by
        let reduce: Reduce = Relation::reduce()
        .input(table.clone())
        .with(("sum_a", AggregateColumn::sum("a")))
        .group_by(expr!(b))
        .build();
        let reduces = reduce.split_distinct_aggregates();
        assert_eq!(reduces.len(), 1);
        assert_eq!(
            reduces[0].data_type(),
            DataType::structured([
                ("sum_a", DataType::float_interval(-2000., 2000.)),
                ("b", DataType::integer_interval(0, 10)),
            ])
        );

        // simple distinct
        let reduce: Reduce = Relation::reduce()
        .input(table.clone())
        .with(("sum_distinct_a", AggregateColumn::sum_distinct("a")))
        .build();
        let reduces = reduce.split_distinct_aggregates();
        assert_eq!(reduces.len(), 1);
        Relation::from(reduces[0].clone()).display_dot().unwrap();
        assert_eq!(
            reduces[0].data_type(),
            DataType::structured([
                ("sum_distinct_a", DataType::float_interval(-2000., 2000.))
            ])
        );

        // simple distinct with group by
        let reduce: Reduce = Relation::reduce()
        .input(table.clone())
        .with(("sum_distinct_a", AggregateColumn::sum_distinct("a")))
        .group_by(expr!(b))
        .build();
        let reduces = reduce.split_distinct_aggregates();
        assert_eq!(reduces.len(), 1);
        Relation::from(reduces[0].clone()).display_dot().unwrap();
        assert_eq!(
            reduces[0].data_type(),
            DataType::structured([
                ("sum_distinct_a", DataType::float_interval(-2000., 2000.)),
                ("b", DataType::integer_interval(0, 10)),
            ])
        );

        // simple distinct with group by
        let reduce: Reduce = Relation::reduce()
        .input(table.clone())
        .with(("sum_distinct_a", AggregateColumn::sum_distinct("a")))
        .with_group_by_column("b")
        .build();
        let reduces = reduce.split_distinct_aggregates();
        assert_eq!(reduces.len(), 1);
        Relation::from(reduces[0].clone()).display_dot().unwrap();
        assert_eq!(
            reduces[0].data_type(),
            DataType::structured([
                ("sum_distinct_a", DataType::float_interval(-2000., 2000.)),
                ("b", DataType::integer_interval(0, 10)),
            ])
        );

        // multi distinct + no group by
        let reduce: Reduce = Relation::reduce()
        .input(table.clone())
        .with(("sum_a", AggregateColumn::sum("a")))
        .with(("sum_distinct_a", AggregateColumn::sum_distinct("a")))
        .with(("count_b", AggregateColumn::count("b")))
        .with(("count_distinct_b", AggregateColumn::count_distinct("b")))
        .build();
        let reduces = reduce.split_distinct_aggregates();
        assert_eq!(reduces.len(), 3);
        Relation::from(reduces[0].clone()).display_dot().unwrap();
        Relation::from(reduces[1].clone()).display_dot().unwrap();
        Relation::from(reduces[2].clone()).display_dot().unwrap();

        // multi distinct + group by
        let reduce: Reduce = Relation::reduce()
        .input(table.clone())
        .with(("sum_a", AggregateColumn::sum("a")))
        .with(("sum_distinct_a", AggregateColumn::sum_distinct("a")))
        .with(("count_b", AggregateColumn::count("b")))
        .with(("count_distinct_b", AggregateColumn::count_distinct("b")))
        .with(("my_c", AggregateColumn::first("c")))
        .group_by(expr!(c))
        .build();
        let reduces = reduce.split_distinct_aggregates();
        assert_eq!(reduces.len(), 3);
        Relation::from(reduces[0].clone()).display_dot().unwrap();
        Relation::from(reduces[1].clone()).display_dot().unwrap();
        Relation::from(reduces[2].clone()).display_dot().unwrap();

        // reduce without any aggregation
        let reduce: Reduce = Relation::reduce()
        .input(table.clone())
        .with_group_by_column("a")
        .with_group_by_column("c")
        .build();
        let reduces = reduce.split_distinct_aggregates();
        assert_eq!(reduces.len(), 1);
    }

    #[test]
    fn test_distinct_differentially_private_aggregates() {
        let schema: Schema = vec![
            ("a", DataType::float_interval(-2., 2.)),
            ("b", DataType::integer_interval(0, 10)),
            ("c", DataType::float_interval(10., 20.)),
            (PrivacyUnit::privacy_unit(), DataType::text()),
            (PrivacyUnit::privacy_unit_weight(), DataType::float()),
        ]
        .into_iter()
        .collect();
        let table: Relation = Relation::table()
            .name("table")
            .schema(schema.clone())
            .size(1000)
            .build();

        let (epsilon, delta) = (1.0, 1e-5);

        // No distinct + no group by
        let reduce: Reduce = Relation::reduce()
        .input(table.clone())
        .with(("sum_a", AggregateColumn::sum("a")))
        .build();
        let dp_relation = reduce.differentially_private_aggregates(epsilon.clone(), delta.clone()).unwrap();
        assert_eq!(
            dp_relation.private_query(),
            &PrivateQuery::gaussian_from_epsilon_delta_sensitivity(
                epsilon.clone(),
                delta.clone(),
                2.
            )
        );
        assert_eq!(
            dp_relation.relation().data_type(),
            DataType::structured([
                ("sum_a", DataType::float_interval(-2000., 2000.))
            ])
        );

        // No distinct + group by
        let reduce: Reduce = Relation::reduce()
        .input(table.clone())
        .with(("sum_a", AggregateColumn::sum("a")))
        .group_by(expr!(b))
        .build();
        let dp_relation = reduce.differentially_private_aggregates(epsilon.clone(), delta.clone()).unwrap();
        assert_eq!(
            dp_relation.private_query(),
            &PrivateQuery::gaussian_from_epsilon_delta_sensitivity(
                epsilon.clone(),
                delta.clone(),
                2.
            )
        );
        assert_eq!(
            dp_relation.relation().data_type(),
            DataType::structured([
                ("sum_a", DataType::float_interval(-2000., 2000.))
            ])
        );

        // simple distinct
        let reduce: Reduce = Relation::reduce()
        .input(table.clone())
        .with(("sum_distinct_a", AggregateColumn::sum_distinct("a")))
        .build();
        let dp_relation = reduce.differentially_private_aggregates(epsilon.clone(), delta.clone()).unwrap();
        //dp_relation.relation().display_dot().unwrap();
        assert_eq!(
            dp_relation.private_query(),
            &PrivateQuery::gaussian_from_epsilon_delta_sensitivity(
                epsilon.clone(),
                delta.clone(),
                2.
            )
        );
        assert_eq!(
            dp_relation.relation().data_type(),
            DataType::structured([
                ("sum_distinct_a", DataType::float_interval(-2000., 2000.))
            ])
        );

        // simple distinct with group by
        let reduce: Reduce = Relation::reduce()
        .input(table.clone())
        .with(("sum_distinct_a", AggregateColumn::sum_distinct("a")))
        .with_group_by_column("b")
        .build();
        let dp_relation = reduce.differentially_private_aggregates(epsilon.clone(), delta.clone()).unwrap();
        //dp_relation.relation().display_dot().unwrap();
        assert_eq!(
            dp_relation.private_query(),
            &PrivateQuery::gaussian_from_epsilon_delta_sensitivity(
                epsilon.clone(),
                delta.clone(),
                2.
            )
        );
        assert_eq!(
            dp_relation.relation().data_type(),
            DataType::structured([
                ("sum_distinct_a", DataType::float_interval(-2000., 2000.)),
                ("b", DataType::integer_interval(0, 10))
            ])
        );

        // multi distinct + no group by
        let reduce: Reduce = Relation::reduce()
        .input(table.clone())
        .with(("sum_a", AggregateColumn::sum("a")))
        .with(("sum_distinct_a", AggregateColumn::sum_distinct("a")))
        .with(("count_b", AggregateColumn::count("b")))
        .with(("count_distinct_b", AggregateColumn::count_distinct("b")))
        .with(("avg_distinct_b", AggregateColumn::mean_distinct("b")))
        .with(("var_distinct_b", AggregateColumn::var_distinct("b")))
        .with(("std_distinct_b", AggregateColumn::std_distinct("b")))
        .build();
        let dp_relation = reduce.differentially_private_aggregates(epsilon.clone(), delta.clone()).unwrap();
        dp_relation.relation().display_dot().unwrap();
        assert_eq!(
            dp_relation.relation().data_type(),
            DataType::structured([
                ("sum_a", DataType::float_interval(-2000., 2000.)),
                ("sum_distinct_a", DataType::float_interval(-2000., 2000.)),
                ("count_b", DataType::integer_interval(0, 1000)),
                ("count_distinct_b", DataType::integer_interval(0, 1000)),
                ("avg_distinct_b", DataType::float_interval(0., 10000.)),
                ("var_distinct_b", DataType::float_interval(0., 100000.)),
                ("std_distinct_b", DataType::float_interval(0., 316.22776601683796)),
            ])
        );

        // multi distinct + group by
        let reduce: Reduce = Relation::reduce()
        .input(table.clone())
        .with(("sum_a", AggregateColumn::sum("a")))
        .with(("sum_distinct_a", AggregateColumn::sum_distinct("a")))
        .with(("count_b", AggregateColumn::count("b")))
        .with(("count_distinct_b", AggregateColumn::count_distinct("b")))
        .with(("my_c", AggregateColumn::first("c")))
        .with(("avg_distinct_b", AggregateColumn::mean_distinct("b")))
        .with(("var_distinct_b", AggregateColumn::var_distinct("b")))
        .with(("std_distinct_b", AggregateColumn::std_distinct("b")))
        .group_by(expr!(c))
        .build();
        let dp_relation = reduce.differentially_private_aggregates(epsilon.clone(), delta.clone()).unwrap();
        dp_relation.relation().display_dot().unwrap();
        assert_eq!(
            dp_relation.relation().data_type(),
            DataType::structured([
                ("sum_a", DataType::float_interval(-2000., 2000.)),
                ("sum_distinct_a", DataType::float_interval(-2000., 2000.)),
                ("count_b", DataType::integer_interval(0, 1000)),
                ("count_distinct_b", DataType::integer_interval(0, 1000)),
                ("my_c", DataType::float_interval(10., 20.)),
                ("avg_distinct_b", DataType::float_interval(0., 10000.)),
                ("var_distinct_b", DataType::float_interval(0., 100000.)),
                ("std_distinct_b", DataType::float_interval(0., 316.22776601683796)),
            ])
        );
    }

    #[test]
    fn test_differentially_private_group_by_aggregate() {
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
            .with_group_by_column("a")
            .input(table.clone())
            .build();
        let (dp_relation, private_query) = reduce
        .differentially_private_aggregates(epsilon.clone(), delta.clone())
        .unwrap()
        .into();
        dp_relation.display_dot().unwrap();
        assert_eq!(
            private_query,
            PrivateQuery::gaussian_from_epsilon_delta_sensitivity(
                epsilon.clone(),
                delta.clone(),
                10.
            )
        );
        assert_eq!(
            dp_relation.data_type(),
            DataType::structured([
                ("sum_a", DataType::float_interval(0., 1000.)),
                ("a", DataType::integer_range(1..=10)
            )])
        );


        let reduce: Reduce = Relation::reduce()
            .name("reduce_relation")
            .with(("sum_a".to_string(), AggregateColumn::sum("a")))
            .group_by(expr!(a))
            .input(table.clone())
            .build();
        let (dp_relation, private_query) = reduce
        .differentially_private_aggregates(epsilon.clone(), delta.clone())
        .unwrap()
        .into();
        dp_relation.display_dot().unwrap();
        assert_eq!(
            private_query,
            PrivateQuery::gaussian_from_epsilon_delta_sensitivity(
                epsilon.clone(),
                delta.clone(),
                10.
            )
        );
        assert_eq!(
            dp_relation.data_type(),
            DataType::structured([("sum_a", DataType::float_interval(0., 1000.)),])
        );
    }
}
