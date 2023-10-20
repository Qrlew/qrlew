use crate::{
    builder::With,
    data_type::DataTyped,
    differential_privacy::private_query::PrivateQuery,
    differential_privacy::{private_query, DPRelation, Error, Result},
    expr::{aggregate, AggregateColumn, Expr},
    protection::PEPRelation,
    relation::{field::Field, Map, Reduce, Relation, Variant as _},
    DataType, Ready,
};
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
        let number_of_agg = bounds.len() as f64;
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
        DPRelation::new(self.add_gaussian_noise(noise_multipliers), private_query)
    }
}

impl PEPRelation {
    /// Builds a DPRelation wrapping a Relation::Reduce
    /// whose `aggregates` are the noisy sums of each column in `sums`
    /// and the group by columns are defined by `group_by_names`
    /// The budget is equally splitted among the sums.
    fn differentially_private_sums(
        self,
        sums: Vec<&str>,
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
            .map(|c| {
                (
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
            self.protected_entity_id(),
            group_by_names,
            input_values_bound.clone(),
        );

        let (dp_clipped_relation, private_query) = clipped_relation
            .gaussian_mechanisms(epsilon, delta, input_values_bound)
            .into();

        Ok(DPRelation::new(dp_clipped_relation, private_query))
    }

    /// Rewrite aggregations as sums and ass noise to that sums.
    /// The budget is equally splitted among the sums.
    pub fn differentially_private_aggregates(
        self,
        named_aggregates: Vec<(&str, AggregateColumn)>,
        group_by: &[Expr],
        epsilon: f64,
        delta: f64,
    ) -> Result<DPRelation> {
        let mut output_builder = Map::builder();
        let mut named_sums = vec![];
        let mut input_builder = Map::builder()
            .with((
                self.protected_entity_id(),
                Expr::col(self.protected_entity_id()),
            ))
            .with((
                self.protected_entity_weight(),
                Expr::col(self.protected_entity_weight()),
            ));

        let mut group_by_names = vec![];
        (input_builder, group_by_names) =
            group_by
                .into_iter()
                .fold((input_builder, group_by_names), |(mut b, mut v), x| {
                    if let Expr::Column(c) = x {
                        b = b.with((c.last().unwrap(), x.clone()));
                        v.push(c.last().unwrap());
                    }
                    (b, v)
                });

        (input_builder, named_sums, output_builder) = named_aggregates.into_iter().fold(
            (input_builder, named_sums, output_builder),
            |(mut input_b, mut sums, mut output_b), (name, aggregate)| {
                let one_col = "_ONE_".to_string();
                let colname = aggregate.column_name().unwrap().to_string();
                let sum_col = format!("_SUM_{}", colname);
                let count_col = format!("_COUNT_{}", colname);
                match aggregate.aggregate() {
                    aggregate::Aggregate::First => {
                        assert!(group_by_names.contains(&colname.as_str()));
                        output_b = output_b.with((name, Expr::col(colname.as_str())))
                    }
                    aggregate::Aggregate::Mean => {
                        input_b = input_b
                            .with((name, Expr::col(colname.as_str())))
                            .with((one_col.as_str(), Expr::val(1.)));
                        sums.push((count_col.clone(), one_col));
                        sums.push((sum_col.clone(), colname));
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
                        output_b = output_b.with((name, Expr::col(count_col)));
                    }
                    aggregate::Aggregate::Sum => {
                        input_b = input_b.with((colname.as_str(), Expr::col(colname.as_str())));
                        sums.push((sum_col.clone(), colname));
                        output_b = output_b.with((name, Expr::col(sum_col)));
                    }
                    aggregate::Aggregate::Std => todo!(),
                    aggregate::Aggregate::Var => todo!(),
                    _ => (),
                }
                (input_b, sums, output_b)
            },
        );

        let input: Relation = input_builder.input(self.deref().clone()).build();
        let pep_input = PEPRelation::try_from(input)?;

        let (dp_relation, private_query) = pep_input
            .differentially_private_sums(
                named_sums
                    .iter() // Convert &str to String
                    .map(|(_, s)| s.as_str())
                    .collect::<Vec<&str>>(),
                group_by_names,
                epsilon,
                delta,
            )?
            .into();

        let names: HashMap<String, String> =
            named_sums.into_iter().map(|(s1, s2)| (s2, s1)).collect();

        let dp_relation = output_builder
            .input(
                dp_relation
                    .rename_fields(|n, _| names.get(n).unwrap_or(&n.to_string()).to_string()),
            )
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
        let pep_input = PEPRelation::try_from(self.input().clone())?;
        pep_input.differentially_private_aggregates(
            self.named_aggregates()
                .into_iter()
                .map(|(n, agg)| (n, agg.clone()))
                .collect(),
            self.group_by(),
            epsilon,
            delta,
        )
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
        protection::{Protection, Strategy},
        sql::parse,
        Relation,
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

        let dp_relation = PEPRelation::try_from(reduce.input().clone())
            .unwrap()
            .differentially_private_sums(vec!["price"], vec![], epsilon, delta)
            .unwrap();
        dp_relation.display_dot().unwrap();
        matches!(dp_relation.schema()[0].data_type(), DataType::Float(_));

        let query: &str = &ast::Query::from(&relation).to_string();
        println!("{query}");
        _ = database
            .query(query)
            .unwrap()
            .iter()
            .map(ToString::to_string);
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
            pep_table.deref().clone().into(),
        );
        let relation = Relation::from(reduce.clone());
        relation.display_dot().unwrap();

        let dp_relation = PEPRelation::try_from(reduce.input().clone())
            .unwrap()
            .differentially_private_sums(vec!["price"], vec!["item"], epsilon, delta)
            .unwrap();
        dp_relation.display_dot().unwrap();
        assert_eq!(dp_relation.schema().len(), 2);
        assert_eq!(dp_relation.schema()[0].data_type(), DataType::text());
        assert!(dp_relation.schema()[1]
            .data_type()
            .is_subset_of(&DataType::float()));

        let query: &str = &ast::Query::from(&relation).to_string();
        println!("{query}");
        _ = database
            .query(query)
            .unwrap()
            .iter()
            .map(ToString::to_string);
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
                ("count_price".to_string(), AggregateColumn::count("price")),
                ("sum_price".to_string(), AggregateColumn::sum("price")),
                ("avg_price".to_string(), AggregateColumn::mean("price")),
            ],
            vec![],
            pep_table.deref().clone().into(),
        );
        let relation = Relation::from(reduce.clone());
        relation.display_dot().unwrap();

        let dp_relation = reduce
            .differentially_private_aggregates(epsilon, delta)
            .unwrap();
        dp_relation.display_dot().unwrap();
        assert_eq!(dp_relation.schema().len(), 3);
        assert!(dp_relation
            .data_type()
            .is_subset_of(&DataType::structured(vec![
                ("count_price", DataType::float()),
                ("sum_price", DataType::float()),
                ("avg_price", DataType::float()),
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
    fn test_differentially_private_aggregates_with_group_by() {
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
                ("count_price".to_string(), AggregateColumn::count("price")),
                ("sum_price".to_string(), AggregateColumn::sum("price")),
                ("avg_price".to_string(), AggregateColumn::mean("price")),
            ],
            vec![expr!(item)],
            pep_table.deref().clone().into(),
        );
        let relation = Relation::from(reduce.clone());
        relation.display_dot().unwrap();

        let dp_relation = reduce
            .differentially_private_aggregates(epsilon, delta)
            .unwrap();
        dp_relation.display_dot().unwrap();
        assert_eq!(dp_relation.schema().len(), 3);
        assert!(dp_relation
            .data_type()
            .is_subset_of(&DataType::structured(vec![
                ("count_price", DataType::float()),
                ("sum_price", DataType::float()),
                ("avg_price", DataType::float()),
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
            vec![Expr::col("item")],
            pep_table.deref().clone().into(),
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
        _ = database
            .query(query)
            .unwrap()
            .iter()
            .map(ToString::to_string);
    }
}
