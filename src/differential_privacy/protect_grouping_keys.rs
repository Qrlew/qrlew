use crate::{
    builder::{Ready, With},
    differential_privacy::{DPRelation, Error, PrivateQuery},
    expr::{aggregate, Expr},
    namer,
    protection::PEPRelation,
    relation::{Join, Map, Reduce, Relation, Variant as _},
};
use std::result;

pub type Result<T> = result::Result<T, Error>;

pub const COUNT_DISTINCT_PE_ID: &str = "_COUNT_DISTINCT_PE_ID_";

impl Reduce {
    pub fn grouping_columns(&self) -> Result<Vec<String>> {
        self.group_by()
            .iter()
            .cloned()
            .map(|x| {
                if let Expr::Column(name) = x {
                    Ok(name.to_string())
                } else {
                    Err(Error::GroupingKeysError(
                        "We support only `Expr::Column` in the group_by".into(),
                    ))
                }
            })
            .collect()
    }

    // Returns a `Relation` outputing all grouping keys that can be safely released
    pub fn grouping_values(
        &self,
        protected_entity_id: &str,
        protected_entity_weight: &str,
        epsilon: f64,
        delta: f64,
    ) -> Result<DPRelation> {
        let grouping_cols = self.grouping_columns()?;
        if !grouping_cols.is_empty() {
            PEPRelation::try_from(self.inputs()[0].clone().filter_fields(|f| {
                grouping_cols.contains(&f.to_string())
                    || f == protected_entity_id
                    || f == protected_entity_weight
            }))?
            .released_values(epsilon, delta)
        } else {
            Err(Error::GroupingKeysError("No grouping keys.".to_string()))
        }
    }

    pub fn protect_grouping_keys(
        self,
        protected_entity_id: &str,
        protected_entity_weight: &str,
        epsilon: f64,
        delta: f64,
    ) -> Result<(PEPRelation, PrivateQuery)> {
        Ok(
            if self.grouping_columns()? == vec![protected_entity_id.to_string()] {
                (
                    PEPRelation::try_from(Relation::from(self.clone()))?,
                    PrivateQuery::null(),
                )
            } else {
                let (grouping_values, private_query) = self
                    .grouping_values(protected_entity_id, protected_entity_weight, epsilon, delta)?
                    .into();
                let input_relation_with_protected_grouping_keys = self
                    .input()
                    .clone()
                    .join_with_grouping_values(grouping_values)?;
                let relation: Relation = Reduce::builder()
                    .with(self.clone())
                    .input(input_relation_with_protected_grouping_keys)
                    .build();
                (PEPRelation::try_from(relation)?, private_query)
            },
        )
    }
}

impl Map {
    pub fn protect_grouping_keys(
        self,
        epsilon: f64,
        delta: f64,
    ) -> Result<(PEPRelation, PrivateQuery)> {
        let (protected_input, private_query) = PEPRelation::try_from(self.inputs()[0].clone())?
            .protect_grouping_keys(epsilon, delta)?;
        let relation: Relation = Map::builder()
            .with(self.clone())
            .input(Relation::from(protected_input))
            .build();
        Ok((PEPRelation::try_from(relation)?, private_query))
    }
}

impl Join {
    pub fn protect_grouping_keys(
        self,
        epsilon: f64,
        delta: f64,
    ) -> Result<(PEPRelation, PrivateQuery)> {
        let (left_dp_relation, left_private_query) =
            PEPRelation::try_from(self.inputs()[0].clone())?
                .protect_grouping_keys(epsilon, delta)?;
        let (right_dp_relation, right_private_query) =
            PEPRelation::try_from(self.inputs()[1].clone())?
                .protect_grouping_keys(epsilon, delta)?;
        let relation: Relation = Join::builder()
            .left(Relation::from(left_dp_relation))
            .right(Relation::from(right_dp_relation))
            .with(self.clone())
            .build();
        Ok((
            PEPRelation::try_from(relation)?,
            vec![left_private_query, right_private_query].into(),
        ))
    }
}

impl PEPRelation {
    pub fn tau_thresholding_values(self, epsilon: f64, delta: f64) -> Result<DPRelation> {
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
            super::mechanisms::gaussian_noise(epsilon, delta, 1.),
        )];
        let rel = rel.add_gaussian_noise(name_sigmas);

        // Returns a `Relation::Map` with the right field names and with `COUNT(DISTINCT PE_ID) > tau`
        let tau = super::mechanisms::gaussian_tau(epsilon, delta, 1.0);
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

    fn with_tau_thresholding_values(
        self,
        public_columns: &Vec<String>,
        epsilon: f64,
        delta: f64,
    ) -> Result<DPRelation> {
        let relation: Relation = Relation::from(self).with_name(namer::new_name("FILTER"));
        let relation_with_private_values =
            Relation::from(relation).filter_fields(|f| !public_columns.contains(&f.to_string()));
        PEPRelation::try_from(relation_with_private_values)?.tau_thresholding_values(epsilon, delta)
    }

    pub fn released_values(self, epsilon: f64, delta: f64) -> Result<DPRelation> {
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
            self.with_tau_thresholding_values(&public_columns, epsilon, delta)
        } else if all_columns_are_public {
            Ok(DPRelation::new(
                self.with_public_values(&public_columns)?,
                PrivateQuery::null(),
            ))
        } else {
            let (relation, private_query) = self
                .clone()
                .with_tau_thresholding_values(&public_columns, epsilon, delta)?
                .into();
            let relation = self
                .with_public_values(&public_columns)?
                .cross_join(relation)?;
            Ok(DPRelation::new(relation, private_query))
        }
    }

    pub fn protect_grouping_keys(
        self,
        epsilon: f64,
        delta: f64,
    ) -> Result<(PEPRelation, PrivateQuery)> {
        let protected_entity_id = self.protected_entity_id().to_string();
        let protected_entity_veight = self.protected_entity_weight().to_string();

        match Relation::from(self) {
            Relation::Table(t) => {
                let (relation, private_query) = PEPRelation(Relation::from(t))
                    .released_values(epsilon, delta)?
                    .into();
                Ok((PEPRelation(relation), private_query))
            }
            Relation::Map(m) => m.protect_grouping_keys(epsilon, delta),
            Relation::Reduce(r) => r.protect_grouping_keys(
                protected_entity_id.as_str(),
                protected_entity_veight.as_str(),
                epsilon,
                delta,
            ),
            Relation::Join(j) => j.protect_grouping_keys(epsilon, delta),
            Relation::Set(_) => todo!(),
            Relation::Values(v) => Ok((PEPRelation(Relation::from(v)), PrivateQuery::null())),
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

    fn join_with_grouping_values(self, grouping_values: Relation) -> Result<Relation> {
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
        data_type::{DataType, DataTyped},
        display::Dot,
        expr::AggregateColumn,
        hierarchy::Hierarchy,
        io::{postgresql, Database},
        protection::{PE_ID, PE_WEIGHT},
        relation::Schema,
        sql::parse,
    };
    use std::{ops::Deref, rc::Rc};

    #[test]
    fn test_tau_thresholded_values() {
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
    fn test_released_values() {
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
        let (rel, pq) = protected_table.released_values(1., 0.003).unwrap().into();
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
        let (rel, pq) = protected_table.released_values(1., 0.003).unwrap().into();
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
        let (rel, pq) = protected_table.released_values(1., 0.003).unwrap().into();
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

    #[test]
    fn test_grouping_values() {
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

        // Without GROUPBY
        let red = Reduce::new(
            "reduce_relation".to_string(),
            vec![("sum_a".to_string(), AggregateColumn::sum("a"))],
            vec![],
            Rc::new(table.clone()),
        );
        assert!(red.grouping_values(PE_ID, PE_WEIGHT, 1., 0.003).is_err());

        // With GROUPBY. Only one column with possible values.
        let red = Reduce::new(
            "reduce_relation".to_string(),
            vec![("sum_a".to_string(), AggregateColumn::sum("a"))],
            vec![Expr::col("b")],
            Rc::new(table.clone()),
        );
        let (rel, pq) = red
            .grouping_values(PE_ID, PE_WEIGHT, 1.0, 0.003)
            .unwrap()
            .into();
        rel.display_dot().unwrap();
        assert_eq!(
            rel.data_type(),
            DataType::structured([("b", DataType::integer_values([1, 2, 5, 6, 7, 8]))])
        );
        assert_eq!(pq, PrivateQuery::null());

        // With GROUPBY. Only one column with tau-thresolding values.
        let red = Reduce::new(
            "reduce_relation".to_string(),
            vec![
                ("sum_a".to_string(), AggregateColumn::sum("a")),
                ("c".to_string(), AggregateColumn::col("c")),
            ],
            vec![Expr::col("c")],
            Rc::new(table.clone()),
        );
        let (rel, pq) = red
            .grouping_values(PE_ID, PE_WEIGHT, 1.0, 0.003)
            .unwrap()
            .into();
        //rel.display_dot();
        assert_eq!(
            rel.data_type(),
            DataType::structured([("c", DataType::integer_range(5..=20)),])
        );
        assert_eq!(pq, PrivateQuery::EpsilonDelta(1., 0.003));

        // With GROUPBY. Columns with both tau-thresolding and possible values.
        let red = Reduce::new(
            "reduce_relation".to_string(),
            vec![
                ("sum_a".to_string(), AggregateColumn::sum("a")),
                ("c".to_string(), AggregateColumn::col("c")),
                ("b".to_string(), AggregateColumn::col("b")),
            ],
            vec![Expr::col("b"), Expr::col("c")],
            Rc::new(table.clone()),
        );
        let (rel, pq) = red
            .grouping_values(PE_ID, PE_WEIGHT, 1.0, 0.003)
            .unwrap()
            .into();
        //rel.display_dot();
        assert_eq!(
            rel.data_type(),
            DataType::structured([
                ("c", DataType::integer_range(5..=20)),
                ("b", DataType::integer_values([1, 2, 5, 6, 7, 8])),
            ])
        );
        assert_eq!(pq, PrivateQuery::EpsilonDelta(1., 0.003));
    }

    #[test]
    fn test_protect_grouping_keys_reduce() {
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

        // Reduce without GROUPBY
        let relation = Relation::from(Reduce::new(
            "reduce_relation".to_string(),
            vec![("sum_a".to_string(), AggregateColumn::sum("a"))],
            vec![],
            Rc::new(table.clone()),
        ));
        let pep_rel = relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]);
        let (protected_pep_rel, pq) = pep_rel.clone().protect_grouping_keys(1., 0.003).unwrap();
        assert_eq!(Relation::from(pep_rel), Relation::from(protected_pep_rel));
        assert_eq!(pq, PrivateQuery::null());

        // With GROUPBY. Only one column with possible values.
        let relation = Relation::from(Reduce::new(
            "reduce_relation".to_string(),
            vec![("sum_a".to_string(), AggregateColumn::sum("a"))],
            vec![Expr::col("b")],
            Rc::new(table.clone()),
        ));
        let pep_rel = relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]);
        //pep_rel.display_dot();
        let (protected_pep_rel, pq) = pep_rel.clone().protect_grouping_keys(1., 0.003).unwrap();
        //protected_pep_rel.display_dot();
        assert_eq!(
            protected_pep_rel.data_type(),
            DataType::structured([
                (PE_ID, DataType::text()),
                (PE_WEIGHT, DataType::integer_min(0)),
                ("sum_a", DataType::integer_min(0))
            ])
        );
        assert_eq!(pq, PrivateQuery::null());

        // With GROUPBY. Only one column with possible values.
        let relation = Relation::from(Reduce::new(
            "reduce_relation".to_string(),
            vec![
                ("sum_a".to_string(), AggregateColumn::sum("a")),
                ("b".to_string(), AggregateColumn::col("b")),
            ],
            vec![Expr::col("b")],
            Rc::new(table.clone()),
        ));
        let pep_rel = relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]);
        //pep_rel.display_dot();
        let (protected_pep_rel, pq) = pep_rel.clone().protect_grouping_keys(1., 0.003).unwrap();
        //protected_pep_rel.display_dot();
        assert_eq!(
            protected_pep_rel.data_type(),
            DataType::structured([
                (PE_ID, DataType::text()),
                (PE_WEIGHT, DataType::integer_min(0)),
                ("sum_a", DataType::integer_min(0)),
                ("b", DataType::integer_values([1, 2, 5, 6, 7, 8]))
            ])
        );
        assert_eq!(pq, PrivateQuery::null());

        // With GROUPBY. Only one column with tau-thresolding values.
        let relation = Relation::from(Reduce::new(
            "reduce_relation".to_string(),
            vec![
                ("sum_a".to_string(), AggregateColumn::sum("a")),
                ("c".to_string(), AggregateColumn::col("c")),
            ],
            vec![Expr::col("c")],
            Rc::new(table.clone()),
        ));
        let pep_rel = relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]);
        //pep_rel.display_dot();
        let (protected_pep_rel, pq) = pep_rel.clone().protect_grouping_keys(1., 0.003).unwrap();
        //protected_pep_rel.display_dot();
        assert_eq!(
            protected_pep_rel.data_type(),
            DataType::structured([
                (PE_ID, DataType::text()),
                (PE_WEIGHT, DataType::integer_min(0)),
                ("sum_a", DataType::integer_min(0)),
                ("c", DataType::integer_range(5..=20))
            ])
        );
        assert_eq!(pq, PrivateQuery::EpsilonDelta(1., 0.003));

        // With GROUPBY. Columns with both tau-thresolding and possible values.
        let relation = Relation::from(Reduce::new(
            "reduce_relation".to_string(),
            vec![
                ("sum_a".to_string(), AggregateColumn::sum("a")),
                ("c".to_string(), AggregateColumn::col("c")),
                ("b".to_string(), AggregateColumn::col("b")),
            ],
            vec![Expr::col("b"), Expr::col("c")],
            Rc::new(table.clone()),
        ));
        let pep_rel = relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]);
        //pep_rel.display_dot();
        let (protected_pep_rel, pq) = pep_rel.clone().protect_grouping_keys(1., 0.003).unwrap();
        protected_pep_rel.display_dot();
        assert_eq!(
            protected_pep_rel.data_type(),
            DataType::structured([
                (PE_ID, DataType::text()),
                (PE_WEIGHT, DataType::integer_min(0)),
                ("sum_a", DataType::integer_min(0)),
                ("c", DataType::integer_range(5..=20)),
                ("b", DataType::integer_values([1, 2, 5, 6, 7, 8]))
            ])
        );
        assert_eq!(pq, PrivateQuery::EpsilonDelta(1., 0.003));
    }

    #[test]
    fn test_protect_grouping_keys_map() {
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

        // Reduce without GROUPBY
        let reduce_relation = Relation::from(Reduce::new(
            "reduce_relation".to_string(),
            vec![
                ("sum_a".to_string(), AggregateColumn::sum("a")),
                ("c".to_string(), AggregateColumn::first("c")),
                ("b".to_string(), AggregateColumn::first("b")),
            ],
            vec![Expr::col("b"), Expr::col("c")],
            Rc::new(table.clone()),
        ));
        let relation: Relation = Map::builder()
            .with(("twice_sum_a", expr!(2 * sum_a)))
            .with(("b", expr!(b)))
            .with(("c", expr!(c)))
            .input(reduce_relation.clone())
            .name("my_map")
            .build();
        relation.display_dot();
        let pep_rel = relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]);
        let (protected_pep_rel, pq) = pep_rel.clone().protect_grouping_keys(1., 0.003).unwrap();
        protected_pep_rel.0.display_dot();
        assert_eq!(pq, PrivateQuery::EpsilonDelta(1., 0.003));
    }

    #[test]
    fn test_protect_grouping_keys_with_where() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        let str_query = "SELECT order_id, sum(price) AS sum_price,
        count(price) AS count_price,
        avg(price) AS mean_price
        FROM item_table WHERE order_id IN (1,2,3,4,5,6,7,8,9,10) GROUP BY order_id";
        let query = parse(str_query).unwrap();
        let relation = Relation::try_from(query.with(&relations)).unwrap();
        relation.display_dot().unwrap();

        let pep_relation = relation.force_protect_from_field_paths(
            &relations,
            vec![
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
            ],
        );
        pep_relation.display_dot().unwrap();

        let (protected_relation, pq) = pep_relation.protect_grouping_keys(1., 1e-3).unwrap();
        protected_relation.display_dot().unwrap();
        // assert_eq!( // TODO: fix size for joins with distinct values
        //     protected_relation.data_type(),
        //     Relation::from(pep_relation).data_type()
        // );
        assert_eq!(protected_relation.schema().len(), 6);
        assert_eq!(
            protected_relation.data_type()["order_id"],
            DataType::integer_values([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        );
        assert_eq!(pq, PrivateQuery::null());
        let protected_query = ast::Query::from(protected_relation.deref());
        database.query(&protected_query.to_string()).unwrap();
    }

    #[test]
    fn test_protect_grouping_keys_simple() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        // GROUPING col in the SELECT clause
        let str_query = "SELECT z, sum(x) AS sum_x FROM table_2 GROUP BY z";
        let query = parse(str_query).unwrap();
        let relation = Relation::try_from(query.with(&relations)).unwrap();

        let pep_relation =
            relation.force_protect_from_field_paths(&relations, vec![("table_2", vec![], "y")]);

        let (protected_relation, private_query) = pep_relation
            .clone()
            .protect_grouping_keys(1., 1e-3)
            .unwrap();
        protected_relation.display_dot().unwrap();
        // assert_eq!( // TODO: fix size for joins with distinct values
        //     protected_relation.data_type(),
        //     Relation::from(pep_relation).data_type()
        // );
        assert_eq!(protected_relation.schema().len(), 4);
        assert_eq!(
            protected_relation.data_type()["z"],
            DataType::text_values(["Foo".into(), "Bar".into()])
        );
        matches!(protected_relation.data_type()["sum_x"], DataType::Float(_));
        assert_eq!(private_query, PrivateQuery::null());
        let dp_query = ast::Query::from(protected_relation.deref());
        database.query(&dp_query.to_string()).unwrap();

        // GROUPING col NOT in the SELECT clause
        let str_query = "SELECT sum(x) AS sum_x FROM table_2 GROUP BY z";
        let query = parse(str_query).unwrap();
        let relation = Relation::try_from(query.with(&relations)).unwrap();

        let pep_relation =
            relation.force_protect_from_field_paths(&relations, vec![("table_2", vec![], "y")]);

        let (protected_relation, private_query) = pep_relation
            .clone()
            .protect_grouping_keys(1., 1e-3)
            .unwrap()
            .into();
        //dp_relation.display_dot().unwrap();
        assert_eq!(private_query, PrivateQuery::null());
        // assert_eq!(
        //     protected_relation.data_type(),
        //     Relation::from(pep_relation).data_type()
        // );
        assert_eq!(protected_relation.schema().len(), 3);
        matches!(protected_relation.data_type()["sum_x"], DataType::Float(_));
        let dp_query = ast::Query::from(protected_relation.deref());
        database.query(&dp_query.to_string()).unwrap();

        // GROUPING col has no possible values
        let str_query = "SELECT y, sum(x) AS sum_x FROM table_2 GROUP BY y";
        let query = parse(str_query).unwrap();
        let relation = Relation::try_from(query.with(&relations)).unwrap();

        let pep_relation =
            relation.force_protect_from_field_paths(&relations, vec![("table_2", vec![], "z")]);
        pep_relation.display_dot().unwrap();

        let (protected_relation, private_query) = pep_relation
            .clone()
            .protect_grouping_keys(1., 1e-3)
            .unwrap()
            .into();
        protected_relation.display_dot().unwrap();
        assert_eq!(private_query, PrivateQuery::EpsilonDelta(1., 1e-3));
        assert_eq!(protected_relation.schema().len(), 4);
        assert_eq!(
            protected_relation.data_type()["y"],
            DataType::optional(DataType::text())
        );
        matches!(protected_relation.data_type()["sum_x"], DataType::Float(_));
        let dp_query = ast::Query::from(protected_relation.deref());
        database.query(&dp_query.to_string()).unwrap();
    }
}
