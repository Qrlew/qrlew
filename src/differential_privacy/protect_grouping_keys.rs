use crate::display::Dot;
use crate::{
    builder::{Ready, With, WithIterator},
    data_type::{
        self,
        intervals::{Bound, Intervals},
        DataTyped,
    },
    expr::{aggregate, Aggregate, Expr, Value},
    hierarchy::Hierarchy,
    protection::{self, PEPRelation, PE_ID, PE_WEIGHT},
    relation::{transforms, Field, Join, Map, Reduce, Relation, Set, Table, Variant as _, Visitor},
    visitor::Acceptor,
    DataType,
};
use std::{error, fmt, ops::Deref, result};

#[derive(Debug, Clone)]
pub enum Error {
    TauThresholdingError(String),
    NonPEPRelationError(String),
    NoGroupingColumnsError,
    Other(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::TauThresholdingError(desc) => {
                writeln!(f, "TauThresholdingError: {}", desc)
            }
            Error::NonPEPRelationError(desc) => {
                writeln!(f, "NonPEPRelationError: {}", desc)
            }
            Error::NoGroupingColumnsError => {
                writeln!(f, "No grouping columns")
            }
            Error::Other(err) => writeln!(f, "{}", err),
        }
    }
}

impl error::Error for Error {}

impl From<transforms::Error> for Error {
    fn from(err: transforms::Error) -> Self {
        Error::Other(err.to_string())
    }
}

impl From<protection::Error> for Error {
    fn from(err: protection::Error) -> Self {
        Error::NonPEPRelationError(err.to_string())
    }
}

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
                    Err(Error::TauThresholdingError(
                        "We support only `Expr::Column` in the group_by".into(),
                    ))
                }
            })
            .collect()
    }

    // Returns a `Relation` outputing all grouping keys that can be safely released
    pub fn grouping_values(&self, epsilon: f64, delta: f64) -> Result<Relation> {
        let grouping_cols = self.grouping_columns()?;
        if !grouping_cols.is_empty() {
            PEPRelation::try_from(self.inputs()[0].clone().filter_fields(|f| {
                grouping_cols.contains(&f.to_string()) || f == PE_ID || f == PE_WEIGHT
            }))?
            .released_values(epsilon, delta)
        } else {
            Err(Error::NoGroupingColumnsError)
        }
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

        let right = Relation::from(self.with_grouping_columns());
        let right_names = right
            .schema()
            .iter()
            .map(|f| f.name().to_string())
            .collect::<Vec<_>>();

        let join_rel: Relation = Relation::join()
            .right(right)
            .right_names(right_names.clone())
            .left(grouping_values)
            .left_outer()
            .on_iter(on)
            .build();

        Ok(join_rel.filter_fields(|f| right_names.contains(&f.to_string())))
    }
}

impl PEPRelation {
    pub fn tau_thresholding_values(self, epsilon: f64, delta: f64) -> Result<Relation> {
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
        Ok(rel
            .filter_columns(filter_column)
            .filter_fields(|f| columns.contains(&f)))
    }

    fn with_tau_thresholding_values(
        self,
        public_columns: &Vec<String>,
        epsilon: f64,
        delta: f64,
    ) -> Result<Relation> {
        let relation_with_private_values =
            Relation::from(self).filter_fields(|f| !public_columns.contains(&f.to_string()));
        PEPRelation::try_from(relation_with_private_values)?.tau_thresholding_values(epsilon, delta)
    }

    pub fn released_values(self, epsilon: f64, delta: f64) -> Result<Relation> {
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
            self.with_public_values(&public_columns)
        } else {
            Ok(self.with_public_values(&public_columns)?.cross_join(
                self.with_tau_thresholding_values(&public_columns, epsilon, delta)?,
            )?)
        }
    }

    pub fn protect_grouping_keys(self, epsilon: f64, delta: f64) -> Result<PEPRelation> {
        let peid = self.protected_entity_id();
        match self.deref() {
            Relation::Table(_) => Ok(PEPRelation::try_from(
                self.released_values(epsilon, delta)?,
            )?),
            Relation::Map(m) => {
                let protected_input = PEPRelation::try_from(m.inputs()[0].clone())?
                    .protect_grouping_keys(epsilon, delta)?;
                let rel: Relation = Map::builder()
                    .with(m.clone())
                    .input(Relation::from(protected_input))
                    .build();
                Ok(PEPRelation::try_from(rel)?)
            }
            Relation::Reduce(r) => Ok(if r.grouping_columns()? == vec![peid.to_string()] {
                self
            } else {
                let columns = r
                    .schema()
                    .iter()
                    .map(|f| f.name().to_string())
                    .collect::<Vec<_>>();
                PEPRelation::try_from(
                    r.clone()
                        .join_with_grouping_values(r.grouping_values(epsilon, delta)?)?
                        .filter_fields(|f| columns.contains(&f.to_string())),
                )?
            }),
            Relation::Join(j) => {
                let rel: Relation = Join::builder()
                    .left(Relation::from(
                        PEPRelation::try_from(j.inputs()[0].clone())?
                            .protect_grouping_keys(epsilon, delta)?,
                    ))
                    .right(Relation::from(
                        PEPRelation::try_from(j.inputs()[1].clone())?
                            .protect_grouping_keys(epsilon, delta)?,
                    ))
                    .with(j.clone())
                    .build();
                Ok(PEPRelation::try_from(rel)?)
            }
            Relation::Set(_) => todo!(),
            Relation::Values(_) => Ok(self),
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{display::Dot, expr::AggregateColumn, namer, relation::Schema};
    use std::rc::Rc;

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

        let rel = protected_table
            .clone()
            .tau_thresholding_values(1., 0.003)
            .unwrap();
        //rel.display_dot();
        assert_eq!(
            rel.data_type(),
            DataType::structured([
                ("a", DataType::integer_range(1..=10)),
                ("b", DataType::integer_values([1, 2, 5, 6, 7, 8])),
                ("c", DataType::integer_range(5..=20))
            ])
        );
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
        let rel = protected_table.released_values(1., 0.003).unwrap();
        matches!(rel, Relation::Join(_));
        //rel.display_dot();
        assert_eq!(
            rel.data_type(),
            DataType::structured([
                ("a", DataType::integer_values([1, 2, 4, 6])),
                ("b", DataType::float_values([1.2, 4.6, 7.8]))
            ])
        );

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
        let rel = protected_table.released_values(1., 0.003).unwrap();
        matches!(rel, Relation::Map(_));
        //rel.display_dot();
        assert_eq!(
            rel.data_type(),
            DataType::structured([
                ("a", DataType::integer_range(1..=10)),
                ("b", DataType::float_range(5.4..=20.))
            ])
        );

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
        let rel = protected_table.released_values(1., 0.003).unwrap();
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
        assert!(red.grouping_values(1., 0.003).is_err());

        // With GROUPBY. Only one column with possible values.
        let red = Reduce::new(
            "reduce_relation".to_string(),
            vec![("sum_a".to_string(), AggregateColumn::sum("a"))],
            vec![Expr::col("b")],
            Rc::new(table.clone()),
        );
        let rel = red.grouping_values(1.0, 0.003).unwrap();
        assert_eq!(
            rel.data_type(),
            DataType::structured([("b", DataType::integer_values([1, 2, 5, 6, 7, 8]))])
        );

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
        let rel = red.grouping_values(1.0, 0.003).unwrap();
        //rel.display_dot();
        assert_eq!(
            rel.data_type(),
            DataType::structured([("c", DataType::integer_range(5..=20)),])
        );

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
        let rel = red.grouping_values(1.0, 0.003).unwrap();
        //rel.display_dot();
        assert_eq!(
            rel.data_type(),
            DataType::structured([
                ("c", DataType::integer_range(5..=20)),
                ("b", DataType::integer_values([1, 2, 5, 6, 7, 8])),
            ])
        );
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
        let pep_rel = relation.force_protect_from_field_paths(&relations, &[("table", &[], "id")]);
        let protected_pep_rel = pep_rel.clone().protect_grouping_keys(1., 0.003).unwrap();
        assert_eq!(Relation::from(pep_rel), Relation::from(protected_pep_rel));

        // With GROUPBY. Only one column with possible values.
        let relation = Relation::from(Reduce::new(
            "reduce_relation".to_string(),
            vec![("sum_a".to_string(), AggregateColumn::sum("a"))],
            vec![Expr::col("b")],
            Rc::new(table.clone()),
        ));
        let pep_rel = relation.force_protect_from_field_paths(&relations, &[("table", &[], "id")]);
        //pep_rel.display_dot();
        let protected_pep_rel = pep_rel.clone().protect_grouping_keys(1., 0.003).unwrap();
        //protected_pep_rel.display_dot();
        assert_eq!(
            protected_pep_rel.data_type(),
            DataType::structured([
                (PE_ID, DataType::text()),
                (PE_WEIGHT, DataType::integer_min(0)),
                ("sum_a", DataType::integer_min(0))
            ])
        );

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
        let pep_rel = relation.force_protect_from_field_paths(&relations, &[("table", &[], "id")]);
        //pep_rel.display_dot();
        let protected_pep_rel = pep_rel.clone().protect_grouping_keys(1., 0.003).unwrap();
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
        let pep_rel = relation.force_protect_from_field_paths(&relations, &[("table", &[], "id")]);
        //pep_rel.display_dot();
        let protected_pep_rel = pep_rel.clone().protect_grouping_keys(1., 0.003).unwrap();
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
        let pep_rel = relation.force_protect_from_field_paths(&relations, &[("table", &[], "id")]);
        //pep_rel.display_dot();
        let protected_pep_rel = pep_rel.clone().protect_grouping_keys(1., 0.003).unwrap();
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
        let pep_rel = relation.force_protect_from_field_paths(&relations, &[("table", &[], "id")]);
        //pep_rel.display_dot();
        namer::reset();
        let protected_pep_rel = pep_rel.clone().protect_grouping_keys(1., 0.003).unwrap();
        namer::reset();
        let correct_protected_rel: Relation = Map::builder()
            .with((PE_ID, Expr::col(PE_ID)))
            .with((PE_WEIGHT, Expr::col(PE_WEIGHT)))
            .with(("twice_sum_a", expr!(2 * sum_a)))
            .with(("b", expr!(b)))
            .with(("c", expr!(c)))
            .name("my_map")
            .input(
                reduce_relation
                    .force_protect_from_field_paths(&relations, &[("table", &[], "id")])
                    .protect_grouping_keys(1., 0.003)
                    .unwrap()
                    .0,
            )
            .build();
        protected_pep_rel.0.display_dot();
        correct_protected_rel.display_dot();
        assert_eq!(protected_pep_rel.0, correct_protected_rel);
    }
}
