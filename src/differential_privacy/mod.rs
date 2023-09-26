//! # Methods to transform `Relation`s into differentially private ones
//!
//! This is experimental and little tested yet.
//!

pub mod mechanisms;
pub mod private_query;
pub mod protect_grouping_keys;

use crate::{
    builder::With,
    data_type::DataTyped,
    differential_privacy::private_query::PrivateQuery,
    expr::{self, aggregate, AggregateColumn, Expr},
    protection::{self, PEPRelation},
    relation::{field::Field, transforms, Join, Map, Reduce, Relation, Variant as _},
    DataType, Ready,
};
use std::collections::{HashMap, HashSet};
use std::{cmp, error, fmt, result};

#[derive(Debug, PartialEq)]
pub enum Error {
    InvalidRelation(String),
    UnsafeGroups(String),
    GroupingKeysError(String),
    Other(String),
}

impl Error {
    pub fn invalid_relation(relation: impl fmt::Display) -> Error {
        Error::InvalidRelation(format!("{relation} is invalid"))
    }
    pub fn unsafe_groups(groups: impl fmt::Display) -> Error {
        Error::UnsafeGroups(format!("{groups} should be public"))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidRelation(relation) => writeln!(f, "{relation} invalid."),
            Error::UnsafeGroups(groups) => writeln!(f, "{groups} should be public."),
            Error::GroupingKeysError(desc) => writeln!(f, "GroupingKeysError: {}", desc),
            Error::Other(err) => writeln!(f, "{}", err),
        }
    }
}

impl From<expr::Error> for Error {
    fn from(err: expr::Error) -> Self {
        Error::Other(err.to_string())
    }
}
impl From<transforms::Error> for Error {
    fn from(err: transforms::Error) -> Self {
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

impl From<DPRelation> for (Relation, PrivateQuery) {
    fn from(value: DPRelation) -> Self {
        (value.relation, value.private_query)
    }
}

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

impl PEPRelation {
    // /// Compile a protected Relation into DP
    // pub fn dp_compile_sums(self, epsilon: f64, delta: f64) -> Result<DPRelation> {// Return a DP relation
    //     let protected_entity_id = self.protected_entity_id().to_string();
    //     let protected_entity_weight = self.protected_entity_weight().to_string();
    //     if let PEPRelation(Relation::Reduce(reduce)) = self {
    //         reduce.dp_compile_sums(&protected_entity_id, &protected_entity_weight, epsilon, delta)
    //     } else {
    //         Err(Error::invalid_relation(self.0))
    //     }
    // }

    pub fn dp_compile_aggregates(self, epsilon: f64, delta: f64) -> Result<DPRelation> {
        let protected_entity_id = self.protected_entity_id().to_string();
        let protected_entity_weight = self.protected_entity_weight().to_string();

        // Return a DP relation
        let (dp_relation, private_query) = match Relation::from(self) {
            Relation::Map(map) => {
                let dp_input = PEPRelation::try_from(map.input().clone())?
                    .dp_compile_aggregates(epsilon, delta)?;
                let relation = Map::builder()
                    .filter_fields_with(map, |f| {
                        f != protected_entity_id.as_str() && f != protected_entity_weight.as_str()
                    })
                    .input(dp_input.relation().clone())
                    .build();
                Ok(DPRelation::new(relation, dp_input.private_query().clone()))
            }
            Relation::Reduce(reduce) => reduce.dp_compile_aggregates(
                &protected_entity_id,
                &protected_entity_weight,
                epsilon,
                delta,
            ),
            Relation::Table(_) => todo!(),
            Relation::Join(j) => {
                let (left_dp_relation, left_private_query) =
                    PEPRelation::try_from(j.inputs()[0].clone())?
                        .dp_compile_aggregates(epsilon, delta)?
                        .into();
                let (right_dp_relation, right_private_query) =
                    PEPRelation::try_from(j.inputs()[1].clone())?
                        .dp_compile_aggregates(epsilon, delta)?
                        .into();
                let relation: Relation = Join::builder()
                    .left(left_dp_relation)
                    .right(right_dp_relation)
                    .with(j.clone())
                    .build();
                Ok(DPRelation::new(
                    relation,
                    vec![left_private_query, right_private_query].into(),
                ))
            }
            Relation::Set(_) => todo!(),
            Relation::Values(_) => todo!(),
        }?
        .into();

        Ok(DPRelation::new(dp_relation, private_query))
    }

    /// Compile a protected Relation into DP
    pub fn dp_compile(
        self,
        epsilon: f64,
        delta: f64,
        epsilon_tau_thresholding: f64,
        delta_tau_thresholding: f64,
    ) -> Result<DPRelation> {
        let (relation_with_protected_keys, private_query_grouping_keys) =
            self.protect_grouping_keys(epsilon_tau_thresholding, delta_tau_thresholding)?;

        let (relation_with_dp_aggs, private_query_dp_aggs) = relation_with_protected_keys
            .dp_compile_aggregates(epsilon, delta)?
            .into();

        Ok(DPRelation::new(
            relation_with_dp_aggs,
            vec![private_query_grouping_keys, private_query_dp_aggs].into(),
        ))
    }
}

/* Reduce
 */
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

    /// Rewrite aggregations as sums and compile sums
    pub fn dp_compile_aggregates(
        self,
        protected_entity_id: &str,
        protected_entity_weight: &str,
        epsilon: f64,
        delta: f64,
    ) -> Result<DPRelation> {
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

impl Relation {
    fn gaussian_mechanism(self, epsilon: f64, delta: f64, bounds: Vec<(&str, f64)>) -> DPRelation {
        let noise_multiplier = mechanisms::gaussian_noise(epsilon, delta, 1.0); // TODO set this properly
        let noise_multipliers = bounds
            .into_iter()
            .map(|(name, bound)| (name, noise_multiplier * bound))
            .collect::<Vec<_>>();
        let private_query = noise_multipliers
            .iter()
            .map(|(_, n)| PrivateQuery::Gaussian(*n))
            .collect::<Vec<_>>()
            .into();
        DPRelation::new(self.add_gaussian_noise(noise_multipliers), private_query)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ast,
        builder::With,
        display::Dot,
        io::{postgresql, Database},
        sql::parse,
        Relation,
    };

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
    fn test_dp_compile() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        let query = parse(
            "SELECT price
        FROM item_table WHERE order_id IN (1,2,3,4,5,6,7,8,9,10)",
        )
        .unwrap();
        let relation = Relation::try_from(query.with(&relations)).unwrap();
        relation.display_dot().unwrap();

        let query = parse(
            "SELECT order_id, sum(price) AS sum_price,
        count(price) AS count_price,
        avg(price) AS mean_price
        FROM item_table WHERE order_id IN (1,2,3,4,5,6,7,8,9,10) GROUP BY order_id",
        )
        .unwrap();
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

        let epsilon = 1.;
        let delta = 1e-3;
        let epsilon_tau_thresholding = 1.;
        let delta_tau_thresholding = 1e-3;
        let (dp_relation, private_query) = pep_relation
            .dp_compile(
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
                PrivateQuery::Gaussian(mechanisms::gaussian_noise(epsilon, delta, 50.0)),
                PrivateQuery::Gaussian(mechanisms::gaussian_noise(epsilon, delta, 1.0)),
            ]
            .into()
        );
        let dp_query = ast::Query::from(&dp_relation);
        for row in database.query(&dp_query.to_string()).unwrap() {
            println!("{row}");
        }
    }

    #[test]
    fn test_dp_compile_simple() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        // GROUPING col in the SELECT clause
        let str_query = "SELECT z, sum(x) AS sum_x FROM table_2 GROUP BY z";
        let query = parse(str_query).unwrap();
        let relation = Relation::try_from(query.with(&relations)).unwrap();

        let pep_relation =
            relation.force_protect_from_field_paths(&relations, vec![("table_2", vec![], "y")]);

        let (dp_relation, private_query) =
            pep_relation.dp_compile(1., 1e-3, 1., 1e-3).unwrap().into();
        dp_relation.display_dot().unwrap();

        assert_eq!(
            dp_relation.data_type()["z"],
            DataType::text_values(["Foo".into(), "Bar".into()])
        );
        assert!(matches!(
            dp_relation.data_type()["sum_x"],
            DataType::Float(_)
        ));
        assert_eq!(
            private_query,
            vec![PrivateQuery::Gaussian(mechanisms::gaussian_noise(
                1., 1e-3, 100.0
            )),]
            .into()
        );
        assert_eq!(dp_relation.schema().len(), 2);
        let dp_query = ast::Query::from(&dp_relation);
        database.query(&dp_query.to_string()).unwrap();

        // GROUPING col NOT in the SELECT clause
        let str_query = "SELECT sum(x) AS sum_x FROM table_2 GROUP BY z";
        let query = parse(str_query).unwrap();
        let relation = Relation::try_from(query.with(&relations)).unwrap();

        let pep_relation =
            relation.force_protect_from_field_paths(&relations, vec![("table_2", vec![], "y")]);

        let (dp_relation, private_query) =
            pep_relation.dp_compile(1., 1e-3, 1., 1e-3).unwrap().into();
        //dp_relation.display_dot().unwrap();
        assert_eq!(
            private_query,
            vec![PrivateQuery::Gaussian(mechanisms::gaussian_noise(
                1., 1e-3, 100.0
            )),]
            .into()
        );
        assert_eq!(dp_relation.schema().len(), 1);
        assert!(matches!(
            dp_relation.data_type()["sum_x"],
            DataType::Float(_)
        ));
        let dp_query = ast::Query::from(&dp_relation);
        database.query(&dp_query.to_string()).unwrap();

        // GROUPING col has no possible values
        let str_query = "SELECT y, sum(x) AS sum_x FROM table_2 GROUP BY y";
        let query = parse(str_query).unwrap();
        let relation = Relation::try_from(query.with(&relations)).unwrap();

        let pep_relation =
            relation.force_protect_from_field_paths(&relations, vec![("table_2", vec![], "z")]);

        let (dp_relation, private_query) =
            pep_relation.dp_compile(1., 1e-3, 1., 1e-3).unwrap().into();
        dp_relation.display_dot().unwrap();
        assert_eq!(
            private_query,
            vec![
                PrivateQuery::EpsilonDelta(1., 1e-3),
                PrivateQuery::Gaussian(mechanisms::gaussian_noise(1., 1e-3, 100.0)),
            ]
            .into()
        );
        assert_eq!(dp_relation.schema().len(), 2);
        assert!(matches!(
            dp_relation.data_type()["sum_x"],
            DataType::Float(_)
        ));
        assert_eq!(
            dp_relation.data_type()["y"],
            DataType::optional(DataType::text())
        );
        let dp_query = ast::Query::from(&dp_relation);
        database.query(&dp_query.to_string()).unwrap();
    }
}
