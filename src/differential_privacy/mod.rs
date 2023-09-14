//! # Methods to transform `Relation`s into differentially private ones
//!
//! This is experimental and little tested yet.
//!

pub mod mechanisms;
pub mod protect_grouping_keys;

use crate::{
    Ready,
    data_type::DataTyped,
    expr::{self, aggregate, Expr, AggregateColumn},
    hierarchy::Hierarchy,
    relation::{field::Field, transforms, Map, Reduce, Relation, Variant as _},
    DataType,
    display::Dot,
    protection::PEPRelation,
    builder::With,
};
use std::collections::{HashMap, HashSet};
use std::ops::Deref;
use std::process::Output;
use std::{cmp, error, fmt, rc::Rc, result};

#[derive(Debug, PartialEq)]
pub enum Error {
    InvalidRelation(String),
    Other(String),
}

impl Error {
    pub fn invalid_relation(relation: impl fmt::Display) -> Error {
        Error::InvalidRelation(format!("{} is invalid", relation))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidRelation(relation) => writeln!(f, "{} invalid.", relation),
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

impl error::Error for Error {}
pub type Result<T> = result::Result<T, Error>;

/// A DP Relation
#[derive(Clone, Debug)]
pub struct DPRelation(pub Relation);

impl From<DPRelation> for Relation {
    fn from(value: DPRelation) -> Self {
        value.0
    }
}

impl Deref for DPRelation {
    type Target = Relation;

    fn deref(&self) -> &Self::Target {
        &self.0
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

    /// Compile a protected Relation into DP
    pub fn dp_compile(self, epsilon: f64, delta: f64) -> Result<DPRelation> {// Return a DP relation
        let protected_entity_id = self.protected_entity_id().to_string();
        let protected_entity_weight = self.protected_entity_weight().to_string();
        match Relation::from(self) {
            Relation::Map(map) => {
                let dp_input = PEPRelation(map.input().clone()).dp_compile(epsilon, delta)?;
                Ok(DPRelation(Map::builder().with(map).input(Relation::from(dp_input)).build()))
            },
            Relation::Reduce(reduce) => reduce.dp_compile_sums(&protected_entity_id, &protected_entity_weight, epsilon, delta),
            relation => Err(Error::invalid_relation(relation))
        }
    }
}

/* Reduce
 */
impl Reduce {
    /// DP compile the sums
    fn dp_compile_sums(self, protected_entity_id: &str, protected_entity_weight: &str, epsilon: f64, delta: f64) -> Result<DPRelation> {
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
            } else if aggregate.aggregate() == &aggregate::Aggregate::Sum && name != protected_entity_weight {// add aggregate
                let input_data_type = self.input().schema()[input_name].data_type();
                let absolute_bound = input_data_type.absolute_upper_bound().unwrap_or(1.0);
                input_values_bound.push((input_name, absolute_bound));
            }
        };
        // Check that groups are public
        if !input_groups.iter().all(|e| match self.input().schema()[*e].data_type() {// TODO improve this
            DataType::Boolean(b) if b.all_values() => true,
            DataType::Integer(i) if i.all_values() => true,
            DataType::Enum(e) => true,
            DataType::Float(f) if f.all_values() => true,
            DataType::Text(t) if t.all_values() => true,
            _ => false,
        }) {
            //return Err(Error::invalid_relation(self));
            println!("GROUPS SHOULD BE PUBLIC")
        };
        // Clip the relation
        let clipped_relation = self.input().clone().l2_clipped_sums(
            input_entities.unwrap(),
            input_groups.into_iter().collect(),
            input_values_bound.iter().cloned().collect(),
        );
        let noise_multiplier = 1.; // TODO set this properly
        let dp_clipped_relation = clipped_relation.add_gaussian_noise(input_values_bound.into_iter().map(|(name, bound)| (name,noise_multiplier*bound)).collect());
        let renamed_dp_clipped_relation = dp_clipped_relation.rename_fields(|n, e| names.get(n).unwrap_or(&n).to_string());
        Ok(DPRelation(renamed_dp_clipped_relation))
    }

    /// Rewrite aggregations as sums and compile sums
    pub fn dp_compile(self, protected_entity_id: &str, protected_entity_weight: &str, epsilon: f64, delta: f64) -> Result<DPRelation> {
        let mut output = Map::builder();
        let mut sums = Reduce::builder();
        // Add aggregate colums
        for (name, aggregate) in self.named_aggregates().into_iter() {
            match aggregate.aggregate() {
                aggregate::Aggregate::First => {
                    sums = sums.with((aggregate.column_name()?, AggregateColumn::col(aggregate.column_name()?)));
                },
                aggregate::Aggregate::Mean => {
                    let sum_col = &format!("_SUM_{}", aggregate.column_name()?);
                    let count_col = &format!("_COUNT_{}", aggregate.column_name()?);
                    sums = sums
                        .with((count_col, Expr::sum(Expr::val(1.))))
                        .with((sum_col, Expr::sum(Expr::col(aggregate.column_name()?))));
                    output = output
                        .with((name, Expr::divide(Expr::col(sum_col), Expr::greatest(Expr::val(1.), Expr::col(count_col)))))
                },
                aggregate::Aggregate::Count => {
                    let count_col = &format!("_COUNT_{}", aggregate.column_name()?);
                    sums = sums.with((count_col, Expr::sum(Expr::val(1.))));
                    output = output.with((name, Expr::col(count_col)));
                },
                aggregate::Aggregate::Sum if aggregate.column_name()? != protected_entity_weight => {
                    let sum_col = &format!("_SUM_{}", aggregate.column_name()?);
                    sums = sums.with((sum_col, Expr::sum(Expr::col(aggregate.column_name()?))));
                    output = output.with((name, Expr::col(sum_col)));
                },
                aggregate::Aggregate::Std => todo!(),
                aggregate::Aggregate::Var => todo!(),
                _ => (),
            }
        }
        sums = sums.group_by_iter(self.group_by().iter().cloned());
        let sums: Reduce = sums.input(self.input().clone()).build();
        let dp_sums: Relation = sums.dp_compile_sums(protected_entity_id, protected_entity_weight, epsilon, delta)?.into();
        Ok(DPRelation(output.input(dp_sums).build()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ast,
        builder::{Ready, With},
        display::Dot,
        io::{postgresql, Database},
        relation::Variant as _,
        sql::parse,
        Relation,
    };
    use colored::Colorize;
    use itertools::Itertools;

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

        let table = relations
            .get(&["item_table".into()])
            .unwrap()
            .as_ref()
            .clone();

        // with GROUP BY
        let relation: Relation = Relation::reduce()
            .input(table.clone())
            .with(("sum_price", Expr::sum(Expr::col("price"))))
            .with(("count_price", Expr::count(Expr::col("price"))))
            .with(("mean_price", Expr::mean(Expr::col("price"))))
            // .with_group_by_column("order_id")
            .group_by(Expr::col("order_id"))
            .build();
        relation.display_dot().unwrap();

        let pep_relation = relation.force_protect_from_field_paths(
            &relations,
            &[
                (
                    "item_table",
                    &[
                        ("order_id", "order_table", "id"),
                        ("user_id", "user_table", "id"),
                    ],
                    "name",
                ),
                ("order_table", &[("user_id", "user_table", "id")], "name"),
                ("user_table", &[], "name"),
            ],
        );
        pep_relation.display_dot().unwrap();

        let epsilon = 1.;
        let delta = 1e-3;
        let dp_relation = pep_relation.dp_compile(epsilon, delta).unwrap();
        dp_relation.display_dot().unwrap();
    }
}
