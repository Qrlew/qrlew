//! # Methods to transform `Relation`s into differentially private ones
//!
//! This is experimental and little tested yet.
//!

pub mod mechanisms;
pub mod protect_grouping_keys;

use crate::data_type::DataTyped;
use crate::{
    expr::{aggregate, Expr},
    hierarchy::Hierarchy,
    protected::PE_ID,
    relation::{field::Field, transforms, Reduce, Relation, Variant as _},
    DataType,
};
use std::{cmp, error, fmt, rc::Rc, result};

#[derive(Debug, PartialEq)]
pub enum Error {
    Other(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Other(err) => writeln!(f, "{}", err),
        }
    }
}

impl From<transforms::Error> for Error {
    fn from(err: transforms::Error) -> Self {
        Error::Other(err.to_string())
    }
}

impl error::Error for Error {}
pub type Result<T> = result::Result<T, Error>;

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

/* Reduce
 */
impl Reduce {
    pub fn dp_compilation<'a>(
        self,
        relations: &'a Hierarchy<Rc<Relation>>,
        protected_entity: &'a [(&'a str, &'a [(&'a str, &'a str, &'a str)], &'a str)],
        epsilon: f64,
        delta: f64,
    ) -> Result<Relation> {
        let multiplicity = 1; // TODO
        let (clipping_values, name_sigmas): (Vec<(String, f64)>, Vec<(String, f64)>) = self
            .schema()
            .clone()
            .iter()
            .zip(self.aggregate().clone().into_iter())
            .fold((vec![], vec![]), |(c, s), (f, x)| {
                if let (name, Expr::Aggregate(agg)) = (f.name(), x) {
                    match agg.aggregate() {
                        aggregate::Aggregate::Sum => {
                            let mut c = c;
                            let cvalue = self
                                .input()
                                .schema()
                                .field(agg.argument_name().unwrap())
                                .unwrap()
                                .clone()
                                .clipping_value(multiplicity);
                            c.push((agg.argument_name().unwrap().to_string(), cvalue));
                            let mut s = s;
                            s.push((
                                name.to_string(),
                                mechanisms::gaussian_noise(epsilon, delta, cvalue),
                            ));
                            (c, s)
                        }
                        _ => (c, s),
                    }
                } else {
                    (c, s)
                }
            });

        let clipping_values = clipping_values
            .iter()
            .map(|(n, v)| (n.as_str(), *v))
            .collect();
        let clipped_relation = self.clip_aggregates(PE_ID, clipping_values)?;

        let name_sigmas = name_sigmas.iter().map(|(n, v)| (n.as_str(), *v)).collect();
        Ok(clipped_relation.add_gaussian_noise(name_sigmas))
    }
}

impl Relation {
    pub fn dp_compilation<'a>(
        self,
        relations: &'a Hierarchy<Rc<Relation>>,
        protected_entity: &'a [(&'a str, &'a [(&'a str, &'a str, &'a str)], &'a str)],
        epsilon: f64,
        delta: f64,
    ) -> Result<Relation> {
        let protected_relation = self.force_protect_from_field_paths(relations, protected_entity);
        match protected_relation {
            Relation::Reduce(reduce) => {
                reduce.dp_compilation(relations, protected_entity, epsilon, delta)
            }
            _ => todo!(),
        }
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
    fn test_dp_compilation() {
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
            .with_group_by_column("order_id")
            .build();

        let epsilon = 1.;
        let delta = 1e-3;
        let dp_relation = relation
            .dp_compilation(
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
                epsilon,
                delta,
            )
            .unwrap();
        dp_relation.display_dot().unwrap();

        let query: &str = &ast::Query::from(&dp_relation).to_string();
        println!("Query: {}", query);
        let my_res = database.query(query).unwrap();
        println!("\n\nresults = {:?}", my_res);
    }
}
