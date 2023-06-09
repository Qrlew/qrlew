use crate::{
    builder::{Ready, With, WithIterator},
    data_type::{
        self,
        intervals::{Bound, Intervals},
    },
    expr::{aggregate, Aggregate, Expr, Value},
    hierarchy::Hierarchy,
    protected::PE_ID,
    relation::{transforms, Join, Map, Reduce, Relation, Set, Table, Variant as _, Visitor},
    visitor::Acceptor,
    DataType,
};
use std::{error, fmt, result};

#[derive(Debug, Clone)]
pub enum Error {
    TauThresholdingError(String),
    Other(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::TauThresholdingError(desc) => {
                writeln!(f, "TauThresholdingError: {}", desc)
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

pub type Result<T> = result::Result<T, Error>;

pub const PE_DISTINCT_COUNT: &str = "_PROTECTED_DISTINCT_COUNT_";

impl Reduce {
    /// Returns a Relation that output the categories for which the noisy count
    /// of DISTINCT PE_ID is greater that tau(epsilon, delat, sensitivty)
    pub fn tau_thresholded_values(
        &self,
        epsilon: f64,
        delta: f64,
        sensitivity: f64,
    ) -> Result<Relation> {
        let tau = super::mechanisms::gaussian_tau(epsilon, delta, sensitivity);
        if self.input.schema().field(PE_ID).is_err() {
            return Err(Error::TauThresholdingError(format!(
                "{PE_ID} column has not been found in the Relation"
            )));
        }

        // compute distinct relation
        let group_by: Vec<String> = self
            .group_by
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
            .collect::<Result<_>>()?;
        let group_by: Vec<&str> = group_by.iter().map(|s| s.as_str()).collect();
        let aggregates = vec![(PE_DISTINCT_COUNT, aggregate::Aggregate::Count)];
        let rel =
            self.input
                .as_ref()
                .clone()
                .distinct_aggregates(PE_ID, group_by.clone(), aggregates);
        let name_sigmas = vec![(
            PE_DISTINCT_COUNT,
            super::mechanisms::gaussian_noise(epsilon, delta, 1.),
        )];
        let rel = rel.add_gaussian_noise(name_sigmas);

        // Returns Map with the right field names and with count(distinct) > tau
        let columns = vec![(PE_DISTINCT_COUNT, Some(tau.into()), None, vec![])];
        Ok(rel.filter_columns(columns))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{display::Dot, relation::Schema};
    use std::rc::Rc;

    #[test]
    fn test_tau_thresholded_values_reduce() {
        let table: Relation = Relation::table()
            .name("table")
            .schema(
                Schema::builder()
                    .with(("a", DataType::integer_range(1..=10)))
                    .with(("b", DataType::integer_values([1, 2, 5, 6, 7, 8])))
                    .with(("c", DataType::integer_range(5..=20)))
                    .with((PE_ID, DataType::integer_range(1..=100)))
                    .build(),
            )
            .build();

        // Without GROUPBY
        let red = Reduce::new(
            "reduce_relation".to_string(),
            vec![("sum_a".to_string(), Expr::sum(Expr::col("a")))],
            vec![],
            Rc::new(table.clone()),
        );
        let rel = red.tau_thresholded_values(1., 0.003, 5.).unwrap();
        rel.display_dot();
        assert_eq!(rel.schema().fields().len(), 1);

        // With GROUPBY
        let red = Reduce::new(
            "reduce_relation".to_string(),
            vec![("sum_a".to_string(), Expr::sum(Expr::col("a")))],
            vec![Expr::col("b")],
            Rc::new(table.clone()),
        );
        let rel = red.tau_thresholded_values(1.0, 0.003, 1.).unwrap();
        rel.display_dot();
        assert_eq!(rel.schema().fields().len(), 2);
    }
}
