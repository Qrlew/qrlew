use std::{fmt, hash, mem};

use itertools::Itertools;

use super::{implementation, Result};
use crate::data_type::{value::Value, DataType};

/// The list of operators
/// inspired by: https://docs.rs/sqlparser/latest/sqlparser/ast/enum.BinaryOperator.html
/// and mostly: https://docs.rs/polars/latest/polars/prelude/enum.AggExpr.html
/// https://docs.rs/polars-lazy/latest/polars_lazy/dsl/enum.AggExpr.html
#[derive(Clone, Debug, PartialEq)]
pub enum Aggregate {
    Min,
    Max,
    Median,
    NUnique,
    First,
    Last,
    Mean,
    List,
    Count,
    Quantile(f64),
    Quantiles(&'static [f64]),
    Sum,
    AggGroups,
    Std,
    Var,
}

// TODO make sure f64::nan do not happen
impl Eq for Aggregate {}

#[allow(clippy::derive_hash_xor_eq)]
impl hash::Hash for Aggregate {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        mem::discriminant(self).hash(state);
        match self {
            Aggregate::Quantile(q) => {
                mem::discriminant(self).hash(state);
                q.to_be_bytes().hash(state);
            }
            Aggregate::Quantiles(v) => {
                mem::discriminant(self).hash(state);
                v.iter().for_each(|q| q.to_be_bytes().hash(state));
            }
            _ => mem::discriminant(self).hash(state),
        }
    }
}

impl Aggregate {
    /// Return the function object implementing the function
    pub fn super_image(self, set: &DataType) -> Result<DataType> {
        Ok(implementation::aggregate(self).super_image(&set)?)
    }

    /// Return the function object implementing the function
    pub fn value(self, arg: &Value) -> Result<Value> {
        Ok(implementation::aggregate(self).value(&arg)?)
    }
}

impl fmt::Display for Aggregate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Aggregate::Min => write!(f, "min"),
            Aggregate::Max => write!(f, "max"),
            Aggregate::Median => write!(f, "median"),
            Aggregate::NUnique => write!(f, "n_unique"),
            Aggregate::First => write!(f, "first"),
            Aggregate::Last => write!(f, "last"),
            Aggregate::Mean => write!(f, "mean"),
            Aggregate::List => write!(f, "list"),
            Aggregate::Count => write!(f, "count"),
            Aggregate::Quantile(q) => write!(f, "quantile<{q}>"),
            Aggregate::Quantiles(v) => write!(
                f,
                "quantiles<{}>",
                v.iter().map(|q| format!("{q}")).join(", ")
            ),
            Aggregate::Sum => write!(f, "sum"),
            Aggregate::AggGroups => write!(f, "agg_groups"),
            Aggregate::Std => write!(f, "std"),
            Aggregate::Var => write!(f, "var"),
        }
    }
}
