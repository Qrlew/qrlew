use itertools::Itertools;
use statrs::{
    distribution::{ContinuousCDF, Normal},
    prec::F64_PREC,
};
use std::fmt;

/// A Private Query
#[derive(Clone, Debug, PartialEq)]
pub enum PrivateQuery {
    Gaussian(f64),
    Laplace(f64),
    EpsilonDelta(f64, f64),
    Composed(Vec<PrivateQuery>),
}

impl PrivateQuery {
    pub fn compose(self, other: Self) -> Self {
        if other.is_null() {
            self
        } else if self.is_null() {
            other
        } else {
            let (v1, v2) = match (self, other) {
                (PrivateQuery::Composed(v1), PrivateQuery::Composed(v2)) => (v1, v2),
                (PrivateQuery::Composed(v), other) => (v, vec![other]),
                (current, PrivateQuery::Composed(v)) => (vec![current], v),
                (current, other) => (vec![current], vec![other]),
            };
            PrivateQuery::Composed(v1.into_iter().chain(v2.into_iter()).collect())
        }
    }

    pub fn null() -> Self {
        Self::EpsilonDelta(0., 0.)
    }

    pub fn is_null(&self) -> bool {
        match self {
            PrivateQuery::Gaussian(n) | PrivateQuery::Laplace(n) => n == &0.0,
            PrivateQuery::EpsilonDelta(e, d) => e == &0. && d == &0.,
            PrivateQuery::Composed(v) => v.iter().all(|q| q.is_null()),
        }
    }

    pub fn gaussian_from_epsilon_delta_sensitivity(
        epsilon: f64,
        delta: f64,
        sensitivity: f64,
    ) -> Self {
        PrivateQuery::Gaussian(gaussian_noise(epsilon, delta, sensitivity))
    }
}

impl fmt::Display for PrivateQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PrivateQuery::Gaussian(n) => writeln!(f, "Gaussian ({})", n),
            PrivateQuery::Laplace(n) => writeln!(f, "Laplace ({})", n),
            PrivateQuery::EpsilonDelta(e, d) => writeln!(f, "EpsilonDelta ({}, {})", e, d),
            PrivateQuery::Composed(v) => write!(
                f,
                "Composed ({})",
                v.iter().map(|pq| format!("{}", pq)).join(", ")
            ),
        }
    }
}

impl From<Vec<PrivateQuery>> for PrivateQuery {
    fn from(v: Vec<PrivateQuery>) -> Self {
        v.into_iter().reduce(|c, q| c.compose(q)).unwrap()
    }
}

pub fn gaussian_noise(epsilon: f64, delta: f64, sensitivity: f64) -> f64 {
    (2. * (1.25_f64 / delta).ln() ).sqrt() * sensitivity / epsilon
}

pub fn gaussian_tau(epsilon: f64, delta: f64, sensitivity: f64) -> f64 {
    let dist = Normal::new(0.0, 1.0).unwrap();
    let scale = gaussian_noise(epsilon, delta, sensitivity);
    // TODO: we want to overestimate tau
    1. + scale * dist.inverse_cdf((1. - delta / 2.).powf(1. / sensitivity)) + F64_PREC
}
