use super::{DpRelation, Ready, Reduce, Relation, Result, With};
use crate::privacy_unit_tracking::PupRelation;
use std::{cmp::Eq, hash::Hash};

/// Represent a simple privacy budget
#[derive(Clone, Debug, PartialEq)]
pub struct DpParameters {
    pub epsilon: f64,
    pub delta: f64,
    /// Tau-thresholding share
    pub tau_thresholding_share: f64,
    /// The concentration parameter used to compute clipping
    pub clipping_concentration: f64,
    /// The quantile parameter used to compute clipping
    pub clipping_quantile: f64,
}

impl DpParameters {
    pub fn new(epsilon: f64, delta: f64, tau_thresholding_share: f64, clipping_concentration: f64, clipping_quantile: f64) -> DpParameters {
        DpParameters { epsilon, delta, tau_thresholding_share, clipping_concentration, clipping_quantile }
    }

    pub fn from_epsilon_delta(epsilon: f64, delta: f64) -> DpParameters {
        DpParameters::new(epsilon, delta, 0.5, 0.01, 0.9)
    }
}

impl DpParameters {
    pub fn reduce(&self, reduce: &Reduce, input: PupRelation) -> Result<DpRelation> {
        let reduce: Reduce = Relation::reduce()
            .with(reduce.clone())
            .input(Relation::from(input))
            .build();
        reduce.differentially_private(&self)
    }
}

impl Hash for DpParameters {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Hash::hash(&self.epsilon.to_be_bytes(), state);
        Hash::hash(&self.delta.to_be_bytes(), state);
    }
}

impl Eq for DpParameters {}
