use super::{DpRelation, Ready, Reduce, Relation, Result, With};
use crate::{privacy_unit_tracking::PupRelation, relation::Variant};
use std::{cmp::Eq, hash::Hash};

/// Represent a simple privacy budget
#[derive(Clone, Debug, PartialEq)]
pub struct DpParameters {
    epsilon: f64,
    delta: f64,
    /// Tau-thresholding share
    tau_thresholding_share: f64,
    /// The concentration parameter used to compute clipping
    clipping_concentration: f64,
    /// The quantile parameter used to compute clipping
    clipping_quantile: f64,
}

impl DpParameters {
    pub fn new(epsilon: f64, delta: f64, tau_thresholding_share: f64, clipping_concentration: f64, clipping_quantile: f64) -> DpParameters {
        DpParameters { epsilon, delta, tau_thresholding_share, clipping_concentration, clipping_quantile }
    }

    pub fn from_epsilon_delta(epsilon: f64, delta: f64) -> DpParameters {
        DpParameters::new(epsilon, delta, 0.5, 0.01, 0.9)
    }

    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    pub fn delta(&self) -> f64 {
        self.delta
    }

    pub fn epsilon_aggregation(&self, tau_thresholding: bool) -> f64 {
        if tau_thresholding {
            self.epsilon*(1.-self.tau_thresholding_share)
        } else {
            self.epsilon
        }
    }

    pub fn delta_aggregation(&self, tau_thresholding: bool) -> f64 {
        if tau_thresholding {
            self.delta*(1.-self.tau_thresholding_share)
        } else {
            self.delta
        }
    }

    pub fn epsilon_tau_thresholding(&self) -> f64 {
        self.epsilon*self.tau_thresholding_share
    }

    pub fn delta_tau_thresholding(&self) -> f64 {
        self.delta*self.tau_thresholding_share
    }
}

impl DpParameters {
    pub fn reduce(&self, reduce: &Reduce, input: PupRelation) -> Result<DpRelation> {
        let reduce: Reduce = Relation::reduce()
            .with(reduce.clone())
            .input(Relation::from(input))
            .build();
        let (epsilon, delta, epsilon_tau_thresholding, delta_tau_thresholding) =
            if reduce.group_by().is_empty() {
                let tau_thresholding = false;
                (self.epsilon_aggregation(tau_thresholding), self.delta_aggregation(tau_thresholding), 0., 0.)
            } else {
                let tau_thresholding = true;
                (self.epsilon_aggregation(tau_thresholding), self.delta_aggregation(tau_thresholding), self.epsilon_tau_thresholding(), self.delta_tau_thresholding())
            };
        reduce.differentially_private(
            epsilon,
            delta,
            epsilon_tau_thresholding,
            delta_tau_thresholding,
        )
    }
}

impl Hash for DpParameters {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Hash::hash(&self.epsilon.to_be_bytes(), state);
        Hash::hash(&self.delta.to_be_bytes(), state);
    }
}

impl Eq for DpParameters {}
