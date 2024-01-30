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
    /// The max_multiplicity in absolute terms
    pub privacy_unit_max_multiplicity: f64,
    /// The max_multiplicity in terms of the dataset size
    pub privacy_unit_max_multiplicity_share: f64,
}

impl DpParameters {
    pub fn new(epsilon: f64, delta: f64, tau_thresholding_share: f64, privacy_unit_max_multiplicity: f64, privacy_unit_max_multiplicity_share: f64) -> DpParameters {
        DpParameters { epsilon, delta, tau_thresholding_share, privacy_unit_max_multiplicity, privacy_unit_max_multiplicity_share }
    }

    pub fn from_epsilon_delta(epsilon: f64, delta: f64) -> DpParameters {
        // These default values are underestimating the bounds
        DpParameters::new(epsilon, delta, 0.5, 100.0, 0.1)
    }

    pub fn with_tau_thresholding_share(self, tau_thresholding_share: f64) -> DpParameters {
        DpParameters { tau_thresholding_share, ..self }
    }

    pub fn with_privacy_unit_max_multiplicity(self, privacy_unit_max_multiplicity: f64) -> DpParameters {
        DpParameters { privacy_unit_max_multiplicity, ..self }
    }

    pub fn with_privacy_unit_max_multiplicity_share(self, privacy_unit_max_multiplicity_share: f64) -> DpParameters {
        DpParameters { privacy_unit_max_multiplicity_share, ..self }
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
