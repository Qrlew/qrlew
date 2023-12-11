use super::{DPRelation, Ready, Reduce, Relation, Result, With};
use crate::{privacy_unit_tracking::PUPRelation, relation::Variant};
use std::{cmp::Eq, hash::Hash};

/// Represent a simple privacy budget
#[derive(Clone, Debug, PartialEq)]
pub struct Budget {
    epsilon: f64,
    delta: f64,
}

impl Budget {
    pub fn new(epsilon: f64, delta: f64) -> Budget {
        Budget { epsilon, delta }
    }

    pub fn delta(&self) -> f64 {
        self.delta
    }

    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    pub fn split_half(&self) -> (f64, f64, f64, f64) {
        (
            self.epsilon / 2.,
            self.delta / 2.,
            self.epsilon / 2.,
            self.delta / 2.,
        )
    }
}

impl Budget {
    pub fn reduce(&self, reduce: &Reduce, input: PUPRelation) -> Result<DPRelation> {
        print!("DEBUG input {}", Relation::from(input.clone()).schema());
        let reduce: Reduce = Relation::reduce()
            .with(reduce.clone())
            .input(Relation::from(input))
            .build();
        print!("DEBUG reduce input {}", reduce.input().schema());
        let (epsilon, delta, epsilon_tau_thresholding, delta_tau_thresholding) =
            if reduce.group_by().is_empty() {
                (self.epsilon, self.delta, 0., 0.)
            } else {
                self.split_half()
            };
        reduce.differentially_private(
            epsilon,
            delta,
            epsilon_tau_thresholding,
            delta_tau_thresholding,
        )
    }
}

impl Hash for Budget {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Hash::hash(&self.epsilon.to_be_bytes(), state);
        Hash::hash(&self.delta.to_be_bytes(), state);
    }
}

impl Eq for Budget {}
