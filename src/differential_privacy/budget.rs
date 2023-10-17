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
}

impl Hash for Budget {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Hash::hash(&self.epsilon.to_be_bytes(), state);
        Hash::hash(&self.delta.to_be_bytes(), state);
    }
}

impl Eq for Budget {}