use std::{
    hash::Hash,
    cmp::Eq,
};

/// Represent a simple privacy budget
#[derive(Clone, Debug, PartialEq)]
pub struct Budget {
    epsilon: f64,
    delta: f64,
}

impl Hash for Budget {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Hash::hash(&self.epsilon.to_be_bytes(), state);
        Hash::hash(&self.delta.to_be_bytes(), state);
    }
}

impl Eq for Budget {}