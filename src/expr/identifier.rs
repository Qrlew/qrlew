use std::{fmt, ops, vec};

use super::{Error, Result};
use crate::{builder::With, hierarchy::Path};

/// The list of operators
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Identifier(Vec<String>);

impl Identifier {
    /// An empty identifier
    pub fn empty() -> Self {
        Identifier(vec![])
    }

    /// Return the first element
    /// Panics if len == 0
    pub fn head(&self) -> Result<String> {
        self.get(0)
            .cloned()
            .ok_or_else(|| Error::invalid_expression("Identifier too short"))
    }

    /// Return the tail
    pub fn tail(&self) -> Result<Identifier> {
        Ok(self
            .0
            .split_first()
            .ok_or(Error::invalid_expression("Identifier too short"))?
            .1
            .iter()
            .collect())
    }

    pub fn from_name<S: Into<String>>(name: S) -> Identifier {
        Identifier(vec![name.into()])
    }

    pub fn from_qualified_name<S: Into<String>>(path: S, name: S) -> Identifier {
        Identifier(vec![path.into(), name.into()])
    }

    pub fn split_first(&self) -> Result<(String, Identifier)> {
        let (head, tail) = self.0.split_first().ok_or(Error::other("Split failed"))?;
        Ok((head.clone(), Identifier::from(tail.to_vec())))
    }

    pub fn split_last(&self) -> Result<(String, Identifier)> {
        let (child, parent) = self.0.split_last().ok_or(Error::other("Split failed"))?;
        Ok((child.clone(), Identifier::from(parent.to_vec())))
    }

    pub fn to_vec(&self) -> Vec<String> {
        self.0.cloned()
    }
}

impl With<String> for Identifier {
    fn with(self, input: String) -> Self {
        let mut result = self.0;
        result.push(input);
        Identifier(result)
    }
}

impl With<(usize, String)> for Identifier {
    fn with(self, (index, input): (usize, String)) -> Self {
        let mut result = self.0;
        result.insert(index, input);
        Identifier(result)
    }
}

impl From<&str> for Identifier {
    fn from(name: &str) -> Self {
        Identifier::from_name(name)
    }
}

impl From<String> for Identifier {
    fn from(name: String) -> Self {
        Identifier::from_name(name)
    }
}

impl From<Vec<String>> for Identifier {
    fn from(qualified_name: Vec<String>) -> Self {
        Identifier(qualified_name)
    }
}

impl<const N: usize> From<[&str; N]> for Identifier {
    fn from(qualified_name: [&str; N]) -> Self {
        Identifier(qualified_name.into_iter().map(String::from).collect())
    }
}

impl<S: Into<String>> FromIterator<S> for Identifier {
    fn from_iter<T: IntoIterator<Item = S>>(iter: T) -> Self {
        Identifier(iter.into_iter().map(|s| s.into()).collect())
    }
}

impl IntoIterator for Identifier {
    type Item = String;
    type IntoIter = vec::IntoIter<String>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl ops::Deref for Identifier {
    type Target = Vec<String>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<[String]> for Identifier {
    fn as_ref(&self) -> &[String] {
        &self.0
    }
}

impl fmt::Display for Identifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.join("."))
    }
}

impl Path for Identifier {
    fn path(self) -> Vec<String> {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identifier() {
        let id: Identifier = ["hello", "world"].into();
        println!("{id}");
        assert!(id.len() == 2)
    }

    #[test]
    fn test_head_tail() {
        let id: Identifier = ["good", "morning", "Qrlew", "!"].into();
        println!("id = {id}");
        println!("head = {}", id.head().unwrap());
        println!("tail = {}", id.tail().unwrap());
        println!("tail^2 = {}", id.tail().unwrap().tail().unwrap());
        assert!(id.tail().unwrap().tail().unwrap().len() == 2)
    }
}
