//! # Hierarchy structure
//!
//! A map with paths as keys.
//! Suffix of paths are valid keys when non-ambiguous.
//!

use core::fmt;
use itertools::Itertools;
use std::{
    collections::BTreeMap,
    iter::Extend,
    ops::{Deref, DerefMut, Index},
};

use crate::builder::With;

/// How many times is the element
enum Found<T> {
    Zero,
    One(T),
    More,
}

/// Found can be converted to Option
impl<T> From<Found<T>> for Option<T> {
    fn from(value: Found<T>) -> Self {
        match value {
            Found::One(t) => Some(t),
            _ => None,
        }
    }
}

/// A trait Path to manage conversions
pub trait Path: Clone {
    fn path(self) -> Vec<String>;
    fn cloned(&self) -> Vec<String> {
        self.clone().path()
    }
}

impl Path for &str {
    fn path(self) -> Vec<String> {
        vec![self.to_string()]
    }
}

impl Path for String {
    fn path(self) -> Vec<String> {
        vec![self]
    }
}

impl<const N: usize> Path for [&str; N] {
    fn path(self) -> Vec<String> {
        self.iter().map(|s| s.to_string()).collect()
    }
}

impl<const N: usize> Path for [String; N] {
    fn path(self) -> Vec<String> {
        self.iter().cloned().collect()
    }
}

impl Path for &[&str] {
    fn path(self) -> Vec<String> {
        self.iter().map(|s| s.to_string()).collect()
    }
}

impl Path for &[String] {
    fn path(self) -> Vec<String> {
        self.iter().cloned().collect()
    }
}

impl Path for Vec<&str> {
    fn path(self) -> Vec<String> {
        self.iter().map(|s| s.to_string()).collect()
    }
}

impl Path for &Vec<String> {
    fn path(self) -> Vec<String> {
        self.clone()
    }
}

impl Path for Vec<String> {
    fn path(self) -> Vec<String> {
        self
    }
}

// Utility fuunctions for path comparisons

fn is_prefix_of(left: &[String], right: &[String]) -> bool {
    left.iter().zip(right.iter()).all(|(pr, pa)| pr == pa)
}

/// A utility function to check a path is a suffix of another
fn is_suffix_of(left: &[String], right: &[String]) -> bool {
    left.iter()
        .rev()
        .zip(right.iter().rev())
        .all(|(s, p)| s == p)
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Hierarchy<T: Clone>(BTreeMap<Vec<String>, T>);

impl<T: Clone> Hierarchy<T> {
    pub fn new(objects: BTreeMap<Vec<String>, T>) -> Self {
        Hierarchy(objects)
    }

    pub fn empty() -> Self {
        Hierarchy::new(BTreeMap::new())
    }

    pub fn chain(self, other: Self) -> Self {
        self.into_iter().chain(other.into_iter()).collect()
    }

    pub fn prepend(self, head: &[String]) -> Self {
        self.into_iter()
            .map(|(s, d)| {
                (
                    head.iter()
                        .map(|s| s.clone())
                        .chain(s.into_iter())
                        .collect::<Vec<String>>(),
                    d,
                )
            })
            .collect()
    }

    pub fn get(&self, path: &[String]) -> Option<&T> {
        self.0.get(path).or_else(|| {
            self.0
                .iter()
                .fold(Found::Zero, |f, (qualified_path, object)| {
                    if is_suffix_of(path, qualified_path) {
                        match f {
                            Found::Zero => Found::One(object),
                            _ => Found::More,
                        }
                    } else {
                        f
                    }
                })
                .into()
        })
    }

    pub fn filter(&self, path: &[String]) -> Self {
        self.iter()
            .filter_map(|(qualified_path, object)| {
                if is_prefix_of(path, qualified_path) {
                    Some((qualified_path.clone(), object.clone()))
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn map<U: Clone, F: Fn(&T) -> U>(&self, f: F) -> Hierarchy<U> {
        self.iter().map(|(p, o)| (p.clone(), f(o))).collect()
    }

    pub fn filter_map<U: Clone, F: Fn(&T) -> Option<U>>(&self, f: F) -> Hierarchy<U> {
        self.iter()
            .filter_map(|(p, o)| Some((p.clone(), f(o)?)))
            .collect()
    }
}

impl<P: Path> Hierarchy<P> {
    pub fn and_then<T: Clone>(&self, h: Hierarchy<T>) -> Hierarchy<T> {
        self.filter_map(|p| h.get(&p.cloned()).cloned())
    }
}

// TODO add filters

impl<T: Clone> Deref for Hierarchy<T> {
    type Target = BTreeMap<Vec<String>, T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Clone> DerefMut for Hierarchy<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: Clone> IntoIterator for Hierarchy<T> {
    type Item = <BTreeMap<Vec<String>, T> as IntoIterator>::Item;
    type IntoIter = <BTreeMap<Vec<String>, T> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

/// Create a hierarchy from an iterator
impl<P: Path, T: Clone> FromIterator<(P, T)> for Hierarchy<T> {
    fn from_iter<I: IntoIterator<Item = (P, T)>>(iter: I) -> Self {
        Hierarchy::new(iter.into_iter().map(|(p, o)| (p.path(), o)).collect())
    }
}

/// Implement Extend
impl<P: Path, T: Clone> Extend<(P, T)> for Hierarchy<T> {
    fn extend<I: IntoIterator<Item = (P, T)>>(&mut self, iter: I) {
        self.0.extend(iter.into_iter().map(|(p, t)| (p.path(), t)))
    }
}

/// Create a hierarchy with an array of objects
impl<P: Path, T: Clone, const N: usize> From<[(P, T); N]> for Hierarchy<T> {
    fn from(value: [(P, T); N]) -> Self {
        Hierarchy::new(value.into_iter().map(|(p, o)| (p.path(), o)).collect())
    }
}

/// Create a new hierarchy with new objects
impl<'a, P: Path, T: Clone, I: IntoIterator<Item = (P, T)>> With<I> for Hierarchy<T> {
    fn with(mut self, input: I) -> Self {
        self.0.append(&mut BTreeMap::from_iter(
            input.into_iter().map(|(p, t)| (p.path(), t)),
        ));
        self
    }
}

/// Index
impl<P: Path, T: Clone> Index<P> for Hierarchy<T> {
    type Output = T;

    fn index(&self, index: P) -> &Self::Output {
        self.get(&index.path()).unwrap()
    }
}

/// Index
impl<T: Clone + fmt::Display> fmt::Display for Hierarchy<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{{\n  {},\n}}",
            self.iter()
                .map(|(p, t)| format!("{} -> {}", p.join("."), t))
                .join(",\n  ")
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::DataType;

    use super::*;

    #[test]
    fn test_hierarchy() {
        let values = Hierarchy::from([
            (vec!["a", "b", "c"], 1),
            (vec!["a", "b", "d"], 2),
            (vec!["a", "c"], 3),
            (vec!["a", "e"], 4),
            (vec!["a", "e", "f"], 5),
            (vec!["b", "c"], 6),
        ]);
        println!("{:?} -> {}", ["a", "c"], values[["a", "c"]]);
        println!("{:?} -> {:?}", ["c"], values.filter(&(["a", "b"].path())));
        println!("{:?} -> {}", ["e"], values[["e"]]);
        println!("{:?} -> {}", ["e", "f"], values[["e", "f"]]);
        println!("{:?} -> {}", ["b", "c"], values[["b", "c"]]);
        println!("{:?} -> {}", ["b", "d"], values[["b", "d"]]);
        println!("{:?} -> {}", ["d"], values[["d"]]);
    }

    #[test]
    fn test_filter() {
        let values = Hierarchy::from([
            (vec!["a", "b", "c"], 1),
            (vec!["a", "b", "d"], 2),
            (vec!["a", "c"], 3),
            (vec!["a", "e"], 4),
            (vec!["a", "e", "f"], 5),
        ]);
        let values = values.with([(vec!["b", "c"], 6), (vec!["b", "d"], 7)]);
        println!("values = {:#?}", values);
        println!(
            "filtered values = {:#?}",
            values.clone().filter(&["a"].path())
        );
        println!(
            "filtered values = {:#?}",
            values.clone().filter(&["b"].path())
        );
        println!(
            "filtered values = {:#?}",
            values.clone().filter(&["a", "b"].path())
        );
    }

    #[test]
    fn test_map() {
        let values = Hierarchy::from([
            (["a", "b", "c"].to_vec(), "hello"),
            (["a", "b", "d"].to_vec(), "hello2"),
            (["a", "c"].to_vec(), "hello3"),
            (["a", "e"].to_vec(), "hello4"),
            (["a", "e", "f"].to_vec(), "hello5"),
        ]);
        let values = values.with([
            (["b", "c"].to_vec(), "hello6"),
            (["b", "d"].to_vec(), "hello7"),
        ]);
        println!("values = {:#?}", values);
        println!("mapped values = {:#?}", values.clone().map(|s| s.len()));
    }

    #[test]
    fn test_composition() {
        let values = Hierarchy::from([
            (["table_1", "a"], ["t1", "a"]),
            (["table_1", "b"], ["t1", "b"]),
            (["table_2", "u"], ["t2", "a"]),
            (["table_2", "v"], ["t2", "c"]),
        ]);
        let map = Hierarchy::from([
            (vec!["t1", "a"], "hello a"),
            (vec!["b"], "hello b"),
            (vec!["t2", "a"], "hello a"),
            // (vec!["t2", "c"], "hello c"),
        ]);

        println!("values = {:#?}", values);
        println!("mapped values = {}", values.clone().and_then(map));
    }

    #[test]
    fn test_chain() {
        let h1 = Hierarchy::from([
            (["table_1", "a"], DataType::float()),
            (["table_1", "b"], DataType::integer()),
        ]);
        let h2 = Hierarchy::from([
            (["table_2", "a"], DataType::float()),
            (["table_2", "c"], DataType::integer()),
        ]);
        let joined_h = h1.chain(h2);
        assert_eq!(
            joined_h,
            Hierarchy::from([
                (["table_1", "a"], DataType::float()),
                (["table_1", "b"], DataType::integer()),
                (["table_2", "a"], DataType::float()),
                (["table_2", "c"], DataType::integer()),
            ])
        )
    }

    #[test]
    fn test_preprend() {
        let h1 = Hierarchy::from([(["a"], DataType::float()), (["b"], DataType::integer())]);
        let prepended_h = h1.prepend(&["table_1".to_string()]);
        assert_eq!(
            prepended_h,
            Hierarchy::from([
                (["table_1", "a"], DataType::float()),
                (["table_1", "b"], DataType::integer()),
            ])
        )
    }
}
