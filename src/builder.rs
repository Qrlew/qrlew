//! # Builder utilities
//! 
//! This module contains utilities to ease and standardize the writing of builders
//! such as: [relation::builder::MapBuilder]
//!

use std::{error, fmt, ops::Deref};

/// A trait for builder ad-hoc polymorphism
pub trait With<Input, Output = Self> {
    fn with(self, input: Input) -> Output;
}

/// Implement With for the unit type
impl<T, W: Default + With<T>> With<T, W> for () {
    fn with(self, input: T) -> W {
        W::default().with(input)
    }
}

pub trait WithIterator<Input> {
    fn with_iter<I: IntoIterator<Item = Input>>(self, iter: I) -> Self;
}

impl<Input, W: With<Input>> WithIterator<Input> for W {
    fn with_iter<I: IntoIterator<Item = Input>>(self, iter: I) -> Self {
        iter.into_iter().fold(self, |w, i| w.with(i))
    }
}

// A builder to add a context to an object

/// A struct holding an object with its context
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct WithContext<O, C> {
    pub object: O,
    pub context: C,
}

impl<O, C> Deref for WithContext<O, C> {
    type Target = O;

    fn deref(&self) -> &Self::Target {
        &self.object
    }
}

impl<O: fmt::Display, C: fmt::Display> fmt::Display for WithContext<O, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.object, self.context)
    }
}

/// A trait to declare a type comes sometime with some context of type C
pub trait WithoutContext: Sized {
    fn with<C>(self, context: C) -> WithContext<Self, C> {
        WithContext {
            object: self,
            context: context,
        }
    }
}

/// Anyone can have a context
// impl<O> WithoutContext for O {}
impl<'a, O> WithoutContext for &'a O {}

/// A trait enabling build when a builder is ready
pub trait Ready<Output>: Sized {
    type Error: error::Error;
    /// Build and panic in case of error
    fn build(self) -> Output {
        self.try_build().unwrap()
    }
    /// Try to build
    fn try_build(self) -> Result<Output, Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;

    impl WithoutContext for String {}
    impl WithoutContext for WithContext<String, i64> {}

    #[test]
    fn test_with_context() {
        let x = "Hello world".to_string();
        let y: WithContext<String, i64> = x.with(5);
        println!("x with context = {}", y);
        println!("x = {}", *y);
        println!("y with context = {}", y.with("Cool"));
    }
}
