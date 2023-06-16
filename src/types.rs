//! # Type utilities
//! 
//! A few types and type composers
//!

/// An abstract product of types
pub trait And<F> {
    type Product;

    fn and(self, other: F) -> Self::Product;
}

pub trait Factor: And<Self, Product = Self> + Sized {
    fn unit() -> Self;

    fn all<I: IntoIterator<Item = Self>>(iter: I) -> Self {
        iter.into_iter()
            .reduce(|p, f| p.and(f))
            .unwrap_or(Self::unit())
    }
}

impl<A: And<A, Product = A> + Default> Factor for A {
    fn unit() -> Self {
        Self::default()
    }
}

/// An abstract sum of types
pub trait Or<T> {
    type Sum;

    fn or(self, other: T) -> Self::Sum;
}

pub trait Term: Or<Self, Sum = Self> + Sized {
    fn unit() -> Self;

    fn any<I: IntoIterator<Item = Self>>(iter: I) -> Self {
        iter.into_iter()
            .reduce(|p, f| p.or(f))
            .unwrap_or(Self::unit())
    }
}

impl<O: Or<O, Sum = O> + Default> Term for O {
    fn unit() -> Self {
        Self::default()
    }
}
