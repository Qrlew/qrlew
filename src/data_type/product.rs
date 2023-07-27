//! A data structure to iterate on a carthesian product of homogenous iterable
//! Typically for Collections of `Intervals<Value>`

use std::{
    clone::Clone,
    convert::{Infallible, TryFrom, TryInto},
    error,
    fmt::{self, Debug, Display},
    ops::BitAnd,
    rc::Rc,
    result,
};

use super::{
    super::data_type,
    intervals::{Bound, Intervals},
    value::{self, Value},
    DataType,
};

/// The errors products can lead to
#[derive(Debug)]
pub enum Error {
    InvalidValue(String),
    InvalidDataType(String),
    Other(String),
}

impl Error {
    pub fn invalid_value(arg: impl fmt::Display) -> Error {
        Error::InvalidValue(format!("invalid value: {}", arg))
    }
    pub fn invalid_data_type(set: impl fmt::Display) -> Error {
        Error::InvalidDataType(format!("invalid data_type: {}", set))
    }
    pub fn other<T: fmt::Display>(desc: T) -> Error {
        Error::Other(desc.to_string())
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidValue(arg) => writeln!(f, "InvalidValue: {}", arg),
            Error::InvalidDataType(set) => writeln!(f, "InvalidDataType: {}", set),
            Error::Other(err) => writeln!(f, "{}", err),
        }
    }
}

impl error::Error for Error {}

impl From<Infallible> for Error {
    fn from(err: Infallible) -> Self {
        Error::Other(err.to_string())
    }
}
impl From<data_type::Error> for Error {
    fn from(err: data_type::Error) -> Self {
        Error::other(err)
    }
}
impl From<value::Error> for Error {
    fn from(err: value::Error) -> Self {
        Error::other(err)
    }
}

type Result<T> = result::Result<T, Error>;

/// A product of homogenous iterable
#[derive(Clone)]
pub struct VecProduct<T: Clone> {
    vecs: Vec<Vec<T>>,
}

impl<T: Clone> VecProduct<T> {
    pub fn new<P: IntoIterator<Item = I>, I: IntoIterator<Item = T>>(vecs: P) -> VecProduct<T> {
        VecProduct {
            vecs: vecs
                .into_iter()
                .map(|vec| vec.into_iter().collect())
                .collect(),
        }
    }

    pub fn empty() -> VecProduct<T> {
        VecProduct { vecs: vec![] }
    }

    pub fn of<I: IntoIterator<Item = T>>(vec: I) -> VecProduct<T> {
        VecProduct::new(vec![vec])
    }

    pub fn times<I: IntoIterator<Item = T>>(self, vec: I) -> VecProduct<T> {
        let mut result = self.vecs;
        result.push(vec.into_iter().collect());
        VecProduct { vecs: result }
    }

    /// Return a vector of equal length vectors
    pub fn vec(&self) -> Vec<Vec<T>> {
        self.vecs
            .iter()
            .fold(vec![vec![]], move |vec_vec, vec_t| {
                vec_vec
                    .into_iter()
                    .flat_map(move |vec| {
                        vec_t.iter().map(move |t| {
                            let mut result = vec.clone();
                            result.push(t.clone());
                            result
                        })
                    })
                    .collect()
            })
    }
}
/// A product of heterogenous types
pub trait Product {}

/// Empty product
#[derive(Clone, Debug)]
pub struct Unit;

impl Product for Unit {}

/// Term of a product
#[derive(Clone, Debug)]
pub struct Term<A, Next: Product> {
    value: A,
    next: Rc<Next>,
}

impl<A, Next: Product> Product for Term<A, Next> {}

impl<A, Next: Product> Term<A, Next> {
    pub fn new(value: A, next: Rc<Next>) -> Term<A, Next> {
        Term { value, next }
    }

    pub fn from_value_next<N: Into<Rc<Next>>>(value: A, next: N) -> Term<A, Next> {
        Term::new(value, next.into())
    }

    pub fn value(&self) -> &A {
        &self.value
    }

    pub fn next(&self) -> &Next {
        &self.next
    }
}

// Construction

pub trait And<A>: Product {
    type Result: Product;

    fn and(self, term: A) -> Self::Result;
}

impl<A> And<A> for Unit {
    type Result = Term<A, Unit>;

    fn and(self, value: A) -> Self::Result {
        Term::from_value_next(value, Unit)
    }
}

impl<A> BitAnd<A> for Unit {
    type Output = Term<A, Unit>;
    fn bitand(self, rhs: A) -> Self::Output {
        self.and(rhs)
    }
}

impl<A: Clone, B: Clone, Next: And<B>> And<B> for Term<A, Next>
where
    Next: Clone,
{
    type Result = Term<A, Next::Result>;

    fn and(self, value: B) -> Self::Result {
        Term::from_value_next(self.value().clone(), self.next().clone().and(value))
    }
}

impl<A: Clone, B: Clone, Next: And<B>> BitAnd<B> for Term<A, Next>
where
    Next: Clone,
{
    type Output = Term<A, Next::Result>;

    fn bitand(self, rhs: B) -> Self::Output {
        self.and(rhs)
    }
}

// Conversions to and from tuples

impl<A> From<(A,)> for Term<A, Unit> {
    fn from(value: (A,)) -> Self {
        Term::from_value_next(value.0, Unit)
    }
}

impl<A, B> From<(A, B)> for Term<A, Term<B, Unit>> {
    fn from(value: (A, B)) -> Self {
        Term::from_value_next(value.0, Term::from_value_next(value.1, Unit))
    }
}

impl<A, B, C> From<(A, B, C)> for Term<A, Term<B, Term<C, Unit>>> {
    fn from(value: (A, B, C)) -> Self {
        Term::from_value_next(
            value.0,
            Term::from_value_next(value.1, Term::from_value_next(value.2, Unit)),
        )
    }
}

impl<A: Clone> From<Term<A, Unit>> for (A,) {
    fn from(value: Term<A, Unit>) -> Self {
        (value.value().clone(),)
    }
}

impl<A: Clone, B: Clone> From<Term<A, Term<B, Unit>>> for (A, B) {
    fn from(value: Term<A, Term<B, Unit>>) -> Self {
        (value.value().clone(), value.next().clone().value)
    }
}

impl<A: Clone, B: Clone, C: Clone> From<Term<A, Term<B, Term<C, Unit>>>> for (A, B, C) {
    fn from(value: Term<A, Term<B, Term<C, Unit>>>) -> Self {
        (
            value.value().clone(),
            value.next().clone().value().clone(),
            value.next().clone().next().clone().value().clone(),
        )
    }
}

/// Product of `ToString`
pub trait ToStringProduct: Product {
    fn to_string_vec(&self) -> Vec<String>;
}

impl ToStringProduct for Unit {
    fn to_string_vec(&self) -> Vec<String> {
        vec![]
    }
}

impl Display for Unit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "()")
    }
}

impl<S: ToString, Next: ToStringProduct> ToStringProduct for Term<S, Next> {
    fn to_string_vec(&self) -> Vec<String> {
        let mut result = vec![];
        result.push(self.value.to_string());
        let mut tail = self.next.to_string_vec();
        result.append(&mut tail);
        result
    }
}

impl<S: ToString, Next: ToStringProduct> Display for Term<S, Next> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({})", self.to_string_vec().join(", "))
    }
}

/// Product of `Bound`
pub trait BoundProduct: Product + Clone {}

impl BoundProduct for Unit {}

impl<B: Bound, Next: BoundProduct> BoundProduct for Term<B, Next> {}

// Some conversions for BoundProduct

/// A -> Term<A, Unit>
impl<A: Bound> From<A> for Term<A, Unit> {
    fn from(value: A) -> Self {
        Term::from_value_next(value, Unit)
    }
}

/// Unit -> Value
impl From<Unit> for Value {
    fn from(_value: Unit) -> Self {
        ().into()
    }
}

/// Term<A, Unit> -> Value
impl<A: Bound> From<Term<A, Unit>> for Value
where
    A: Into<Value>,
{
    fn from(value: Term<A, Unit>) -> Self {
        (<(A,)>::from(value)).into()
    }
}

/// Term<A, Term<B, Unit>> -> Value
impl<A: Bound, B: Bound> From<Term<A, Term<B, Unit>>> for Value
where
    A: Into<Value>,
    B: Into<Value>,
{
    fn from(value: Term<A, Term<B, Unit>>) -> Self {
        (<(A, B)>::from(value)).into()
    }
}

/// Term<A, Term<B, Term<C, Unit>>> -> Value
impl<A: Bound, B: Bound, C: Bound> From<Term<A, Term<B, Term<C, Unit>>>> for Value
where
    A: Into<Value>,
    B: Into<Value>,
    C: Into<Value>,
{
    fn from(value: Term<A, Term<B, Term<C, Unit>>>) -> Self {
        (<(A, B, C)>::from(value)).into()
    }
}

/// Value -> Term<A, Unit>
impl<A: Bound> TryFrom<Value> for Term<A, Unit>
where
    Value: TryInto<A, Error = value::Error>,
{
    type Error = Error;
    fn try_from(value: Value) -> Result<Self> {
        Ok(<(A,)>::try_from(value)?.into())
    }
}

/// Value -> Term<A, Term<B, Unit>>
impl<A: Bound, B: Bound> TryFrom<Value> for Term<A, Term<B, Unit>>
where
    Value: TryInto<A, Error = value::Error> + TryInto<B, Error = value::Error>,
{
    type Error = Error;
    fn try_from(value: Value) -> Result<Self> {
        Ok(<(A, B)>::try_from(value)?.into())
    }
}

/// Value -> Term<A, Term<B, Term<C, Unit>>>
impl<A: Bound, B: Bound, C: Bound> TryFrom<Value> for Term<A, Term<B, Term<C, Unit>>>
where
    Value: TryInto<A, Error = value::Error>
        + TryInto<B, Error = value::Error>
        + TryInto<C, Error = value::Error>,
{
    type Error = Error;
    fn try_from(value: Value) -> Result<Self> {
        Ok(<(A, B, C)>::try_from(value)?.into())
    }
}

/// Product of simple intervals [B,B]
pub trait IntervalProduct: Product + Clone {
    type BoundProduct: BoundProduct;
    /// Iterate over all combinations
    fn iter(&self) -> std::vec::IntoIter<Self::BoundProduct>;
}
impl IntervalProduct for Unit {
    type BoundProduct = Unit;

    fn iter(&self) -> std::vec::IntoIter<Self::BoundProduct> {
        vec![Unit].into_iter()
    }
}
impl<B: Bound, Next: IntervalProduct> IntervalProduct for Term<[B; 2], Next> {
    type BoundProduct = Term<B, Next::BoundProduct>;

    fn iter(&self) -> std::vec::IntoIter<Self::BoundProduct> {
        let result: Vec<Self::BoundProduct> = self
            .next
            .iter()
            .flat_map(move |prod| {
                self.value
                    .iter()
                    .map(move |bound| Term::from_value_next(bound.clone(), prod.clone()))
            })
            .collect();
        result.into_iter()
    }
}

/// Product of `Intervals<B>`
pub trait IntervalsProduct: Product + Clone {
    type IntervalProduct: IntervalProduct;
    /// Iterate on interval products
    fn iter(&self) -> std::vec::IntoIter<Self::IntervalProduct>;
    /// Empty `IntervalsProduct``
    fn empty() -> Self;
    /// Union
    fn union(&self, other: &Self) -> Self;
    /// Intersection
    fn intersection(&self, other: &Self) -> Self;
}

impl IntervalsProduct for Unit {
    type IntervalProduct = Unit;

    fn iter(&self) -> std::vec::IntoIter<Self::IntervalProduct> {
        vec![Unit].into_iter()
    }

    fn empty() -> Self {
        Unit
    }

    fn union(&self, _other: &Self) -> Self {
        Unit
    }

    fn intersection(&self, _other: &Self) -> Self {
        Unit
    }
}

impl<B: Bound, Next: IntervalsProduct> IntervalsProduct for Term<Intervals<B>, Next> {
    type IntervalProduct = Term<[B; 2], Next::IntervalProduct>;

    fn iter(&self) -> std::vec::IntoIter<Self::IntervalProduct> {
        let result: Vec<Self::IntervalProduct> = self
            .next
            .iter()
            .flat_map(move |prod| {
                self.value
                    .iter()
                    .map(move |inter| Term::from_value_next(inter.clone(), prod.clone()))
            })
            .collect();
        result.into_iter()
    }

    fn empty() -> Self {
        Term::from_value_next(Intervals::empty(), Next::empty())
    }

    fn union(&self, other: &Self) -> Self {
        Term::from_value_next(
            self.value.clone().union(other.clone().value),
            self.next.union(&other.next),
        )
    }

    fn intersection(&self, other: &Self) -> Self {
        Term::from_value_next(
            self.value.clone().intersection(other.clone().value),
            self.next.intersection(&other.next),
        )
    }
}

// Some conversions for IntervalsProduct

/// Intervals<A> -> Term<Intervals<A>, Unit>
impl<A: Bound> From<Intervals<A>> for Term<Intervals<A>, Unit> {
    fn from(value: Intervals<A>) -> Self {
        Term::from_value_next(value, Unit)
    }
}

/// Term<Intervals<A>, Unit> -> Intervals<A>
impl<A: Bound> From<Term<Intervals<A>, Unit>> for Intervals<A> {
    fn from(value: Term<Intervals<A>, Unit>) -> Self {
        value.value().clone()
    }
}

/// DataType -> Term<Intervals<A>, Unit>
impl<A: Bound> TryFrom<DataType> for Term<Intervals<A>, Unit>
where
    Intervals<A>: TryFrom<DataType, Error = data_type::Error>,
{
    type Error = Error;
    fn try_from(value: DataType) -> Result<Self> {
        Ok(<(Intervals<A>,)>::try_from(value)?.into())
    }
}

/// DataType -> Term<Intervals<A>, Term<Intervals<B>, Unit>>
impl<A: Bound, B: Bound> TryFrom<DataType> for Term<Intervals<A>, Term<Intervals<B>, Unit>>
where
    Intervals<A>: TryFrom<DataType, Error = data_type::Error>,
    Intervals<B>: TryFrom<DataType, Error = data_type::Error>,
{
    type Error = Error;
    fn try_from(value: DataType) -> Result<Self> {
        Ok(<(Intervals<A>, Intervals<B>)>::try_from(value)?.into())
    }
}

/// DataType -> Term<Intervals<A>, Term<Intervals<B>, Term<Intervals<C>, Unit>>>
impl<A: Bound, B: Bound, C: Bound> TryFrom<DataType>
    for Term<Intervals<A>, Term<Intervals<B>, Term<Intervals<C>, Unit>>>
where
    Intervals<A>: TryFrom<DataType, Error = data_type::Error>,
    Intervals<B>: TryFrom<DataType, Error = data_type::Error>,
    Intervals<C>: TryFrom<DataType, Error = data_type::Error>,
{
    type Error = Error;
    fn try_from(value: DataType) -> Result<Self> {
        Ok(<(Intervals<A>, Intervals<B>, Intervals<C>)>::try_from(value)?.into())
    }
}

#[cfg(test)]
mod tests {
    use super::{super::intervals::Intervals, *};

    #[test]
    fn test_product() {
        let simple_term: Term<_, _> = (1,).into();
        println!("{simple_term}");
        let pair: Term<i32, _> = Term::from((1, 2.5));
        println!("{pair}");
        let triple = pair.and("hello");
        println!("{triple}");
        let triple = Unit & '>' & 1 & 2.5 & "hello";
        println!("{triple}");
    }

    #[test]
    fn test_interval_product() {
        let intervals = Unit
            & Intervals::from(0..=4)
            & Intervals::from_values(["A".to_string(), "B".to_string(), "C".to_string()])
            & (Intervals::from(-0.1..=2.3) | Intervals::from(5.0) | Intervals::from(10.1..=20.3));
        println!("{intervals}");
        for inter in intervals.iter() {
            println!("\nIntervals");
            for bound in inter.iter() {
                println!("{bound}");
            }
        }
    }

    #[test]
    fn test_interval_product_conversion() {
        let floats: DataType = data_type::Float::from_interval(0., 1.).into();
        let integers: DataType = data_type::Integer::from_interval(2, 3).into();
        let texts: DataType =
            data_type::Text::from_values([String::from("Hello"), String::from("World")]).into();
        let structured = data_type::Struct::from_data_types(&[floats, integers, texts]);
        let set = DataType::from(structured);
        let intervals: Term<
            data_type::Float,
            Term<data_type::Integer, Term<data_type::Text, Unit>>,
        > = set.try_into().unwrap();
        println!("intervals from datatype = {intervals}");
    }

    #[test]
    fn test_interval_union_inter() {
        let left = Unit
            & Intervals::from(0.0..=4.0)
            & Intervals::from_values([String::from("A"), String::from("B")])
            & Intervals::from(0..=5);
        let right = Unit
            & Intervals::from(-2.0..=2.0)
            & Intervals::from_values([String::from("B"), String::from("C")])
            & Intervals::from(-2..=3);
        // Print union and intersection
        println!("{} ∪ {} = {}", &left, &right, left.union(&right));
        println!("{} ∩ {} = {}", &left, &right, left.intersection(&right));
    }

    #[test]
    fn test_vec_product() {
        let u = Intervals::from_intervals([[1.0, 5.0], [6.0, 6.0]]);
        let v: Intervals<f64> = Intervals::from_intervals([[-100., -100.], [-5., 3.], [4., 5.]]);
        let w: Intervals<f64> = Intervals::from_values([0., 1.]);
        let prod = VecProduct::of(u.clone()).times(v.clone()).times(w.clone());
        println!("{} x {} x {} = {:?}", &u, &v, &w, &prod.vec());
        for i in prod.vec() {
            println!("{i:?}");
        }
    }
}
