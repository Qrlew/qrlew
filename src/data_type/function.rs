use std::{
    cmp, collections,
    convert::{Infallible, TryFrom, TryInto},
    error, fmt,
    hash::Hasher,
    ops::Deref,
    rc::Rc,
    result,
};

use itertools::Itertools;

use super::{
    super::data_type,
    injection,
    intervals::{Bound, Intervals},
    product::{self, IntervalProduct, IntervalsProduct, Term, Unit},
    value::{self, Value, Variant as _},
    DataType, DataTyped, Integer, List, Variant,
};

use crate::{
    builder::With,
    encoder::{Encoder, BASE_64},
};

/// Inspiration from:
/// - https://www.postgresql.org/docs/9.1/functions-math.html
/// - https://docs.rs/sqlparser/latest/sqlparser/ast/enum.Expr.html
///

/// The errors functions can lead to
#[derive(Debug)]
pub enum Error {
    ArgumentOutOfRange(String),
    SetOutOfRange(String),
    InvalidFunction(String),
    Other(String),
}

impl Error {
    pub fn argument_out_of_range(arg: impl fmt::Display, range: impl fmt::Display) -> Error {
        Error::ArgumentOutOfRange(format!("{} not in {}", arg, range))
    }
    pub fn set_out_of_range(set: impl fmt::Display, range: impl fmt::Display) -> Error {
        Error::SetOutOfRange(format!("{} not in {}", set, range))
    }
    pub fn invalid_function(domain: impl fmt::Display, co_domain: impl fmt::Display) -> Error {
        Error::InvalidFunction(format!(
            "Invalid function from {} into {}",
            domain, co_domain
        ))
    }
    pub fn other<T: fmt::Display>(desc: T) -> Error {
        Error::Other(desc.to_string())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::ArgumentOutOfRange(arg) => writeln!(f, "ArgumentOutOfRange: {}", arg),
            Error::SetOutOfRange(set) => writeln!(f, "SetOutOfRange: {}", set),
            Error::InvalidFunction(set) => writeln!(f, "InvalidFunction: {}", set),
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
impl From<injection::Error> for Error {
    fn from(err: injection::Error) -> Self {
        Error::other(err)
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
impl From<product::Error> for Error {
    fn from(err: product::Error) -> Self {
        Error::other(err)
    }
}

type Result<T> = result::Result<T, Error>;

/// A function computing a value and the image of some DataType
pub trait Function: fmt::Debug + fmt::Display {
    /// The domain, given as a Cartesian product
    fn domain(&self) -> DataType;
    /// The co-domain
    fn co_domain(&self) -> DataType {
        self.super_image(&self.domain()).unwrap()
    }
    /// A super-image of a set (a set containing the image of the set and included in the co-domain)
    fn super_image(&self, set: &DataType) -> Result<DataType>;
    /// The actual implementation of the function
    fn value(&self, arg: &Value) -> Result<Value>;
}

impl<Fun: Function + ?Sized> DataTyped for Fun {
    fn data_type(&self) -> DataType {
        DataType::from(super::Function::from((self.domain(), self.co_domain())))
    }
}

/// A wrapper to convert an injection into a function
#[derive(Debug, Clone)]
pub struct Injection<Inj: injection::Injection>(Inj);

impl<Inj: injection::Injection> fmt::Display for Injection<Inj> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "injection{{{}}}", self.0)
    }
}

impl<Inj: injection::Injection> Function for Injection<Inj>
where
    Error: From<<Inj::Domain as TryFrom<DataType>>::Error>,
    Error: From<<<Inj::Domain as Variant>::Element as TryFrom<Value>>::Error>,
{
    fn domain(&self) -> DataType {
        self.0.domain().into()
    }

    fn super_image(&self, set: &DataType) -> Result<DataType> {
        let set = Inj::Domain::try_from(set.clone())?;
        Ok(self.0.super_image(&set).map(|image| image.into())?)
    }

    fn value(&self, arg: &Value) -> Result<Value> {
        let arg = <Inj::Domain as Variant>::Element::try_from(arg.clone())?;
        Ok(self.0.value(&arg).map(|value| value.into())?)
    }
}

/// A function defined by its type signature and indicative value function without any other particular properties
/// In particular, no range computation is done
#[derive(Clone)]
pub struct Simple {
    domain: DataType,
    co_domain: DataType,
    value: Rc<dyn Fn(Value) -> Value>,
}

impl Simple {
    /// Constructor for Generic
    pub fn new(domain: DataType, co_domain: DataType, value: Rc<dyn Fn(Value) -> Value>) -> Self {
        Simple {
            domain,
            co_domain,
            value,
        }
    }
}

impl fmt::Debug for Simple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "simple{{{} -> {}}}", self.domain(), self.co_domain())
    }
}

impl fmt::Display for Simple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "simple{{{} -> {}}}", self.domain(), self.co_domain())
    }
}

impl Function for Simple {
    fn domain(&self) -> DataType {
        self.domain.clone()
    }

    fn super_image(&self, _set: &DataType) -> Result<DataType> {
        Ok(self.co_domain.clone())
    }

    fn value(&self, arg: &Value) -> Result<Value> {
        Ok((*self.value)(arg.clone()))
    }
}

/// A function defined pointwise without any other particular properties
/// Range computation is done on finite ranges
#[derive(Clone)]
pub struct Pointwise {
    domain: DataType,
    co_domain: DataType,
    value: Rc<dyn Fn(Value) -> Value>,
}

impl Pointwise {
    /// Constructor for Generic
    pub fn new(domain: DataType, co_domain: DataType, value: Rc<dyn Fn(Value) -> Value>) -> Self {
        Pointwise {
            domain,
            co_domain,
            value,
        }
    }
    /// Build univariate pointwise function
    pub fn univariate<A: Variant, B: Variant>(
        domain: A,
        co_domain: B,
        value: impl Fn(<A::Element as value::Variant>::Wrapped) -> <B::Element as value::Variant>::Wrapped
            + 'static,
    ) -> Self
    where
        <<A::Element as value::Variant>::Wrapped as TryFrom<Value>>::Error: fmt::Debug,
    {
        Self::new(
            domain.into(),
            co_domain.into(),
            Rc::new(move |a| {
                let a = <A::Element as value::Variant>::Wrapped::try_from(a).unwrap();
                value(a).into()
            }),
        )
    }
    /// Build bivariate pointwise function
    pub fn bivariate<A: Variant, B: Variant, C: Variant>(
        domain: (A, B),
        co_domain: C,
        value: impl Fn(
                <A::Element as value::Variant>::Wrapped,
                <B::Element as value::Variant>::Wrapped,
            ) -> <C::Element as value::Variant>::Wrapped
            + 'static,
    ) -> Self
    where
        <A::Element as value::Variant>::Wrapped: TryFrom<Value>,
        <B::Element as value::Variant>::Wrapped: TryFrom<Value>,
        <<A::Element as value::Variant>::Wrapped as TryFrom<Value>>::Error: fmt::Debug,
        <<B::Element as value::Variant>::Wrapped as TryFrom<Value>>::Error: fmt::Debug,
        <C::Element as value::Variant>::Wrapped: Into<Value>,
    {
        let domain = data_type::Struct::from_data_types(&[domain.0.into(), domain.1.into()]);
        Self::new(
            domain.into(),
            co_domain.into(),
            Rc::new(move |ab| {
                let ab = value::Struct::try_from(ab).unwrap();
                let a = <A::Element as value::Variant>::Wrapped::try_from(ab[0].as_ref().clone())
                    .unwrap();
                let b = <B::Element as value::Variant>::Wrapped::try_from(ab[1].as_ref().clone())
                    .unwrap();
                value(a, b).into()
            }),
        )
    }
    /// Build variadic pointwise function
    pub fn variadic<D: Variant, C: Variant>(
        domain: Vec<D>,
        co_domain: C,
        value: impl Fn(
                Vec<<D::Element as value::Variant>::Wrapped>,
            ) -> <C::Element as value::Variant>::Wrapped
            + 'static,
    ) -> Self
    where
        <D::Element as value::Variant>::Wrapped: TryFrom<Value>,
        <<D::Element as value::Variant>::Wrapped as TryFrom<Value>>::Error: fmt::Debug,
        <C::Element as value::Variant>::Wrapped: Into<Value>,
    {
        let domain = data_type::Struct::from_data_types(&domain.iter());
        Self::new(
            domain.into(),
            co_domain.into(),
            Rc::new(move |v| {
                let v = value::Struct::try_from(v).unwrap();
                let v: Vec<<D::Element as value::Variant>::Wrapped> = v
                    .into_iter()
                    .map(|(_n, v)| v.as_ref().clone().try_into().unwrap())
                    .collect();
                value(v).into()
            }),
        )
    }
}

impl fmt::Debug for Pointwise {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "pointwise{{{} -> {}}}", self.domain(), self.co_domain())
    }
}

impl fmt::Display for Pointwise {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "pointwise{{{} -> {}}}", self.domain(), self.co_domain())
    }
}

impl Function for Pointwise {
    fn domain(&self) -> DataType {
        self.domain.clone()
    }

    fn super_image(&self, set: &DataType) -> Result<DataType> {
        if !set.is_subset_of(&self.domain()) {
            Err(Error::set_out_of_range(set, self.domain()))
        } else {
            Ok(match set {
                DataType::Null => DataType::Null,
                DataType::Unit(_) => self
                    .value(&Value::unit())
                    .map(Into::into)
                    .unwrap_or_else(|_| self.co_domain()),
                DataType::Boolean(b) if b.all_values() => {
                    b.iter().map(|[v, _]| (*self.value)((*v).into())).collect()
                }
                DataType::Integer(i) if i.all_values() => {
                    i.iter().map(|[v, _]| (*self.value)((*v).into())).collect()
                }
                DataType::Enum(e) => e
                    .values
                    .iter()
                    .map(|(_, i)| (*self.value)((*i, e.values.clone()).into()))
                    .collect(),
                DataType::Float(f) if f.all_values() => {
                    f.iter().map(|[v, _]| (*self.value)((*v).into())).collect()
                }
                DataType::Text(t) if t.all_values() => t
                    .iter()
                    .map(|[v, _]| (*self.value)((v.to_string()).into()))
                    .collect(),
                DataType::Date(d) if d.all_values() => {
                    d.iter().map(|[v, _]| (*self.value)((*v).into())).collect()
                }
                DataType::Time(t) if t.all_values() => {
                    t.iter().map(|[v, _]| (*self.value)((*v).into())).collect()
                }
                DataType::DateTime(d) if d.all_values() => {
                    d.iter().map(|[v, _]| (*self.value)((*v).into())).collect()
                }
                DataType::Duration(d) if d.all_values() => {
                    d.iter().map(|[v, _]| (*self.value)((*v).into())).collect()
                }
                _ => self.co_domain.clone(),
            })
        }
    }

    fn value(&self, arg: &Value) -> Result<Value> {
        Ok((*self.value)(arg.clone()))
    }
}

/// Partitionned monotonic function (plus some complex periodic cases)
/// The domain is a (cartesian) product of Intervals<B> types
/// P and T are convenient representations of the product and elements of the product
/// The partition function maps a product into a vector of products where the value function is supposed to be monotonic
#[derive(Clone)]
pub struct PartitionnedMonotonic<P, T, Prod: IntervalsProduct, U: Bound>
where
    P: From<Prod> + Into<Prod>,
    T: From<<Prod::IntervalProduct as IntervalProduct>::BoundProduct>,
{
    domain: Prod,
    partition: Rc<dyn Fn(P) -> Vec<P>>,
    value: Rc<dyn Fn(T) -> U>,
}

impl<P, T, Prod: IntervalsProduct, U: Bound> PartitionnedMonotonic<P, T, Prod, U>
where
    P: From<Prod> + Into<Prod>,
    T: From<<Prod::IntervalProduct as IntervalProduct>::BoundProduct>,
{
    /// Constructor for Base Maps
    pub fn new(
        domain: Prod,
        partition: Rc<dyn Fn(P) -> Vec<P>>,
        value: Rc<dyn Fn(T) -> U>,
    ) -> Self {
        PartitionnedMonotonic {
            domain,
            partition,
            value,
        }
    }

    pub fn from_intervals(domain: P, value: impl Fn(T) -> U + 'static) -> Self
    where
        P: Clone + 'static,
    {
        Self::new(
            domain.clone().into(),
            Rc::new(move |set: P| vec![set.into().intersection(&domain.clone().into()).into()]),
            Rc::new(value),
        )
    }

    pub fn from_partitions(
        partitions: impl AsRef<[P]> + 'static,
        value: impl Fn(T) -> U + 'static,
    ) -> Self
    where
        P: Clone,
    {
        let domain = partitions
            .as_ref()
            .iter()
            .fold(Prod::empty(), |domain, partition| {
                domain.union(&partition.clone().into())
            });
        let partition = move |set: P| {
            partitions
                .as_ref()
                .iter()
                .map(move |partition| {
                    set.clone()
                        .into()
                        .intersection(&partition.clone().into())
                        .into()
                })
                .collect()
        };
        Self::new(domain, Rc::new(partition), Rc::new(value))
    }
}

impl<A: Bound + 'static, B: Bound>
    PartitionnedMonotonic<Intervals<A>, (A,), Term<Intervals<A>, Unit>, B>
{
    pub fn univariate(domain: Intervals<A>, value: impl Fn(A) -> B + 'static) -> Self {
        Self::new(
            domain.clone().into(),
            Rc::new(move |set: Intervals<A>| vec![set.intersection(domain.clone())]),
            Rc::new(move |arg: (A,)| value(arg.0)),
        )
    }

    pub fn piecewise_univariate<const N: usize>(
        partitions: [Intervals<A>; N],
        value: impl Fn(A) -> B + 'static,
    ) -> Self {
        Self::from_partitions(partitions, move |(a,)| value(a))
    }
}

impl PartitionnedMonotonic<Intervals<f64>, (f64,), Term<Intervals<f64>, Unit>, f64> {
    pub fn periodic_univariate<const N: usize>(
        partitions: [Intervals<f64>; N],
        value: impl Fn(f64) -> f64 + 'static,
    ) -> Self {
        let domain: Intervals<f64> = partitions
            .iter()
            .fold(Intervals::empty(), |union, intervals| {
                union.union(intervals.clone())
            });
        let min = *domain.min().unwrap();
        let max = *domain.max().unwrap();
        let period = max - min;
        // Compute the shifted version of the set (by an integer number of period) and intersect with partitions
        let partition = move |set: Intervals<f64>| {
            let shift = ((*set.min().unwrap() - min) / period).floor();
            let shifted = set
                .clone()
                .map_bounds(move |b| b - shift * period)
                .union(set.map_bounds(|b| b - (shift + 1.) * period));
            partitions
                .as_ref()
                .iter()
                .map(move |partition| shifted.clone().intersection(partition.clone()))
                .collect()
        };
        Self::new(
            Term::from_value_next(Intervals::default(), Unit),
            Rc::new(partition),
            Rc::new(move |(a,)| value(a)),
        )
    }
}

impl<A: Bound + 'static, B: Bound + 'static, C: Bound>
    PartitionnedMonotonic<
        (Intervals<A>, Intervals<B>),
        (A, B),
        Term<Intervals<A>, Term<Intervals<B>, Unit>>,
        C,
    >
{
    pub fn bivariate(
        domain: (Intervals<A>, Intervals<B>),
        value: impl Fn(A, B) -> C + 'static,
    ) -> Self {
        Self::from_intervals(domain, move |(a, b)| value(a, b))
    }

    pub fn piecewise_bivariate<const N: usize>(
        partitions: [(Intervals<A>, Intervals<B>); N],
        value: impl Fn(A, B) -> C + 'static,
    ) -> Self {
        Self::from_partitions(partitions, move |(a, b)| value(a, b))
    }
}

impl<P, T, Prod: IntervalsProduct, U: Bound> PartitionnedMonotonic<P, T, Prod, U>
where
    P: From<Prod> + Into<Prod>,
    T: From<<Prod::IntervalProduct as IntervalProduct>::BoundProduct>,
    Self: Function,
{
    // Utility functions to check the consistency of values

    /// Check the image function
    fn checked_image(&self, set: &DataType, super_image: DataType) -> Result<DataType> {
        if !set.is_subset_of(&self.domain()) {
            Err(Error::set_out_of_range(set, self.domain()))
        } else {
            Ok(super_image)
        }
    }

    /// Check the value function
    fn checked_value(&self, arg: &Value, value: Value) -> Result<Value> {
        if !self.domain().contains(arg) {
            Err(Error::argument_out_of_range(arg, self.domain()))
        } else if !self.co_domain().contains(&value) {
            Err(Error::argument_out_of_range(value, self.co_domain()))
        } else {
            Ok(value)
        }
    }
}

impl<P, T, Prod: IntervalsProduct, U: Bound> fmt::Debug for PartitionnedMonotonic<P, T, Prod, U>
where
    P: From<Prod> + Into<Prod>,
    T: From<<Prod::IntervalProduct as IntervalProduct>::BoundProduct>,
    Self: Function,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "partitionned_monotonic{{{} -> {}}}",
            self.domain(),
            self.co_domain()
        )
    }
}

impl<P, T, Prod: IntervalsProduct, U: Bound> fmt::Display for PartitionnedMonotonic<P, T, Prod, U>
where
    P: From<Prod> + Into<Prod>,
    T: From<<Prod::IntervalProduct as IntervalProduct>::BoundProduct>,
    Self: Function,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "partitionned_monotonic{{{} -> {}}}",
            self.domain(),
            self.co_domain()
        )
    }
}

impl<P, T, Prod: IntervalsProduct, U: Bound> Function for PartitionnedMonotonic<P, T, Prod, U>
where
    P: From<Prod> + Into<Prod> + Into<DataType> + TryFrom<DataType, Error = data_type::Error>,
    T: From<<Prod::IntervalProduct as IntervalProduct>::BoundProduct>
        + TryFrom<Value, Error = value::Error>,
    Intervals<U>: Into<DataType>,
    U: Into<Value>,
{
    fn domain(&self) -> DataType {
        let p: P = self.domain.clone().into();
        p.into()
    }

    fn super_image(&self, set: &DataType) -> Result<DataType> {
        // The monotonicity is constant on each bloc
        // First try to convert into the right datatype
        let converted_set = &set.into_data_type(&self.domain())?;
        // Then express in a more suitable form
        let p: P = converted_set.clone().try_into()?;
        let partitions: Vec<Prod> = (self.partition)(p).into_iter().map(|p| p.into()).collect();
        let result: Intervals<U> = partitions
            .iter()
            .flat_map(|prod| {
                prod.iter().map(|inter| {
                    let mut sorted: Vec<U> = inter
                        .iter()
                        .map(|bound| (self.value)(bound.into()))
                        .collect();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(cmp::Ordering::Equal));
                    [sorted[0].clone(), sorted[sorted.len() - 1].clone()]
                })
            })
            .collect();
        self.checked_image(converted_set, result.into())
    }

    fn value(&self, arg: &Value) -> Result<Value> {
        // First try to convert into the right datatype
        let converted_arg = &arg.as_data_type(&self.domain())?;
        // Then express in a more suitable form
        let t: T = converted_arg.clone().try_into()?;
        self.checked_value(converted_arg, (self.value)(t).into())
    }
}

/// # Extended function
/// Functions can be wrapped with this `Extended` object.
/// The co_domain is usually Option<original CoDomain> unless the domain is included in the original domain.
#[derive(Debug)]
pub struct Extended<F: Function> {
    function: F,
    domain: DataType,
}

impl<F: Function> Extended<F> {
    pub fn new(function: F, domain: DataType) -> Extended<F> {
        Extended { function, domain }
    }
}

impl<F: Function> fmt::Display for Extended<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "extended{{{} -> {}}}", self.domain(), self.co_domain())
    }
}

impl<F: Function + Clone> Clone for Extended<F> {
    fn clone(&self) -> Self {
        Extended::new(self.function.clone(), self.domain.clone())
    }
}

impl<F: Function> Function for Extended<F> {
    fn domain(&self) -> DataType {
        self.domain.clone()
    }

    fn super_image(&self, set: &DataType) -> Result<DataType> {
        if set.is_subset_of(&self.function.domain()) {
            // If the set is included in the original domain -> fall back to the original function
            self.function.super_image(set)
        } else if set.is_subset_of(&self.domain) {
            // The set is valid
            set.super_intersection(&self.function.domain())
                .and_then(|set_into_domain| {
                    Ok(DataType::optional(
                        self.function.super_image(&set_into_domain)?,
                    ))
                })
                .or_else(|_err| Ok(DataType::optional(self.function.co_domain())))
        } else {
            Err(Error::set_out_of_range(set, &self.domain))
        }
    }

    fn value(&self, arg: &Value) -> Result<Value> {
        if self.domain.contains(arg) {
            // The arg is valid
            self.function.value(arg).or_else(|_err| Ok(Value::none()))
        } else {
            Err(Error::argument_out_of_range(arg, &self.domain))
        }
    }
}

/// A function is extensible if it can be extended to a different domain
pub trait Extensible {
    type Extended;

    fn extend(self, domain: DataType) -> Self::Extended;
}

// Implement extensible for all borrowed function
impl<'a, F: Function + Clone> Extensible for &'a F {
    type Extended = Extended<F>;

    fn extend(self: &'a F, domain: DataType) -> Self::Extended {
        Extended::new(self.clone(), domain)
    }
}

impl<F: Function> Extensible for Extended<F> {
    type Extended = Extended<F>;

    fn extend(self: Extended<F>, domain: DataType) -> Extended<F> {
        Extended::new(self.function, domain)
    }
}

/// A function defined pointwise without any other particular properties
#[derive(Clone)]
pub struct Aggregate<A: Variant, B: Variant>
where
    A::Element: TryFrom<Value>,
    Error: From<<A::Element as TryFrom<Value>>::Error>,
    B::Element: Into<Value>,
    A: Into<DataType> + TryFrom<DataType>,
    Error: From<<A as TryFrom<DataType>>::Error>,
    B: Into<DataType>,
{
    aggregation_domain: A,
    value: Rc<dyn Fn(Vec<A::Element>) -> B::Element>,
    super_image: Rc<dyn Fn((A, Integer)) -> Result<B>>,
}

impl<A: Variant, B: Variant> Aggregate<A, B>
where
    A::Element: TryFrom<Value>,
    Error: From<<A::Element as TryFrom<Value>>::Error>,
    B::Element: Into<Value>,
    A: Into<DataType> + TryFrom<DataType>,
    Error: From<<A as TryFrom<DataType>>::Error>,
    B: Into<DataType>,
{
    /// Constructor for Generic
    pub fn new(
        aggregation_domain: A,
        value: Rc<dyn Fn(Vec<A::Element>) -> B::Element>,
        super_image: Rc<dyn Fn((A, Integer)) -> Result<B>>,
    ) -> Self {
        Aggregate {
            aggregation_domain,
            value,
            super_image,
        }
    }

    /// Constructor for Generic
    pub fn from(
        aggregation_domain: A,
        value: impl Fn(Vec<A::Element>) -> B::Element + 'static,
        super_image: impl Fn((A, Integer)) -> Result<B> + 'static,
    ) -> Self {
        Aggregate::new(aggregation_domain, Rc::new(value), Rc::new(super_image))
    }
}

impl<A: Variant, B: Variant> fmt::Debug for Aggregate<A, B>
where
    A::Element: TryFrom<Value>,
    Error: From<<A::Element as TryFrom<Value>>::Error>,
    B::Element: Into<Value>,
    A: Into<DataType> + TryFrom<DataType>,
    Error: From<<A as TryFrom<DataType>>::Error>,
    B: Into<DataType>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "aggregate{{{} -> {}}}", self.domain(), self.co_domain())
    }
}

impl<A: Variant, B: Variant> fmt::Display for Aggregate<A, B>
where
    A::Element: TryFrom<Value>,
    Error: From<<A::Element as TryFrom<Value>>::Error>,
    B::Element: Into<Value>,
    A: Into<DataType> + TryFrom<DataType>,
    Error: From<<A as TryFrom<DataType>>::Error>,
    B: Into<DataType>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "aggregate{{{} -> {}}}", self.domain(), self.co_domain())
    }
}

impl<A: Variant, B: Variant> Function for Aggregate<A, B>
where
    A::Element: TryFrom<Value>,
    Error: From<<A::Element as TryFrom<Value>>::Error>,
    B::Element: Into<Value>,
    A: Into<DataType> + TryFrom<DataType>,
    Error: From<<A as TryFrom<DataType>>::Error>,
    B: Into<DataType>,
{
    fn domain(&self) -> DataType {
        List::from_data_type(self.aggregation_domain.clone().into()).into()
    }

    fn super_image(&self, set: &DataType) -> Result<DataType> {
        let set = set.clone().into_data_type(&self.domain())?;
        if let DataType::List(List { data_type, size }) = set {
            (*self.super_image)(((*data_type).clone().try_into()?, size.clone())).map(Into::into)
        } else {
            Err(Error::set_out_of_range(set, self.domain()))
        }
    }

    fn value(&self, arg: &Value) -> Result<Value> {
        let list: value::List = arg.clone().try_into()?;
        let vals: Result<Vec<A::Element>> = list
            .iter()
            .map(|value| Ok(A::Element::try_from(value.clone())?))
            .collect();
        Ok((*self.value)(vals?).into())
    }
}

// Today, float -> float would take ints but always return float
// pub struct Polymorphic<Domain, ValueFunctions> {
//     domain: Domain,
//     value: ValueFunction,
// }
#[derive(Clone, Debug, Default)]
pub struct Polymorphic(Vec<Rc<dyn Function>>);

impl Polymorphic {
    /// Constructor for Polymorphic
    pub fn new(implementations: Vec<Rc<dyn Function>>) -> Self {
        Polymorphic(implementations)
    }
}

impl<F: Function + 'static, G: Function + 'static> From<(F, G)> for Polymorphic {
    fn from((f, g): (F, G)) -> Self {
        Polymorphic(vec![Rc::new(f), Rc::new(g)])
    }
}

impl<F: Function + 'static, G: Function + 'static, H: Function + 'static> From<(F, G, H)>
    for Polymorphic
{
    fn from((f, g, h): (F, G, H)) -> Self {
        Polymorphic(vec![Rc::new(f), Rc::new(g), Rc::new(h)])
    }
}

impl<F: Function + 'static> With<F> for Polymorphic {
    fn with(mut self, input: F) -> Self {
        self.0.push(Rc::new(input));
        self
    }
}

impl<const N: usize> From<[Rc<dyn Function>; N]> for Polymorphic {
    fn from(fs: [Rc<dyn Function>; N]) -> Self {
        Polymorphic(fs.into_iter().map(|f| f).collect())
    }
}

impl Function for Polymorphic {
    fn domain(&self) -> DataType {
        DataType::sum(self.0.iter().map(|implementation| implementation.domain()))
    }

    fn super_image(&self, set: &DataType) -> Result<DataType> {
        if let DataType::Union(union) = set {
            let result: Result<data_type::Union> = union
                .fields
                .iter()
                .map(|(f, dt)| Ok((f.clone(), self.super_image(dt)?)))
                .collect();
            result.map(Into::into)
        } else {
            self.0
                .iter()
                .find_map(|implementation| implementation.super_image(set).ok())
                .ok_or_else(|| Error::set_out_of_range(set, self.domain()))
        }
    }

    fn value(&self, arg: &Value) -> Result<Value> {
        self.0
            .iter()
            .find_map(|implementation| implementation.value(arg).ok())
            .ok_or_else(|| Error::argument_out_of_range(arg, self.domain()))
    }
}

impl fmt::Display for Polymorphic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "polymorphic{{{}}}",
            self.0.iter().map(|i| i.to_string()).join(" | ")
        )
    }
}

// CASE WHEN ... THEN ... ELSE END
#[derive(Clone, Debug)]
pub struct Case;

impl fmt::Display for Case {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "case")
    }
}

impl Function for Case {
    fn domain(&self) -> DataType {
        DataType::from(data_type::Struct::from_data_types(&[
            DataType::boolean(),
            DataType::Any,
            DataType::Any,
        ]))
    }

    fn super_image(&self, set: &DataType) -> Result<DataType> {
        if !set.is_subset_of(&self.domain()) {
            Err(Error::set_out_of_range(set, self.domain()))
        } else {
            if let DataType::Struct(struct_data_type) = set {
                let when_condition = match struct_data_type.field_from_index(0).1.as_ref().clone() {
                    DataType::Boolean(bool_datatype) => bool_datatype,
                    _ => return Err(Error::argument_out_of_range(set, self.domain())),
                };

                if when_condition.is_empty() {
                    Ok(DataType::Null)
                } else if when_condition == data_type::Boolean::from_value(false) {
                    Ok(struct_data_type.field_from_index(2).1.as_ref().clone())
                } else if when_condition == data_type::Boolean::from_value(true) {
                    Ok(struct_data_type.field_from_index(1).1.as_ref().clone())
                } else {
                    Ok(struct_data_type
                        .field_from_index(1)
                        .1
                        .as_ref()
                        .clone()
                        .super_union(struct_data_type.field_from_index(2).1.as_ref())?)
                }
            } else {
                Err(Error::argument_out_of_range(set, self.domain()))
            }
        }
    }

    fn value(&self, arg: &Value) -> Result<Value> {
        if let Value::Struct(struct_values) = arg {
            if struct_values.field_from_index(0).1 == Rc::new(Value::boolean(true)) {
                Ok(struct_values.field_from_index(1).1.as_ref().clone())
            } else {
                Ok(struct_values.field_from_index(2).1.as_ref().clone())
            }
        } else {
            Err(Error::argument_out_of_range(arg, self.domain()))
        }
    }
}

/*
We list here all the functions to expose
*/

// Invalid function
pub fn null() -> impl Function + Clone {
    PartitionnedMonotonic::univariate(data_type::Text::default(), |_x| "null".to_string())
}

// Unary operators

/// Builds the minus `Function`
pub fn opposite() -> impl Function + Clone {
    PartitionnedMonotonic::univariate(data_type::Float::default(), |x| -x)
}
/// Builds the minus `Function`
pub fn not() -> impl Function + Clone {
    PartitionnedMonotonic::univariate(data_type::Boolean::default(), |x| !x)
}

// Arithmetic binary operators

/// The sum (polymorphic)
pub fn plus() -> impl Function + Clone {
    Polymorphic::from((
        PartitionnedMonotonic::bivariate(
            (data_type::Integer::default(), data_type::Integer::default()),
            |x, y| x.saturating_add(y),
        ),
        PartitionnedMonotonic::bivariate(
            (data_type::Float::default(), data_type::Float::default()),
            |x, y| (x + y).clamp(<f64 as Bound>::min(), <f64 as Bound>::max()),
        ),
    ))
}

/// The difference
pub fn minus() -> impl Function + Clone {
    Polymorphic::from((
        PartitionnedMonotonic::bivariate(
            (data_type::Integer::default(), data_type::Integer::default()),
            |x, y| x.saturating_sub(y),
        ),
        PartitionnedMonotonic::bivariate(
            (data_type::Float::default(), data_type::Float::default()),
            |x, y| (x - y).clamp(<f64 as Bound>::min(), <f64 as Bound>::max()),
        ),
    ))
}

/// The product (the domain is partitionned)
pub fn multiply() -> impl Function + Clone {
    Polymorphic::from((
        // Integer implementation
        PartitionnedMonotonic::piecewise_bivariate(
            [
                (
                    data_type::Integer::from_min(0),
                    data_type::Integer::from_min(0),
                ),
                (
                    data_type::Integer::from_min(0),
                    data_type::Integer::from_max(0),
                ),
                (
                    data_type::Integer::from_max(0),
                    data_type::Integer::from_min(0),
                ),
                (
                    data_type::Integer::from_max(0),
                    data_type::Integer::from_max(0),
                ),
            ],
            |x, y| x.saturating_mul(y),
        ),
        // Float implementation
        PartitionnedMonotonic::piecewise_bivariate(
            [
                (
                    data_type::Float::from_min(0.0),
                    data_type::Float::from_min(0.0),
                ),
                (
                    data_type::Float::from_min(0.0),
                    data_type::Float::from_max(0.0),
                ),
                (
                    data_type::Float::from_max(0.0),
                    data_type::Float::from_min(0.0),
                ),
                (
                    data_type::Float::from_max(0.0),
                    data_type::Float::from_max(0.0),
                ),
            ],
            |x, y| (x * y).clamp(<f64 as Bound>::min(), <f64 as Bound>::max()),
        ),
    ))
}

/// The division (the domain is partitionned)
pub fn divide() -> impl Function + Clone {
    Polymorphic::from((
        // Integer implementation
        PartitionnedMonotonic::piecewise_bivariate(
            [
                (
                    data_type::Integer::from_min(0),
                    data_type::Integer::from_min(0),
                ),
                (
                    data_type::Integer::from_min(0),
                    data_type::Integer::from_max(0),
                ),
                (
                    data_type::Integer::from_max(0),
                    data_type::Integer::from_min(0),
                ),
                (
                    data_type::Integer::from_max(0),
                    data_type::Integer::from_max(0),
                ),
            ],
            |x, y| x.saturating_div(y),
        ),
        // Float implementation
        PartitionnedMonotonic::piecewise_bivariate(
            [
                (
                    data_type::Float::from_min(0.0),
                    data_type::Float::from_min(0.0),
                ),
                (
                    data_type::Float::from_min(0.0),
                    data_type::Float::from_max(0.0),
                ),
                (
                    data_type::Float::from_max(0.0),
                    data_type::Float::from_min(0.0),
                ),
                (
                    data_type::Float::from_max(0.0),
                    data_type::Float::from_max(0.0),
                ),
            ],
            |x, y| (x / y).clamp(<f64 as Bound>::min(), <f64 as Bound>::max()),
        ),
    ))
}

/// The modulo
pub fn modulo() -> impl Function + Clone {
    Pointwise::bivariate(
        (data_type::Integer::default(), data_type::Integer::default()),
        data_type::Integer::default(),
        |a, b| (a % b).into(),
    )
}

pub fn string_concat() -> impl Function + Clone {
    Pointwise::bivariate(
        (data_type::Text::default(), data_type::Text::default()),
        data_type::Text::default(),
        |a, b| (a + &b).into(),
    )
}

pub fn concat(n: usize) -> impl Function + Clone {
    Pointwise::variadic(vec![DataType::Any; n], data_type::Text::default(), |v| {
        v.into_iter().map(|v| v.to_string()).join("")
    })
}

pub fn md5() -> impl Function + Clone {
    Simple::new(
        DataType::text(),
        DataType::text(),
        Rc::new(|v| {
            let mut s = collections::hash_map::DefaultHasher::new();
            Bound::hash((value::Text::try_from(v).unwrap()).deref(), &mut s);
            Encoder::new(BASE_64, 10).encode(s.finish()).into()
        }),
    )
}

pub fn gt() -> impl Function + Clone {
    Polymorphic::default()
        .with(Pointwise::bivariate(
            (data_type::Integer::default(), data_type::Integer::default()),
            data_type::Boolean::default(),
            |a, b| (a > b).into(),
        ))
        .with(Pointwise::bivariate(
            (data_type::Float::default(), data_type::Float::default()),
            data_type::Boolean::default(),
            |a, b| (a > b).into(),
        ))
        .with(Pointwise::bivariate(
            (data_type::Date::default(), data_type::Date::default()),
            data_type::Boolean::default(),
            |a, b| (a > b).into(),
        ))
        .with(Pointwise::bivariate(
            (data_type::Time::default(), data_type::Time::default()),
            data_type::Boolean::default(),
            |a, b| (a > b).into(),
        ))
        .with(Pointwise::bivariate(
            (
                data_type::DateTime::default(),
                data_type::DateTime::default(),
            ),
            data_type::Boolean::default(),
            |a, b| (a > b).into(),
        ))
        .with(Pointwise::bivariate(
            (data_type::Text::default(), data_type::Text::default()),
            data_type::Boolean::default(),
            |a, b| (a > b).into(),
        ))
}

pub fn lt() -> impl Function + Clone {
    Polymorphic::default()
        .with(Pointwise::bivariate(
            (data_type::Integer::default(), data_type::Integer::default()),
            data_type::Boolean::default(),
            |a, b| (a < b).into(),
        ))
        .with(Pointwise::bivariate(
            (data_type::Float::default(), data_type::Float::default()),
            data_type::Boolean::default(),
            |a, b| (a < b).into(),
        ))
        .with(Pointwise::bivariate(
            (data_type::Date::default(), data_type::Date::default()),
            data_type::Boolean::default(),
            |a, b| (a < b).into(),
        ))
        .with(Pointwise::bivariate(
            (data_type::Time::default(), data_type::Time::default()),
            data_type::Boolean::default(),
            |a, b| (a < b).into(),
        ))
        .with(Pointwise::bivariate(
            (
                data_type::DateTime::default(),
                data_type::DateTime::default(),
            ),
            data_type::Boolean::default(),
            |a, b| (a < b).into(),
        ))
        .with(Pointwise::bivariate(
            (data_type::Text::default(), data_type::Text::default()),
            data_type::Boolean::default(),
            |a, b| (a < b).into(),
        ))
}

pub fn gt_eq() -> impl Function + Clone {
    Polymorphic::default()
        .with(Pointwise::bivariate(
            (data_type::Integer::default(), data_type::Integer::default()),
            data_type::Boolean::default(),
            |a, b| (a >= b).into(),
        ))
        .with(Pointwise::bivariate(
            (data_type::Float::default(), data_type::Float::default()),
            data_type::Boolean::default(),
            |a, b| (a >= b).into(),
        ))
        .with(Pointwise::bivariate(
            (data_type::Date::default(), data_type::Date::default()),
            data_type::Boolean::default(),
            |a, b| (a >= b).into(),
        ))
        .with(Pointwise::bivariate(
            (data_type::Time::default(), data_type::Time::default()),
            data_type::Boolean::default(),
            |a, b| (a >= b).into(),
        ))
        .with(Pointwise::bivariate(
            (
                data_type::DateTime::default(),
                data_type::DateTime::default(),
            ),
            data_type::Boolean::default(),
            |a, b| (a >= b).into(),
        ))
        .with(Pointwise::bivariate(
            (data_type::Text::default(), data_type::Text::default()),
            data_type::Boolean::default(),
            |a, b| (a >= b).into(),
        ))
}

pub fn lt_eq() -> impl Function + Clone {
    Polymorphic::default()
        .with(Pointwise::bivariate(
            (data_type::Integer::default(), data_type::Integer::default()),
            data_type::Boolean::default(),
            |a, b| (a <= b).into(),
        ))
        .with(Pointwise::bivariate(
            (data_type::Float::default(), data_type::Float::default()),
            data_type::Boolean::default(),
            |a, b| (a <= b).into(),
        ))
        .with(Pointwise::bivariate(
            (data_type::Date::default(), data_type::Date::default()),
            data_type::Boolean::default(),
            |a, b| (a <= b).into(),
        ))
        .with(Pointwise::bivariate(
            (data_type::Time::default(), data_type::Time::default()),
            data_type::Boolean::default(),
            |a, b| (a <= b).into(),
        ))
        .with(Pointwise::bivariate(
            (
                data_type::DateTime::default(),
                data_type::DateTime::default(),
            ),
            data_type::Boolean::default(),
            |a, b| (a <= b).into(),
        ))
        .with(Pointwise::bivariate(
            (data_type::Text::default(), data_type::Text::default()),
            data_type::Boolean::default(),
            |a, b| (a <= b).into(),
        ))
}

pub fn eq() -> impl Function + Clone {
    Pointwise::bivariate(
        (DataType::Any, DataType::Any),
        data_type::Boolean::default(),
        |a, b| (a == b).into(),
    )
}

pub fn not_eq() -> impl Function + Clone {
    Pointwise::bivariate(
        (DataType::Any, DataType::Any),
        data_type::Boolean::default(),
        |a, b| (a != b).into(),
    )
}

// Boolean binary operators

/// The conjunction
pub fn and() -> impl Function + Clone {
    PartitionnedMonotonic::bivariate(
        (data_type::Boolean::default(), data_type::Boolean::default()),
        |x, y| x && y,
    )
}
/// The disjunction
pub fn or() -> impl Function + Clone {
    PartitionnedMonotonic::bivariate(
        (data_type::Boolean::default(), data_type::Boolean::default()),
        |x, y| x || y,
    )
}
/// The exclusive or
pub fn xor() -> impl Function + Clone {
    PartitionnedMonotonic::bivariate(
        (data_type::Boolean::default(), data_type::Boolean::default()),
        |x, y| x ^ y,
    )
}

// Bitwise binary operators

pub fn bitwise_or() -> impl Function + Clone {
    Pointwise::bivariate(
        (data_type::Boolean::default(), data_type::Boolean::default()),
        data_type::Boolean::default(),
        |a, b| (a | b).into(),
    )
}

pub fn bitwise_and() -> impl Function + Clone {
    Pointwise::bivariate(
        (data_type::Boolean::default(), data_type::Boolean::default()),
        data_type::Boolean::default(),
        |a, b| (a & b).into(),
    )
}

pub fn bitwise_xor() -> impl Function + Clone {
    Pointwise::bivariate(
        (data_type::Boolean::default(), data_type::Boolean::default()),
        data_type::Boolean::default(),
        |a, b| (a ^ b).into(),
    )
}

// Real functions

/// Builds the exponential `Function`
pub fn exp() -> impl Function + Clone {
    PartitionnedMonotonic::univariate(data_type::Float::default(), |x| {
        x.exp().clamp(0.0, <f64 as Bound>::max())
    })
}

/// Builds the logarithm `Function`
pub fn ln() -> impl Function + Clone {
    PartitionnedMonotonic::univariate(data_type::Float::from(0.0..), |x| {
        x.ln().clamp(<f64 as Bound>::min(), <f64 as Bound>::max())
    })
}

/// Builds the decimal logarithm `Function`
pub fn log() -> impl Function + Clone {
    PartitionnedMonotonic::univariate(data_type::Float::from(0.0..), |x| {
        x.log(10.)
            .clamp(<f64 as Bound>::min(), <f64 as Bound>::max())
    })
}

/// Builds the sqrt `Function`
pub fn sqrt() -> impl Function + Clone {
    PartitionnedMonotonic::univariate(data_type::Float::from(0.0..), |x| {
        x.sqrt().clamp(<f64 as Bound>::min(), <f64 as Bound>::max())
    })
}

/// The pow function
pub fn pow() -> impl Function + Clone {
    PartitionnedMonotonic::piecewise_bivariate(
        [
            (
                data_type::Float::from_min(0.0),
                data_type::Float::from_min(0.0),
            ),
            (
                data_type::Float::from_min(0.0),
                data_type::Float::from_max(0.0),
            ),
        ],
        |x, n| {
            x.powf(n)
                .clamp(<f64 as Bound>::min(), <f64 as Bound>::max())
        },
    )
}

/// Builds the abs `Function`, a piecewise monotonic function
pub fn abs() -> impl Function + Clone {
    PartitionnedMonotonic::piecewise_univariate(
        [
            data_type::Float::from(..=0.0),
            data_type::Float::from(0.0..),
        ],
        |x| x.abs(),
    )
}

/// sine
pub fn sin() -> impl Function + Clone {
    PartitionnedMonotonic::periodic_univariate(
        [
            data_type::Float::from(-0.5 * std::f64::consts::PI..=0.5 * std::f64::consts::PI),
            data_type::Float::from(0.5 * std::f64::consts::PI..=1.5 * std::f64::consts::PI),
        ],
        |x| x.sin(),
    )
}

/// cosine
pub fn cos() -> impl Function + Clone {
    PartitionnedMonotonic::periodic_univariate(
        [
            data_type::Float::from(0.0..=std::f64::consts::PI),
            data_type::Float::from(std::f64::consts::PI..=2.0 * std::f64::consts::PI),
        ],
        |x| x.cos(),
    )
}

pub fn bivariate_min() -> impl Function + Clone {
    Polymorphic::from((
        PartitionnedMonotonic::bivariate(
            (data_type::Integer::default(), data_type::Integer::default()),
            |x, y| x.min(y),
        ),
        PartitionnedMonotonic::bivariate(
            (data_type::Float::default(), data_type::Float::default()),
            |x, y| x.min(y),
        ),
    ))
}

pub fn bivariate_max() -> impl Function + Clone {
    Polymorphic::from((
        PartitionnedMonotonic::bivariate(
            (data_type::Integer::default(), data_type::Integer::default()),
            |x, y| x.max(y),
        ),
        PartitionnedMonotonic::bivariate(
            (data_type::Float::default(), data_type::Float::default()),
            |x, y| x.max(y),
        ),
    ))
}

// String functions

/// Builds the lower `Function`
pub fn lower() -> impl Function + Clone {
    PartitionnedMonotonic::univariate(data_type::Text::default(), |x| x.to_lowercase())
}

/// Builds the upper `Function`
pub fn upper() -> impl Function + Clone {
    PartitionnedMonotonic::univariate(data_type::Text::default(), |x| x.to_uppercase())
}

/// Builds the char_length `Function`
pub fn char_length() -> impl Function + Clone {
    Pointwise::univariate(
        data_type::Text::default(),
        data_type::Integer::default(),
        |a| a.len().try_into().unwrap(),
    )
}

/// Builds the position `Function`
pub fn position() -> impl Function + Clone {
    Pointwise::bivariate(
        (data_type::Text::default(), data_type::Text::default()),
        DataType::optional(DataType::integer()),
        |a, b| {
            Value::Optional(value::Optional::new(
                a.find(&b)
                    .map(|v| Rc::new(Value::integer(v.try_into().unwrap()))),
            ))
        },
    )
}

// Case function
pub fn case() -> impl Function + Clone {
    Case
}

/*
Aggregation functions
 */

/// Median aggregation
pub fn median() -> impl Function + Clone {
    null()
}

pub fn n_unique() -> impl Function + Clone {
    null()
}

/// First element in group
pub fn first() -> impl Function + Clone {
    Aggregate::from(
        DataType::Any,
        |values| values.first().unwrap().clone(),
        |(dt, _size)| match dt {
            DataType::List(list) => Ok(list.data_type().clone()),
            dt => Ok(dt),
        },
    )
}

/// Last element in group
pub fn last() -> impl Function + Clone {
    Aggregate::from(
        DataType::Any,
        |values| values.last().unwrap().clone(),
        |(dt, _size)| match dt {
            DataType::List(list) => Ok(list.data_type().clone()),
            dt => Ok(dt),
        },
    )
}

/// Mean aggregation
pub fn mean() -> impl Function + Clone {
    // Only works on types that can be converted to floats
    Aggregate::from(
        data_type::Float::full(),
        |values| {
            let (count, sum) = values.into_iter().fold((0.0, 0.0), |(count, sum), value| {
                (count + 1.0, sum + f64::from(value))
            });
            (sum / count).into()
        },
        |(intervals, _size)| Ok(intervals.into_interval()),
    )
}

/// Aggregate as a list
pub fn list() -> impl Function + Clone {
    null()
}

/// Count aggregation
pub fn count() -> impl Function + Clone {
    Polymorphic::from((
        // Any implementation
        Aggregate::from(
            DataType::Any,
            |values| (values.len() as i64).into(),
            |(_dt, size)| Ok(size),
        ),
        // Optional implementation
        Aggregate::from(
            data_type::Optional::from(DataType::Any),
            |values| {
                values
                    .iter()
                    .filter_map(|value| value.as_ref().and(Some(1)))
                    .sum::<i64>()
                    .into()
            },
            |(_dt, size)| Ok(data_type::Integer::from_interval(0, *size.max().unwrap())),
        ),
    ))
}

/// Min aggregation
pub fn min() -> impl Function + Clone {
    Polymorphic::from((
        // Integer implementation
        Aggregate::from(
            data_type::Integer::full(),
            |values| {
                values
                    .into_iter()
                    .map(|f| *f)
                    .min()
                    .unwrap_or(<i64 as Bound>::max())
                    .into()
            },
            |(intervals, _size)| Ok(intervals),
        ),
        // Float implementation
        Aggregate::from(
            data_type::Float::full(),
            |values| {
                values
                    .into_iter()
                    .map(|f| *f)
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(cmp::Ordering::Equal))
                    .unwrap_or(<f64 as Bound>::max())
                    .into()
            },
            |(intervals, _size)| Ok(intervals),
        ),
    ))
}

/// Max aggregation
pub fn max() -> impl Function + Clone {
    Polymorphic::from((
        // Integer implementation
        Aggregate::from(
            data_type::Integer::full(),
            |values| {
                values
                    .into_iter()
                    .map(|f| *f)
                    .max()
                    .unwrap_or(<i64 as Bound>::min())
                    .into()
            },
            |(intervals, _size)| Ok(intervals),
        ),
        // Float implementation
        Aggregate::from(
            data_type::Float::full(),
            |values| {
                values
                    .into_iter()
                    .map(|f| *f)
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(cmp::Ordering::Equal))
                    .unwrap_or(<f64 as Bound>::min())
                    .into()
            },
            |(intervals, _size)| Ok(intervals),
        ),
    ))
}

/// Quantile aggregation
pub fn quantile(_p: f64) -> impl Function + Clone {
    null()
}

/// Multi-quantileq aggregation
pub fn quantiles(_p: Vec<f64>) -> impl Function + Clone {
    null()
}

/// Sum aggregation
pub fn sum() -> impl Function + Clone {
    Polymorphic::from((
        // Integer implementation
        Aggregate::from(
            data_type::Integer::full(),
            |values| values.into_iter().map(|f| *f).sum::<i64>().into(),
            |(intervals, size)| {
                Ok(data_type::Integer::try_from(multiply().super_image(
                    &DataType::structured_from_data_types([intervals.into(), size.into()]),
                )?)?)
            },
        ),
        // Float implementation
        Aggregate::from(
            data_type::Float::full(),
            |values| values.into_iter().map(|f| *f).sum::<f64>().into(),
            |(intervals, size)| {
                Ok(data_type::Float::try_from(multiply().super_image(
                    &DataType::structured_from_data_types([intervals.into(), size.into()]),
                )?)?)
            },
        ),
    ))
}

/// Agg groups aggregation
pub fn agg_groups() -> impl Function + Clone {
    null()
}

/// Standard deviation aggregation
pub fn std() -> impl Function + Clone {
    // Only works on types that can be converted to floats
    Aggregate::from(
        data_type::Float::full(),
        |values| {
            let (count, sum, sum_2) =
                values
                    .into_iter()
                    .fold((0.0, 0.0, 0.0), |(count, sum, sum_2), value| {
                        let value: f64 = value.into();
                        (
                            count + 1.0,
                            sum + f64::from(value),
                            sum_2 + (f64::from(value) * f64::from(value)),
                        )
                    });
            ((sum_2 - sum * sum / count) / (count - 1.)).sqrt().into()
        },
        |(intervals, _size)| match (intervals.min(), intervals.max()) {
            (Some(&min), Some(&max)) => Ok(data_type::Float::from_interval(0., (max - min) / 2.)),
            _ => Ok(data_type::Float::from_min(0.)),
        },
    )
}

/// Variance aggregation
pub fn var() -> impl Function + Clone {
    // Only works on types that can be converted to floats
    Aggregate::from(
        data_type::Float::full(),
        |values| {
            let (count, sum, sum_2) =
                values
                    .into_iter()
                    .fold((0.0, 0.0, 0.0), |(count, sum, sum_2), value| {
                        let value: f64 = value.into();
                        (
                            count + 1.0,
                            sum + f64::from(value),
                            sum_2 + (f64::from(value) * f64::from(value)),
                        )
                    });
            ((sum_2 - sum * sum / count) / (count - 1.)).into()
        },
        |(intervals, _size)| match (intervals.min(), intervals.max()) {
            (Some(&min), Some(&max)) => Ok(data_type::Float::from_interval(
                0.,
                ((max - min) / 2.).powi(2),
            )),
            _ => Ok(data_type::Float::from_min(0.)),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::{
        super::{value::Value, Struct},
        *,
    };

    #[test]
    fn test_argument_conversion() {
        let fun = exp();
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());

        let set = DataType::integer_interval(-1, 5);
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(matches!(im, DataType::Float(_)));

        let set = DataType::float_interval(-1., 5.);
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(matches!(im, DataType::Float(_)));
    }

    #[test]
    fn test_eq() {
        println!("Test eq");
        let fun = eq();
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());

        let set = DataType::float_values([1., 2.]) & DataType::float_values([1., 2.]);
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(matches!(im, DataType::Boolean(_)));
        let arg = Value::float(1.) & Value::float(1.);
        let val = fun.value(&arg).unwrap();
        println!("val({}) = {}", arg, val);
    }

    #[test]
    fn test_exp() {
        println!("Test exp");
        let fun = Value::function(exp());
        println!("type = {}", fun);
        let fun = exp();
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());

        let set = DataType::float_interval(1., 2.);
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(matches!(im, DataType::Float(_)));
    }

    #[test]
    fn test_ln() {
        println!("Test ln");
        let fun = ln();
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());

        let set = DataType::float_interval(0., 2.);
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(matches!(im, DataType::Float(_)));
    }

    #[test]
    fn test_log() {
        println!("Test log");
        let fun = log();
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());

        let set = DataType::float_interval(0., 2.);
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(matches!(im, DataType::Float(_)));
    }

    #[test]
    fn test_abs() {
        println!("Test abs");
        let fun = abs();
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());

        let set = DataType::float_interval(-3., 2.);
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(matches!(im, DataType::Float(_)));
        // Test an alternative (wrong implementation) of abs
        let wrong_fun = PartitionnedMonotonic::univariate(data_type::Float::default(), |x| x.abs());
        let wrong_im = wrong_fun.super_image(&set).unwrap();
        println!("wrong im({}) = {}", set, wrong_im);
        assert!(im != wrong_im);
    }

    #[test]
    fn test_sin() {
        println!("Test sin");
        let fun = sin();
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());

        let set = DataType::float_interval(-0.1, 0.1);
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(matches!(im, DataType::Float(_)));

        let set = DataType::float_interval(0., 8.0);
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(matches!(im, DataType::Float(_)));

        let set =
            DataType::float_interval(4. * std::f64::consts::PI, 4. * std::f64::consts::PI + 0.6);
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(matches!(im, DataType::Float(_)));
    }

    #[test]
    fn test_sqrt() {
        println!("Test sqrt");
        let fun = sqrt();
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());

        let set = DataType::float_interval(0.01, 100.);
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(matches!(im, DataType::Float(_)));

        let set = DataType::integer_interval(0, 8);
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(matches!(im, DataType::Float(_)));
    }

    #[test]
    fn test_plus() {
        println!("Test plus");
        // Test a bivariate monotonic function
        let fun = plus();
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());
        let set = DataType::from(Struct::from_data_types(&[
            DataType::from(data_type::Float::from_intervals([
                [0., 2.],
                [5., 5.],
                [10., 10.],
            ])),
            DataType::float_interval(2.9, 3.),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(matches!(im, DataType::Float(_)));
    }

    #[test]
    fn test_mult() {
        println!("Test mult");
        // Test a bivariate monotonic function
        let fun = multiply();
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());
        // First test
        let set = DataType::from(Struct::from_data_types(&[
            DataType::from(data_type::Float::from_intervals([
                [0., 2.],
                [5., 5.],
                [10., 10.],
            ])),
            DataType::float_interval(-3., 3.),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        // Two intervals accross 0
        let set = DataType::from(Struct::from_data_types(&[
            DataType::float_interval(-1., 10.),
            DataType::float_interval(-1., 10.),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        // Test with values
        let set = DataType::from(Struct::from_data_types(&[
            DataType::float_values([0., 1., 3.]),
            DataType::float_values([-5., 5.]),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        // Test with values and thin intervals
        let set = DataType::from(Struct::from_data_types(&[
            DataType::float_values([-5., 1., 3.]),
            DataType::float_interval(-1.1, -1.),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(matches!(im, DataType::Float(_)));
        // Test with integers
        let set = DataType::from(Struct::from_data_types(&[
            DataType::integer_values([-5, 1, 4]),
            DataType::integer_interval(1, 3),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(matches!(im, DataType::Integer(_)));
    }

    #[test]
    fn test_concat() {
        println!("Test concat");
        // Test a bivariate monotonic function
        let cc = concat(3);
        println!("concat = {}", cc);
        println!(
            "concat(set) = {}",
            concat(3)
                .super_image(
                    &(DataType::float_values([0.0, 0.1])
                        & DataType::float_values([0.0, 0.1])
                        & DataType::float_values([0.0, 0.1]))
                )
                .unwrap()
        );
        println!(
            r#"concat(5, "hello", 12.5) = {}"#,
            concat(3)
                .value(&Value::structured_from_values(&[
                    5.into(),
                    "hello".to_string().into(),
                    12.5.into()
                ]))
                .unwrap()
        );
    }

    #[test]
    fn test_md5() {
        println!("Test md5");
        // Test a bivariate monotonic function
        let m = md5();
        println!("md5 = {}", m);
        println!(
            "md5(set) = {}",
            md5()
                .super_image(&DataType::structured_from_data_types([
                    DataType::text_values(["hello".into(), "world".into()])
                ]))
                .unwrap()
        );
        println!(
            r#"md5("hello") = {}"#,
            md5().value(&Value::text("hello")).unwrap()
        );
    }

    #[test]
    fn test_extended() {
        println!("Test extended");
        let extended_cos = cos().extend(DataType::Any);
        println!("cos = {}", cos());
        println!("extended cos = {}", extended_cos);
        println!(
            "extended extended cos = {}",
            extended_cos.clone().extend(DataType::integer())
        );
        println!(
            "extended extended cos = {}",
            extended_cos.clone().extend(DataType::Any)
        );
        assert_eq!(
            extended_cos.clone().extend(DataType::integer()).co_domain(),
            DataType::float_range(-1.0..=1.0)
        );
        assert_eq!(
            extended_cos.extend(DataType::Any).co_domain(),
            DataType::optional(DataType::float_range(-1.0..=1.0))
        );
    }

    #[test]
    fn test_extended_binary() {
        println!("Test extended");
        // Test a bivariate monotonic function
        let extended_add = plus().extend(DataType::Any & DataType::Any);
        println!("add = {}", plus());
        println!("extended add = {}", extended_add);
        println!(
            "extended extended add = {}",
            extended_add
                .clone()
                .extend(DataType::integer() & DataType::integer())
        );
        println!(
            "extended extended add = {}",
            extended_add.extend(DataType::Any & DataType::Any)
        );
    }

    #[test]
    fn test_extended_plus() {
        println!("Test extended");
        // Test a bivariate monotonic function
        let extended_plus = plus().extend(DataType::Any & DataType::Any);
        println!("plus = {}", plus());
        println!("extended plus = {}", extended_plus);
        println!(
            "plus(set) = {}",
            plus()
                .super_image(
                    &(DataType::float_interval(0.0, 0.1) & DataType::float_interval(0.0, 0.1))
                )
                .unwrap()
        );
        println!(
            "extended_plus(set) = {}",
            extended_plus
                .super_image(
                    &(DataType::float_interval(0.0, 0.1) & DataType::float_interval(0.0, 0.1))
                )
                .unwrap()
        );
    }

    #[test]
    fn test_aggregate_count() {
        println!("Test count aggregate");
        // Test an aggregate function
        let count = count();
        println!("count = {}", count);
        let list = DataType::list(DataType::float_interval(-1., 2.), 2, 20);
        println!("count({}) = {}", list, count.super_image(&list).unwrap());
        assert_eq!(
            count.super_image(&list).unwrap(),
            DataType::integer_interval(2, 20)
        );
        let list = DataType::list(DataType::integer_interval(1, 10), 2, 20);
        println!("count({}) = {}", list, count.super_image(&list).unwrap());
        assert_eq!(
            count.super_image(&list).unwrap(),
            DataType::integer_interval(2, 20)
        );
    }

    #[test]
    fn test_aggregate_sum() {
        println!("Test sum aggregate");
        // Test an aggregate function
        let sum = sum();
        println!("sum = {}", sum);
        let list = DataType::list(DataType::float_interval(-1., 2.), 2, 20);
        println!("sum({}) = {}", list, sum.super_image(&list).unwrap());
        assert_eq!(
            sum.super_image(&list).unwrap(),
            DataType::float_interval(-20., 40.)
        );
        let list = DataType::list(DataType::integer_interval(1, 10), 2, 20);
        println!("sum({}) = {}", list, sum.super_image(&list).unwrap());
        assert_eq!(
            sum.super_image(&list).unwrap(),
            DataType::integer_interval(2, 200)
        );
        let list = DataType::integer_interval(1, 10);
        println!("sum({}) = {}", list, sum.super_image(&list).unwrap());
        assert_eq!(
            sum.super_image(&list).unwrap(),
            DataType::integer_interval(1, 10)
        );
    }

    #[test]
    fn test_aggregate_var() {
        println!("Test var aggregate");
        // Test an aggregate function
        let var = var();
        println!("var = {}", var);
        let list = DataType::list(DataType::float_interval(-1., 2.), 2, 20);
        println!("var({}) = {}", list, var.super_image(&list).unwrap());
        let list = DataType::list(DataType::integer_interval(1, 10), 2, 20);
        println!("var({}) = {}", list, var.super_image(&list).unwrap());
    }

    #[test]
    fn test_extended_aggregate_sum() {
        println!("Test extended");
        // Test a bivariate monotonic function
        let extended_sum = sum().extend(DataType::Any);
        println!("sum = {}", sum());
        println!("sum domain = {}", sum().domain());
        println!("extended sum = {}", extended_sum);
        let set = DataType::list(DataType::float_interval(0.0, 0.1), 1, 10);
        let super_set = DataType::list(DataType::float(), 0, 100);
        println!("sum(set) = {}", sum().super_image(&set).unwrap());
        println!(
            "extended_sum(set) = {}",
            extended_sum.super_image(&set).unwrap()
        );
        println!(
            "{} is subset of {} = {}",
            set,
            super_set,
            set.is_subset_of(&super_set)
        );
        assert!(set.is_subset_of(&super_set));
        println!(
            "{} is subset of {} = {}",
            set,
            sum().domain(),
            set.is_subset_of(&sum().domain())
        );
        assert!(set.is_subset_of(&sum().domain()));
    }

    #[test]
    fn test_case() {
        println!("Test case");
        let fun = case();
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());

        // true, int, int
        let set = DataType::from(Struct::from_data_types(&[
            DataType::boolean_value(true),
            DataType::from(data_type::Integer::from_intervals([
                [0, 2],
                [5, 5],
                [10, 10],
            ])),
            DataType::integer_values(vec![10, 15, 30]),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(
            im == DataType::from(data_type::Integer::from_intervals([
                [0, 2],
                [5, 5],
                [10, 10]
            ]))
        );

        // false, int, int
        let set = DataType::from(Struct::from_data_types(&[
            DataType::boolean_value(false),
            DataType::from(data_type::Integer::from_intervals([
                [0, 2],
                [5, 5],
                [10, 10],
            ])),
            DataType::integer_values(vec![10, 15, 30]),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(im == DataType::integer_values(vec!(10, 15, 30)));

        // none, int, int
        let set = DataType::from(Struct::from_data_types(&[
            DataType::from(data_type::Boolean::empty()),
            DataType::from(data_type::Integer::from_intervals([
                [0, 2],
                [5, 5],
                [10, 10],
            ])),
            DataType::integer_values(vec![10, 15, 30]),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(im == DataType::Null);

        // {false, true}, int, int
        let set = DataType::from(Struct::from_data_types(&[
            DataType::from(data_type::Boolean::default()),
            DataType::from(data_type::Integer::from_intervals([
                [0, 2],
                [5, 5],
                [10, 10],
            ])),
            DataType::integer_values(vec![10, 15, 30]),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(matches!(im, DataType::Integer(_)));

        // {false, true}, int, float
        let set = DataType::from(Struct::from_data_types(&[
            DataType::from(data_type::Boolean::default()),
            DataType::from(data_type::Integer::from_intervals([
                [0, 2],
                [5, 5],
                [10, 10],
            ])),
            DataType::float_values(vec![10., 15., 30.56]),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(matches!(im, DataType::Float(_)));

        // {false, true}, int, text
        let set = DataType::from(Struct::from_data_types(&[
            DataType::from(data_type::Boolean::default()),
            DataType::from(data_type::Integer::from_intervals([
                [0, 2],
                [5, 5],
                [10, 10],
            ])),
            DataType::text_values(vec!["a".to_string(), "b".to_string()]),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(matches!(im, DataType::Text(_)));

        // {false, true}, int, text
        let date_a = chrono::NaiveDate::from_isoywd_opt(2022, 10, chrono::Weekday::Mon).unwrap();
        let date_b = date_a + chrono::Duration::days(10);
        let set = DataType::from(Struct::from_data_types(&[
            DataType::from(data_type::Boolean::default()),
            DataType::date_interval(date_a, date_b),
            DataType::from(data_type::Integer::from_intervals([
                [0, 2],
                [5, 5],
                [10, 10],
            ])),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(im == DataType::Any);
    }

    #[test]
    fn test_lower() {
        println!("Test lower");
        let fun = lower();
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());

        let set: DataType =
            data_type::Text::from_values([String::from("Hello"), String::from("World")]).into();
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(matches!(im, DataType::Text(_)));
    }

    #[test]
    fn test_upper() {
        println!("Test uppeer");
        let fun = upper();
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());

        let set: DataType =
            data_type::Text::from_values([String::from("Hello"), String::from("World")]).into();
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(matches!(im, DataType::Text(_)));
    }

    #[test]
    fn test_char_length() {
        println!("Test char_length");
        let fun = char_length();
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());

        let set: DataType =
            data_type::Text::from_values([String::from("Hello"), String::from("World!")]).into();
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(matches!(im, DataType::Integer(_)));
    }

    #[test]
    fn test_position() {
        println!("Test position");
        let fun = position();
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());

        let set = DataType::from(Struct::from_data_types(&[
            DataType::from(data_type::Text::from_values([
                String::from("Hello"),
                String::from("World"),
            ])),
            DataType::from(data_type::Text::from_values([
                String::from("e"),
                String::from("z"),
            ])),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(matches!(im, DataType::Optional(_)));

        let set = DataType::from(Struct::from_data_types(&[
            DataType::from(data_type::Text::from_values([
                String::from("Hello"),
                String::from("World"),
            ])),
            DataType::from(data_type::Text::from_values([String::from("l")])),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(matches!(im, DataType::Optional(_)));
    }
}
