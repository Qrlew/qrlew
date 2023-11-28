use std::{
    borrow::BorrowMut,
    cell::RefCell,
    cmp, collections,
    convert::{Infallible, TryFrom, TryInto},
    error, fmt,
    hash::Hasher,
    ops::Deref,
    result,
    sync::{Arc, Mutex},
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
pub trait Function: fmt::Debug + fmt::Display + Sync + Send {
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

impl<Inj: injection::Injection + Sync + Send> Function for Injection<Inj>
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

/// A function defined by its type signature and potentially stateful value function without any other particular properties
/// In particular, no range computation is done
/// Note that stateful computations should be avoided and reserved to pseudorandom functions//TODO remove this feature?
#[derive(Clone)]
pub struct Stateful {
    domain: DataType,
    co_domain: DataType,
    value: Arc<Mutex<RefCell<dyn FnMut(Value) -> Value + Send>>>,
}

impl Stateful {
    /// Constructor for Generic
    pub fn new(
        domain: DataType,
        co_domain: DataType,
        value: Arc<Mutex<RefCell<dyn FnMut(Value) -> Value + Send>>>,
    ) -> Self {
        Stateful {
            domain,
            co_domain,
            value,
        }
    }
}

impl fmt::Debug for Stateful {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "simple{{{} -> {}}}", self.domain(), self.co_domain())
    }
}

impl fmt::Display for Stateful {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "simple{{{} -> {}}}", self.domain(), self.co_domain())
    }
}

impl Function for Stateful {
    fn domain(&self) -> DataType {
        self.domain.clone()
    }

    fn super_image(&self, _set: &DataType) -> Result<DataType> {
        Ok(self.co_domain.clone())
    }

    fn value(&self, arg: &Value) -> Result<Value> {
        let locked_value = self.value.lock().unwrap();
        let mut borrowed_value = (*locked_value).borrow_mut();
        Ok((*borrowed_value)(arg.clone()))
    }
}

/// A function defined pointwise without any other particular properties
/// Range computation is done on finite ranges
#[derive(Clone)]
pub struct Pointwise {
    domain: DataType,
    co_domain: DataType,
    value: Arc<dyn Fn(Value) -> Result<Value> + Sync + Send>,
}

impl Pointwise {
    /// Constructor for Generic
    pub fn new(
        domain: DataType,
        co_domain: DataType,
        value: Arc<dyn Fn(Value) -> Result<Value> + Sync + Send>,
    ) -> Self {
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
            + Sync
            + Send
            + 'static,
    ) -> Self
    where
        <<A::Element as value::Variant>::Wrapped as TryFrom<Value>>::Error: fmt::Debug,
        Error: From<<<A::Element as value::Variant>::Wrapped as TryFrom<Value>>::Error>,
    {
        Self::new(
            domain.into(),
            co_domain.into(),
            Arc::new(move |a| {
                Ok(
                    <A::Element as value::Variant>::Wrapped::try_from(a)
                        .map(|a| value(a).into())?,
                )
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
            + Sync
            + Send
            + 'static,
    ) -> Self
    where
        <A::Element as value::Variant>::Wrapped: TryFrom<Value>,
        <B::Element as value::Variant>::Wrapped: TryFrom<Value>,
        <<A::Element as value::Variant>::Wrapped as TryFrom<Value>>::Error: fmt::Debug,
        Error: From<<<A::Element as value::Variant>::Wrapped as TryFrom<Value>>::Error>,
        <<B::Element as value::Variant>::Wrapped as TryFrom<Value>>::Error: fmt::Debug,
        Error: From<<<B::Element as value::Variant>::Wrapped as TryFrom<Value>>::Error>,
        <C::Element as value::Variant>::Wrapped: Into<Value>,
    {
        let domain = data_type::Struct::from_data_types(&[domain.0.into(), domain.1.into()]);
        Self::new(
            domain.into(),
            co_domain.into(),
            Arc::new(move |ab| {
                let ab = value::Struct::try_from(ab).unwrap();
                let a = <A::Element as value::Variant>::Wrapped::try_from(ab[0].as_ref().clone());
                let b = <B::Element as value::Variant>::Wrapped::try_from(ab[1].as_ref().clone());
                Ok(a.map(|a| b.map(|b| value(a, b).into()))??)
            }),
        )
    }

    /// Build trivariate pointwise function
    pub fn trivariate<A: Variant, B: Variant, C: Variant, D: Variant>(
        domain: (A, B, C),
        co_domain: D,
        value: impl Fn(
                <A::Element as value::Variant>::Wrapped,
                <B::Element as value::Variant>::Wrapped,
                <C::Element as value::Variant>::Wrapped,
            ) -> <D::Element as value::Variant>::Wrapped
            + Sync
            + Send
            + 'static,
    ) -> Self
    where
        <A::Element as value::Variant>::Wrapped: TryFrom<Value>,
        <B::Element as value::Variant>::Wrapped: TryFrom<Value>,
        <C::Element as value::Variant>::Wrapped: TryFrom<Value>,
        <<A::Element as value::Variant>::Wrapped as TryFrom<Value>>::Error: fmt::Debug,
        Error: From<<<A::Element as value::Variant>::Wrapped as TryFrom<Value>>::Error>,
        <<B::Element as value::Variant>::Wrapped as TryFrom<Value>>::Error: fmt::Debug,
        Error: From<<<B::Element as value::Variant>::Wrapped as TryFrom<Value>>::Error>,
        <<C::Element as value::Variant>::Wrapped as TryFrom<Value>>::Error: fmt::Debug,
        Error: From<<<C::Element as value::Variant>::Wrapped as TryFrom<Value>>::Error>,
        <D::Element as value::Variant>::Wrapped: Into<Value>,
    {
        let domain = data_type::Struct::from_data_types(&[
            domain.0.into(),
            domain.1.into(),
            domain.2.into(),
        ]);
        Self::new(
            domain.into(),
            co_domain.into(),
            Arc::new(move |ab| {
                let abc = value::Struct::try_from(ab).unwrap();
                let a = <A::Element as value::Variant>::Wrapped::try_from(abc[0].as_ref().clone());
                let b = <B::Element as value::Variant>::Wrapped::try_from(abc[1].as_ref().clone());
                let c = <C::Element as value::Variant>::Wrapped::try_from(abc[2].as_ref().clone());
                Ok(a.map(|a| b.map(|b| c.map(|c| value(a, b, c).into())))???)
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
            + Sync
            + Send
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
            Arc::new(move |v| {
                let vec = value::Struct::try_from(v)
                    .unwrap()
                    .into_iter()
                    .map(|(_n, v)| {
                        <D::Element as value::Variant>::Wrapped::try_from(v.as_ref().clone())
                    }) //.unwrap())
                    .collect::<Vec<_>>();
                if vec.iter().all(|v| v.is_ok()) {
                    let v = vec.into_iter().map(|v| v.unwrap()).collect();
                    Ok(value(v).into())
                } else {
                    Err(Error::other("Argument out of range"))
                }
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
        let converted_set = &set.into_data_type(&self.domain())?;
        let super_image = if let Ok(vec) = TryInto::<Vec<Value>>::try_into(converted_set.clone()) {
            vec.into_iter()
                .map(|v| (*self.value)(v))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .collect()
        } else {
            self.co_domain.clone()
        };
        if !converted_set.is_subset_of(&self.domain()) {
            Err(Error::set_out_of_range(converted_set, self.domain()))
        } else {
            Ok(super_image)
        }
    }

    fn value(&self, arg: &Value) -> Result<Value> {
        let converted_arg = &arg.as_data_type(&self.domain())?;
        let value = (*self.value)(converted_arg.clone())?;
        if !self.domain().contains(converted_arg) {
            Err(Error::argument_out_of_range(converted_arg, self.domain()))
        } else if !self.co_domain().contains(&value) {
            Err(Error::argument_out_of_range(value, self.co_domain()))
        } else {
            Ok(value)
        }
    }
}

/// Partitionned monotonic function (plus some complex periodic cases).
/// The domain is a (cartesian) product of `Intervals<B>` types.
/// `P` and `T` are convenient representations of the product and elements of the product.
/// The partition function maps a product into a vector of products where the value function is supposed to be monotonic.
#[derive(Clone)]
pub struct PartitionnedMonotonic<P, T, Prod: IntervalsProduct, U: Bound>
where
    P: From<Prod> + Into<Prod>,
    T: From<<Prod::IntervalProduct as IntervalProduct>::BoundProduct>,
{
    domain: Prod,
    partition: Arc<dyn Fn(P) -> Vec<P> + Sync + Send>,
    value: Arc<dyn Fn(T) -> U + Sync + Send>,
}

impl<P, T, Prod: IntervalsProduct, U: Bound> PartitionnedMonotonic<P, T, Prod, U>
where
    P: From<Prod> + Into<Prod>,
    T: From<<Prod::IntervalProduct as IntervalProduct>::BoundProduct>,
{
    /// Constructor for Base Maps
    pub fn new(
        domain: Prod,
        partition: Arc<dyn Fn(P) -> Vec<P> + Sync + Send>,
        value: Arc<dyn Fn(T) -> U + Sync + Send>,
    ) -> Self {
        PartitionnedMonotonic {
            domain,
            partition,
            value,
        }
    }

    pub fn from_intervals(domain: P, value: impl Fn(T) -> U + Sync + Send + 'static) -> Self
    where
        P: Clone + Sync + Send + 'static,
    {
        Self::new(
            domain.clone().into(),
            Arc::new(move |set: P| vec![set.into().intersection(&domain.clone().into()).into()]),
            Arc::new(value),
        )
    }

    pub fn from_partitions(
        partitions: impl AsRef<[P]> + Sync + Send + 'static,
        value: impl Fn(T) -> U + Sync + Send + 'static,
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
        Self::new(domain, Arc::new(partition), Arc::new(value))
    }
}

impl<A: Bound + Sync + Send + 'static, B: Bound + Sync + Send>
    PartitionnedMonotonic<Intervals<A>, (A,), Term<Intervals<A>, Unit>, B>
{
    pub fn univariate(
        domain: Intervals<A>,
        value: impl Fn(A) -> B + Sync + Send + 'static,
    ) -> Self {
        Self::new(
            domain.clone().into(),
            Arc::new(move |set: Intervals<A>| vec![set.intersection(domain.clone())]),
            Arc::new(move |arg: (A,)| value(arg.0)),
        )
    }

    pub fn piecewise_univariate<const N: usize>(
        partitions: [Intervals<A>; N],
        value: impl Fn(A) -> B + Sync + Send + 'static,
    ) -> Self {
        Self::from_partitions(partitions, move |(a,)| value(a))
    }
}

impl PartitionnedMonotonic<Intervals<f64>, (f64,), Term<Intervals<f64>, Unit>, f64> {
    pub fn periodic_univariate<const N: usize>(
        partitions: [Intervals<f64>; N],
        value: impl Fn(f64) -> f64 + Sync + Send + 'static,
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
            Arc::new(partition),
            Arc::new(move |(a,)| value(a)),
        )
    }
}

impl<A: Bound + Sync + Send + 'static, B: Bound + Sync + Send + 'static, C: Bound>
    PartitionnedMonotonic<
        (Intervals<A>, Intervals<B>),
        (A, B),
        Term<Intervals<A>, Term<Intervals<B>, Unit>>,
        C,
    >
{
    pub fn bivariate(
        domain: (Intervals<A>, Intervals<B>),
        value: impl Fn(A, B) -> C + Sync + Send + 'static,
    ) -> Self {
        Self::from_intervals(domain, move |(a, b)| value(a, b))
    }

    pub fn piecewise_bivariate<const N: usize>(
        partitions: [(Intervals<A>, Intervals<B>); N],
        value: impl Fn(A, B) -> C + Sync + Send + 'static,
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

impl<P, T, Prod: IntervalsProduct + Sync + Send, U: Bound> Function
    for PartitionnedMonotonic<P, T, Prod, U>
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
pub struct Optional<F: Function>(F);

impl<F: Function> Optional<F> {
    pub fn new(function: F) -> Optional<F> {
        Optional(function)
    }
}

impl<F: Function> fmt::Display for Optional<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "optional{{{} -> {}}}", self.domain(), self.co_domain())
    }
}

impl<F: Function> Function for Optional<F> {
    fn domain(&self) -> DataType {
        DataType::Any
    }

    fn co_domain(&self) -> DataType {
        DataType::optional(self.0.co_domain()).flatten_optional()
    }

    fn super_image(&self, set: &DataType) -> Result<DataType> {
        let set = set.flatten_optional();
        match set {
            DataType::Optional(optional_set) => self
                .0
                .super_image(optional_set.data_type())
                .map(|dt| DataType::optional(dt)),
            set => self.0.super_image(&set),
        }
        .or_else(|err| Ok(self.co_domain()))
    }

    fn value(&self, arg: &Value) -> Result<Value> {
        match arg {
            Value::Optional(optional_arg) => match optional_arg.as_deref() {
                Some(arg) => self.0.value(arg).map(Value::some),
                None => Ok(Value::none()),
            },
            arg => self.0.value(arg),
        }
        .or_else(|err| Ok(Value::none()))
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

/// A function defined pointwise without any other particular properties
#[derive(Clone)]
pub struct Aggregate<A: Variant + Sync + Send, B: Variant + Sync + Send>
where
    A::Element: TryFrom<Value>,
    Error: From<<A::Element as TryFrom<Value>>::Error>,
    B::Element: Into<Value>,
    A: Into<DataType> + TryFrom<DataType>,
    Error: From<<A as TryFrom<DataType>>::Error>,
    B: Into<DataType>,
{
    aggregation_domain: A,
    value: Arc<dyn Fn(Vec<A::Element>) -> B::Element + Sync + Send>,
    super_image: Arc<dyn Fn((A, Integer)) -> Result<B> + Sync + Send>,
}

impl<A: Variant + Sync + Send, B: Variant + Sync + Send> Aggregate<A, B>
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
        value: Arc<dyn Fn(Vec<A::Element>) -> B::Element + Sync + Send>,
        super_image: Arc<dyn Fn((A, Integer)) -> Result<B> + Sync + Send>,
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
        value: impl Fn(Vec<A::Element>) -> B::Element + Sync + Send + 'static,
        super_image: impl Fn((A, Integer)) -> Result<B> + Sync + Send + 'static,
    ) -> Self {
        Aggregate::new(aggregation_domain, Arc::new(value), Arc::new(super_image))
    }
}

impl<A: Variant + Sync + Send, B: Variant + Sync + Send> fmt::Debug for Aggregate<A, B>
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

impl<A: Variant + Sync + Send, B: Variant + Sync + Send> fmt::Display for Aggregate<A, B>
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

impl<A: Variant + Sync + Send, B: Variant + Sync + Send> Function for Aggregate<A, B>
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
#[derive(Debug, Default)]
pub struct Polymorphic(Vec<Arc<dyn Function + Sync + Send>>);

impl Polymorphic {
    /// Constructor for Polymorphic
    pub fn new(implementations: Vec<Arc<dyn Function + Sync + Send>>) -> Self {
        Polymorphic(implementations)
    }
}

impl<F: Function + Sync + 'static, G: Function + Sync + 'static> From<(F, G)> for Polymorphic {
    fn from((f, g): (F, G)) -> Self {
        Polymorphic(vec![Arc::new(f), Arc::new(g)])
    }
}

impl<F: Function + Sync + 'static, G: Function + Sync + 'static, H: Function + Sync + 'static>
    From<(F, G, H)> for Polymorphic
{
    fn from((f, g, h): (F, G, H)) -> Self {
        Polymorphic(vec![Arc::new(f), Arc::new(g), Arc::new(h)])
    }
}

impl<F: Function + 'static> With<F> for Polymorphic {
    fn with(mut self, input: F) -> Self {
        self.0.push(Arc::new(input));
        self
    }
}

impl<const N: usize> From<[Arc<dyn Function + Sync + Send>; N]> for Polymorphic {
    fn from(fs: [Arc<dyn Function + Sync + Send>; N]) -> Self {
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
                    _ => return Err(Error::set_out_of_range(set, self.domain())),
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
                Err(Error::set_out_of_range(set, self.domain()))
            }
        }
    }

    fn value(&self, arg: &Value) -> Result<Value> {
        if let Value::Struct(struct_values) = arg {
            if struct_values.field_from_index(0).1 == Arc::new(Value::boolean(true)) {
                Ok(struct_values.field_from_index(1).1.as_ref().clone())
            } else {
                Ok(struct_values.field_from_index(2).1.as_ref().clone())
            }
        } else {
            Err(Error::argument_out_of_range(arg, self.domain()))
        }
    }
}

// TODO
#[derive(Clone, Debug)]
pub struct UserDefineFunction {
    name: String,
    domain: DataType,
    co_domain: DataType
}

impl fmt::Display for UserDefineFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl Function for UserDefineFunction {
    fn domain(&self) -> DataType {
        self.domain.clone()
    }

    fn super_image(&self, set: &DataType) -> Result<DataType> {
        if !set.is_subset_of(&self.domain()) {
            Err(Error::set_out_of_range(set, self.domain()))
        } else {
            Ok(self.co_domain.clone())
        }
    }

    fn value(&self, arg: &Value) -> Result<Value> {
        todo!()
    }
}

impl UserDefineFunction {
    pub fn new(name: String, domain: DataType, co_domain: DataType) -> UserDefineFunction {
        UserDefineFunction {name, domain, co_domain}
    }
}

// IN (..)
#[derive(Clone, Debug)]
pub struct InList(DataType);

impl fmt::Display for InList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "in")
    }
}

impl Function for InList {
    fn domain(&self) -> DataType {
        DataType::from(data_type::Struct::from_data_types(&[
            self.0.clone(),
            DataType::list(self.0.clone(), 1, i64::MAX as usize),
        ]))
    }

    fn super_image(&self, set: &DataType) -> Result<DataType> {
        if !set.is_subset_of(&self.domain()) {
            Err(Error::set_out_of_range(set, self.domain()))
        } else {
            if let DataType::Struct(struct_data_type) = set {
                assert_eq!(struct_data_type.len(), 2);
                if let DataType::List(List { data_type, .. }) = struct_data_type[1].as_ref() {
                    Ok(
                        if struct_data_type[0].as_ref().super_intersection(data_type)?
                            == DataType::Null
                        {
                            DataType::boolean_value(false)
                        } else {
                            DataType::boolean()
                        },
                    )
                } else {
                    Err(Error::set_out_of_range(set, self.domain()))
                }
            } else {
                Err(Error::set_out_of_range(set, self.domain()))
            }
        }
    }

    fn value(&self, arg: &Value) -> Result<Value> {
        let domain = self.domain();
        let arg = &arg.as_data_type(&domain)?;
        if let Value::Struct(args) = arg {
            assert_eq!(args.len(), 2);
            if let Value::List(list) = args[1].as_ref() {
                Ok(if list.iter().any(|v| v == args[0].as_ref()) {
                    Value::boolean(true)
                } else {
                    Value::boolean(false)
                })
            } else {
                Err(Error::argument_out_of_range(arg, self.domain()))
            }
        } else {
            Err(Error::argument_out_of_range(arg, self.domain()))
        }
    }
}

#[derive(Clone, Debug)]
pub struct Coalesce;

impl fmt::Display for Coalesce {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "coalesce")
    }
}

impl Function for Coalesce {
    fn domain(&self) -> DataType {
        DataType::from(data_type::Struct::from_data_types(&[
            DataType::Any,
            DataType::Any,
        ]))
    }

    fn super_image(&self, set: &DataType) -> Result<DataType> {
        if !set.is_subset_of(&self.domain()) {
            Err(Error::set_out_of_range(set, self.domain()))
        } else {
            if let DataType::Struct(struct_data_type) = set {
                let data_type_1 = struct_data_type.field_from_index(0).1.as_ref().clone();
                let data_type_2 = struct_data_type.field_from_index(1).1.as_ref().clone();

                Ok(if let DataType::Optional(o) = data_type_1 {
                    o.data_type().super_union(&data_type_2)?
                } else {
                    data_type_1
                })
            } else {
                Err(Error::set_out_of_range(set, self.domain()))
            }
        }
    }

    fn value(&self, arg: &Value) -> Result<Value> {
        if let Value::Struct(struct_values) = arg {
            if struct_values.field_from_index(0).1 == Arc::new(Value::none()) {
                Ok(struct_values.field_from_index(1).1.as_ref().clone())
            } else {
                Ok(struct_values.field_from_index(0).1.as_ref().clone())
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
pub fn null() -> impl Function {
    PartitionnedMonotonic::univariate(data_type::Text::default(), |_x| "null".to_string())
}

/*
Conversion function
 */

/// Builds the cast operator
pub fn cast(into: DataType) -> impl Function {
    match into {
        DataType::Text(t) if t == data_type::Text::full() => {
            Pointwise::univariate(
                //DataType::Any,
                DataType::Any,
                DataType::text(),
                |v| v.to_string().into())
        }
        DataType::Float(f) if f == data_type::Float::full() => {
            Pointwise::univariate(
                DataType::text(),
                DataType::float(),
                |v| v.to_string().parse::<f64>().unwrap().into()
            )
        }
        DataType::Integer(i) if i == data_type::Integer::full() => {
            Pointwise::univariate(
                DataType::text(),
                DataType::integer(),
                |v| v.to_string().parse::<i64>().unwrap().into()
            )
        }
        DataType::Boolean(b) if b == data_type::Boolean::full() => {
            Pointwise::univariate(
                DataType::text(),
                DataType::boolean(),
                |v| {
                    let true_list = vec![
                        "t".to_string(), "tr".to_string(), "tru".to_string(), "true".to_string(),
                        "y".to_string(), "ye".to_string(), "yes".to_string(),
                        "on".to_string(),
                        "1".to_string()
                    ];
                    let false_list = vec![
                        "f".to_string(), "fa".to_string(), "fal".to_string(), "fals".to_string(), "false".to_string(),
                        "n".to_string(), "no".to_string(),
                        "off".to_string(),
                        "0".to_string()
                    ];
                    if true_list.contains(&v.to_string().to_lowercase()) {
                        true.into()
                    } else if false_list.contains(&v.to_string().to_lowercase()) {
                        false.into()
                    } else {
                        panic!()
                    }
                }
            )
        }
        DataType::Date(d) if d == data_type::Date::full() => {
            Pointwise::univariate(
                DataType::text(),
                DataType::date(),
                |v| todo!()
            )
        }
        DataType::DateTime(d) if d == data_type::DateTime::full() => {
            Pointwise::univariate(
                DataType::text(),
                DataType::date_time(),
                |v| todo!()
            )
        }
        DataType::Time(t) if t == data_type::Time::full() => {
            Pointwise::univariate(
                DataType::text(),
                DataType::time(),
                |v| todo!()
            )
        }
        _ => todo!(),
    }
}

// Unary operators

/// Builds the minus `Function`
pub fn opposite() -> impl Function {
    PartitionnedMonotonic::univariate(data_type::Float::default(), |x| -x)
}
/// Builds the minus `Function`
pub fn not() -> impl Function {
    PartitionnedMonotonic::univariate(data_type::Boolean::default(), |x| !x)
}

// Arithmetic binary operators

/// The sum (polymorphic)
pub fn plus() -> impl Function {
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
pub fn minus() -> impl Function {
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
pub fn multiply() -> impl Function {
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
pub fn divide() -> impl Function {
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
pub fn modulo() -> impl Function {
    Pointwise::bivariate(
        (data_type::Integer::default(), data_type::Integer::default()),
        data_type::Integer::default(),
        |a, b| (a % b).into(),
    )
}

pub fn string_concat() -> impl Function {
    Pointwise::bivariate(
        (data_type::Text::default(), data_type::Text::default()),
        data_type::Text::default(),
        |a, b| (a + &b).into(),
    )
}

pub fn rtrim() -> impl Function {
    Pointwise::bivariate(
        (data_type::Text::default(), data_type::Text::default()),
        data_type::Text::default(),
        |a, b| a.as_str().trim_end_matches(b.as_str()).into(),
    )
}

pub fn ltrim() -> impl Function {
    Pointwise::bivariate(
        (data_type::Text::default(), data_type::Text::default()),
        data_type::Text::default(),
        |a, b| a.as_str().trim_start_matches(b.as_str()).into(),
    )
}

pub fn substr() -> impl Function {
    Pointwise::bivariate(
        (data_type::Text::default(), data_type::Integer::default()),
        data_type::Text::default(),
        |a, b| {
            let start = b as usize;
            a.as_str().get(start..).unwrap_or("").to_string()
        },
    )
}

pub fn substr_with_size() -> impl Function {
    Pointwise::trivariate(
        (
            data_type::Text::default(),
            data_type::Integer::default(),
            data_type::Integer::default(),
        ),
        data_type::Text::default(),
        |a, b, c| {
            let start = b as usize;
            let end = cmp::min((b + c) as usize, a.len());
            a.as_str().get(start..end).unwrap_or("").to_string()
        },
    )
}

pub fn concat(n: usize) -> impl Function {
    Pointwise::variadic(vec![DataType::Any; n], data_type::Text::default(), |v| {
        v.into_iter().map(|v| v.to_string()).join("")
    })
}

pub fn md5() -> impl Function {
    Stateful::new(
        DataType::text(),
        DataType::text(),
        Arc::new(Mutex::new(RefCell::new(|v| {
            let mut s = collections::hash_map::DefaultHasher::new();
            Bound::hash((value::Text::try_from(v).unwrap()).deref(), &mut s);
            Encoder::new(BASE_64, 10).encode(s.finish()).into()
        }))),
    )
}

pub fn random<R: rand::Rng + Send + 'static>(mut rng: Mutex<R>) -> impl Function {
    Stateful::new(
        DataType::unit(),
        DataType::float_interval(0., 1.),
        Arc::new(Mutex::new(RefCell::new(move |v| {
            rng.lock().unwrap().borrow_mut().gen::<f64>().into()
        }))),
    )
}

pub fn pi() -> impl Function {
    Stateful::new(
        DataType::unit(),
        DataType::float_value(3.141592653589793),
        Arc::new(Mutex::new(RefCell::new(move |_| 3.141592653589793.into()))),
    )
}

pub fn gt() -> impl Function {
    Polymorphic::default()
        .with(PartitionnedMonotonic::bivariate(
            (data_type::Float::default(), data_type::Float::default()),
            |a, b| (a > b),
        ))
        .with(PartitionnedMonotonic::bivariate(
            (data_type::Date::default(), data_type::Date::default()),
            |a, b| (a > b),
        ))
        .with(PartitionnedMonotonic::bivariate(
            (data_type::Time::default(), data_type::Time::default()),
            |a, b| (a > b),
        ))
        .with(PartitionnedMonotonic::bivariate(
            (
                data_type::DateTime::default(),
                data_type::DateTime::default(),
            ),
            |a, b| (a > b),
        ))
        .with(PartitionnedMonotonic::bivariate(
            (data_type::Text::default(), data_type::Text::default()),
            |a, b| (a > b),
        ))
}

pub fn lt() -> impl Function {
    Polymorphic::default()
        .with(PartitionnedMonotonic::bivariate(
            (data_type::Float::default(), data_type::Float::default()),
            |a, b| (a < b),
        ))
        .with(PartitionnedMonotonic::bivariate(
            (data_type::Date::default(), data_type::Date::default()),
            |a, b| (a < b),
        ))
        .with(PartitionnedMonotonic::bivariate(
            (data_type::Time::default(), data_type::Time::default()),
            |a, b| (a < b),
        ))
        .with(PartitionnedMonotonic::bivariate(
            (
                data_type::DateTime::default(),
                data_type::DateTime::default(),
            ),
            |a, b| (a < b),
        ))
        .with(PartitionnedMonotonic::bivariate(
            (data_type::Text::default(), data_type::Text::default()),
            |a, b| (a < b),
        ))
}

pub fn gt_eq() -> impl Function {
    Polymorphic::default()
        .with(PartitionnedMonotonic::bivariate(
            (data_type::Float::default(), data_type::Float::default()),
            |a, b| (a >= b),
        ))
        .with(PartitionnedMonotonic::bivariate(
            (data_type::Date::default(), data_type::Date::default()),
            |a, b| (a >= b),
        ))
        .with(PartitionnedMonotonic::bivariate(
            (data_type::Time::default(), data_type::Time::default()),
            |a, b| (a >= b),
        ))
        .with(PartitionnedMonotonic::bivariate(
            (
                data_type::DateTime::default(),
                data_type::DateTime::default(),
            ),
            |a, b| (a >= b),
        ))
        .with(PartitionnedMonotonic::bivariate(
            (data_type::Text::default(), data_type::Text::default()),
            |a, b| (a >= b),
        ))
}

pub fn lt_eq() -> impl Function {
    Polymorphic::default()
        .with(PartitionnedMonotonic::bivariate(
            (data_type::Float::default(), data_type::Float::default()),
            |a, b| (a <= b),
        ))
        .with(PartitionnedMonotonic::bivariate(
            (data_type::Date::default(), data_type::Date::default()),
            |a, b| (a <= b),
        ))
        .with(PartitionnedMonotonic::bivariate(
            (data_type::Time::default(), data_type::Time::default()),
            |a, b| (a <= b),
        ))
        .with(PartitionnedMonotonic::bivariate(
            (
                data_type::DateTime::default(),
                data_type::DateTime::default(),
            ),
            |a, b| (a <= b),
        ))
        .with(PartitionnedMonotonic::bivariate(
            (data_type::Text::default(), data_type::Text::default()),
            |a, b| (a <= b),
        ))
}

pub fn eq() -> impl Function {
    Pointwise::bivariate(
        (DataType::Any, DataType::Any),
        data_type::Boolean::default(),
        |a, b| (a == b).into(),
    )
}

pub fn not_eq() -> impl Function {
    Pointwise::bivariate(
        (DataType::Any, DataType::Any),
        data_type::Boolean::default(),
        |a, b| (a != b).into(),
    )
}

// Boolean binary operators

/// The conjunction
pub fn and() -> impl Function {
    PartitionnedMonotonic::bivariate(
        (data_type::Boolean::default(), data_type::Boolean::default()),
        |x, y| x && y,
    )
}
/// The disjunction
pub fn or() -> impl Function {
    PartitionnedMonotonic::bivariate(
        (data_type::Boolean::default(), data_type::Boolean::default()),
        |x, y| x || y,
    )
}
/// The exclusive or
pub fn xor() -> impl Function {
    PartitionnedMonotonic::bivariate(
        (data_type::Boolean::default(), data_type::Boolean::default()),
        |x, y| x ^ y,
    )
}

// Bitwise binary operators

pub fn bitwise_or() -> impl Function {
    Pointwise::bivariate(
        (data_type::Boolean::default(), data_type::Boolean::default()),
        data_type::Boolean::default(),
        |a, b| (a | b).into(),
    )
}

pub fn bitwise_and() -> impl Function {
    Pointwise::bivariate(
        (data_type::Boolean::default(), data_type::Boolean::default()),
        data_type::Boolean::default(),
        |a, b| (a & b).into(),
    )
}

pub fn bitwise_xor() -> impl Function {
    Pointwise::bivariate(
        (data_type::Boolean::default(), data_type::Boolean::default()),
        data_type::Boolean::default(),
        |a, b| (a ^ b).into(),
    )
}

// Real functions

/// Builds the exponential `Function`
pub fn exp() -> impl Function {
    PartitionnedMonotonic::univariate(data_type::Float::default(), |x| {
        x.exp().clamp(0.0, <f64 as Bound>::max())
    })
}

/// Builds the logarithm `Function`
pub fn ln() -> impl Function {
    PartitionnedMonotonic::univariate(data_type::Float::from(0.0..), |x| {
        x.ln().clamp(<f64 as Bound>::min(), <f64 as Bound>::max())
    })
}

/// Builds the decimal logarithm `Function`
pub fn log() -> impl Function {
    PartitionnedMonotonic::univariate(data_type::Float::from(0.0..), |x| {
        x.log(10.)
            .clamp(<f64 as Bound>::min(), <f64 as Bound>::max())
    })
}

/// Builds the sqrt `Function`
pub fn sqrt() -> impl Function {
    PartitionnedMonotonic::univariate(data_type::Float::from(0.0..), |x| {
        x.sqrt().clamp(<f64 as Bound>::min(), <f64 as Bound>::max())
    })
}

/// The pow function
pub fn pow() -> impl Function {
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
pub fn abs() -> impl Function {
    PartitionnedMonotonic::piecewise_univariate(
        [
            data_type::Float::from(..=0.0),
            data_type::Float::from(0.0..),
        ],
        |x| x.abs(),
    )
}

/// sine
pub fn sin() -> impl Function {
    PartitionnedMonotonic::periodic_univariate(
        [
            data_type::Float::from(-0.5 * std::f64::consts::PI..=0.5 * std::f64::consts::PI),
            data_type::Float::from(0.5 * std::f64::consts::PI..=1.5 * std::f64::consts::PI),
        ],
        |x| x.sin(),
    )
}

/// cosine
pub fn cos() -> impl Function {
    PartitionnedMonotonic::periodic_univariate(
        [
            data_type::Float::from(0.0..=std::f64::consts::PI),
            data_type::Float::from(std::f64::consts::PI..=2.0 * std::f64::consts::PI),
        ],
        |x| x.cos(),
    )
}

pub fn least() -> impl Function {
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

pub fn greatest() -> impl Function {
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
pub fn lower() -> impl Function {
    PartitionnedMonotonic::univariate(data_type::Text::default(), |x| x.to_lowercase())
}

/// Builds the upper `Function`
pub fn upper() -> impl Function {
    PartitionnedMonotonic::univariate(data_type::Text::default(), |x| x.to_uppercase())
}

/// Builds the char_length `Function`
pub fn char_length() -> impl Function {
    Pointwise::univariate(
        data_type::Text::default(),
        data_type::Integer::default(),
        |a| a.len().try_into().unwrap(),
    )
}

/// Builds the position `Function`
pub fn position() -> impl Function {
    Pointwise::bivariate(
        (data_type::Text::default(), data_type::Text::default()),
        DataType::optional(DataType::integer()),
        |a, b| {
            Value::Optional(value::Optional::new(
                a.find(&b)
                    .map(|v| Arc::new(Value::integer(v.try_into().unwrap()))),
            ))
        },
    )
}

/// Regexp contains
pub fn regexp_contains() -> impl Function {
    UserDefineFunction::new(
        "regexp_contains".to_string(),
        DataType::structured_from_data_types([DataType::text(), DataType::text()]),
        DataType::boolean()
    )
}

/// Regexp extract
pub fn regexp_extract() -> impl Function {
    UserDefineFunction::new(
        "regexp_extract".to_string(),
        DataType::structured_from_data_types([DataType::text(), DataType::text(), DataType::integer(), DataType::integer()]),
        DataType::optional(DataType::text())
    )
}

/// Regexp replace
pub fn regexp_replace() -> impl Function {
    UserDefineFunction::new(
        "regexp_replace".to_string(),
        DataType::structured_from_data_types([DataType::text(), DataType::text(), DataType::text()]),
        DataType::text()
    )
}

/// Transact newid
pub fn newid() -> impl Function {
    UserDefineFunction::new(
        "newid".to_string(),
        DataType::unit(),
        DataType::text()
    )
}

/// MySQL encode
pub fn encode() -> impl Function {
    UserDefineFunction::new(
        "encode".to_string(),
        DataType::structured_from_data_types([DataType::text(), DataType::text()]),
        DataType::text()
    )
}

/// MySQL decode
pub fn decode() -> impl Function {
    UserDefineFunction::new(
        "decode".to_string(),
        DataType::structured_from_data_types([DataType::text(), DataType::text()]),
        DataType::text()
    )
}

/// MySQL unhex
pub fn unhex() -> impl Function {
    UserDefineFunction::new(
        "unhex".to_string(),
        DataType::text(),
        DataType::text()
    )
}

// Case function
pub fn case() -> impl Function {
    Case
}

// In operator
pub fn in_list() -> impl Function {
    Polymorphic::from((
        InList(data_type::Integer::default().into()),
        InList(data_type::Float::default().into()),
        InList(data_type::Text::default().into()),
    ))
}

// Coalesce function
pub fn coalesce() -> impl Function {
    Coalesce
}

// Ceil function
pub fn ceil() -> impl Function {
    PartitionnedMonotonic::univariate(
        data_type::Float::default(),
        |a| a.ceil(),
    )
}

// Floor function
pub fn floor() -> impl Function {
    PartitionnedMonotonic::univariate(
        data_type::Float::default(),
        |a| a.floor(),
    )
}

// Round function
// monotonic for the 1st variable but not for the second => Pointwise
pub fn round() -> impl Function {
    Pointwise::bivariate(
        (data_type::Float::default(), data_type::Integer::default()),
        data_type::Float::default(),
        |a, b| {
            let multiplier = 10.0_f64.powi(b as i32);
            (a * multiplier).round() / multiplier
        }
    )
}

// Trunc function
// monotonic for the 1st variable but not for the second (eg: when the 2nd arg is negative )=> Pointwise
pub fn trunc() -> impl Function {
    Pointwise::bivariate(
        (data_type::Float::default(), data_type::Integer::default()),
        data_type::Float::default(),
        |a, b| {
            let multiplier = 10.0_f64.powi(b as i32);
            (a * multiplier).trunc() / multiplier
        }
    )
}

// Sign function
pub fn sign() -> impl Function {
    PartitionnedMonotonic::univariate(
        data_type::Float::default(),
        |a| if a == 0. {0} else if a < 0. {-1} else {1}
    )
}

/*
Aggregation functions
 */

/// Median aggregation
pub fn median() -> impl Function {
    null()
}

pub fn n_unique() -> impl Function {
    null()
}

/// First element in group
pub fn first() -> impl Function {
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
pub fn last() -> impl Function {
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
pub fn mean() -> impl Function {
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
pub fn list() -> impl Function {
    null()
}

/// Count aggregation
pub fn count() -> impl Function {
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
pub fn min() -> impl Function {
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
pub fn max() -> impl Function {
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
pub fn quantile(_p: f64) -> impl Function {
    null()
}

/// Multi-quantileq aggregation
pub fn quantiles(_p: Vec<f64>) -> impl Function {
    null()
}

/// Sum aggregation
pub fn sum() -> impl Function {
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
pub fn agg_groups() -> impl Function {
    null()
}

/// Standard deviation aggregation
pub fn std() -> impl Function {
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
pub fn var() -> impl Function {
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
    use chrono;

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

        // false or true
        let set = DataType::float_values([1., 2.]) & DataType::float_values([1., 2.]);
        let im = fun.super_image(&set).unwrap();
        println!("\nim({}) = {}", set, im);
        assert_eq!(im, DataType::boolean_values([false, true]));
        let arg = Value::float(1.) & Value::float(1.);
        let val = fun.value(&arg).unwrap();
        println!("val({}) = {}", arg, val);
        assert_eq!(val, Value::from(true));
        let arg = Value::float(1.) & Value::float(2.);
        let val = fun.value(&arg).unwrap();
        println!("val({}) = {}", arg, val);
        assert_eq!(val, Value::from(false));

        // false
        let set = DataType::float_values([1., 2.]) & DataType::float_values([4., 5.]);
        let im = fun.super_image(&set).unwrap();
        println!("\nim({}) = {}", set, im);
        assert_eq!(im, DataType::boolean_value(false));
        let arg = Value::float(1.) & Value::float(5.);
        let val = fun.value(&arg).unwrap();
        assert_eq!(val, Value::from(false));

        // true
        let set = DataType::float_value(1.) & DataType::float_value(1.);
        let im = fun.super_image(&set).unwrap();
        println!("\nim({}) = {}", set, im);
        assert_eq!(im, DataType::boolean_value(true));
        let arg = Value::float(1.) & Value::float(1.);
        let val = fun.value(&arg).unwrap();
        assert_eq!(val, Value::from(true));
    }

    #[test]
    fn test_gt() {
        println!("Test gt");
        let fun = gt();
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());

        // false or true
        let set = DataType::float_interval(1., 5.) & DataType::float_interval(3., 4.);
        let im = fun.super_image(&set).unwrap();
        println!("\nim({}) = {}", set, im);
        assert_eq!(im, DataType::boolean());
        let arg = Value::integer(4) & Value::integer(3);
        let val = fun.value(&arg).unwrap();
        println!("val({}) = {}", arg, val);
        assert_eq!(val, Value::from(true));
        let arg = Value::float(1.1) & Value::float(3.1);
        let val = fun.value(&arg).unwrap();
        println!("val({}) = {}", arg, val);
        assert_eq!(val, Value::from(false));

        // false
        let set = DataType::float_values([1.1, 2.2]) & DataType::float_values([3.01, 4.1]);
        let im = fun.super_image(&set).unwrap();
        println!("\nim({}) = {}", set, im);
        assert_eq!(im, DataType::boolean_value(false));
        let set = DataType::float_interval(1., 2.) & DataType::float_interval(3.01, 4.1);
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert_eq!(im, DataType::boolean_value(false));

        // true
        let set = DataType::float_values([4.1, 5.03]) & DataType::float_values([3., 4.]);
        let im = fun.super_image(&set).unwrap();
        println!("\nim({}) = {}", set, im);
        assert_eq!(im, DataType::boolean_value(true));
        let set = DataType::float_interval(4.1, 5.03) & DataType::float_interval(3., 4.);
        let im = fun.super_image(&set).unwrap();
        println!("\nim({}) = {}", set, im);
        assert_eq!(im, DataType::boolean_value(true));

        // true
        let set = DataType::integer_values([5, 7]) & DataType::float_values([3., 4.]);
        let im = fun.super_image(&set).unwrap();
        println!("\nim({}) = {}", set, im);
        assert_eq!(im, DataType::boolean_value(true));

        // false
        let set = DataType::float_values([1., 2.3]) & DataType::integer_values([10]);
        let im = fun.super_image(&set).unwrap();
        println!("\nim({}) = {}", set, im);
        assert_eq!(im, DataType::boolean_value(false));
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

        // float + integer
        let set = DataType::from(Struct::from_data_types(&[
            DataType::from(data_type::Integer::from_intervals([
                [0, 2],
                [5, 5],
                [10, 10],
            ])),
            DataType::float_interval(2.9, 3.),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(matches!(im, DataType::Float(_)));

        // im(struct{0: float[1, 100], 1: integer{-30, 0, 20}}) = float[-29, 120]
        let set: DataType = DataType::structured_from_data_types([
            DataType::float_interval(1.0, 100.0),
            DataType::integer_values([20, 0, -30]),
        ]);
        let im = fun.super_image(&set).unwrap();
        println!("\nim({}) = {}", set, im);
        assert_eq!(im, DataType::float_interval(-29.0, 120.0));

        // im(struct{0: float[1, 100], 1: float{-30, 0, 20}}) = float[-29, 120]
        let set: DataType = DataType::structured_from_data_types([
            DataType::float_interval(1.0, 100.0),
            DataType::float_values([20.0, 0.0, -30.0]),
        ]);
        let im = fun.super_image(&set).unwrap();
        println!("\nim({}) = {}", set, im);
        assert_eq!(im, DataType::float_interval(-29.0, 120.0));

        // im(struct{0: float[1, 10], 1: float{-30, 0, 20}}) = float[-29.0, -20.0] U float[1.0, 10.0] U float[21.0, 30.0]
        let set: DataType = DataType::structured_from_data_types([
            DataType::float_interval(1.0, 10.0),
            DataType::float_values([20.0, 0.0, -30.0]),
        ]);
        let im = fun.super_image(&set).unwrap();
        println!("\nim({}) = {}", set, im);
        assert_eq!(
            im,
            DataType::from(data_type::Float::from_intervals([
                [-29.0, -20.0],
                [1.0, 10.0],
                [21.0, 30.0],
            ]))
        );
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
    fn test_optional() {
        println!("Test optional");
        let optional_greatest = Optional::new(greatest());
        println!("greatest = {}", greatest());
        println!("optional greatest = {}", optional_greatest);
        println!(
            "super_image([0,1] & [-5,2]) = {}",
            optional_greatest
                .super_image(
                    &(DataType::float_interval(0., 1.) & DataType::float_interval(-5., 2.))
                )
                .unwrap()
        );
        println!(
            "super_image(optional([0,1] & [-5,2])) = {}",
            optional_greatest
                .super_image(&DataType::optional(
                    (DataType::float_interval(0., 1.) & DataType::float_interval(-5., 2.))
                ))
                .unwrap()
        );
        println!(
            "super_image(optional([0,1]) & [-5,2]) = {}",
            optional_greatest
                .super_image(
                    &(DataType::optional(DataType::float_interval(0., 1.))
                        & DataType::float_interval(-5., 2.))
                )
                .unwrap()
        );
        assert_eq!(
            optional_greatest
                .super_image(&DataType::optional(
                    (DataType::float_interval(0., 1.) & DataType::float_interval(-5., 2.))
                ))
                .unwrap(),
            optional_greatest
                .super_image(
                    &(DataType::optional(DataType::float_interval(0., 1.))
                        & DataType::float_interval(-5., 2.))
                )
                .unwrap(),
        );
        println!(
            "super_image(text) = {}",
            optional_greatest.super_image(&DataType::text()).unwrap()
        );
    }

    #[test]
    fn test_extended() {
        println!("Test extended");
        let extended_cos = Extended::new(cos(), DataType::Any);
        println!("cos = {}", cos());
        println!("extended cos = {}", extended_cos);
        assert_eq!(
            extended_cos.co_domain(),
            DataType::optional(DataType::float_range(-1.0..=1.0))
        );
    }

    #[test]
    fn test_optional_aggregate_sum() {
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
        let opt_sum = Optional::new(sum);
        println!("opt_sum = {}", opt_sum);
        let list = DataType::optional(DataType::list(DataType::float_interval(-1., 2.), 2, 20));
        println!(
            "\n{} is_subset_of {} = {}",
            list,
            opt_sum.domain(),
            list.is_subset_of(&opt_sum.domain())
        );
        println!(
            "\nopt_sum({}) = {}",
            list,
            opt_sum.super_image(&list).unwrap()
        );
        let list = DataType::list(DataType::optional(DataType::float_interval(-1., 2.)), 2, 20);
        println!(
            "\n{} is_subset_of {} = {}",
            list,
            opt_sum.domain(),
            list.is_subset_of(&opt_sum.domain())
        );
        println!(
            "\nopt_sum({}) = {}",
            list,
            opt_sum.super_image(&list).unwrap()
        );
        let list = DataType::list(DataType::float_interval(-1., 2.), 2, 20);
        println!(
            "\n{} is_subset_of {} = {}",
            list,
            opt_sum.domain(),
            list.is_subset_of(&opt_sum.domain())
        );
        println!(
            "\nopt_sum({}) = {}",
            list,
            opt_sum.super_image(&list).unwrap()
        );
    }

    #[test]
    fn test_extended_binary() {
        println!("Test extended");
        // Test a bivariate monotonic function
        let extended_add = Extended::new(plus(), DataType::Any & DataType::Any);
        println!("add = {}", plus());
        println!("extended add = {}", extended_add);
    }

    #[test]
    fn test_extended_plus() {
        println!("Test extended");
        // Test a bivariate monotonic function
        let extended_plus = Extended::new(plus(), DataType::Any & DataType::Any);
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
        let extended_sum = Extended::new(sum(), DataType::Any);
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
    fn test_in_list() {
        println!("Test in_list");
        let fun = in_list();
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());

        // 10 in (10)
        let set = DataType::from(Struct::from_data_types(&[
            DataType::integer_value(10),
            DataType::list(DataType::integer_values(vec![8, 10, 15, 30]), 3, 3),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("\nim({}) = {}", set, im);
        assert!(im == DataType::boolean());
        let arg = Value::structured_from_values([Value::from(10), Value::list([Value::from(10)])]);
        let val = fun.value(&arg).unwrap();
        println!("value({}) = {}", arg, val);
        assert_eq!(val, Value::from(true));

        // integer in (integer[8, 30])
        let set = DataType::from(Struct::from_data_types(&[
            DataType::integer(),
            DataType::list(DataType::integer_interval(8, 30), 3, 3),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("\nim({}) = {}", set, im);
        assert!(im == DataType::boolean());
        let arg = Value::structured_from_values([Value::from(10), Value::list([Value::from(10)])]);
        let val = fun.value(&arg).unwrap();
        println!("value({}) = {}", arg, val);
        assert_eq!(val, Value::from(true));
        let arg = Value::structured_from_values([Value::from(100), Value::list([Value::from(10)])]);
        let val = fun.value(&arg).unwrap();
        println!("value({}) = {}", arg, val);
        assert_eq!(val, Value::from(false));

        // integer[1, 5] in (integer[8, 30])
        let set = DataType::from(Struct::from_data_types(&[
            DataType::integer_interval(1, 5),
            DataType::list(DataType::integer_interval(8, 30), 3, 3),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("\nim({}) = {}", set, im);
        assert!(im == DataType::boolean_value(false));
        let arg = Value::structured_from_values([Value::from(1), Value::list([Value::from(10)])]);
        let val = fun.value(&arg).unwrap();
        println!("value({}) = {}", arg, val);
        assert_eq!(val, Value::from(false));

        // integer[1, 5] in (float[2., 30.])
        let set = DataType::from(Struct::from_data_types(&[
            DataType::integer_interval(1, 5),
            DataType::list(DataType::float_interval(2., 30.), 3, 3),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("\nim({}) = {}", set, im);
        assert_eq!(im, DataType::boolean());
        let arg = Value::structured_from_values([
            Value::from(3),
            Value::list([Value::from(2.), Value::from(3.)]),
        ]);
        let val = fun.value(&arg).unwrap();
        println!("value({}) = {}", arg, val);
        assert_eq!(val, Value::from(true));
        let arg = Value::structured_from_values([Value::from(1), Value::list([Value::from(3.)])]);
        let val = fun.value(&arg).unwrap();
        println!("value({}) = {}", arg, val);
        assert_eq!(val, Value::from(false));

        // float[1., 5.] in (integer[2, 30])
        let set = DataType::from(Struct::from_data_types(&[
            DataType::float_interval(1., 5.),
            DataType::list(DataType::integer_interval(2, 30), 3, 3),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("\nim({}) = {}", set, im);
        assert_eq!(im, DataType::boolean());
        let arg = Value::structured_from_values([
            Value::from(3.),
            Value::list([Value::from(2), Value::from(3)]),
        ]);
        let val = fun.value(&arg).unwrap();
        println!("value({}) = {}", arg, val);
        assert_eq!(val, Value::from(true));
        let arg = Value::structured_from_values([Value::from(1.), Value::list([Value::from(15)])]);
        let val = fun.value(&arg).unwrap();
        println!("value({}) = {}", arg, val);
        assert_eq!(val, Value::from(false));

        // text['1', '5'] in (integer[2, '30])
        let set = DataType::from(Struct::from_data_types(&[
            DataType::text_values(["3".to_string(), "a".to_string()]),
            DataType::list(DataType::integer_interval(2, 30), 3, 3),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("\nim({}) = {}", set, im);
        assert_eq!(im, DataType::boolean());
        let arg = Value::structured_from_values([
            Value::from("3".to_string()),
            Value::list([Value::from(2), Value::from(3)]),
        ]);
        let val = fun.value(&arg).unwrap();
        println!("value({}) = {}", arg, val);
        assert_eq!(val, Value::from(true));
        let arg = Value::structured_from_values([
            Value::from("a".to_string()),
            Value::list([Value::from(15)]),
        ]);
        let val = fun.value(&arg).unwrap();
        println!("value({}) = {}", arg, val);
        assert_eq!(val, Value::from(false));
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
        println!("Test upper");
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

    #[test]
    fn test_least() {
        println!("Test least");
        let fun = least();
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());

        // im(struct{0: float[1, 100], 1: float{-30, 0, 20}}) = float{-30, 0} U float[1, 20]
        let set: DataType = DataType::structured_from_data_types([
            DataType::float_values([100.0, 1.0]),
            DataType::float_values([20.0, 0.0, -30.0]),
        ]);
        let im = fun.super_image(&set).unwrap();
        println!("\nim({}) = {}", set, im);
        assert_eq!(im, DataType::float_values([-30.0, 0.0, 1.0, 20.]));

        // im(struct{0: float[1, 100], 1: float{-30, 0, 20}}) = float{-30, 0} U float[1, 20]
        let set: DataType = DataType::structured_from_data_types([
            DataType::float_interval(1.0, 100.),
            DataType::float_values([20.0, 0.0, -30.0]),
        ]);
        let im = fun.super_image(&set).unwrap();
        println!("\nim({}) = {}", set, im);
        assert_eq!(
            im,
            DataType::float_values([-30.0, 0.0])
                .super_union(&DataType::float_interval(1., 20.))
                .unwrap()
        );

        // im(struct{0: float[1, +∞), 1: float(-∞, 100]}) = float(-∞, 100]
        let set: DataType = DataType::structured_from_data_types([
            DataType::float_min(1.0),
            DataType::float_max(100.0),
        ]);
        let im = fun.super_image(&set).unwrap();
        println!("\nim({}) = {}", set, im);
        assert_eq!(im, DataType::float_max(100.0));

        // im(struct{0: float{1}, 1: float{100}}) = int{1}
        let set: DataType = DataType::structured_from_data_types([
            DataType::float_value(1.0),
            DataType::float_value(100.0),
        ]);
        let im = fun.super_image(&set).unwrap();
        println!("\nim({}) = {}", set, im);
        assert_eq!(im, DataType::float_value(1.0));

        // im(struct{0: float(-∞, 10], 1: float[100, +∞)}) = float(-∞, 10]
        let set: DataType = DataType::structured_from_data_types([
            DataType::float_max(10.0),
            DataType::float_min(100.0),
        ]);
        let im = fun.super_image(&set).unwrap();
        println!("\nim({}) = {}", set, im);
        assert_eq!(im, DataType::float_max(10.0));

        // im(struct{0: float[1 10], 1: int[100, 200]}) = float[1 10]
        let set: DataType = DataType::structured_from_data_types([
            DataType::float_interval(1., 10.),
            DataType::integer_interval(100, 200),
        ]);
        let im = fun.super_image(&set).unwrap();
        println!("\nim({}) = {}", set, im);
        assert_eq!(im, DataType::float_interval(1., 10.));

        // im(struct{0: int[1 10], 1: float[100, +∞)}) = int[1, 10]
        let set: DataType = DataType::structured_from_data_types([
            DataType::integer_interval(1, 10),
            DataType::float_min(100.0),
        ]);
        let im = fun.super_image(&set).unwrap();
        println!("\nim({}) = {}", set, im);
        assert_eq!(im, DataType::integer_interval(1, 10));

        // im(struct{0: float(-∞, 10], 1: int[2 100]}) = float(-∞, 10]
        let set: DataType = DataType::structured_from_data_types([
            DataType::float_max(10.0),
            DataType::integer_interval(2, 100),
        ]);
        let im = fun.super_image(&set).unwrap();
        println!("\nim({}) = {}", set, im);
        assert_eq!(im, DataType::float_max(10.0));
    }

    #[test]
    fn test_greatest() {
        println!("Test greatest");
        let fun = greatest();
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());

        // im(struct{0: float{1, 100}, 1: float{-30, 0, 20}}) = float{1, 20, 100}
        let set: DataType = DataType::structured_from_data_types([
            DataType::float_values([100.0, 1.0]),
            DataType::float_values([20.0, 0.0, -30.0]),
        ]);
        let im = fun.super_image(&set).unwrap();
        println!("\nim({}) = {}", set, im);
        assert_eq!(im, DataType::float_values([1., 20., 100.]));

        // im(struct{0: float[1, 100], 1: float{-30, 0, 20}}) = float[1, 100]
        let set: DataType = DataType::structured_from_data_types([
            DataType::float_interval(1.0, 100.0),
            DataType::float_values([20.0, 0.0, -30.0]),
        ]);
        let im = fun.super_image(&set).unwrap();
        println!("\nim({}) = {}", set, im);
        assert_eq!(im, DataType::float_interval(1.0, 100.));

        // im(struct{0: float[1, +∞), 1: float(-∞, 100]}) = float[1, ∞)
        let set: DataType = DataType::structured_from_data_types([
            DataType::float_min(1.0),
            DataType::float_max(100.0),
        ]);
        let im = fun.super_image(&set).unwrap();
        println!("\nim({}) = {}", set, im);
        assert_eq!(im, DataType::float_min(1.0));

        // im(struct{0: float{1}, 1: float{100}}) = int{100}
        let set: DataType = DataType::structured_from_data_types([
            DataType::float_value(1.0),
            DataType::float_value(100.0),
        ]);
        let im = fun.super_image(&set).unwrap();
        println!("\nim({}) = {}", set, im);
        assert_eq!(im, DataType::float_value(100.0));

        // im(struct{0: float(-∞, 10], 1: float[100, +∞)}) = float[100, +∞)
        let set: DataType = DataType::structured_from_data_types([
            DataType::float_max(10.0),
            DataType::float_min(100.0),
        ]);
        let im = fun.super_image(&set).unwrap();
        println!("\nim({}) = {}", set, im);
        assert_eq!(im, DataType::float_min(100.0));

        // im(struct{0: float[1 10], 1: int[100, 200]}) = float[1 10]
        let set: DataType = DataType::structured_from_data_types([
            DataType::float_interval(1., 10.),
            DataType::integer_interval(100, 200),
        ]);
        let im = fun.super_image(&set).unwrap();
        println!("\nim({}) = {}", set, im);
        assert_eq!(im, DataType::integer_interval(100, 200));

        // im(struct{0: int[1 10], 1: float[100, +∞)}) = int[1, 10]
        let set: DataType = DataType::structured_from_data_types([
            DataType::integer_interval(1, 10),
            DataType::float_min(100.0),
        ]);
        let im = fun.super_image(&set).unwrap();
        println!("\nim({}) = {}", set, im);
        assert_eq!(im, DataType::float_min(100.0));

        // im(struct{0: float(-∞, 10], 1: int[2 100]}) = float(-∞, 10]
        let set: DataType = DataType::structured_from_data_types([
            DataType::float_max(10.0),
            DataType::integer_interval(2, 100),
        ]);
        let im = fun.super_image(&set).unwrap();
        println!("\nim({}) = {}", set, im);
        assert_eq!(
            im,
            DataType::float_interval(2., 10.)
                .super_union(&DataType::integer_interval(10, 100))
                .unwrap()
        );
    }

    #[test]
    fn test_coalesce() {
        println!("Test coalesce");
        let fun = coalesce();
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());

        let set = DataType::from(Struct::from_data_types(&[
            DataType::integer(),
            DataType::text(),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(im == DataType::integer());

        let set = DataType::from(Struct::from_data_types(&[
            DataType::optional(DataType::integer()),
            DataType::text(),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(im == DataType::text());

        let set = DataType::from(Struct::from_data_types(&[
            DataType::optional(DataType::integer_interval(1, 5)),
            DataType::integer_value(20),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert_eq!(
            im,
            DataType::integer_interval(1, 5)
                .super_union(&DataType::integer_value(20))
                .unwrap()
        );

        let set = DataType::from(Struct::from_data_types(&[
            DataType::optional(DataType::integer()),
            DataType::optional(DataType::text()),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(im == DataType::optional(DataType::text()));
    }

    #[test]
    fn test_rtrim() {
        println!("Test rtrim");
        let fun = rtrim();
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());

        let set = DataType::from(Struct::from_data_types(&[
            DataType::text(),
            DataType::text(),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(im == DataType::text());

        let set = DataType::from(Struct::from_data_types(&[
            DataType::text_values([
                "aba".to_string(),
                "aa".to_string(),
                "baaa".to_string(),
                "ba".to_string(),
                "mc".to_string(),
            ]),
            DataType::text_values(["a".to_string(), "c".to_string()]),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(
            im == DataType::text_values([
                "".to_string(),
                "aa".to_string(),
                "ab".to_string(),
                "aba".to_string(),
                "b".to_string(),
                "ba".to_string(),
                "baaa".to_string(),
                "m".to_string(),
                "mc".to_string()
            ])
        );

        let arg = Value::text("sarusss".to_string()) & Value::text("s".to_string());
        let val = fun.value(&arg).unwrap();
        println!("val({}) = {}", arg, val);
        assert_eq!(val, Value::from("saru".to_string()));
    }

    #[test]
    fn test_ltrim() {
        println!("Test ltrim");
        let fun = ltrim();
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());

        let set = DataType::from(Struct::from_data_types(&[
            DataType::text(),
            DataType::text(),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(im == DataType::text());

        let set = DataType::from(Struct::from_data_types(&[
            DataType::text_values([
                "aba".to_string(),
                "aa".to_string(),
                "baaa".to_string(),
                "ba".to_string(),
                "mc".to_string(),
            ]),
            DataType::text_values(["a".to_string(), "c".to_string()]),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(
            im == DataType::text_values([
                "".to_string(),
                "aa".to_string(),
                "aba".to_string(),
                "ba".to_string(),
                "baaa".to_string(),
                "mc".to_string()
            ])
        );

        let arg = Value::text("sarus".to_string()) & Value::text("s".to_string());
        let val = fun.value(&arg).unwrap();
        println!("val({}) = {}", arg, val);
        assert_eq!(val, Value::from("arus".to_string()));
    }

    #[test]
    fn test_substr() {
        println!("Test substr");
        let fun = substr();
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());

        let set = DataType::from(Struct::from_data_types(&[
            DataType::text(),
            DataType::integer(),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(im == DataType::text());

        let set = DataType::from(Struct::from_data_types(&[
            DataType::text_values(["abcdefg".to_string(), "hijklmno".to_string()]),
            DataType::integer_values([3, 6, 10]),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert_eq!(
            im,
            DataType::text_values([
                "".to_string(),
                "defg".to_string(),
                "g".to_string(),
                "klmno".to_string(),
                "no".to_string()
            ])
        );
    }

    #[test]
    fn test_substr_with_size() {
        println!("Test substr_with_size");
        let fun = substr_with_size();
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());

        let set = DataType::from(Struct::from_data_types(&[
            DataType::text(),
            DataType::integer(),
            DataType::integer(),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(im == DataType::text());

        let set = DataType::from(Struct::from_data_types(&[
            DataType::text_values(["abcdefg".to_string(), "hijklmno".to_string()]),
            DataType::integer_values([3, 6, 10]),
            DataType::integer_value(2),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert_eq!(
            im,
            DataType::text_values([
                "".to_string(),
                "de".to_string(),
                "g".to_string(),
                "kl".to_string(),
                "no".to_string()
            ])
        );
    }

    #[test]
    fn test_ceil() {
        println!("Test ceil");
        let fun = ceil();
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());

        let set = DataType::float_values([9., 9.1, 9.5, 10.5]);
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(im == DataType::integer_values([9, 10, 11]));

        let set = DataType::integer_values([9, 10]);
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(im == DataType::integer_values([9, 10]));
    }

    #[test]
    fn test_floor() {
        println!("Test floor");
        let fun = floor();
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());

        let set = DataType::float_values([9., 9.1, 9.5, 10.5]);
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(im == DataType::integer_values([9, 10]));

        let set = DataType::integer_values([9, 10]);
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(im == DataType::integer_values([9, 10]));
    }

    #[test]
    fn test_round() {
        println!("Test round");
        let fun = round();
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());

        let set = DataType::from(Struct::from_data_types(&[
            DataType::float_values([8.1, 9.16, 10.226, 11.333]),
            DataType::integer_values([0, 2]),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(im == DataType::float_values([8., 9., 10., 11., 8.1, 9.16, 10.23, 11.33]));

        let set = DataType::from(Struct::from_data_types(&[
            DataType::integer_values([9, 10]),
            DataType::integer_values([0, 2]),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(im == DataType::integer_values([9, 10]));
    }

    #[test]
    fn test_trunc() {
        println!("Test trunc");
        let fun = trunc();
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());

        let set = DataType::from(Struct::from_data_types(&[
            DataType::float_values([8.1, 9.16, 10.226, 11.333]),
            DataType::integer_values([0, 2]),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(im == DataType::float_values([8., 9., 10., 11., 8.1, 9.16, 10.22, 11.33]));

        let set = DataType::from(Struct::from_data_types(&[
            DataType::integer_values([9, 10]),
            DataType::integer_values([0, 2]),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(im == DataType::integer_values([9, 10]));
    }

    #[test]
    fn test_cast_as_text() {
        println!("Test cast as text");
        let fun = cast(DataType::text());
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());
        println!("data_type = {}", fun.data_type());

        let set = DataType::integer_values([1, 3, 4]);
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(im == DataType::text_values(["1".to_string(), "3".to_string(), "4".to_string()]));

        let set = DataType::integer_values([1, 3, 4]);
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(im == DataType::text_values(["1".to_string(), "3".to_string(), "4".to_string()]));

        let set = DataType::date_value(chrono::NaiveDate::from_ymd_opt(2015, 6, 3).unwrap());
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(im == DataType::text_values(["2015-06-03".to_string()]));
    }

    #[test]
    fn test_cast_as_float() {
        println!("Test cast as float");
        let fun = cast(DataType::float());
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());
        println!("data_type = {}", fun.data_type());

        let set = DataType::text_values(["1.5".to_string(), "3".to_string(), "4.555".to_string()]);
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(im == DataType::float_values([1.5, 3., 4.555]));
    }

    #[test]
    fn test_cast_as_integer() {
        println!("\nTest cast as integer");
        let fun = cast(DataType::integer());
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());
        println!("data_type = {}", fun.data_type());

        let set = DataType::text_values(["1".to_string(), "3".to_string(), "4".to_string()]);
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(im == DataType::integer_values([1, 3, 4]));
    }

    #[test]
    fn test_cast_to_boolean() {
        println!("\nTest cast as boolean");
        let fun = cast(DataType::boolean());
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());
        println!("data_type = {}", fun.data_type());

        let set = DataType::text_values(["1".to_string(), "tru".to_string()]);
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(im == DataType::boolean_value(true));
    }

    #[test]
    fn test_sign() {
        println!("\nTest sign");
        let fun = sign();
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());
        println!("data_type = {}", fun.data_type());

        let set = DataType::float_interval(-5., 5.);
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(im == DataType::integer_interval(-1, 1));

        let set = DataType::float_value(0.);
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(im == DataType::integer_value(0));
    }

    #[test]
    fn test_regexp_contains() {
        println!("\nTest regexp_contains");
        let fun = regexp_contains();
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());
        println!("data_type = {}", fun.data_type());

        let set = DataType::from(Struct::from_data_types(&[
            DataType::text_values(["foo@example.com".to_string(), "bar@example.org".to_string(), "www.example.net".to_string()]),
            DataType::text_value(r"@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+".to_string())
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(im == DataType::boolean());
    }

    #[test]
    fn test_regexp_extract() {
        println!("\nTest regexp_extract");
        let fun = regexp_extract();
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());
        println!("data_type = {}", fun.data_type());

        let set = DataType::from(Struct::from_data_types(&[
            DataType::text_value("Hello Helloo and Hellooo".to_string()),
            DataType::text_value("H?ello+".to_string()),
            DataType::integer_value(3),
            DataType::integer_value(1)
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(im == DataType::optional(DataType::text()));
    }

    #[test]
    fn test_regexp_replace() {
        println!("\nTest regexp_replace");
        let fun = regexp_replace();
        println!("type = {}", fun);
        println!("domain = {}", fun.domain());
        println!("co_domain = {}", fun.co_domain());
        println!("data_type = {}", fun.data_type());

        let set = DataType::from(Struct::from_data_types(&[
            DataType::text_values(["# Heading".to_string(), "# Another heading".to_string()]),
            DataType::text_value(r"^# ([a-zA-Z0-9\s]+$)".to_string()),
            DataType::text_value(r"<h1>\\1</h1>".to_string()),
        ]));
        let im = fun.super_image(&set).unwrap();
        println!("im({}) = {}", set, im);
        assert!(im == DataType::text());
    }
}
