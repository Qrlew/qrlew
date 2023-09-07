//! # The data types used in Qrlew
//!
//! A DataType may be represented in many Variants
//!
//! ## Variants
//!
//! Each variant guarantees its own consistency as a variant.
//! The following operations are alowed inside the Variant
//!
//! ### Predicates
//!
//! * contains
//! * is_subset_of
//! * is_superset_of
//!
//! They return True if they can prove the assertion and False otherwise
//!
//! ### Approximate boolean operations
//!
//! * super_union
//! * super_intersection
//!
//! They return a superset of the actual boolean op. An upper bound approximation from a set perspective
//!
//! ### Variant characterization
//!
//! * minimal_subset
//! * maximal_superset
//!
//! They return extremal sets in the given Variant
//!
//! ### Cross Variant operations
//!
//! * into_data_type
//! * into_variant
//!
//! The first one maps the source dataset into the target dataset if there is an injection from one to the other
//! The second maps the source dataset into the largest set in the Variant of the target sety and maps it into it.

pub mod function;
pub mod generator;
#[allow(clippy::type_complexity)]
pub mod injection;
pub mod intervals;
pub mod product;
pub mod sql;
pub mod value;

use chrono;
use itertools::Itertools;
use paste::paste;
use std::{
    cmp,
    collections::{BTreeSet, HashSet},
    convert::Infallible,
    error, fmt, hash,
    marker::Copy,
    ops::{self, Deref, Index},
    rc::Rc,
    result,
};

use crate::{
    hierarchy::{Hierarchy, Path},
    namer,
    types::{And, Or},
    visitor::{self, Acceptor},
};
use injection::{Base, InjectInto, Injection};
use intervals::{Bound, Intervals};

pub use generator::Generator;
pub use value::Value;

// Error handling

/// The errors data_types can lead to
#[derive(Debug)]
pub enum Error {
    NoSuperset(String),
    InvalidConversion(String),
    InvalidField(String),
    Other(String),
}

impl Error {
    pub fn no_superset(left: impl fmt::Display, right: impl fmt::Display) -> Error {
        Error::NoSuperset(format!(
            "No superset including {} and {} found",
            left, right
        ))
    }
    pub fn invalid_conversion(this: impl fmt::Display, that: impl fmt::Display) -> Error {
        Error::InvalidConversion(format!("Cannot convert {} into {}", this, that))
    }
    pub fn invalid_field(field: impl fmt::Display) -> Error {
        Error::InvalidField(format!("{} is missing", field))
    }
    pub fn other(desc: impl fmt::Display) -> Error {
        Error::Other(format!("{}", desc))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NoSuperset(desc) => writeln!(f, "NoSuperset: {}", desc),
            Error::InvalidConversion(desc) => writeln!(f, "InvalidConversion: {}", desc),
            Error::InvalidField(desc) => writeln!(f, "InvalidField: {}", desc),
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
impl From<function::Error> for Error {
    fn from(err: function::Error) -> Self {
        Error::Other(err.to_string())
    }
}
impl From<injection::Error> for Error {
    fn from(err: injection::Error) -> Self {
        Error::Other(err.to_string())
    }
}
impl From<value::Error> for Error {
    fn from(err: value::Error) -> Self {
        Error::Other(err.to_string())
    }
}

pub type Result<T> = result::Result<T, Error>;

// Some macros

#[allow(unused_macros)]
/// A quick macro to test all possible conversions
macro_rules! for_all_pairs {
    (@inner $fun:expr, $left:expr, ($($right:expr),*)) => {
        $($fun($left, $right);)*
    };
    (@expand $fun:expr, ($($left:expr),*), $right:tt) => {
        $(
            for_all_pairs!(@inner $fun, $left, $right);
        )*
    };
    ($fun:expr, $($value:expr),*) => {
        for_all_pairs!(@expand $fun, ($($value),*), ($($value),*));
    };
}

/// Invoke the same method, no matter the variant
macro_rules! for_all_variants {
    ($data_type:expr, $variant:ident, $fun:expr, [$($Variant:ident),*], $default:expr) => {
        match $data_type {
            $(DataType::$Variant($variant) => $fun,)*
            _ => $default,
        }
    };
    ($data_type:expr, $variant:ident, $fun:expr, [$($Variant:ident),*]) => {
        match $data_type {
            $(DataType::$Variant($variant) => $fun,)*
        }
    };
}

/// Invoke the same method, no matter the variant
macro_rules! for_all_variant_pairs {
    ($left:expr, $right:expr, $left_variant:ident, $right_variant:ident, $fun:expr, [$($Variant:ident),*]) => {
        match ($left, $right) {
            $((DataType::$Variant($left_variant), DataType::$Variant($right_variant)) => $fun,)*
        }
    };
    ($left:expr, $right:expr, $left_variant:ident, $right_variant:ident, $fun:expr, [$($Variant:ident),*], $default:expr) => {
        match ($left, $right) {
            $((DataType::$Variant($left_variant), DataType::$Variant($right_variant)) => $fun,)*
            _ => $default,
        }
    };
}

/// An object with an associated type
pub trait DataTyped {
    /// Return the DataType atached to the object
    fn data_type(&self) -> DataType;

    /// Return whether the object has exactly the given type
    fn has_data_type(&self, data_type: &DataType) -> bool {
        &self.data_type() == data_type
    }
    /// Return whether the object has a type contained in the given type
    fn is_contained_by(&self, data_type: &DataType) -> bool {
        self.data_type().is_subset_of(data_type)
    }
}

impl<D: DataTyped> From<D> for DataType {
    fn from(d: D) -> Self {
        d.data_type()
    }
}

/// Types from a Variant can be converted to
/// types of other Variants.
///
/// The conversion of a type: _a_ of Variant: _A_ into _b_ of Variant _B_
/// returns _a'_ of variant _B_. It means there is an injection
/// from _a_ to _a'_.
///
/// This is where cross variant conversions are defined
/// It is specific to each variant.
pub trait Variant:
    Into<DataType>
    + TryFrom<DataType>
    + From<Self::Element>
    + Clone
    + hash::Hash
    + cmp::PartialEq
    + cmp::PartialOrd
    + fmt::Debug
    + fmt::Display
{
    type Element: value::Variant;

    /// Tests if an element is in `self`
    fn contains(&self, element: &Self::Element) -> bool;

    /// Test if elements of `self` are in `other`
    fn is_subset_of(&self, other: &Self) -> bool;

    /// Test if elements of `other` are in `self`
    fn is_superset_of(&self, other: &Self) -> bool {
        other.is_subset_of(self)
    }

    /// A simplified union.
    /// The result is a datatype containing both datatypes, not necessarily the smallest possible.
    /// The operation can fail.
    fn super_union(&self, other: &Self) -> Result<Self>;

    /// A simplified intersection.
    /// The result is a datatype containing the intersection, not necessarily the smallest possible
    /// The operation can fail.
    fn super_intersection(&self, other: &Self) -> Result<Self>;

    /// Convert type _a_ of Variant _A_ into another type _b_ of Variant _B_,
    /// only if there is an injection from _a_ to _b_.
    /// The conversion may fail.
    /// This is consistent with the inject_into method.
    fn into_data_type(&self, other: &DataType) -> Result<DataType>
    where
        Self: InjectInto<DataType>,
    {
        Ok(self.inject_into(other)?.super_image(self)?)
    }

    /// Return a small data_set in this variant
    fn minimal_subset(&self) -> Result<Self> {
        Err(Error::other("Cannot build a minimal DataType"))
    }

    /// Return a large data_set in this variant
    fn maximal_superset(&self) -> Result<Self> {
        Err(Error::other("Cannot build a maximal DataType"))
    }

    /// Convert type _a_ of Variant _A_ into a similar type of Variant _B_, when possible.
    /// The conversion may fail.
    fn into_variant(&self, variant: &DataType) -> Result<DataType>
    where
        Self: InjectInto<DataType>,
    {
        variant
            .maximal_superset()
            .and_then(|var| self.into_data_type(&var))
    }
}

// A few basic shared implementations

/// A basic implementation of partial_cmp
fn partial_cmp<V: Variant>(variant: &V, other: &V) -> Option<cmp::Ordering> {
    match (variant.is_subset_of(other), other.is_subset_of(variant)) {
        (true, true) => Some(cmp::Ordering::Equal),
        (true, false) => Some(cmp::Ordering::Less),
        (false, true) => Some(cmp::Ordering::Greater),
        (false, false) => None,
    }
}

// Label the types with traits to manipulate them by block

pub trait Specific: Variant {}
impl Specific for Unit {}
impl Specific for Boolean {}
impl Specific for Integer {}
impl Specific for Enum {}
impl Specific for Float {}
impl Specific for Text {}
impl Specific for Bytes {}
impl Specific for Date {}
impl Specific for Time {}
impl Specific for DateTime {}
impl Specific for Duration {}
impl Specific for Id {}
impl Specific for Struct {}
impl Specific for Union {}
impl Specific for Optional {}
impl Specific for List {}
impl Specific for Set {}
impl Specific for Array {}
impl Specific for Function {}

pub trait Generic: Variant {}
impl Generic for DataType {}

// TODO have default data_types for Primitive
// If A::default() -inj-> B::default() then a ∩ b = inj(a) ∩ b
// And a ∪ b = inj(a) ∪ b
pub trait Primitive: Specific {}
impl Primitive for Unit {}
impl Primitive for Boolean {}
impl Primitive for Integer {}
impl Primitive for Enum {}
impl Primitive for Float {}
impl Primitive for Text {}
impl Primitive for Bytes {}
impl Primitive for Date {}
impl Primitive for Time {}
impl Primitive for DateTime {}
impl Primitive for Duration {}
impl Primitive for Id {}

pub trait Composite: Specific {}
impl Composite for Struct {}
impl Composite for Union {}
impl Composite for Optional {}
impl Composite for List {}
impl Composite for Set {}
impl Composite for Array {}
impl Composite for Function {}

/// Unit variant
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd)]
pub struct Unit;

impl Default for Unit {
    fn default() -> Self {
        Unit
    }
}

impl fmt::Display for Unit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "()")
    }
}

impl Or<DataType> for Unit {
    type Sum = Optional;
    fn or(self, other: DataType) -> Self::Sum {
        match other {
            DataType::Null | DataType::Unit(_) => Optional::from_data_type(DataType::Null),
            DataType::Optional(o) => o,
            o => Optional::from_data_type(o),
        }
    }
}

impl InjectInto<DataType> for Unit {
    type Injection = Base<Self, DataType>;
    fn inject_into(&self, other: &DataType) -> injection::Result<Base<Self, DataType>> {
        injection::From(self.clone()).into(other.clone())
    }
}

impl From<value::Unit> for Unit {
    fn from(_value: value::Unit) -> Self {
        Unit
    }
}

impl Variant for Unit {
    type Element = value::Unit;

    fn contains(&self, _element: &Self::Element) -> bool {
        true
    }

    fn is_subset_of(&self, _other: &Self) -> bool {
        true
    }

    fn super_union(&self, _other: &Self) -> Result<Self> {
        Ok(Unit)
    }

    fn super_intersection(&self, _other: &Self) -> Result<Self> {
        Ok(Unit)
    }

    fn is_superset_of(&self, other: &Self) -> bool {
        other.is_subset_of(self)
    }

    fn minimal_subset(&self) -> Result<Self> {
        Ok(Unit)
    }

    fn maximal_superset(&self) -> Result<Self> {
        Ok(Unit)
    }
}

/// Boolean variant
pub type Boolean = Intervals<bool>;

impl Variant for Boolean {
    type Element = value::Boolean;

    fn contains(&self, element: &Self::Element) -> bool {
        self.contains(&**element)
    }

    fn is_subset_of(&self, other: &Self) -> bool {
        self.is_subset_of(other)
    }

    fn super_union(&self, other: &Self) -> Result<Self> {
        Ok(self.clone().union(other.clone()))
    }

    fn super_intersection(&self, other: &Self) -> Result<Self> {
        Ok(self.clone().intersection(other.clone()))
    }

    fn minimal_subset(&self) -> Result<Self> {
        Ok(Self::empty())
    }

    fn maximal_superset(&self) -> Result<Self> {
        Ok(Self::full())
    }
}

impl InjectInto<DataType> for Boolean {
    type Injection = Base<Self, DataType>;
    fn inject_into(&self, other: &DataType) -> injection::Result<Base<Self, DataType>> {
        injection::From(self.clone()).into(other.clone())
    }
}

impl From<value::Boolean> for Boolean {
    fn from(value: value::Boolean) -> Self {
        Boolean::from_value(*value)
    }
}

/// Integer variant
pub type Integer = Intervals<i64>;

impl Variant for Integer {
    type Element = value::Integer;

    fn contains(&self, element: &Self::Element) -> bool {
        self.contains(&**element)
    }

    fn is_subset_of(&self, other: &Self) -> bool {
        self.is_subset_of(other)
    }

    fn super_union(&self, other: &Self) -> Result<Self> {
        Ok(self.clone().union(other.clone()))
    }

    fn super_intersection(&self, other: &Self) -> Result<Self> {
        Ok(self.clone().intersection(other.clone()))
    }

    fn minimal_subset(&self) -> Result<Self> {
        Ok(Self::empty())
    }

    fn maximal_superset(&self) -> Result<Self> {
        Ok(Self::full())
    }
}

impl InjectInto<DataType> for Integer {
    type Injection = Base<Self, DataType>;
    fn inject_into(&self, other: &DataType) -> injection::Result<Base<Self, DataType>> {
        injection::From(self.clone()).into(other.clone())
    }
}

impl From<value::Integer> for Integer {
    fn from(value: value::Integer) -> Self {
        Integer::from_value(*value)
    }
}

impl From<Boolean> for Integer {
    fn from(_: Boolean) -> Self {
        Integer::from_values(&[0, 1])
    }
}

impl From<Enum> for Integer {
    fn from(e: Enum) -> Self {
        e.values.into_iter().map(|(_, i)| i).collect()
    }
}

/// Enum variant
#[derive(Debug, Clone, Hash, PartialEq)]
pub struct Enum {
    values: Rc<[(String, i64)]>,
}

impl Enum {
    pub fn new(values: Rc<[(String, i64)]>) -> Enum {
        assert!(!values.is_empty()); // An Enum should not be empty
        let codes: BTreeSet<i64> = values.iter().map(|(_, i)| *i).collect();
        assert!(values.len() == codes.len()); // Codes must be distinct
        Enum { values }
    }

    pub fn values(&self) -> BTreeSet<(String, i64)> {
        self.values.iter().cloned().collect()
    }
    pub fn value(&self) -> &(String, i64) {
        self.values.iter().next().unwrap()
    }

    pub fn encode(&self, key: String) -> Result<i64> {
        Ok(self
            .values
            .iter()
            .find(|(k, _)| &key == k)
            .ok_or(Error::invalid_field(key))?
            .1)
    }

    pub fn decode(&self, value: i64) -> Result<String> {
        Ok(self
            .values
            .iter()
            .find(|(_, v)| &value == v)
            .ok_or(Error::invalid_field(value))?
            .0
            .clone())
    }
}

/// To ease iteration
impl Deref for Enum {
    type Target = [(String, i64)];

    fn deref(&self) -> &Self::Target {
        &*self.values
    }
}

impl<S: Clone + Into<String>> From<&[S]> for Enum {
    fn from(values: &[S]) -> Self {
        Enum::new(
            values
                .iter()
                .enumerate()
                .map(|(i, s)| (s.clone().into(), i as i64))
                .collect(),
        )
    }
}

impl<S: Into<String>> FromIterator<(S, i64)> for Enum {
    fn from_iter<I: IntoIterator<Item = (S, i64)>>(iter: I) -> Self {
        Enum::new(
            iter.into_iter()
                .map(|(s, i)| (s.into(), i as i64))
                .collect(),
        )
    }
}

impl cmp::PartialOrd for Enum {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        partial_cmp(self, other)
    }
}

impl fmt::Display for Enum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "enum({})",
            self.values
                .iter()
                .map(|(s, i)| format!(r#""{}" ({})"#, s, i))
                .join(", ")
        )
    }
}

impl Variant for Enum {
    type Element = value::Enum;

    fn contains(&self, element: &Self::Element) -> bool {
        if let Ok(key) = element.decode() {
            self.values.contains(&(key, element.0))
        } else {
            false
        }
    }

    fn is_subset_of(&self, other: &Self) -> bool {
        self.values().is_subset(&other.values())
    }

    fn super_union(&self, other: &Self) -> Result<Self> {
        Ok(self.values().union(&other.values()).cloned().collect())
    }

    fn super_intersection(&self, other: &Self) -> Result<Self> {
        Ok(self
            .values()
            .intersection(&other.values())
            .cloned()
            .collect())
    }

    fn minimal_subset(&self) -> Result<Self> {
        Ok(Self::new(Rc::new([])))
    }
}

impl InjectInto<DataType> for Enum {
    type Injection = Base<Self, DataType>;
    fn inject_into(&self, other: &DataType) -> injection::Result<Base<Self, DataType>> {
        injection::From(self.clone()).into(other.clone())
    }
}

impl From<value::Enum> for Enum {
    fn from(value: value::Enum) -> Self {
        Enum::new(value.1.clone())
    }
}

/// Float variant
pub type Float = Intervals<f64>;

impl Variant for Float {
    type Element = value::Float;

    fn contains(&self, element: &Self::Element) -> bool {
        self.contains(&**element)
    }

    fn is_subset_of(&self, other: &Self) -> bool {
        self.is_subset_of(other)
    }

    fn super_union(&self, other: &Self) -> Result<Self> {
        Ok(self.clone().union(other.clone()))
    }

    fn super_intersection(&self, other: &Self) -> Result<Self> {
        Ok(self.clone().intersection(other.clone()))
    }

    fn minimal_subset(&self) -> Result<Self> {
        Ok(Self::empty())
    }

    fn maximal_superset(&self) -> Result<Self> {
        Ok(Self::full())
    }
}

impl InjectInto<DataType> for Float {
    type Injection = Base<Self, DataType>;
    fn inject_into(&self, other: &DataType) -> injection::Result<Base<Self, DataType>> {
        injection::From(self.clone()).into(other.clone())
    }
}

impl From<value::Float> for Float {
    fn from(value: value::Float) -> Self {
        Float::from_value(*value)
    }
}

/// Text variant
pub type Text = Intervals<String>;

impl Variant for Text {
    type Element = value::Text;

    fn contains(&self, element: &Self::Element) -> bool {
        self.contains(&**element)
    }

    fn is_subset_of(&self, other: &Self) -> bool {
        self.is_subset_of(other)
    }

    fn super_union(&self, other: &Self) -> Result<Self> {
        Ok(self.clone().union(other.clone()))
    }

    fn super_intersection(&self, other: &Self) -> Result<Self> {
        Ok(self.clone().intersection(other.clone()))
    }

    fn minimal_subset(&self) -> Result<Self> {
        Ok(Self::empty())
    }

    fn maximal_superset(&self) -> Result<Self> {
        Ok(Self::full())
    }
}

impl InjectInto<DataType> for Text {
    type Injection = Base<Self, DataType>;
    fn inject_into(&self, other: &DataType) -> injection::Result<Base<Self, DataType>> {
        injection::From(self.clone()).into(other.clone())
    }
}

impl From<value::Text> for Text {
    fn from(value: value::Text) -> Self {
        Text::from_value((*value).clone())
    }
}

/// Bytes variant
#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd)]
pub struct Bytes;

impl Default for Bytes {
    fn default() -> Self {
        Bytes
    }
}

impl fmt::Display for Bytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bytes")
    }
}

impl Variant for Bytes {
    type Element = value::Bytes;

    fn contains(&self, _: &Self::Element) -> bool {
        true
    }

    fn is_subset_of(&self, _: &Self) -> bool {
        true
    }

    fn super_union(&self, _: &Self) -> Result<Self> {
        Ok(Bytes)
    }

    fn super_intersection(&self, _: &Self) -> Result<Self> {
        Ok(Bytes)
    }

    fn minimal_subset(&self) -> Result<Self> {
        Ok(Bytes)
    }

    fn maximal_superset(&self) -> Result<Self> {
        Ok(Bytes)
    }
}

impl InjectInto<DataType> for Bytes {
    type Injection = Base<Self, DataType>;
    fn inject_into(&self, other: &DataType) -> injection::Result<Base<Self, DataType>> {
        injection::From(self.clone()).into(other.clone())
    }
}

impl From<value::Bytes> for Bytes {
    fn from(_value: value::Bytes) -> Self {
        Bytes
    }
}

/// Struct variant
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Struct {
    fields: Vec<(String, Rc<DataType>)>,
}

impl Struct {
    /// Create a Struct from a rc slice of fields
    pub fn new(fields: Vec<(String, Rc<DataType>)>) -> Struct {
        let mut uniques = HashSet::new();
        assert!(fields.iter().all(move |(f, _)| uniques.insert(f.clone())));
        Struct { fields }
    }
    /// An empty struct (a neutral element for the cartesian product)
    pub fn unit() -> Struct {
        Struct::new(vec![])
    }
    /// Create from one field
    pub fn from_field<S: Into<String>, T: Into<Rc<DataType>>>(s: S, t: T) -> Struct {
        Struct::new(vec![(s.into(), t.into())])
    }
    /// Create from one datatype
    pub fn from_data_type(data_type: DataType) -> Struct {
        Struct::default().and(data_type)
    }
    /// Create from a slice of datatypes
    pub fn from_data_types<T: Clone + Into<DataType>, A: AsRef<[T]>>(data_types: A) -> Struct {
        data_types
            .as_ref()
            .iter()
            .fold(Struct::default(), |s, t| s.and(t.clone().into()))
    }
    /// Get all the fields
    pub fn fields(&self) -> &[(String, Rc<DataType>)] {
        self.fields.as_ref()
    }
    /// Get the field
    pub fn field(&self, name: &str) -> Result<&(String, Rc<DataType>)> {
        self.fields
            .iter()
            .find(|(f, _)| f == name)
            .map_or(Err(Error::invalid_field(name).into()), |f| Ok(f))
    }
    /// Get the DataType associated with the field
    pub fn data_type(&self, name: &str) -> Rc<DataType> {
        self.fields
            .iter()
            .find(|(f, _)| f == name)
            .map_or(Rc::new(DataType::Any), |(_, t)| t.clone())
    }
    /// Find the index of the field with the given name
    pub fn index_from_name(&self, name: &str) -> Result<usize> {
        self.fields
            .iter()
            .position(|(s, _t)| s == name)
            .map_or(Err(Error::InvalidField(name.to_string()).into()), |i| Ok(i))
    }
    /// Access a field by index
    pub fn field_from_index(&self, index: usize) -> &(String, Rc<DataType>) {
        &self.fields[index]
    }
    /// Build the type of a DataFrame (column based data)
    pub fn from_schema_size(schema: Struct, size: &Integer) -> Struct {
        schema
            .fields
            .into_iter()
            .map(|(s, t)| (s, Rc::new(List::new(t, size.clone()).into())))
            .collect()
    }
    // TODO This could be implemented with a visitor (it would not fail on cyclic cases)
    /// Produce a Hierarchy of subtypes to access them in a smart way (unambiguous prefix can be omited)
    pub fn hierarchy(&self) -> Hierarchy<&DataType> {
        self.iter().fold(
            self.iter()
                .map(|(s, d)| (vec![s.clone()], d.as_ref()))
                .collect(),
            |h, (s, d)| h.chain(d.hierarchy().prepend(&[s.clone()])),
        )
    }
}

// This is a Unit
impl Default for Struct {
    fn default() -> Self {
        Struct::unit()
    }
}

#[allow(clippy::derive_hash_xor_eq)]
impl hash::Hash for Struct {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.fields.iter().for_each(|(f, t)| {
            Bound::hash(f, state);
            DataType::hash(t, state);
        });
    }
}

/// To ease iteration
impl Deref for Struct {
    type Target = [(String, Rc<DataType>)];

    fn deref(&self) -> &Self::Target {
        &self.fields
    }
}

/// This is the core operation to build a Struct
impl<S: Into<String>, T: Into<Rc<DataType>>> And<(S, T)> for Struct {
    type Product = Struct;
    fn and(self, other: (S, T)) -> Self::Product {
        let field: String = other.0.into();
        let data_type: Rc<DataType> = other.1.into();
        let mut push_other = true;
        // Remove existing elements with the same name
        let mut fields: Vec<(String, Rc<DataType>)> = self
            .fields
            .iter()
            .map(|(f, t)| {
                //(&field != f).then_some((f.clone(), t.clone()))
                if &field != f {
                    (f.clone(), t.clone())
                } else if let (&DataType::Struct(_), &DataType::Struct(_)) =
                    (data_type.as_ref(), t.as_ref())
                {
                    push_other = false;
                    (
                        f.clone(),
                        Rc::new(data_type.as_ref().clone().and(t.as_ref().clone())),
                    )
                } else {
                    push_other = false;
                    (
                        f.clone(),
                        Rc::new(data_type.as_ref().super_intersection(t.as_ref()).unwrap()),
                    )
                }
            })
            .collect();
        if push_other {
            fields.push((field, data_type))
        }
        Struct::new(fields.into())
    }
}

impl<T: Into<Rc<DataType>>> And<(T,)> for Struct {
    type Product = Struct;
    fn and(self, other: (T,)) -> Self::Product {
        let field = namer::new_name_outside("", self.fields.iter().map(|(f, _t)| f));
        let data_type: Rc<DataType> = other.0.into();
        self.and((field, data_type))
    }
}

impl And<Struct> for Struct {
    type Product = Struct;
    fn and(self, other: Struct) -> Self::Product {
        let mut result = self;
        for field in other.fields() {
            result = result.and(field.clone())
        }
        result
    }
}

impl And<DataType> for Struct {
    type Product = Struct;
    fn and(self, other: DataType) -> Self::Product {
        // Simplify in the case of struct and Unit
        match other {
            //DataType::Unit(_u) => self, // TODO remove that ?
            DataType::Struct(s) => self.and(s),
            other => self.and((other,)),
        }
    }
}

impl<S: Into<String>, T: Into<Rc<DataType>>> From<(S, T)> for Struct {
    fn from(field: (S, T)) -> Self {
        Struct::from_field(field.0, field.1)
    }
}

impl<S: Clone + Into<String>, T: Clone + Into<Rc<DataType>>> From<&[(S, T)]> for Struct {
    fn from(values: &[(S, T)]) -> Self {
        Struct::new(
            values
                .into_iter()
                .map(|(f, t)| (f.clone().into(), t.clone().into()))
                .collect(),
        )
    }
}

impl From<Unit> for Struct {
    fn from(_value: Unit) -> Self {
        Struct::unit()
    }
}

impl From<value::Struct> for Struct {
    fn from(value: value::Struct) -> Self {
        (*value)
            .iter()
            .map(|(s, v)| (s.clone(), v.data_type()))
            .collect()
    }
}

impl<S: Into<String>, T: Into<Rc<DataType>>> FromIterator<(S, T)> for Struct {
    fn from_iter<I: IntoIterator<Item = (S, T)>>(iter: I) -> Self {
        iter.into_iter().fold(Struct::unit(), |s, f| s.and(f))
    }
}

impl cmp::PartialOrd for Struct {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        partial_cmp(self, other)
    }
}

impl fmt::Display for Struct {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "struct{{{}}}",
            self.fields
                .iter()
                .map(|(f, t)| { format!("{}: {}", f, t).to_string() })
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}

impl Variant for Struct {
    type Element = value::Struct;

    fn contains(&self, element: &Self::Element) -> bool {
        self.fields.iter().all(|(s, d)| {
            element
                .value(s)
                .map_or(false, |v| d.as_ref().contains(v.as_ref()))
        })
    }

    /// A struct in self should be expressible in other
    /// It is similar for struct to the relation is-subclass-of
    fn is_subset_of(&self, other: &Self) -> bool {
        other.fields.iter().all(
            |(other_field, other_data_type)| {
                self.data_type(other_field).is_subset_of(other_data_type)
            }, // If a field in other is not present in self the inclusion is rejected
        )
    }

    fn super_union(&self, other: &Self) -> Result<Self> {
        let self_fields: BTreeSet<String> = self.fields.iter().map(|(f, _)| f.clone()).collect();
        let other_fields: BTreeSet<String> = other.fields.iter().map(|(f, _)| f.clone()).collect();
        let fields = self_fields.intersection(&other_fields);
        fields
            .into_iter()
            .map(|f| {
                Ok((
                    f.clone(),
                    self.data_type(&f).super_union(&other.data_type(&f))?,
                ))
            })
            .collect()
    }

    fn super_intersection(&self, other: &Self) -> Result<Self> {
        let self_fields: BTreeSet<String> = self.fields.iter().map(|(f, _)| f.clone()).collect();
        let other_fields: BTreeSet<String> = other.fields.iter().map(|(f, _)| f.clone()).collect();
        let fields = self_fields.union(&other_fields);
        fields
            .into_iter()
            .map(|f| {
                Ok((
                    f.clone(),
                    self.data_type(&f)
                        .super_intersection(&other.data_type(&f))?,
                ))
            })
            .collect()
    }

    fn maximal_superset(&self) -> Result<Self> {
        Ok(Struct::new(vec![]))
    }
}

impl InjectInto<DataType> for Struct {
    type Injection = Base<Self, DataType>;
    fn inject_into(&self, other: &DataType) -> injection::Result<Base<Self, DataType>> {
        injection::From(self.clone()).into(other.clone())
    }
}

// Index Structs
impl<P: Path> Index<P> for Struct {
    type Output = DataType;

    fn index(&self, index: P) -> &Self::Output {
        self.hierarchy()[index]
    }
}

impl Index<usize> for Struct {
    type Output = Rc<DataType>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.field_from_index(index).1
    }
}

/// Union variant
#[derive(Debug, Clone, PartialEq)]
pub struct Union {
    fields: Vec<(String, Rc<DataType>)>,
}

impl Union {
    pub fn new(fields: Vec<(String, Rc<DataType>)>) -> Union {
        let mut uniques = HashSet::new();
        assert!(fields.iter().all(move |(f, _)| uniques.insert(f.clone())));
        Union { fields }
    }
    /// An empty union (a neutral element for the disjoint union)
    pub fn null() -> Union {
        Union::new(vec![])
    }
    /// Create from one field
    pub fn from_field<S: Into<String>, T: Into<Rc<DataType>>>(s: S, t: T) -> Union {
        Union::new(vec![(s.into(), t.into())])
    }
    /// Create from one datatype
    pub fn from_data_type(data_type: DataType) -> Union {
        Union::default().or(data_type)
    }
    /// Create from a slice of datatypes
    pub fn from_data_types(data_types: &[DataType]) -> Union {
        data_types
            .iter()
            .fold(Union::default(), |s, t| s.or(t.clone()))
    }
    /// Get all the fields
    pub fn fields(&self) -> &[(String, Rc<DataType>)] {
        self.fields.as_ref()
    }
    /// Get the field
    pub fn field(&self, name: &str) -> Result<&(String, Rc<DataType>)> {
        self.fields
            .iter()
            .find(|(f, _)| f == name)
            .map_or(Err(Error::InvalidField(name.to_string()).into()), |f| Ok(f))
    }
    /// Get the DataType associated with the field
    pub fn data_type(&self, name: &str) -> Rc<DataType> {
        self.fields
            .iter()
            .find(|(f, _)| f == name)
            .map_or(Rc::new(DataType::Null), |(_, t)| t.clone())
    }
    /// Find the index of the field with the given name
    pub fn index_from_name(&self, name: &str) -> Result<usize> {
        self.fields
            .iter()
            .position(|(s, _t)| s == name)
            .map_or(Err(Error::InvalidField(name.to_string()).into()), |i| Ok(i))
    }
    /// Access a field by index
    pub fn field_from_index(&self, index: usize) -> &(String, Rc<DataType>) {
        &self.fields[index]
    }
    // TODO This could be implemented with a visitor (it would not fail on cyclic cases)
    /// Produce a Hierarchy of subtypes to access them in a smart way (unambiguous prefix can be omited)
    pub fn hierarchy(&self) -> Hierarchy<&DataType> {
        self.iter().fold(
            self.iter()
                .map(|(s, d)| (vec![s.clone()], d.as_ref()))
                .collect(),
            |h, (s, d)| h.chain(d.hierarchy().prepend(&[s.clone()])),
        )
    }
}

// This is a Null
impl Default for Union {
    fn default() -> Self {
        Union::null()
    }
}

#[allow(clippy::derive_hash_xor_eq)]
impl hash::Hash for Union {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.fields.iter().for_each(|(f, t)| {
            Bound::hash(f, state);
            DataType::hash(t, state);
        });
    }
}

/// To ease iteration
impl Deref for Union {
    type Target = [(String, Rc<DataType>)];

    fn deref(&self) -> &Self::Target {
        &self.fields
    }
}

/// This is the core operation to build a Union
impl<S: Into<String>, T: Into<Rc<DataType>>> Or<(S, T)> for Union {
    type Sum = Union;
    fn or(self, other: (S, T)) -> Self::Sum {
        let field: String = other.0.into();
        let data_type: Rc<DataType> = other.1.into();
        // Remove existing elements with the same name
        let mut fields: Vec<(String, Rc<DataType>)> = self
            .fields
            .iter()
            .filter_map(|(f, t)| (&field != f).then_some((f.clone(), t.clone())))
            .collect();
        fields.push((field, data_type));
        Union::new(fields.into())
    }
}

impl<T: Into<Rc<DataType>>> Or<(T,)> for Union {
    type Sum = Union;
    fn or(self, other: (T,)) -> Self::Sum {
        let field = namer::new_name_outside("", self.fields.iter().map(|(f, _t)| f));
        let data_type: Rc<DataType> = other.0.into();
        self.or((field, data_type))
    }
}

impl Or<Unit> for Union {
    type Sum = Optional;
    fn or(self, _other: Unit) -> Self::Sum {
        Optional::from_data_type(DataType::from(self))
    }
}

impl Or<Optional> for Union {
    type Sum = Optional;
    fn or(self, other: Optional) -> Self::Sum {
        Optional::from_data_type(DataType::from(self.or(other.data_type().clone())))
    }
}

impl Or<Union> for Union {
    type Sum = Union;
    fn or(self, other: Union) -> Self::Sum {
        let mut result = self;
        for field in other.fields() {
            result = result.or(field.clone())
        }
        result
    }
}

impl Or<DataType> for Union {
    type Sum = Union;
    fn or(self, other: DataType) -> Self::Sum {
        // Simplify in the case of union and Null
        match other {
            DataType::Null => self,
            DataType::Union(u) => self.or(u),
            other => self.or((other,)),
        }
    }
}

impl<S: Into<String>, T: Into<Rc<DataType>>> From<(S, T)> for Union {
    fn from(field: (S, T)) -> Self {
        Union::from_field(field.0, field.1)
    }
}

impl<S: Clone + Into<String>, T: Clone + Into<Rc<DataType>>> From<&[(S, T)]> for Union {
    fn from(values: &[(S, T)]) -> Self {
        Union::new(
            values
                .into_iter()
                .map(|(f, t)| (f.clone().into(), t.clone().into()))
                .collect(),
        )
    }
}

impl From<value::Union> for Union {
    fn from(value: value::Union) -> Self {
        ((*value).0.clone(), (*value).1.data_type().clone()).into()
    }
}

impl<S: Into<String>, T: Into<Rc<DataType>>> FromIterator<(S, T)> for Union {
    fn from_iter<I: IntoIterator<Item = (S, T)>>(iter: I) -> Self {
        iter.into_iter().fold(Union::null(), |s, f| s.or(f))
    }
}

impl cmp::PartialOrd for Union {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        partial_cmp(self, other)
    }
}

impl fmt::Display for Union {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "union{{{}}}",
            self.fields
                .iter()
                .map(|(f, t)| { format!("{}: {}", f, t).to_string() })
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}

impl Variant for Union {
    type Element = value::Union;

    fn contains(&self, element: &Self::Element) -> bool {
        self.fields
            .iter()
            .any(|(s, d)| &element.0 == s && d.contains(&*element.1))
    }

    fn is_subset_of(&self, other: &Self) -> bool {
        let fields: BTreeSet<String> = self.fields.iter().map(|(f, _)| f.to_string()).collect();
        let other_fields: BTreeSet<String> =
            other.fields.iter().map(|(f, _)| f.to_string()).collect();
        fields.is_subset(&other_fields)
            && self
                .fields
                .iter()
                .all(|(f, t)| t.is_subset_of(&self.data_type(f)))
    }

    fn super_union(&self, other: &Self) -> Result<Self> {
        let self_fields: BTreeSet<String> = self.fields.iter().map(|(f, _)| f.clone()).collect();
        let other_fields: BTreeSet<String> = other.fields.iter().map(|(f, _)| f.clone()).collect();
        let fields = self_fields.union(&other_fields);
        fields
            .into_iter()
            .map(|f| {
                Ok((
                    f.clone(),
                    self.data_type(&f).super_union(&other.data_type(&f))?,
                ))
            })
            .collect()
    }

    fn super_intersection(&self, other: &Self) -> Result<Self> {
        let self_fields: BTreeSet<String> = self.fields.iter().map(|(f, _)| f.clone()).collect();
        let other_fields: BTreeSet<String> = other.fields.iter().map(|(f, _)| f.clone()).collect();
        let fields = self_fields.intersection(&other_fields);
        fields
            .into_iter()
            .map(|f| {
                Ok((
                    f.clone(),
                    self.data_type(&f)
                        .super_intersection(&other.data_type(&f))?,
                ))
            })
            .collect()
    }

    fn minimal_subset(&self) -> Result<Self> {
        Ok(Union::new(vec![]))
    }
}

impl InjectInto<DataType> for Union {
    type Injection = Base<Self, DataType>;
    fn inject_into(&self, other: &DataType) -> injection::Result<Base<Self, DataType>> {
        injection::From(self.clone()).into(other.clone())
    }
}

// Index Unions
impl<P: Path> Index<P> for Union {
    type Output = DataType;

    fn index(&self, index: P) -> &Self::Output {
        self.hierarchy()[index]
    }
}

impl Index<usize> for Union {
    type Output = Rc<DataType>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.field_from_index(index).1
    }
}

/// Optional variant
#[derive(Debug, Clone, PartialEq)]
pub struct Optional {
    data_type: Rc<DataType>,
}

impl Optional {
    pub fn new(data_type: Rc<DataType>) -> Optional {
        Optional { data_type }
    }

    pub fn data_type(&self) -> &DataType {
        self.data_type.as_ref()
    }

    pub fn from_data_type<T: Into<DataType>>(data_type: T) -> Optional {
        let data_type = data_type.into();
        match data_type {
            DataType::Optional(o) => o,
            _ => Optional::new(Rc::new(data_type)),
        }
    }
}

impl From<Rc<DataType>> for Optional {
    fn from(data_type: Rc<DataType>) -> Self {
        Optional::new(data_type)
    }
}

impl From<DataType> for Optional {
    fn from(data_type: DataType) -> Self {
        Optional::from_data_type(data_type)
    }
}

#[allow(clippy::derive_hash_xor_eq)]
impl hash::Hash for Optional {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data_type.hash(state);
    }
}

impl cmp::PartialOrd for Optional {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        partial_cmp(self, other)
    }
}

impl fmt::Display for Optional {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "option({})", self.data_type)
    }
}

impl Or<Optional> for Optional {
    type Sum = Optional;
    fn or(self, other: Optional) -> Self::Sum {
        Optional::from_data_type(self.data_type().clone().or(other.data_type().clone()))
    }
}

impl Or<DataType> for Optional {
    type Sum = Optional;
    fn or(self, other: DataType) -> Self::Sum {
        match other {
            DataType::Null | DataType::Unit(_) => self,
            DataType::Optional(o) => self.or(o),
            o => Optional::from_data_type(self.data_type().clone().or(o)),
        }
    }
}

impl Variant for Optional {
    type Element = value::Optional;

    fn contains(&self, element: &Self::Element) -> bool {
        element
            .as_ref()
            .map_or(true, |v| self.data_type.contains(v))
    }

    fn is_subset_of(&self, other: &Self) -> bool {
        self.data_type <= other.data_type
    }

    fn super_union(&self, other: &Self) -> Result<Self> {
        Ok(Optional::from(
            self.data_type().super_union(other.data_type())?,
        ))
    }

    fn super_intersection(&self, other: &Self) -> Result<Self> {
        Ok(Optional::from(
            self.data_type().super_intersection(other.data_type())?,
        ))
    }

    fn minimal_subset(&self) -> Result<Self> {
        Ok(Optional::from_data_type(DataType::Null))
    }

    fn maximal_superset(&self) -> Result<Self> {
        Ok(Optional::from_data_type(DataType::Any))
    }
}

impl InjectInto<DataType> for Optional {
    type Injection = Base<Self, DataType>;
    fn inject_into(&self, other: &DataType) -> injection::Result<Base<Self, DataType>> {
        injection::From(self.clone()).into(other.clone())
    }
}

impl From<value::Optional> for Optional {
    fn from(value: value::Optional) -> Self {
        (*value).clone().map_or(Optional::from(DataType::Any), |o| {
            Optional::from(o.as_ref().data_type())
        })
    }
}

/// List variant
/// Contrary to Structs List DataType is covariant in both data_type and max_size
#[derive(Debug, Clone, Hash, PartialEq)]
pub struct List {
    data_type: Rc<DataType>,
    size: Integer,
}

impl List {
    pub fn new(data_type: Rc<DataType>, size: Integer) -> List {
        List {
            data_type,
            size: size.intersection(Integer::from_min(0)),
        }
    }

    pub fn from_data_type_size(data_type: DataType, size: Integer) -> List {
        List::new(Rc::new(data_type), size)
    }

    pub fn from_data_type(data_type: DataType) -> List {
        List::from_data_type_size(data_type, Integer::from_min(0))
    }

    pub fn size(&self) -> &Integer {
        &self.size
    }

    pub fn data_type(&self) -> &DataType {
        self.data_type.as_ref()
    }
}

impl From<(Rc<DataType>, Integer)> for List {
    fn from(data_type_size: (Rc<DataType>, Integer)) -> Self {
        List::new(data_type_size.0, data_type_size.1)
    }
}

impl From<(DataType, Integer)> for List {
    fn from(data_type_size: (DataType, Integer)) -> Self {
        List::new(Rc::new(data_type_size.0), data_type_size.1)
    }
}

impl cmp::PartialOrd for List {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        partial_cmp(self, other)
    }
}

impl fmt::Display for List {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "list({}, size ∈ {})", self.data_type, self.size)
    }
}

impl Variant for List {
    type Element = value::List;

    fn contains(&self, element: &Self::Element) -> bool {
        self.size.contains(&(element.len() as i64))
            && element.iter().all(|v| self.data_type.contains(v))
    }

    fn is_subset_of(&self, other: &Self) -> bool {
        self.data_type <= other.data_type && self.size <= other.size
    }

    fn super_union(&self, other: &Self) -> Result<Self> {
        Ok(List::from((
            self.data_type().super_union(other.data_type())?,
            self.size().super_union(other.size())?,
        )))
    }

    fn super_intersection(&self, other: &Self) -> Result<Self> {
        Ok(List::from((
            self.data_type().super_intersection(other.data_type())?,
            self.size().super_intersection(other.size())?,
        )))
    }

    fn minimal_subset(&self) -> Result<Self> {
        Ok(List::from_data_type_size(DataType::Null, Integer::empty()))
    }

    fn maximal_superset(&self) -> Result<Self> {
        Ok(List::from_data_type_size(
            DataType::Any,
            Integer::from_max(i64::MAX),
        ))
    }
}

impl InjectInto<DataType> for List {
    type Injection = Base<Self, DataType>;
    fn inject_into(&self, other: &DataType) -> injection::Result<Base<Self, DataType>> {
        injection::From(self.clone()).into(other.clone())
    }
}

impl From<value::List> for List {
    fn from(value: value::List) -> Self {
        List::from((
            (*value).iter().fold(DataType::Null, |s, d| {
                s.super_union(&d.data_type()).unwrap_or(DataType::Any)
            }),
            Integer::from_value((*value).len() as i64),
        ))
    }
}

/// Set variant
/// Contrary to Structs Set DataType is covariant in both data_type and max_size
#[derive(Debug, Clone, Hash, PartialEq)]
pub struct Set {
    data_type: Rc<DataType>,
    size: Integer,
}

impl Set {
    pub fn new(data_type: Rc<DataType>, size: Integer) -> Set {
        Set {
            data_type,
            size: size.intersection(Integer::from_min(0)),
        }
    }

    pub fn from_data_type_size(data_type: DataType, size: Integer) -> Set {
        Set::new(Rc::new(data_type), size)
    }

    pub fn from_data_type(data_type: DataType) -> Set {
        Set::from_data_type_size(data_type, Integer::from_max(i64::MAX))
    }

    pub fn size(&self) -> &Integer {
        &self.size
    }

    pub fn data_type(&self) -> &DataType {
        self.data_type.as_ref()
    }
}

impl From<(Rc<DataType>, Integer)> for Set {
    fn from(data_type_size: (Rc<DataType>, Integer)) -> Self {
        Set::new(data_type_size.0, data_type_size.1)
    }
}

impl From<(DataType, Integer)> for Set {
    fn from(data_type_size: (DataType, Integer)) -> Self {
        Set::new(Rc::new(data_type_size.0), data_type_size.1)
    }
}

impl cmp::PartialOrd for Set {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        partial_cmp(self, other)
    }
}

impl fmt::Display for Set {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "set({}, size ∈ {})", self.data_type, self.size)
    }
}

impl Variant for Set {
    type Element = value::Set;

    fn contains(&self, element: &Self::Element) -> bool {
        self.size.contains(&(element.len() as i64))
            && element.iter().all(|v| self.data_type.contains(v))
    }

    fn is_subset_of(&self, other: &Self) -> bool {
        self.data_type <= other.data_type && self.size <= other.size
    }

    fn super_union(&self, other: &Self) -> Result<Self> {
        Ok(Set::from((
            self.data_type().super_union(other.data_type())?,
            self.size().super_union(other.size())?,
        )))
    }

    fn super_intersection(&self, other: &Self) -> Result<Self> {
        Ok(Set::from((
            self.data_type().super_intersection(other.data_type())?,
            self.size().super_intersection(other.size())?,
        )))
    }

    fn minimal_subset(&self) -> Result<Self> {
        Ok(Set::from_data_type_size(DataType::Null, Integer::empty()))
    }

    fn maximal_superset(&self) -> Result<Self> {
        Ok(Set::from_data_type_size(
            DataType::Any,
            Integer::from_max(i64::MAX),
        ))
    }
}

impl InjectInto<DataType> for Set {
    type Injection = Base<Self, DataType>;
    fn inject_into(&self, other: &DataType) -> injection::Result<Base<Self, DataType>> {
        injection::From(self.clone()).into(other.clone())
    }
}

impl From<value::Set> for Set {
    fn from(value: value::Set) -> Self {
        Set::from((
            (*value).iter().fold(DataType::Null, |s, d| {
                s.super_union(&d.data_type()).unwrap_or(DataType::Any)
            }),
            Integer::from_value((*value).len() as i64),
        ))
    }
}

/// Array variant
#[derive(Debug, Clone, Hash, PartialEq)]
pub struct Array {
    data_type: Rc<DataType>,
    shape: Rc<[usize]>,
}

impl Array {
    // TODO do as lists and sets
    pub fn new(data_type: Rc<DataType>, shape: Rc<[usize]>) -> Array {
        Array { data_type, shape }
    }

    pub fn from_data_type_shape<S: AsRef<[usize]>>(data_type: DataType, shape: S) -> Array {
        Array::new(Rc::new(data_type), Rc::from(shape.as_ref()))
    }

    pub fn data_type(&self) -> &DataType {
        self.data_type.as_ref()
    }

    pub fn shape(&self) -> &[usize] {
        self.shape.as_ref()
    }
}

impl From<(Rc<DataType>, &[usize])> for Array {
    fn from(data_type_shape: (Rc<DataType>, &[usize])) -> Self {
        Array::new(data_type_shape.0, Rc::from(data_type_shape.1))
    }
}

impl From<(DataType, &[usize])> for Array {
    fn from(data_type_shape: (DataType, &[usize])) -> Self {
        Array::new(Rc::new(data_type_shape.0), Rc::from(data_type_shape.1))
    }
}

impl cmp::PartialOrd for Array {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        partial_cmp(self, other)
    }
}

impl fmt::Display for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "array({}, shape<({}))",
            self.data_type,
            self.shape
                .iter()
                .map(|s| { format!("{}", s) })
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}

impl Variant for Array {
    type Element = value::Array;

    fn contains(&self, element: &Self::Element) -> bool {
        element.0.len() == self.shape.iter().product::<usize>()
            && element.1.iter().zip(self.shape.iter()).all(|(e, d)| e == d)
            && element.0.iter().all(|v| self.data_type.contains(v))
    }

    /// Only arrays of the same size can be compared (it differs from lists and sets)
    fn is_subset_of(&self, other: &Self) -> bool {
        self.data_type <= other.data_type && self.shape == other.shape
    }

    fn super_union(&self, other: &Self) -> Result<Self> {
        Ok(Array::from((
            self.data_type().super_union(other.data_type())?,
            self.shape(),
        )))
    }

    fn super_intersection(&self, other: &Self) -> Result<Self> {
        Ok(Array::from((
            self.data_type().super_intersection(other.data_type())?,
            self.shape(),
        )))
    }
}

impl InjectInto<DataType> for Array {
    type Injection = Base<Self, DataType>;
    fn inject_into(&self, other: &DataType) -> injection::Result<Base<Self, DataType>> {
        injection::From(self.clone()).into(other.clone())
    }
}

impl From<value::Array> for Array {
    fn from(value: value::Array) -> Self {
        Array::from((
            (*value).0.iter().fold(DataType::Null, |s, d| {
                s.super_union(&d.data_type()).unwrap_or(DataType::Any)
            }),
            (*value).1.as_ref(),
        ))
    }
}

/// Date variant
pub type Date = Intervals<chrono::NaiveDate>;

impl Variant for Date {
    type Element = value::Date;

    fn contains(&self, element: &Self::Element) -> bool {
        self.contains(&**element)
    }

    fn is_subset_of(&self, other: &Self) -> bool {
        self.is_subset_of(other)
    }

    fn super_union(&self, other: &Self) -> Result<Self> {
        Ok(self.clone().union(other.clone()))
    }

    fn super_intersection(&self, other: &Self) -> Result<Self> {
        Ok(self.clone().intersection(other.clone()))
    }

    fn minimal_subset(&self) -> Result<Self> {
        Ok(Self::empty())
    }

    fn maximal_superset(&self) -> Result<Self> {
        Ok(Self::full())
    }
}

impl InjectInto<DataType> for Date {
    type Injection = Base<Self, DataType>;
    fn inject_into(&self, other: &DataType) -> injection::Result<Base<Self, DataType>> {
        injection::From(self.clone()).into(other.clone())
    }
}

impl From<value::Date> for Date {
    fn from(value: value::Date) -> Self {
        Date::from_value(*value)
    }
}

/// Time variant
pub type Time = Intervals<chrono::NaiveTime>;

impl Variant for Time {
    type Element = value::Time;

    fn contains(&self, element: &Self::Element) -> bool {
        self.contains(&**element)
    }

    fn is_subset_of(&self, other: &Self) -> bool {
        self.is_subset_of(other)
    }

    fn super_union(&self, other: &Self) -> Result<Self> {
        Ok(self.clone().union(other.clone()))
    }

    fn super_intersection(&self, other: &Self) -> Result<Self> {
        Ok(self.clone().intersection(other.clone()))
    }

    fn minimal_subset(&self) -> Result<Self> {
        Ok(Self::empty())
    }

    fn maximal_superset(&self) -> Result<Self> {
        Ok(Self::full())
    }
}

impl InjectInto<DataType> for Time {
    type Injection = Base<Self, DataType>;
    fn inject_into(&self, other: &DataType) -> injection::Result<Base<Self, DataType>> {
        injection::From(self.clone()).into(other.clone())
    }
}

impl From<value::Time> for Time {
    fn from(value: value::Time) -> Self {
        Time::from_value(*value)
    }
}

/// DateTime variant
pub type DateTime = Intervals<chrono::NaiveDateTime>;

impl Variant for DateTime {
    type Element = value::DateTime;

    fn contains(&self, element: &Self::Element) -> bool {
        self.contains(&**element)
    }

    fn is_subset_of(&self, other: &Self) -> bool {
        self.is_subset_of(other)
    }

    fn super_union(&self, other: &Self) -> Result<Self> {
        Ok(self.clone().union(other.clone()))
    }

    fn super_intersection(&self, other: &Self) -> Result<Self> {
        Ok(self.clone().intersection(other.clone()))
    }

    fn minimal_subset(&self) -> Result<Self> {
        Ok(Self::empty())
    }

    fn maximal_superset(&self) -> Result<Self> {
        Ok(Self::full())
    }
}

impl InjectInto<DataType> for DateTime {
    type Injection = Base<Self, DataType>;
    fn inject_into(&self, other: &DataType) -> injection::Result<Base<Self, DataType>> {
        injection::From(self.clone()).into(other.clone())
    }
}

impl From<value::DateTime> for DateTime {
    fn from(value: value::DateTime) -> Self {
        DateTime::from_value(*value)
    }
}

/// Duration variant
pub type Duration = Intervals<chrono::Duration>;

impl Variant for Duration {
    type Element = value::Duration;

    fn contains(&self, element: &Self::Element) -> bool {
        self.contains(&**element)
    }

    fn is_subset_of(&self, other: &Self) -> bool {
        self.is_subset_of(other)
    }

    fn super_union(&self, other: &Self) -> Result<Self> {
        Ok(self.clone().union(other.clone()))
    }

    fn super_intersection(&self, other: &Self) -> Result<Self> {
        Ok(self.clone().intersection(other.clone()))
    }

    fn minimal_subset(&self) -> Result<Self> {
        Ok(Self::empty())
    }

    fn maximal_superset(&self) -> Result<Self> {
        Ok(Self::full())
    }
}

impl InjectInto<DataType> for Duration {
    type Injection = Base<Self, DataType>;
    fn inject_into(&self, other: &DataType) -> injection::Result<Base<Self, DataType>> {
        injection::From(self.clone()).into(other.clone())
    }
}

impl From<value::Duration> for Duration {
    fn from(value: value::Duration) -> Self {
        Duration::from_value(*value)
    }
}

/// Id variant
#[derive(Default, Debug, Clone, Hash, PartialEq)]
pub struct Id {
    /// This should be None if Id is an identifier or the type referred to
    reference: Option<Rc<Id>>,
    /// If entries are unique
    unique: bool,
}

impl Id {
    pub fn new(reference: Option<Rc<Id>>, unique: bool) -> Id {
        Id { reference, unique }
    }

    pub fn reference(&self) -> Option<&Id> {
        self.reference.as_deref()
    }

    pub fn unique(&self) -> bool {
        self.unique
    }
}

impl From<(Option<Rc<Id>>, bool)> for Id {
    fn from(ref_unique: (Option<Rc<Id>>, bool)) -> Self {
        let (reference, unique) = ref_unique;
        Id::new(reference, unique)
    }
}

impl From<(Option<Id>, bool)> for Id {
    fn from(ref_unique: (Option<Id>, bool)) -> Self {
        let (reference, unique) = ref_unique;
        Id::new(reference.map(Rc::new), unique)
    }
}

impl cmp::PartialOrd for Id {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        partial_cmp(self, other)
    }
}

impl fmt::Display for Id {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "id")
    }
}

impl Variant for Id {
    type Element = value::Id;

    fn contains(&self, _: &Self::Element) -> bool {
        true
    }

    fn is_subset_of(&self, _: &Self) -> bool {
        true
    }

    fn super_union(&self, other: &Self) -> Result<Self> {
        Ok(Id::new(
            if self.reference == other.reference {
                self.reference.clone()
            } else {
                None
            },
            false,
        ))
    }

    fn super_intersection(&self, other: &Self) -> Result<Self> {
        Ok(Id::new(
            if self.reference == other.reference {
                self.reference.clone()
            } else {
                None
            },
            self.unique && other.unique,
        ))
    }

    fn maximal_superset(&self) -> Result<Self> {
        Ok(Id::new(None, false))
    }
}

impl From<value::Id> for Id {
    fn from(_value: value::Id) -> Self {
        Id::default()
    }
}

impl InjectInto<DataType> for Id {
    type Injection = Base<Self, DataType>;
    fn inject_into(&self, other: &DataType) -> injection::Result<Base<Self, DataType>> {
        injection::From(self.clone()).into(other.clone())
    }
}

/// Function variant
#[derive(Debug, Clone, Hash, PartialEq)]
pub struct Function {
    pub domain: Rc<DataType>,
    pub co_domain: Rc<DataType>,
}

impl Function {
    pub fn new(from: Rc<DataType>, to: Rc<DataType>) -> Function {
        Function {
            domain: from,
            co_domain: to,
        }
    }

    pub fn from_data_types(from: DataType, to: DataType) -> Function {
        Function::new(Rc::new(from), Rc::new(to))
    }

    pub fn domain(&self) -> &DataType {
        self.domain.as_ref()
    }

    pub fn co_domain(&self) -> &DataType {
        self.co_domain.as_ref()
    }
}

impl From<(Rc<DataType>, Rc<DataType>)> for Function {
    fn from(dom_co_dom: (Rc<DataType>, Rc<DataType>)) -> Self {
        let (dom, co_dom) = dom_co_dom;
        Function::new(dom, co_dom)
    }
}

impl From<(DataType, DataType)> for Function {
    fn from(dom_co_dom: (DataType, DataType)) -> Self {
        let (dom, co_dom) = dom_co_dom;
        Function::from_data_types(dom, co_dom)
    }
}

impl cmp::PartialOrd for Function {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        partial_cmp(self, other)
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} -> {}", self.domain, self.co_domain)
    }
}

impl Variant for Function {
    type Element = value::Function;

    fn contains(&self, element: &Self::Element) -> bool {
        element.domain() >= *self.domain && element.co_domain() <= *self.co_domain
    }

    fn is_subset_of(&self, other: &Self) -> bool {
        self.domain >= other.domain && self.co_domain <= other.co_domain
    }

    fn super_union(&self, other: &Self) -> Result<Self> {
        if self.domain() == other.domain() {
            Ok(Function::from((
                self.domain().clone(),
                self.co_domain().super_union(other.co_domain())?,
            )))
        } else {
            Err(Error::no_superset(self, other))
        }
    }

    fn super_intersection(&self, other: &Self) -> Result<Self> {
        Ok(Function::from((
            self.domain().super_union(other.domain())?,
            self.co_domain().super_intersection(other.co_domain())?,
        )))
    }

    fn minimal_subset(&self) -> Result<Self> {
        Ok(Function::from_data_types(DataType::Any, DataType::Null))
    }

    fn maximal_superset(&self) -> Result<Self> {
        Ok(Function::from_data_types(DataType::Null, DataType::Any))
    }
}

impl InjectInto<DataType> for Function {
    type Injection = Base<Self, DataType>;
    fn inject_into(&self, other: &DataType) -> injection::Result<Base<Self, DataType>> {
        injection::From(self.clone()).into(other.clone())
    }
}

impl From<value::Function> for Function {
    fn from(value: value::Function) -> Self {
        Function::from(((*value).domain(), (*value).co_domain()))
    }
}

#[allow(clippy::derive_hash_xor_eq)]
/// DataType definition
/// Vaguely related to: https://docs.rs/sqlparser/0.25.0/sqlparser/ast/enum.DataType.html
#[derive(Debug, Clone, Hash)]
pub enum DataType {
    /// This is an empty type no data can be of this type, same as an empty Union (Union(A,Null) ~ A)
    Null,
    /// This is a type with one value (e.g. NA) same as an empty Struct (Struct(A,Null) ~ Null and Struct(A,Unit) ~ A)
    Unit(Unit),
    /// Boolean
    Boolean(Boolean),
    /// Integer
    Integer(Integer),
    /// Enum with named values
    Enum(Enum),
    /// Floating point value
    Float(Float),
    /// Text
    Text(Text),
    /// Array of byte
    Bytes(Bytes),
    /// Struct (not used for now)
    Struct(Struct),
    /// Union (not used for now)
    Union(Union),
    /// Value with potential missing values (not used for now)
    Optional(Optional),
    /// Repeated value with order (not used for now)
    List(List),
    /// Repeated value without order (not used for now)
    Set(Set),
    /// Multidimensional array (not used for now)
    Array(Array),
    /// Day
    Date(Date),
    /// Time of day
    Time(Time),
    /// Date and Time
    DateTime(DateTime),
    /// Difference between date and time
    Duration(Duration),
    /// Reference
    Id(Id),
    /// Function type (not used for now) Function([A,B]) ~ A -> B
    /// Function([A,B,C]) ~ A -> B -> C ~ (A,B) -> C
    Function(Function),
    /// Nothing is known about a data
    Any,
}

impl DataType {
    /// Return the default datatype of the same variant
    pub fn default(&self) -> DataType {
        match self {
            DataType::Unit(_) => DataType::from(Unit::default()),
            DataType::Boolean(_) => DataType::from(Boolean::default()),
            DataType::Integer(_) => DataType::from(Integer::default()),
            DataType::Float(_) => DataType::from(Float::default()),
            DataType::Text(_) => DataType::from(Text::default()),
            DataType::Bytes(_) => DataType::from(Bytes::default()),
            DataType::Date(_) => DataType::from(Date::default()),
            DataType::Time(_) => DataType::from(Time::default()),
            DataType::DateTime(_) => DataType::from(DateTime::default()),
            DataType::Duration(_) => DataType::from(Duration::default()),
            DataType::Id(_) => DataType::from(Id::default()),
            _ => self.clone(),
        }
    }

    /// Return a super data_type where both types can map into
    pub fn into_common_super_variant(
        left: &DataType,
        right: &DataType,
    ) -> Result<(DataType, DataType)> {
        match (left.into_variant(right), right.into_variant(left)) {
            (Ok(l), Ok(r)) => {
                let l_into_left = left.maximal_superset().and_then(|t| l.into_data_type(&t));
                if l_into_left.map_or(false, |t| &t == left) {
                    Ok((l, right.clone()))
                } else {
                    Ok((left.clone(), r))
                }
            }
            (Ok(l), Err(_)) => Ok((l, right.clone())),
            (Err(_), Ok(r)) => Ok((left.clone(), r)),
            (Err(_), Err(_)) => Err(Error::other("No common variant")),
        }
    }

    // Return a sub data_type where both types can map into
    pub fn into_common_sub_variant(
        left: &DataType,
        right: &DataType,
    ) -> Result<(DataType, DataType)> {
        match (left.into_variant(right), right.into_variant(left)) {
            (Ok(l), Ok(r)) => {
                let l_into_left = left.minimal_subset().and_then(|t| l.into_data_type(&t));
                if l_into_left.map_or(false, |t| &t == left) {
                    Ok((l, right.clone()))
                } else {
                    Ok((left.clone(), r))
                }
            }
            _ => Err(Error::other("No common variant")),
        }
    }
    // TODO This could be implemented with a visitor (it would not fail on cyclic cases)
    /// Produce a Hierarchy of subtypes to access them in a smart way (unambiguous prefix can be omited)
    fn hierarchy(&self) -> Hierarchy<&DataType> {
        for_all_variants!(
            self,
            x,
            x.hierarchy(),
            [Struct, Union],
            Hierarchy::from([(Vec::<&str>::new(), self)])
        )
    }
}

impl Variant for DataType {
    type Element = value::Value;

    fn contains(&self, element: &Self::Element) -> bool {
        match (self, element) {
            // If self and other are from the same variant
            (DataType::Null, _) => false, // Any element of self is also in other
            (DataType::Unit(_), value::Value::Unit(_)) => true,
            (DataType::Boolean(s), value::Value::Boolean(e)) => s.contains(e),
            (DataType::Integer(s), value::Value::Integer(e)) => s.contains(e),
            (DataType::Enum(s), value::Value::Enum(e)) => s.contains(e),
            (DataType::Float(s), value::Value::Float(e)) => s.contains(e),
            (DataType::Text(s), value::Value::Text(e)) => s.contains(e),
            (DataType::Bytes(_), value::Value::Bytes(_)) => true,
            (DataType::Struct(s), value::Value::Struct(e)) => s.contains(e),
            (DataType::Union(s), value::Value::Union(e)) => s.contains(e),
            (DataType::Optional(s), value::Value::Optional(e)) => s.contains(e),
            (DataType::List(s), value::Value::List(e)) => s.contains(e),
            (DataType::Set(s), value::Value::Set(e)) => s.contains(e),
            (DataType::Array(s), value::Value::Array(e)) => s.contains(e),
            (DataType::Date(s), value::Value::Date(e)) => s.contains(e),
            (DataType::Time(s), value::Value::Time(e)) => s.contains(e),
            (DataType::DateTime(s), value::Value::DateTime(e)) => s.contains(e),
            (DataType::Duration(s), value::Value::Duration(e)) => s.contains(e),
            (DataType::Id(s), value::Value::Id(e)) => s.contains(e),
            (DataType::Function(s), value::Value::Function(e)) => s.contains(e),
            (DataType::Any, _) => true,
            (s, e) => s
                .clone()
                .into_data_type(&e.data_type())
                .map_or(false, |s| s.contains(e)),
        }
    }

    fn is_subset_of(&self, other: &Self) -> bool {
        // If self and other are from the same variant
        for_all_variant_pairs!(
            self,
            other,
            s,
            o,
            s.is_subset_of(o),
            [
                Boolean, Integer, Enum, Float, Text, Struct, Union, Optional, List, Set, Array,
                Date, Time, DateTime, Duration, Id, Function
            ],
            {
                match (self, other) {
                    // If self and other are from different variants
                    (DataType::Null, _) => true, // Any element of self is also in other
                    (DataType::Unit(_), DataType::Unit(_))
                    | (DataType::Unit(_), DataType::Optional(_)) => true,
                    (DataType::Bytes(_), DataType::Bytes(_)) => true,
                    (_, DataType::Any) => true,
                    (DataType::Any, _) => false,
                    (s, o) => s
                        .clone()
                        .into_data_type(o)
                        .map_or(false, |s| s.is_subset_of(o)),
                }
            }
        )
    }

    fn super_union(&self, other: &Self) -> Result<Self> {
        for_all_variant_pairs!(
            self,
            other,
            s,
            o,
            Ok(DataType::from(s.super_union(o)?)),
            [
                Boolean, Integer, Enum, Float, Text, Struct, Union, Optional, List, Set, Array,
                Date, Time, DateTime, Duration, Id, Function
            ],
            {
                match (self, other) {
                    (DataType::Null, o) => Ok(o.clone()),
                    (s, DataType::Null) => Ok(s.clone()),
                    (DataType::Unit(_), DataType::Unit(_)) => Ok(DataType::from(Unit)),
                    (DataType::Bytes(_), DataType::Bytes(_)) => Ok(DataType::from(Bytes)),
                    (DataType::Any, _) => Ok(DataType::Any),
                    (_, DataType::Any) => Ok(DataType::Any),
                    // If self and other are from different variants
                    (s, o) => DataType::into_common_super_variant(s, o)
                        .and_then(|(s, o)| s.super_union(&o))
                        .or(Ok(DataType::Any)),
                }
            }
        )
    }

    fn super_intersection(&self, other: &Self) -> Result<Self> {
        for_all_variant_pairs!(
            self,
            other,
            s,
            o,
            Ok(DataType::from(s.super_intersection(o)?)),
            [
                Boolean, Integer, Enum, Float, Text, Struct, Union, Optional, List, Set, Array,
                Date, Time, DateTime, Duration, Id, Function
            ],
            {
                match (self, other) {
                    (DataType::Null, _) => Ok(DataType::Null),
                    (_, DataType::Null) => Ok(DataType::Null),
                    (DataType::Unit(_), DataType::Unit(_)) => Ok(DataType::unit()),
                    (DataType::Bytes(_), DataType::Bytes(_)) => Ok(DataType::bytes()),
                    (DataType::Any, o) => Ok(o.clone()),
                    (s, DataType::Any) => Ok(s.clone()),
                    (
                        DataType::Optional(Optional { data_type: l }),
                        DataType::Optional(Optional { data_type: r }),
                    ) => DataType::super_intersection(l.as_ref(), r.as_ref()),
                    (DataType::Optional(Optional { data_type: l }), _) => {
                        DataType::super_intersection(l.as_ref(), other)
                    }
                    (_, DataType::Optional(Optional { data_type: r })) => {
                        DataType::super_intersection(self, r.as_ref())
                    }
                    // If self and other are from different variants
                    (s, o) => DataType::into_common_sub_variant(s, o)
                        .or(DataType::into_common_super_variant(s, o))
                        .and_then(|(s, o)| s.super_intersection(&o))
                        .or(Ok(DataType::Any)),
                }
            }
        )
    }

    fn minimal_subset(&self) -> Result<Self> {
        for_all_variants!(
            self,
            x,
            Ok(x.minimal_subset()?.into()),
            [
                Unit, Boolean, Integer, Enum, Float, Text, Bytes, Struct, Union, Optional, List,
                Set, Array, Date, Time, DateTime, Duration, Id, Function
            ],
            Ok(DataType::Null)
        )
    }

    fn maximal_superset(&self) -> Result<Self> {
        for_all_variants!(
            self,
            x,
            Ok(x.maximal_superset()?.into()),
            [
                Unit, Boolean, Integer, Enum, Float, Text, Bytes, Struct, Union, Optional, List,
                Set, Array, Date, Time, DateTime, Duration, Id, Function
            ],
            Ok(DataType::Any)
        )
    }
}

impl InjectInto<DataType> for DataType {
    type Injection = Base<Self, DataType>;
    fn inject_into(&self, other: &DataType) -> injection::Result<Base<Self, DataType>> {
        injection::From(self.clone()).into(other.clone())
    }
}

macro_rules! impl_from {
    ( $Variant:ident ) => {
        impl From<$Variant> for DataType {
            fn from(x: $Variant) -> DataType {
                DataType::$Variant(x)
            }
        }
    };
}

macro_rules! impl_conversions {
    ( $Variant:ident ) => {
        impl_from!($Variant);

        impl TryFrom<DataType> for $Variant {
            type Error = Error;
            fn try_from(x: DataType) -> Result<Self> {
                match x {
                    DataType::$Variant(t) => Ok(t),
                    _ => Err(Error::invalid_conversion(x, stringify!($Variant))),
                }
            }
        }

        impl TryFrom<&DataType> for $Variant {
            type Error = Error;
            fn try_from(x: &DataType) -> Result<Self> {
                match x {
                    DataType::$Variant(t) => Ok(t.clone()),
                    _ => Err(Error::invalid_conversion(x, stringify!($Variant))),
                }
            }
        }
    };
}

impl_conversions!(Unit);
impl_conversions!(Boolean);
impl_conversions!(Integer);
impl_conversions!(Enum);
impl_conversions!(Float);
impl_conversions!(Text);
impl_conversions!(Bytes);
impl_conversions!(Struct);
impl_conversions!(Union);
impl_from!(Optional);
impl_conversions!(List);
impl_conversions!(Set);
impl_conversions!(Array);
impl_conversions!(Date);
impl_conversions!(Time);
impl_conversions!(DateTime);
impl_conversions!(Duration);
impl_conversions!(Id);
impl_conversions!(Function);

macro_rules! impl_into_values {
    ( $Variant:ident ) => {
        impl TryInto<Vec<Value>> for $Variant {
            type Error = Error;

            fn try_into(self) -> Result<Vec<Value>> {
                if self.all_values() {
                    Ok(self.into_iter().map(|[v, _]| Value::from(v)).collect())
                } else {
                    Err(Error::invalid_conversion(
                        stringify!($Variant),
                        "Vec<Value>",
                    ))
                }
            }
        }
    };
}

impl_into_values!(Boolean);
impl_into_values!(Integer);
impl_into_values!(Float);
impl_into_values!(Text);
impl_into_values!(Date);
impl_into_values!(Time);
impl_into_values!(DateTime);
impl_into_values!(Duration);

impl TryInto<Vec<Value>> for DataType {
    type Error = Error;

    fn try_into(self) -> Result<Vec<Value>> {
        match self {
            DataType::Boolean(b) => b.try_into(),
            DataType::Integer(i) => i.try_into(),
            DataType::Float(f) => f.try_into(),
            DataType::Text(t) => t.try_into(),
            DataType::Date(d) => d.try_into(),
            DataType::Time(t) => t.try_into(),
            DataType::DateTime(d) => d.try_into(),
            DataType::Duration(d) => d.try_into(),
            _ => Err(Error::invalid_conversion(
                stringify!($Variant),
                "Vec<Value>",
            )),
        }
    }
}

impl cmp::PartialEq for DataType {
    fn eq(&self, other: &Self) -> bool {
        matches!(self.partial_cmp(other), Some(cmp::Ordering::Equal))
    }
}

// TODO make sure this is the case
impl cmp::Eq for DataType {}

impl cmp::PartialOrd for DataType {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        partial_cmp(self, other)
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for_all_variants!(
            self,
            x,
            write!(f, "{}", x),
            [
                Unit, Boolean, Integer, Enum, Float, Text, Bytes, Struct, Union, Optional, List,
                Set, Array, Date, Time, DateTime, Duration, Id, Function
            ],
            {
                match self {
                    DataType::Null => write!(f, "null"),
                    DataType::Any => write!(f, "any"),
                    _ => write!(f, ""),
                }
            }
        )
    }
}

macro_rules! impl_default_builder {
    ( $Variant:ident ) => {
        impl DataType {
            paste! {
                pub fn [<$Variant:snake>]() -> DataType { DataType::from($Variant::default()) }
            }
        }
    };
}

macro_rules! impl_interval_builders {
    ( $Intervals:ident, $Bound:ty ) => {
        impl DataType {
            paste! {
                pub fn [<$Intervals:snake>]() -> DataType { DataType::from($Intervals::default()) }
                pub fn [<$Intervals:snake _interval>](start: $Bound, end: $Bound) -> DataType { DataType::from($Intervals::from_interval(start, end)) }
                pub fn [<$Intervals:snake _min>](min: $Bound) -> DataType { DataType::from($Intervals::from_min(min)) }
                pub fn [<$Intervals:snake _max>](max: $Bound) -> DataType { DataType::from($Intervals::from_max(max)) }
                pub fn [<$Intervals:snake _value>](value: $Bound) -> DataType { DataType::from($Intervals::from_value(value)) }
                pub fn [<$Intervals:snake _values>]<V: AsRef<[$Bound]>>(values: V) -> DataType { DataType::from($Intervals::from_values(values)) }
                pub fn [<$Intervals:snake _range>]<R: ops::RangeBounds<$Bound>>(range: R) -> DataType { DataType::from($Intervals::from_range(range)) }
            }
        }
    };
}

// Some builders
impl_default_builder!(Unit);
impl_interval_builders!(Boolean, bool);
impl_interval_builders!(Integer, i64);
impl_interval_builders!(Float, f64);
impl_interval_builders!(Text, String);
impl_default_builder!(Bytes);
impl_interval_builders!(Date, chrono::NaiveDate);
impl_interval_builders!(Time, chrono::NaiveTime);
impl_interval_builders!(DateTime, chrono::NaiveDateTime);
impl_interval_builders!(Duration, chrono::Duration);
impl_default_builder!(Id);

/// Implements a few more builders
impl DataType {
    pub fn enumeration<S: Clone + Into<String>, V: AsRef<[S]>>(values: V) -> DataType {
        DataType::from(Enum::from(values.as_ref()))
    }

    pub fn structured<
        S: Clone + Into<String>,
        T: Clone + Into<Rc<DataType>>,
        F: AsRef<[(S, T)]>,
    >(
        fields: F,
    ) -> DataType {
        DataType::from(Struct::from(fields.as_ref()))
    }

    pub fn structured_from_data_types<F: AsRef<[DataType]>>(fields: F) -> DataType {
        DataType::from(Struct::from_data_types(fields))
    }

    pub fn union<S: Clone + Into<String>, F: AsRef<[(S, DataType)]>>(fields: F) -> DataType {
        DataType::from(Union::from(fields.as_ref()))
    }

    pub fn optional(data_type: DataType) -> DataType {
        DataType::from(Optional::from(data_type))
    }

    pub fn list(data_type: DataType, min_size: usize, max_size: usize) -> DataType {
        DataType::from(List::from((
            data_type,
            Integer::from_interval(min_size as i64, max_size as i64),
        )))
    }

    pub fn set(data_type: DataType, min_size: usize, max_size: usize) -> DataType {
        DataType::from(Set::from((
            data_type,
            Integer::from_interval(min_size as i64, max_size as i64),
        )))
    }

    pub fn array<S: AsRef<[usize]>>(data_type: DataType, shape: &[usize]) -> DataType {
        DataType::from(Array::from((data_type, shape)))
    }

    pub fn function(domain: DataType, co_domain: DataType) -> DataType {
        DataType::from(Function::from((domain, co_domain)))
    }
}

impl<P: Path> Index<P> for DataType {
    type Output = DataType;

    fn index(&self, index: P) -> &Self::Output {
        self.hierarchy()[index]
    }
}

// Some more conversions

/// DataType -> (Intervals<A>)
impl<A: Bound> TryFrom<DataType> for (Intervals<A>,)
where
    Intervals<A>: TryFrom<DataType, Error = Error>,
{
    type Error = Error;
    fn try_from(value: DataType) -> Result<Self> {
        let intervals: Intervals<A> = value.try_into()?;
        Ok((intervals,))
    }
}

/// DataType -> (Intervals<A>, Intervals<B>)
impl<A: Bound, B: Bound> TryFrom<DataType> for (Intervals<A>, Intervals<B>)
where
    Intervals<A>: TryFrom<DataType, Error = Error>,
    Intervals<B>: TryFrom<DataType, Error = Error>,
{
    type Error = Error;
    fn try_from(value: DataType) -> Result<Self> {
        let structured: Struct = value.try_into()?;
        let left: Intervals<A> = (*structured.data_type("0")).clone().try_into()?;
        let right: Intervals<B> = (*structured.data_type("1")).clone().try_into()?;
        Ok((left, right))
    }
}

/// DataType -> (Intervals<A>, Intervals<B>, Intervals<C>)
impl<A: Bound, B: Bound, C: Bound> TryFrom<DataType> for (Intervals<A>, Intervals<B>, Intervals<C>)
where
    Intervals<A>: TryFrom<DataType, Error = Error>,
    Intervals<B>: TryFrom<DataType, Error = Error>,
    Intervals<C>: TryFrom<DataType, Error = Error>,
{
    type Error = Error;
    fn try_from(value: DataType) -> Result<Self> {
        let structured: Struct = value.try_into()?;
        let inter_a: Intervals<A> = (*structured.data_type("0")).clone().try_into()?;
        let inter_b: Intervals<B> = (*structured.data_type("1")).clone().try_into()?;
        let inter_c: Intervals<C> = (*structured.data_type("2")).clone().try_into()?;
        Ok((inter_a, inter_b, inter_c))
    }
}

/// (Intervals<A>) -> DataType
impl<A: Bound> From<(Intervals<A>,)> for DataType
where
    Intervals<A>: Into<DataType>,
{
    fn from(value: (Intervals<A>,)) -> Self {
        value.0.into()
    }
}

/// (Intervals<A>, Intervals<B>) -> DataType
impl<A: Bound, B: Bound> From<(Intervals<A>, Intervals<B>)> for DataType
where
    Intervals<A>: Into<DataType>,
    Intervals<B>: Into<DataType>,
{
    fn from(value: (Intervals<A>, Intervals<B>)) -> Self {
        DataType::from(Struct::from_data_types(&[value.0.into(), value.1.into()]))
    }
}

/// (Intervals<A>, Intervals<B>, Intervals<C>) -> DataType
impl<A: Bound, B: Bound, C: Bound> From<(Intervals<A>, Intervals<B>, Intervals<C>)> for DataType
where
    Intervals<A>: Into<DataType>,
    Intervals<B>: Into<DataType>,
    Intervals<C>: Into<DataType>,
{
    fn from(value: (Intervals<A>, Intervals<B>, Intervals<C>)) -> Self {
        DataType::from(Struct::from_data_types(&[
            value.0.into(),
            value.1.into(),
            value.2.into(),
        ]))
    }
}

/*
Function has many implementations:
(int, int), (float, float)
Test inclusion ine each successuvely.
If not included go to next
If included inject into the type
Implement this in functions
 */

// DataType algebra

impl And<DataType> for DataType {
    type Product = DataType;
    fn and(self, other: DataType) -> Self::Product {
        // Simplify in the case of struct and Unit
        match self {
            DataType::Null => DataType::Null,
            //DataType::Unit(_u) => other, // TODO: reactivate ?
            DataType::Struct(s) => s.and(other).into(),
            s => Struct::from_data_type(s).and(other).into(),
        }
    }
}

impl<S: Into<String>, T: Into<Rc<DataType>>> And<(S, T)> for DataType {
    type Product = DataType;
    fn and(self, other: (S, T)) -> Self::Product {
        self.and(DataType::from(Struct::from(other)))
    }
}

impl<T> ops::BitAnd<T> for DataType
where
    Self: And<T>,
{
    type Output = <Self as And<T>>::Product;

    fn bitand(self, rhs: T) -> Self::Output {
        self.and(rhs)
    }
}

impl Or<DataType> for DataType {
    type Sum = DataType;
    fn or(self, other: DataType) -> Self::Sum {
        match (self, other) {
            (DataType::Null, d) => d,
            (DataType::Unit(_), DataType::Unit(_)) => DataType::unit(),
            (DataType::Unit(u), d) | (d, DataType::Unit(u)) => u.or(d).into(),
            (DataType::Optional(o), d) | (d, DataType::Optional(o)) => o.or(d).into(),
            (s, o) => Union::from_data_type(s).or(o).into(),
        }
    }
}

impl<S: Into<String>, T: Into<Rc<DataType>>> Or<(S, T)> for DataType {
    type Sum = DataType;
    fn or(self, other: (S, T)) -> Self::Sum {
        self.or(DataType::from(Union::from(other)))
    }
}

impl<T> ops::BitOr<T> for DataType
where
    Self: Or<T>,
{
    type Output = <Self as Or<T>>::Sum;

    fn bitor(self, rhs: T) -> Self::Output {
        self.or(rhs)
    }
}

/// Implements a few more builders
impl DataType {
    /// Sum type
    pub fn sum<I: IntoIterator<Item = DataType>>(data_types: I) -> DataType {
        data_types.into_iter().fold(DataType::Null, |s, t| s.or(t))
    }

    /// Product type
    pub fn product<I: IntoIterator<Item = DataType>>(data_types: I) -> DataType {
        data_types
            .into_iter()
            .fold(DataType::unit(), |s, t| s.and(t))
    }
}

// A few useful ad-hoc conversions

impl FromIterator<value::Value> for DataType {
    fn from_iter<T: IntoIterator<Item = value::Value>>(iter: T) -> Self {
        // Look at the first element and assume all the others will have the same type
        iter.into_iter().fold(DataType::Null, |data_type, value| {
            data_type
                .super_union(&value.into())
                .unwrap_or(DataType::Any)
        })
    }
}

/// Implement the Acceptor trait
impl<'a> Acceptor<'a> for DataType {
    fn dependencies(&'a self) -> visitor::Dependencies<'a, Self> {
        match self {
            DataType::Struct(s) => s.fields.iter().map(|(_, t)| t.as_ref()).collect(),
            DataType::Union(u) => u.fields.iter().map(|(_, t)| t.as_ref()).collect(),
            DataType::Optional(o) => visitor::Dependencies::from([o.data_type.as_ref()]),
            DataType::List(l) => visitor::Dependencies::from([l.data_type.as_ref()]),
            DataType::Set(s) => visitor::Dependencies::from([s.data_type.as_ref()]),
            DataType::Array(a) => visitor::Dependencies::from([a.data_type.as_ref()]),
            DataType::Function(f) => {
                visitor::Dependencies::from([f.domain.as_ref(), f.co_domain.as_ref()])
            }
            _ => visitor::Dependencies::empty(),
        }
    }
}

// TODO Write tests for all types
#[cfg(test)]
mod tests {
    use std::convert::TryFrom;

    use statrs::statistics::Data;

    use super::*;

    #[test]
    fn test_null() {
        // All text
        let null = DataType::Null;
        println!("type = {}", null);
        let a_type = DataType::float_interval(-1., 3.);
        assert!(null <= a_type);
        assert!(!(a_type <= null));
        assert!(a_type <= a_type);
        assert!(null <= null);
        assert!(a_type == a_type);
        assert!(null == null);
        assert!(null <= DataType::Any);
    }

    #[test]
    fn test_text() {
        // All text
        let all_text = DataType::text();
        println!("type = {}", all_text);
        match &all_text {
            DataType::Text(t) => assert_eq!(t, &Text::full()),
            _ => (),
        }
        let some_text = DataType::from(Text::from_values(
            [String::from("Hello"), String::from("World")].as_ref(),
        ));
        println!("type = {}", some_text);
        match &some_text {
            DataType::Text(t) => assert_ne!(t, &Text::full()),
            _ => (),
        }
        assert!(some_text.is_subset_of(&all_text));
        assert!(some_text.is_subset_of(&some_text));
        assert!(all_text.is_subset_of(&all_text));
        assert!(some_text == some_text);
        assert!(all_text == all_text);
        assert!(!all_text.is_subset_of(&some_text));
        // Test some custom text conversions
        println!(
            "{}",
            DataType::integer_values(&[0, 2, 1, 3])
                .into_data_type(&DataType::text())
                .unwrap()
        );
        assert_eq!(
            DataType::integer_values(&[0, 2, 1, 3])
                .into_data_type(&DataType::text())
                .unwrap(),
            DataType::from(Text::from_values([
                String::from("0"),
                String::from("1"),
                String::from("2"),
                String::from("3")
            ]))
        )
    }

    #[test]
    fn test_build() {
        let data_type = DataType::structured(&[
            ("i", DataType::integer()),
            ("j", DataType::from(Integer::from_interval(5, 20))),
            (
                "k",
                DataType::from(Integer::from_intervals(&[
                    [Bound::min(), 2],
                    [3, 4],
                    [7, Bound::max()],
                ])),
            ),
            ("l", DataType::from(Integer::from_values(&[5, -2, 20]))),
        ]);
        println!("type = {}", data_type);
        assert_eq!(
            format!("{}", data_type),
            "struct{i: int, j: int[5 20], k: int(-∞, 2]∪[3 4]∪[7, +∞), l: int{-2, 5, 20}}",
        )
    }

    #[test]
    fn test_equalities() {
        let empty_interval = DataType::from(Intervals::<f64>::empty());
        println!(
            "{} == {} is {}",
            empty_interval,
            DataType::Null,
            empty_interval == DataType::Null
        );
        println!(
            "{} == {} is {}",
            DataType::Null,
            empty_interval,
            DataType::Null == empty_interval
        );
        println!(
            "{}.cmp({}) = {:?}",
            empty_interval,
            DataType::Null,
            empty_interval.partial_cmp(&DataType::Null)
        );
        println!(
            "{}.cmp({}) = {:#?}",
            DataType::Null,
            empty_interval,
            DataType::Null.partial_cmp(&empty_interval)
        );
        assert_eq!(empty_interval, DataType::Null);
        assert_eq!(DataType::Null, empty_interval);
    }

    #[test]
    fn test_inequalities() {
        let empty_interval = DataType::from(Intervals::<f64>::empty());
        assert!(empty_interval <= DataType::text());
        println!(
            "{} <= {} is {}",
            empty_interval,
            DataType::Null,
            empty_interval <= DataType::Null
        );
        println!(
            "{} <= {} is {}",
            DataType::Null,
            empty_interval,
            DataType::Null <= empty_interval
        );
        println!(
            "{} <= {} is {}",
            empty_interval,
            DataType::text(),
            empty_interval <= DataType::text()
        );
        println!(
            "{} <= {} is {}",
            DataType::Null,
            DataType::text(),
            DataType::Null <= DataType::text()
        );
        assert!(DataType::Null <= DataType::text());
        assert!(DataType::text() <= DataType::text());
        assert!(
            DataType::from(Text::from_values([
                String::from("Qrlew"),
                String::from("Code")
            ])) <= DataType::text()
        );
        println!(
            "{} <= {} is {}",
            DataType::text(),
            empty_interval,
            DataType::text() <= empty_interval
        );
        println!(
            "{} <= {} is {}",
            DataType::text(),
            DataType::Null,
            DataType::text() <= DataType::Null
        );
        assert!(!(DataType::text() <= empty_interval));
        assert!(!(DataType::text() <= DataType::Null));
        println!(
            "{} <= {} is {}",
            DataType::float(),
            DataType::optional(DataType::float()),
            DataType::float() <= DataType::optional(DataType::float())
        );
        assert!(DataType::float() <= DataType::optional(DataType::float()));
        println!(
            "{} <= {} is {}",
            DataType::unit(),
            DataType::optional(DataType::float()),
            DataType::unit() <= DataType::optional(DataType::float())
        );
        assert!(DataType::unit() <= DataType::optional(DataType::float()));
    }

    #[test]
    fn test_string_key() {
        let data_type = DataType::structured(&[
            ("i", DataType::integer()),
            ("j", DataType::from(Integer::from_interval(5, 20))),
            (
                "k",
                DataType::from(Integer::from_intervals(&[
                    [Bound::min(), 2],
                    [3, 4],
                    [7, Bound::max()],
                ])),
            ),
            ("l", DataType::from(Integer::from_values(&[5, -2, 20]))),
        ]);
        println!("type = {}", data_type);
    }

    #[test]
    fn test_match() {
        let data_type = DataType::from(Integer::from_interval(5, 20));
        println!("type = {}", data_type);
        if let DataType::Integer(intervals) = data_type.clone() {
            println!("min = {}", intervals.min().unwrap());
        }
        if let DataType::Integer(i) = data_type {
            println!("max = {}", i.max().unwrap());
        }
    }

    #[test]
    fn test_partial_ord() {
        let a = DataType::integer_interval(0, 10);
        let b = DataType::float_interval(0., 10.);
        println!("{} <= {} is {}", a, b, a <= b);
        assert!(a <= b);

        let a = DataType::integer_values(&[1, 10, 20]);
        let b = DataType::float_interval(0., 10.);
        println!("{} <= {} is {}", a, b, a <= b);
        assert!(!(a <= b));

        let a = DataType::integer_values(&[1, 10, 20]);
        let b = DataType::float_interval(0., 30.);
        println!("{} <= {} is {}", a, b, a <= b);
        assert!(a <= b);

        // TODO Fix this
        let date_a = chrono::NaiveDate::from_isoywd_opt(2022, 10, chrono::Weekday::Mon).unwrap();
        let date_b = date_a + chrono::Duration::days(10);
        let a = DataType::date_interval(date_a, date_b);
        let b = DataType::date_time_interval(
            date_a.and_hms_opt(0, 0, 0).unwrap(),
            date_b.and_hms_opt(0, 0, 0).unwrap(),
        );
        println!("{} <= {} is {}", a, b, a <= b);
        assert!(a <= b);
    }

    /// Utility function
    fn print_conversions(a: &DataType, b: &DataType) {
        let (ca, cb) = if let Ok((ca, cb)) = DataType::into_common_super_variant(a, b) {
            (ca, cb)
        } else {
            (DataType::Null, DataType::Null)
        };
        println!(
            "a = {}, b = {}, a.into(b) = {}, b.into(a) = {}, unified(a,b) = ({},{})",
            a,
            b,
            a.clone().into_data_type(b).unwrap_or(DataType::Null),
            b.clone().into_data_type(a).unwrap_or(DataType::Null),
            ca,
            cb,
        );
    }

    #[test]
    fn test_unify() {
        let i = DataType::integer();
        let f = DataType::float();
        let t = DataType::text();
        let iv = DataType::integer_interval(5, 10);
        let fv = DataType::float_values(&[5.7, 10.1]);
        let d = DataType::date();
        print_conversions(&i, &f);
        print_conversions(&i, &t);
        print_conversions(&f, &t);
        print_conversions(&i, &iv);
        print_conversions(&t, &fv);
        print_conversions(&i, &d);
    }

    #[test]
    fn test_unify_null_struct() {
        let n = DataType::Null;
        let s = DataType::from(Struct::from_data_types(&[
            DataType::float(),
            DataType::integer(),
        ]));
        println!("{} <= {} = {}", &n, &s, n.is_subset_of(&s));
        println!("{} >= {} = {}", &n, &s, s.is_subset_of(&n));
        print_conversions(&n, &s);
    }

    #[test]
    fn test_display() {
        let data_type = DataType::function(
            DataType::structured(&[
                ("name", DataType::list(DataType::text(), 100, 100)),
                ("age", DataType::optional(DataType::integer())),
            ]),
            DataType::float(),
        );
        println!("type = {}", data_type);
    }

    #[test]
    #[should_panic]
    fn test_inconsistent_type() {
        let data_type = DataType::from(Float::from_interval(15.0, 7.0));
        println!("type = {}", data_type);
    }

    #[test]
    fn test_struct_use() {
        let data_type = DataType::from(Struct::from_data_types(&[
            DataType::integer(),
            DataType::date(),
        ]));
        println!("type = {}", data_type);
        let structured: Struct = Struct::try_from(data_type).unwrap();
        assert_eq!(
            structured
                .fields()
                .iter()
                .map(|(s, _)| s.clone())
                .collect_vec(),
            vec!["0".to_string(), "1".to_string()]
        )
    }

    #[test]
    fn test_struct_inclusion() {
        let type_a = DataType::from(Struct::from_data_types(&[
            DataType::float_interval(0., 2.),
            DataType::float_interval(-3., 3.),
            DataType::float_values(&[-5., 5.]),
        ]));
        let type_b = DataType::from(Struct::from_data_types(&[
            DataType::float(),
            DataType::float_interval(-3., 3.),
        ]));
        let type_c = DataType::from(Struct::from_data_types(&[
            DataType::float(),
            DataType::float(),
        ]));
        println!("a = {}, b = {}, c = {}", &type_a, &type_b, &type_c);
        assert!(type_a.is_subset_of(&type_b));
        assert!(type_a.is_subset_of(&type_c));
        assert!(type_b.is_subset_of(&type_c));
    }

    #[test]
    fn test_struct_any_inclusion() {
        let struct_any = DataType::Any & DataType::Any;
        let struct_float = DataType::float_values([1., 2., 3.]) & DataType::float();
        println!("struct_any = {}", struct_any);
        println!("struct_float = {}", struct_float);
        println!(
            "struct_float ⊂ struct_any = {}",
            struct_float.is_subset_of(&struct_any)
        );
        assert!(struct_float.is_subset_of(&struct_any));
    }

    #[test]
    fn test_optional_inclusion() {
        let typ = DataType::float();
        let opt = DataType::optional(typ.clone());
        println!("typ = {}", typ);
        println!("opt = {}", opt);
        println!(
            "typ.clone().into_data_type(opt) {}",
            typ.clone().into_data_type(&opt).unwrap()
        );
        println!("typ ⊂ opt = {}", typ.is_subset_of(&opt));
        // assert!(typ.is_subset_of(&opt));
        println!("opt ⊄ typ = {}", opt.is_subset_of(&typ));
        assert!(!opt.is_subset_of(&typ));
    }

    #[test]
    fn test_struct_and() {
        let a = Struct::default()
            .and(("a", DataType::float_interval(1., 3.)))
            .and(("a", DataType::integer_interval(-10, 10)));
        println!("a = {a}");
        assert_eq!(
            a,
            Struct::from_field("a", DataType::float_values([1., 2., 3.]))
        );

        let a = Struct::default()
            .and(DataType::float())
            .and(("a", DataType::integer_interval(-10, 10)))
            .and(DataType::float())
            .and(DataType::float())
            .and(DataType::float());
        let b = Struct::default()
            .and(("b", DataType::integer()))
            .and(("c", DataType::float()))
            .and(("d", DataType::float()))
            .and(("d", DataType::float()))
            .and(("a", DataType::float_interval(1., 3.)));
        println!("a = {a}");
        println!("b = {b}");

        // a and b
        let c = a.clone().and(b.clone());
        println!("\na and b = {c}");
        assert_eq!(
            c,
            Struct::default()
                .and(("0", DataType::float()))
                .and(("a", DataType::float_values([1., 2., 3.])))
                .and(("1", DataType::float()))
                .and(("2", DataType::float()))
                .and(("3", DataType::float()))
                .and(("b", DataType::integer()))
                .and(("c", DataType::float()))
                .and(("d", DataType::float()))
        );

        // a and unit
        let d = a.clone().and(DataType::unit());
        println!("\na and unit = {d}");
        assert_eq!(
            d,
            Struct::default()
                .and(("0", DataType::float()))
                .and(("a", DataType::integer_interval(-10, 10)))
                .and(("1", DataType::float()))
                .and(("2", DataType::float()))
                .and(("3", DataType::float()))
                .and(("4", DataType::unit()))
        );

        // a and DataType(b)
        let e = a.clone().and(DataType::Struct(b.clone()));
        println!("\na and b = {e}");
        assert_eq!(e.fields().len(), 8);
        assert_eq!(
            e,
            Struct::default()
                .and(("0", DataType::float()))
                .and(("a", DataType::integer_interval(1, 3)))
                .and(("1", DataType::float()))
                .and(("2", DataType::float()))
                .and(("3", DataType::float()))
                .and(("b", DataType::integer()))
                .and(("c", DataType::float()))
                .and(("d", DataType::float()))
        );

        //struct(table1: a) and b
        let f = DataType::structured([("table1", DataType::Struct(a.clone()))])
            .and(DataType::Struct(b.clone()));
        println!("\na and struct(table1: b) = {f}");
        assert_eq!(
            f,
            DataType::structured([
                (
                    "table1",
                    DataType::structured([
                        ("0", DataType::float()),
                        ("a", DataType::integer_interval(-10, 10)),
                        ("1", DataType::float()),
                        ("2", DataType::float()),
                        ("3", DataType::float())
                    ])
                ),
                ("b", DataType::integer()),
                ("c", DataType::float()),
                ("d", DataType::float()),
                ("a", DataType::float_interval(1., 3.))
            ])
        );

        //struct(table1: a) and struct(table1: b)
        let g = DataType::structured([("table1", DataType::Struct(a.clone()))]).and(
            DataType::structured([("table1", DataType::Struct(b.clone()))]),
        );
        println!("\nstruct(table1: a) and struct(table1: b) = {g}");
        assert_eq!(
            g,
            DataType::structured([(
                "table1",
                DataType::structured([
                    ("0", DataType::float()),
                    ("a", DataType::float_values([1., 2., 3.])),
                    ("1", DataType::float()),
                    ("2", DataType::float()),
                    ("3", DataType::float()),
                    ("b", DataType::integer()),
                    ("c", DataType::float()),
                    ("d", DataType::float()),
                ])
            )])
        );

        // struct(table1: a) and struct(table2: b)
        let h = DataType::structured([("table1", DataType::Struct(a))])
            .and(DataType::structured([("table2", DataType::Struct(b))]));
        println!("\nstruct(table1: a) and struct(table2: b) = {h}");
        assert_eq!(
            h,
            DataType::structured([
                (
                    "table1",
                    DataType::structured([
                        ("0", DataType::float()),
                        ("a", DataType::integer_interval(-10, 10)),
                        ("1", DataType::float()),
                        ("2", DataType::float()),
                        ("3", DataType::float())
                    ])
                ),
                (
                    "table2",
                    DataType::structured([
                        ("b", DataType::integer()),
                        ("c", DataType::float()),
                        ("d", DataType::float()),
                        ("a", DataType::float_interval(1., 3.))
                    ])
                )
            ])
        );
    }

    #[test]
    fn test_and() {
        let a = DataType::unit()
            .and(DataType::boolean())
            .and(DataType::boolean())
            .and(DataType::float());
        println!("a = {}", &a);
        let b = DataType::unit()
            & ("a", DataType::boolean())
            & DataType::unit()
            & a
            & ("c", DataType::boolean())
            & ("d", DataType::float());
        println!("b = {b}");
        assert_eq!(Struct::try_from(b).unwrap().fields.len(), 7);
    }

    #[test]
    fn test_index() {
        let dt = DataType::float();
        assert_eq!(dt[Vec::<String>::new()], dt);
        let dt1 = DataType::structured([("a", DataType::integer()), ("b", DataType::boolean())]);
        let dt2 = DataType::structured([("a", DataType::float()), ("c", DataType::integer())]);
        let dt = DataType::Null | ("table1", dt1.clone()) | ("table2", dt2.clone());
        assert_eq!(dt["table1"], dt1);
        assert_eq!(dt["table2"], dt2);
        assert_eq!(dt[["table1", "a"]], DataType::integer());
        assert_eq!(dt[["table2", "a"]], DataType::float());
        assert_eq!(dt["b"], DataType::boolean());
        assert_eq!(dt[["c"]], DataType::integer());

        let a = DataType::structured([
            ("a_0", DataType::integer_min(-10)),
            ("a_1", DataType::integer()),
        ]);
        let b = DataType::structured([
            ("b_0", DataType::float()),
            ("b_1", DataType::float_interval(0., 1.)),
        ]);
        let x = DataType::structured([("a", a.clone()), ("b", b)]);
        println!("{}", x);
        assert_eq!(x[["a"]], a);
        assert_eq!(x[["a", "a_1"]], DataType::integer());
    }

    #[test]
    fn test_union_or() {
        let a = Union::default()
            .or(DataType::float())
            .or(("a", DataType::integer()))
            .or(DataType::float())
            .or(DataType::float())
            .or(DataType::float());
        let b = Union::default()
            .or(("b", DataType::integer()))
            .or(("c", DataType::float()))
            .or(("d", DataType::float()))
            .or(("d", DataType::float()));
        let c = a.clone().or(b.clone());
        let d = a.clone().or(DataType::Null);
        let e = a.or(DataType::Union(b));
        println!("{c}");
        println!("{d}");
        println!("{e}");
        assert_eq!(e.fields().len(), 8);
    }

    #[test]
    fn test_union_unit() {
        let a = DataType::unit().or(DataType::float());
        println!("{:?}", a);

        let a = DataType::unit().and(DataType::float());
        println!("{:?}", a);

        let a = DataType::float().or(DataType::unit());
        println!("{:?}", a);

        let a = DataType::float().and(DataType::unit());
        println!("{:?}", a);
    }

    #[test]
    fn test_union_inclusion() {
        let type_a = DataType::float() & DataType::float();
        let type_b = DataType::integer() & DataType::integer();
        let union_c = Union::null().or(type_a.clone()).or(type_b.clone());
        let union_a = Union::from_field("0", type_a);
        let union_b = Union::from_field("1", type_b);
        println!("a = {}, b = {}, c = {}", &union_a, &union_b, &union_c);
        assert!(union_a.is_subset_of(&union_c));
        assert!(union_b.is_subset_of(&union_c));
    }

    #[test]
    fn test_or() {
        let a = DataType::Null
            .or(DataType::boolean())
            .or(DataType::boolean())
            .or(DataType::float());
        println!("a = {}", &a);

        let b = DataType::Null
            | ("a", DataType::boolean())
            | a
            | ("c", DataType::boolean())
            | ("d", DataType::float());
        println!("b = {b}");
        assert_eq!(Union::try_from(b).unwrap().fields.len(), 6);

        // unit | float
        assert_eq!(
            DataType::unit() | DataType::float(),
            DataType::optional(DataType::float())
        );

        // float | unit
        assert_eq!(
            DataType::float() | DataType::unit(),
            DataType::optional(DataType::float())
        );

        // unit | unit
        assert_eq!(DataType::unit() | DataType::unit(), DataType::unit());

        // option(float) | float
        assert_eq!(
            DataType::optional(DataType::float()) | DataType::float(),
            DataType::optional(
                Union::from_data_types(vec!(DataType::float(), DataType::float()).as_slice())
                    .into()
            )
        );

        // float | option(float)
        assert_eq!(
            DataType::float() | DataType::optional(DataType::float()),
            DataType::optional(
                Union::from_data_types(vec!(DataType::float(), DataType::float()).as_slice())
                    .into()
            )
        );

        // option(integer) | option(float)
        assert_eq!(
            DataType::optional(DataType::float()) | DataType::optional(DataType::float()),
            DataType::optional(
                Union::from_data_types(vec!(DataType::float(), DataType::float()).as_slice())
                    .into()
            )
        );
    }

    #[test]
    fn test_intersection() {
        let left = DataType::float_interval(1., 3.);
        let right = DataType::integer_interval(-10, 10);
        let inter = left.super_intersection(&right).unwrap();
        println!("{left} ∩ {right} = {inter}");
        assert_eq!(inter, DataType::integer_interval(1, 3));
        assert_eq!(inter, right.super_intersection(&left).unwrap());

        let left = DataType::integer_interval(0, 10);
        let right = DataType::float_interval(5., 12.);

        let intersection = left.super_intersection(&DataType::Any).unwrap();
        println!("left ∩ any = {}", intersection);
        assert_eq!(intersection, left);

        let intersection = right.super_intersection(&DataType::Any).unwrap();
        println!("right ∩ any = {}", intersection);
        assert_eq!(intersection, right);

        let intersection = left.super_intersection(&DataType::Null).unwrap();
        println!("left ∩ ∅ = {}", intersection);
        assert_eq!(intersection, DataType::Null);

        let intersection = right.super_intersection(&DataType::Null).unwrap();
        println!("right ∩ ∅ = {}", intersection);
        assert_eq!(intersection, DataType::Null);

        // int[0 10] ∩ float[5 12] = int{5}
        let intersection = left.super_intersection(&right).unwrap();
        println!("{} ∩ {} = {}", left, right, intersection);
        assert_eq!(intersection, DataType::integer_interval(5, 10));

        // int[0 10] ∩ float{5, 8} = int{5, 8}
        let left = DataType::integer_interval(0, 10);
        let right = DataType::float_values([5., 8.]);
        let intersection = left.super_intersection(&right).unwrap();
        println!("{} ∩ {} = {}", left, right, intersection);
        assert_eq!(intersection, DataType::integer_values([5, 8]));

        // optional(int[0 10]) ∩ float{5, 8} = int{5, 8}
        let left = DataType::optional(DataType::integer_interval(0, 10));
        let right = DataType::float_values([5., 8.]);
        let intersection = left.super_intersection(&right).unwrap();
        println!("{} ∩ {} = {}", left, right, intersection);
        assert_eq!(intersection, DataType::integer_values([5, 8]));

        // int[0 10] ∩ optional(float{5, 8}) = int{5, 8}
        let left = DataType::integer_interval(0, 10);
        let right = DataType::optional(DataType::float_values([5., 8.]));
        let intersection = left.super_intersection(&right).unwrap();
        println!("{} ∩ {} = {}", left, right, intersection);
        assert_eq!(intersection, DataType::integer_values([5, 8]));

        // optional(int[0 10]) ∩ optional(float{5, 8}) = optional(int{5, 8})
        let left = DataType::optional(DataType::integer_interval(0, 10));
        let right = DataType::optional(DataType::float_values([5., 8.]));
        let intersection = left.super_intersection(&right).unwrap();
        println!("{} ∩ {} = {}", left, right, intersection);
        assert_eq!(
            intersection,
            DataType::optional(DataType::integer_values([5, 8]))
        );
    }

    #[test]
    fn test_union() {
        let left = DataType::integer_interval(0, 10);
        let right = DataType::float_interval(5., 12.);

        let union = left.super_union(&DataType::Null).unwrap();
        println!("left ∪ ∅ = {}", union);
        assert_eq!(union, left);

        let union = right.super_union(&DataType::Null).unwrap();
        println!("right ∪ ∅ = {}", union);
        assert_eq!(union, right);

        let union = left.super_union(&DataType::Any).unwrap();
        println!("left ∪ any = {}", union);
        assert_eq!(union, DataType::Any);

        let union = right.super_union(&DataType::Any).unwrap();
        println!("right ∪ any = {}", union);
        assert_eq!(union, DataType::Any);

        // int[0 10] ∪ float[5 12] = float{0}∪{1}∪{2}∪{3}∪{4}∪[5 12]
        let union = left.super_union(&right).unwrap();
        println!("{} ∪ {} = {}", left, right, union);
        assert!(left.is_subset_of(&union));
        assert!(right.is_subset_of(&union));
        assert_eq!(
            union,
            DataType::float_values([0., 1., 2., 3., 4.])
                .super_union(&DataType::float_interval(5., 12.))
                .unwrap()
        );

        // int[0 10] ∪ float{5, 8} = int[0 10]
        let left = DataType::integer_interval(0, 10);
        let right = DataType::float_values([5., 8.]);
        let union = left.super_union(&right).unwrap();
        println!("{} ∪ {} = {}", left, right, union);
        assert!(left.is_subset_of(&union));
        assert!(right.is_subset_of(&union));
        assert_eq!(union, DataType::integer_interval(0, 10));

        // optional(int[0 10]) ∪ float{5, 8} = optional(int[0 10])
        let left = DataType::optional(DataType::integer_interval(0, 10));
        let right = DataType::float_values([5., 8.]);
        let intersection = left.super_intersection(&right).unwrap();
        println!("{} ∩ {} = {}", left, right, intersection);
        assert_eq!(intersection, DataType::integer_values([5, 8]));

        // int[0 10] ∪ optional(float{5, 8}) = optional(int[0 10])
        let left = DataType::integer_interval(0, 10);
        let right = DataType::optional(DataType::float_values([5., 8.]));
        let intersection = left.super_intersection(&right).unwrap();
        println!("{} ∩ {} = {}", left, right, intersection);
        assert_eq!(intersection, DataType::integer_values([5, 8]));

        // optional(int[0 10]) ∪ optional(float{5, 8}) = optional(int[0 10])
        let left = DataType::optional(DataType::integer_interval(0, 10));
        let right = DataType::optional(DataType::float_values([5., 8.]));
        let intersection = left.super_intersection(&right).unwrap();
        println!("{} ∩ {} = {}", left, right, intersection);
        assert_eq!(
            intersection,
            DataType::optional(DataType::integer_values([5, 8]))
        );
    }

    #[test]
    fn test_inclusion_in_union() {
        let left = DataType::structured_from_data_types([DataType::float(), DataType::float()]);
        let right =
            DataType::structured_from_data_types([DataType::integer(), DataType::integer()]);
        let union_type = left.clone() | right.clone();
        println!("Union = {}", union_type);
        assert!(left.is_subset_of(&union_type));
        assert!(right.is_subset_of(&union_type));
        let set = DataType::structured_from_data_types([
            DataType::integer_interval(0, 5),
            DataType::integer_interval(-3, 2),
        ]);
        assert!(set.is_subset_of(&union_type));
    }

    #[test]
    fn test_struct_intersection() {
        let left = DataType::unit()
            & ("a", DataType::integer_interval(0, 10))
            & ("b", DataType::integer_interval(-5, 0))
            & ("c", DataType::integer_interval(-1, 1));
        println!("left = {}", left);
        let right = DataType::unit()
            & ("a", DataType::float_interval(-2., 2.))
            & ("b", DataType::float_interval(-2., 2.))
            & ("d", DataType::float_interval(-2., 2.));
        println!("right = {}", right);
        println!(
            "intersection = {}",
            left.super_intersection(&right).unwrap()
        );
        println!("union = {}", left.super_union(&right).unwrap());
        assert!(left.is_subset_of(&left.super_union(&right).unwrap()));
        assert!(right.is_subset_of(&left.super_union(&right).unwrap()));
    }

    #[test]
    fn test_list() {
        let il = List::from_data_type_size(
            DataType::integer_interval(0, 10),
            Integer::from_interval(1, 100),
        );
        println!("il = {il}");
        let fl = List::from_data_type_size(DataType::float(), Integer::from_interval(1, 100));
        println!("fl = {fl}");
        println!("il <= fl = {}", il.is_subset_of(&fl));
        let fld: DataType = fl.clone().into();
        let l = il.into_data_type(&fld).unwrap();
        println!("l = {l}");
    }

    // Test round trip
    fn print_common_variant(left: DataType, right: DataType) {
        let common = DataType::into_common_super_variant(&left, &right)
            .unwrap_or((DataType::Null, DataType::Null));
        println!("({left}, {right}) ~ ({}, {})", common.0, common.1);
    }

    #[test]
    fn test_some_common_variant() {
        print_common_variant(DataType::integer_interval(0, 20), DataType::text());
        print_common_variant(DataType::integer_interval(0, 100), DataType::float());
        print_common_variant(DataType::integer_interval(0, 10000), DataType::float());
    }

    #[test]
    fn test_all_common_variant() {
        for_all_pairs!(
            print_common_variant,
            DataType::Null,
            DataType::unit(),
            DataType::boolean(),
            DataType::integer(),
            DataType::float(),
            DataType::text(),
            DataType::date(),
            DataType::time(),
            DataType::date_time(),
            DataType::duration(),
            DataType::Any
        );
    }

    #[test]
    fn test_from_values() {
        // Floats
        let values: Vec<value::Value> = [0.0, 1.0, 2.0]
            .iter()
            .map(|x| value::Value::from(*x))
            .collect();
        println!(
            "values = {}",
            values.iter().map(ToString::to_string).join(", ")
        );
        let data_type: DataType = values.into_iter().collect();
        println!("data_type = {data_type}");
        // Ints
        let values: Vec<value::Value> = [3, 4, 5, 6, 7]
            .iter()
            .map(|x| value::Value::from(*x))
            .collect();
        println!(
            "values = {}",
            values.iter().map(ToString::to_string).join(", ")
        );
        let data_type: DataType = values.into_iter().collect();
        println!("data_type = {data_type}");
        // Text
        let values: Vec<value::Value> = ["A", "B", "C", "Hello", "World"]
            .iter()
            .map(|x| value::Value::from(x.to_string()))
            .collect();
        println!(
            "values = {}",
            values.iter().map(ToString::to_string).join(", ")
        );
        let data_type: DataType = values.into_iter().collect();
        println!("data_type = {data_type}");
        // Datetime
        let values: Vec<value::Value> = [
            chrono::NaiveDate::from_ymd_opt(2000, 1, 12)
                .unwrap()
                .and_hms_opt(2, 0, 0)
                .unwrap(),
            chrono::NaiveDate::from_ymd_opt(2020, 1, 12)
                .unwrap()
                .and_hms_opt(2, 0, 0)
                .unwrap(),
        ]
        .iter()
        .map(|x| value::Value::from(*x))
        .collect();
        println!(
            "values = {}",
            values.iter().map(ToString::to_string).join(", ")
        );
        let data_type: DataType = values.into_iter().collect();
        println!("data_type = {data_type}");
    }

    #[test]
    fn test_try_into_values() {
        let dt = DataType::float_values([1., 2., 3.]);
        assert_eq!(
            TryInto::<Vec<Value>>::try_into(dt).unwrap(),
            vec![1.0.into(), 2.0.into(), 3.0.into()]
        );

        let dt = DataType::float_interval(1., 1.);
        assert_eq!(
            TryInto::<Vec<Value>>::try_into(dt).unwrap(),
            vec![1.0.into()]
        );

        let dt = DataType::float_interval(1., 3.);
        assert!(TryInto::<Vec<Value>>::try_into(dt).is_err());
    }

    #[test]
    fn test_into_common_super_variant() {
        // Integer, Integer
        let left = DataType::integer_interval(3, 7);
        let right = DataType::integer_interval(2, 5);
        let (new_left, new_right) = DataType::into_common_super_variant(&left, &right).unwrap();
        println!("( {}, {} ) -> ( {}, {} )", left, right, new_left, new_right);
        assert_eq!(new_right, right);
        assert_eq!(new_left, left);

        // Integer, Float
        let left = DataType::float_values([7., 10.5]);
        let right = DataType::integer_interval(2, 5);
        let (new_left, new_right) = DataType::into_common_super_variant(&left, &right).unwrap();
        println!("( {}, {} ) -> ( {}, {} )", left, right, new_left, new_right);
        assert_eq!(new_left, left);
        assert_eq!(new_right, DataType::float_values([2., 3., 4., 5.]));

        // Optional(Integer), Optional(Integer)
        let left = DataType::optional(DataType::integer_interval(3, 7));
        let right = DataType::optional(DataType::integer_interval(2, 5));
        let (new_left, new_right) = DataType::into_common_super_variant(&left, &right).unwrap();
        println!("( {}, {} ) -> ( {}, {} )", left, right, new_left, new_right);
        assert_eq!(new_left, left);
        assert_eq!(new_right, right);

        // Integer, Optional(Integer)
        let left = DataType::integer_interval(3, 7);
        let right = DataType::optional(DataType::integer_interval(2, 5));
        let (new_left, new_right) = DataType::into_common_super_variant(&left, &right).unwrap();
        println!("( {}, {} ) -> ( {}, {} )", left, right, new_left, new_right);
        assert_eq!(new_left, DataType::optional(left));
        assert_eq!(new_right, right);
    }

    #[test]
    fn test_into_common_sub_variant() {
        // Integer, Integer
        let left = DataType::integer_interval(3, 7);
        let right = DataType::integer_interval(2, 5);
        let (new_left, new_right) = DataType::into_common_sub_variant(&left, &right).unwrap();
        println!("( {}, {} ) -> ( {}, {} )", left, right, new_left, new_right);
        assert_eq!(new_right, right);
        assert_eq!(new_left, left);

        // Float, Float
        let left = DataType::float_interval(0., 10.);
        let right = DataType::float_max(9.);
        let (new_left, new_right) = DataType::into_common_sub_variant(&left, &right).unwrap();
        println!("( {}, {} ) -> ( {}, {} )", left, right, new_left, new_right);
        assert_eq!(new_right, right);
        assert_eq!(new_left, left);

        // Integer, Float with integers
        let left = DataType::float_values([7., 10.]);
        let right = DataType::integer_interval(2, 5);
        let (new_left, new_right) = DataType::into_common_sub_variant(&left, &right).unwrap();
        println!("( {}, {} ) -> ( {}, {} )", left, right, new_left, new_right);
        assert_eq!(new_left, DataType::integer_values([7, 10]));
        assert_eq!(new_right, right);

        // Integer, Float (any))
        let left = DataType::float();
        let right = DataType::integer_interval(2, 5);
        assert!(DataType::into_common_sub_variant(&left, &right).is_err());

        // Optional(Integer), Optional(Integer)
        let left = DataType::optional(DataType::integer_interval(3, 7));
        let right = DataType::optional(DataType::integer_interval(2, 5));
        let (new_left, new_right) = DataType::into_common_sub_variant(&left, &right).unwrap();
        println!("( {}, {} ) -> ( {}, {} )", left, right, new_left, new_right);
        assert_eq!(new_left, left);
        assert_eq!(new_right, right);

        // Integer, Optional(Integer)
        let left = DataType::integer_interval(3, 7);
        let right = DataType::optional(DataType::integer_interval(2, 5));
        assert!(DataType::into_common_sub_variant(&left, &right).is_err());
    }

    #[test]
    fn test_hierarchy() {
        let dt_float = DataType::float();
        let dt_int = DataType::integer();
        let struct_dt =
            DataType::structured([("a", DataType::float()), ("b", DataType::integer())]);
        println!("{}", struct_dt.hierarchy());
        let correct_hierarchy = Hierarchy::from([(vec!["a"], &dt_float), (vec!["b"], &dt_int)]);
        assert_eq!(struct_dt.hierarchy(), correct_hierarchy);
        let struct_dt2 =
            DataType::structured([("a", DataType::integer()), ("c", DataType::integer())]);
        let union_dt = DataType::union([
            ("table1", struct_dt.clone()),
            ("table2", struct_dt2.clone()),
        ]);
        let correct_hierarchy = Hierarchy::from([
            (vec!["table1"], &struct_dt),
            (vec!["table2"], &struct_dt2),
            (vec!["table1", "a"], &dt_float),
            (vec!["table1", "b"], &dt_int),
            (vec!["table2", "a"], &dt_int),
            (vec!["table2", "c"], &dt_int),
        ]);
        let h = union_dt.hierarchy();
        println!("{}", h);
        assert_eq!(h, correct_hierarchy);
    }
}
