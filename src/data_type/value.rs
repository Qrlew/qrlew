//! # The values assoctated with types used in Sarus SQL
//!
//! A typed value with runtime type checking
//!

use crate::namer;
use chrono;
use itertools::Itertools;
use serde::{ser::SerializeStruct, Deserialize, Serialize};
use std::{
    cmp,
    collections::{BTreeSet, HashSet},
    convert::Infallible,
    error, fmt, hash,
    ops::{self, Deref, Index},
    rc::Rc,
    result,
};

use super::{
    super::data_type,
    function,
    injection::{self, InjectInto, Injection},
    intervals::Bound,
    And, DataType, DataTyped, Hierarchy, Path, Variant as _,
};

// Error handling

/// The errors values can lead to
#[derive(Debug)]
pub enum Error {
    Value(String),
    Conversion(String),
    Other(String),
}

impl Error {
    pub fn value(err: impl fmt::Display) -> Error {
        Error::Value(format!("Error: {}", err))
    }
    pub fn conversion(err: impl fmt::Display) -> Error {
        Error::Conversion(format!("Error: {}", err))
    }
    pub fn other(err: impl fmt::Display) -> Error {
        Error::Other(format!("Error: {}", err))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Value(desc) => writeln!(f, "Value: {}", desc),
            Error::Conversion(desc) => writeln!(f, "Conversion: {}", desc),
            Error::Other(desc) => writeln!(f, "{}", desc),
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
        Error::Conversion(err.to_string())
    }
}
impl From<injection::Error> for Error {
    fn from(err: injection::Error) -> Self {
        Error::Conversion(err.to_string())
    }
}

type Result<T> = result::Result<T, Error>;

/// Invoke the same method, no matter the variant
macro_rules! for_all_variants {
    ($value:expr, $variant:ident, $fun:expr, [$($Variant:ident),*]) => {
        match $value {
            $(Value::$Variant($variant) => $fun,)*
        }
    };
    ($value:expr, $variant:ident, $fun:expr, [$($Variant:ident),*], $default:expr) => {
        match $value {
            $(Value::$Variant($variant) => $fun,)*
            _ => $default,
        }
    };
}

/// Invoke the same method, no matter the variant
macro_rules! for_all_variant_pairs {
    ($left:expr, $right:expr, $left_variant:ident, $right_variant:ident, $fun:expr, [$($Variant:ident),*]) => {
        match ($left, $right) {
            $((Value::$Variant($left_variant), Value::$Variant($right_variant)) => $fun,)*
        }
    };
    ($left:expr, $right:expr, $left_variant:ident, $right_variant:ident, $fun:expr, [$($Variant:ident),*], $default:expr) => {
        match ($left, $right) {
            $((Value::$Variant($left_variant), Value::$Variant($right_variant)) => $fun,)*
            _ => $default,
        }
    };
}

pub trait Variant:
    TryFrom<Value>
    + Into<Value>
    + Clone
    + fmt::Debug
    + fmt::Display
    + hash::Hash
    + cmp::PartialEq
    + cmp::PartialOrd
    + DataTyped
// where // TODO
//     <Self as TryFrom<Value>>::Error: Into<Error>,
//     <Self::Wrapped as TryFrom<Value>>::Error: Into<Error>,
{
    type Wrapped: From<Self> + Into<Self> + TryFrom<Value> + Into<Value>;

    /// Build a value of a given type from a bunch of values
    fn from_values<V: AsRef<[Value]>>(values: V) -> Result<Self>
    where
        Self: TryFrom<Value, Error = Error>,
    {
        let slice = values.as_ref();
        if slice.len() == 1 {
            Ok(TryFrom::try_from(slice[0].clone())?)
        } else {
            Err(Error::conversion(
                "Cannot convert a slice of many values into a simple type",
            ))
        }
    }

    /// Build a value of a given type from a bunch of values (mostly useful for composite datatypes or conversions)
    fn from_data_typed_values<V: AsRef<[Value]>>(values: V, data_type: &DataType) -> Result<Self>
    where
        Self: TryFrom<Value, Error = Error>,
    {
        let slice = values.as_ref();
        if slice.len() == 1 {
            Ok(TryFrom::try_from(slice[0].as_data_type(data_type)?)?)
        } else {
            Err(Error::conversion(
                "Cannot convert a slice of many values into a simple type",
            ))
        }
    }

    /// Convert the value in a different DataType if possible
    fn as_data_type(&self, data_type: &DataType) -> Result<Value> {
        Ok(self
            .data_type()
            .inject_into(data_type)?
            .value(&self.clone().into())?)
    }
}

macro_rules! impl_wrapped_conversions {
    ( $Variant:ident ) => {
        impl From<$Variant> for <$Variant as Variant>::Wrapped {
            fn from(v: $Variant) -> Self {
                v.0
            }
        }

        impl From<<$Variant as Variant>::Wrapped> for $Variant {
            fn from(w: <$Variant as Variant>::Wrapped) -> Self {
                $Variant(w)
            }
        }
    };
}

macro_rules! impl_variant_conversions {
    ( $Variant:ident ) => {
        impl_wrapped_conversions!($Variant);

        impl TryFrom<Value> for <$Variant as Variant>::Wrapped {
            type Error = Error;

            fn try_from(v: Value) -> Result<Self> {
                Ok(<$Variant as Variant>::Wrapped::from($Variant::try_from(v)?))
            }
        }

        impl From<<$Variant as Variant>::Wrapped> for Value {
            fn from(w: <$Variant as Variant>::Wrapped) -> Self {
                Value::from($Variant::from(w))
            }
        }
    };
}

/// Unit value
#[derive(Clone, Hash, PartialEq, PartialOrd, Debug, Deserialize, Serialize)]
pub struct Unit(());

impl DataTyped for Unit {
    fn data_type(&self) -> DataType {
        DataType::from(data_type::Unit::from(self.clone()))
    }
}

impl Deref for Unit {
    type Target = ();

    fn deref(&self) -> &Self::Target {
        &()
    }
}

impl fmt::Display for Unit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NULL")
    }
}

impl Variant for Unit {
    type Wrapped = ();
}

impl_variant_conversions!(Unit);

/// Boolean value
#[derive(Clone, Hash, PartialEq, PartialOrd, Debug, Deserialize, Serialize)]
pub struct Boolean(bool);

impl DataTyped for Boolean {
    fn data_type(&self) -> DataType {
        DataType::from(data_type::Boolean::from(self.clone()))
    }
}

impl Deref for Boolean {
    type Target = bool;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Display for Boolean {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Variant for Boolean {
    type Wrapped = bool;
}

impl_variant_conversions!(Boolean);

/// Integer value
#[derive(Clone, Hash, PartialEq, PartialOrd, Debug, Deserialize, Serialize)]
pub struct Integer(i64);

impl DataTyped for Integer {
    fn data_type(&self) -> DataType {
        DataType::from(data_type::Integer::from(self.clone()))
    }
}

impl Deref for Integer {
    type Target = i64;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Display for Integer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Variant for Integer {
    type Wrapped = i64;
}

impl_variant_conversions!(Integer);

/// Enum value
#[derive(Clone, Hash, PartialEq, PartialOrd, Debug, Deserialize, Serialize)]
pub struct Enum((i64, Rc<[(String, i64)]>));

impl Enum {
    pub fn decode(&self) -> Result<String> {
        Ok(data_type::Enum::new(self.0 .1.clone()).decode(self.0 .0)?)
    }
}

impl DataTyped for Enum {
    fn data_type(&self) -> DataType {
        DataType::from(data_type::Enum::new(self.0 .1.clone()))
    }
}

impl Deref for Enum {
    type Target = (i64, Rc<[(String, i64)]>);

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Display for Enum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({})",
            self.0 .0,
            self.decode().unwrap_or_else(|_| "Error".into())
        )
    }
}

impl Variant for Enum {
    type Wrapped = (i64, Rc<[(String, i64)]>);
}

impl_variant_conversions!(Enum);

/// Float value
#[derive(Clone, PartialEq, PartialOrd, Debug, Deserialize, Serialize)]
pub struct Float(f64);

#[allow(clippy::derive_hash_xor_eq)]
impl hash::Hash for Float {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl DataTyped for Float {
    fn data_type(&self) -> DataType {
        DataType::from(data_type::Float::from(self.clone()))
    }
}

impl Deref for Float {
    type Target = f64;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Display for Float {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Variant for Float {
    type Wrapped = f64;
}

impl_variant_conversions!(Float);

/// Text value
#[derive(Clone, Hash, PartialEq, PartialOrd, Debug, Deserialize, Serialize)]
pub struct Text(String);

impl DataTyped for Text {
    fn data_type(&self) -> DataType {
        DataType::from(data_type::Text::from(self.clone()))
    }
}

impl Deref for Text {
    type Target = String;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Display for Text {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Variant for Text {
    type Wrapped = String;
}

impl_variant_conversions!(Text);

/// Bytes value
#[derive(Clone, Hash, PartialEq, PartialOrd, Debug, Deserialize, Serialize)]
pub struct Bytes(Vec<u8>);

impl DataTyped for Bytes {
    fn data_type(&self) -> DataType {
        DataType::from(data_type::Bytes::from(self.clone()))
    }
}

impl Deref for Bytes {
    type Target = Vec<u8>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Display for Bytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            self.0.iter().map(|b| format!("{:02x}", b)).join(" ")
        )
    }
}

impl Variant for Bytes {
    type Wrapped = Vec<u8>;
}

impl_variant_conversions!(Bytes);

/// Struct value
#[derive(Clone, Hash, PartialEq, PartialOrd, Debug, Deserialize, Serialize)]
pub struct Struct(Vec<(String, Rc<Value>)>);

impl Struct {
    /// Create a Struct from a rc slice of fields
    pub fn new(fields: Vec<(String, Rc<Value>)>) -> Struct {
        let mut uniques = HashSet::new();
        assert!(fields.iter().all(move |(f, _)| uniques.insert(f.clone())));
        Struct(fields)
    }
    /// An empty struct (a neutral element for the cartesian product)
    pub fn unit() -> Struct {
        Struct::new(vec![])
    }
    /// Create from one field
    pub fn from_field<S: Into<String>, V: Into<Rc<Value>>>(s: S, v: V) -> Struct {
        Struct::new(vec![(s.into(), v.into())])
    }
    /// Create from one datatype
    pub fn from_value(value: Value) -> Struct {
        Struct::default().and(value)
    }
    /// Create from a slice of datatypes
    pub fn from_values(values: &[Value]) -> Struct {
        values
            .iter()
            .fold(Struct::default(), |s, v| s.and(v.clone()))
    }
    /// Get all the fields
    pub fn fields(&self) -> &[(String, Rc<Value>)] {
        self.0.as_ref()
    }
    /// Get the field
    pub fn field(&self, name: &str) -> Result<&(String, Rc<Value>)> {
        self.0
            .iter()
            .find(|(f, _)| f == name)
            .ok_or_else(|| Error::value("Invalid field"))
    }
    /// Get the Value associated with the field
    pub fn value(&self, name: &str) -> Result<&Rc<Value>> {
        self.0
            .iter()
            .find(|(f, _)| f == name)
            .map_or(Err(Error::value("Invalid field")), |(_, v)| Ok(&v))
    }
    /// Find the index of the field with the given name
    pub fn index_from_name(&self, name: &str) -> Result<usize> {
        self.0
            .iter()
            .position(|(s, _t)| s == name)
            .ok_or_else(|| Error::value("Invalid field"))
    }
    /// Access a field by index
    pub fn field_from_index(&self, index: usize) -> &(String, Rc<Value>) {
        &self.0[index]
    }
    pub fn hierarchy(&self) -> Hierarchy<&Value> {
        let h: Hierarchy<&Value> = self
            .iter()
            .map(|(s, v)| (vec![s.to_string()], v.as_ref()))
            .collect();
        self.iter().fold(h, |acc, (s, v)| {
            acc.chain(v.hierarchy().prepend(&[s.to_string()]))
        })
    }
}

// This is a Unit
impl Default for Struct {
    fn default() -> Self {
        Struct::unit()
    }
}

impl DataTyped for Struct {
    fn data_type(&self) -> DataType {
        DataType::from(data_type::Struct::from(self.clone()))
    }
}

impl Deref for Struct {
    type Target = [(String, Rc<Value>)];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// This is the core operation to build a Struct
impl<S: Into<String>, V: Into<Rc<Value>>> And<(S, V)> for Struct {
    type Product = Struct;
    fn and(self, other: (S, V)) -> Self::Product {
        let field: String = other.0.into();
        let value: Rc<Value> = other.1.into();
        // Remove existing elements with the same name
        let mut fields: Vec<(String, Rc<Value>)> = self
            .0
            .iter()
            .filter_map(|(f, v)| (&field != f).then_some((f.clone(), v.clone())))
            .collect();
        fields.push((field, value));
        Struct::new(fields.into())
    }
}

impl<V: Into<Value>> And<(V,)> for Struct {
    type Product = Struct;
    fn and(self, other: (V,)) -> Self::Product {
        let field = namer::new_name_outside("", self.0.iter().map(|(f, _v)| f));
        let value: Value = other.0.into();
        self.and((field, value))
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

impl And<Value> for Struct {
    type Product = Struct;
    fn and(self, other: Value) -> Self::Product {
        // Simplify in the case of struct and Unit
        match other {
            Value::Unit(_u) => self,
            Value::Struct(s) => self.and(s),
            other => self.and((other,)),
        }
    }
}

impl<S: Into<String>, V: Into<Rc<Value>>> From<(S, V)> for Struct {
    fn from(field: (S, V)) -> Self {
        Struct::from_field(field.0, field.1)
    }
}

impl<S: Clone + Into<String>, V: Clone + Into<Rc<Value>>> From<&[(S, V)]> for Struct {
    fn from(values: &[(S, V)]) -> Self {
        Struct::new(
            values
                .iter()
                .map(|(f, v)| (f.clone().into(), v.clone().into()))
                .collect(),
        )
    }
}

impl From<Unit> for Struct {
    fn from(_value: Unit) -> Self {
        Struct::unit()
    }
}

impl<S: Into<String>, V: Into<Rc<Value>>> FromIterator<(S, V)> for Struct {
    fn from_iter<I: IntoIterator<Item = (S, V)>>(iter: I) -> Self {
        iter.into_iter().fold(Struct::unit(), |s, f| s.and(f))
    }
}

impl fmt::Display for Struct {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{{{}}}",
            self.0
                .iter()
                .map(|(k, v)| format!("{}: {}", k, v))
                .join(", ")
        )
    }
}

impl Variant for Struct {
    type Wrapped = Vec<(String, Rc<Value>)>;

    /// Build a value of a given type from a bunch of values
    fn from_values<V: AsRef<[Value]>>(values: V) -> Result<Self> {
        let slice = values.as_ref();
        Ok(Struct(
            slice
                .iter()
                .enumerate()
                .map(|(index, value)| (format!("{index}"), Rc::new(value.clone())))
                .collect(),
        ))
    }

    /// Build a value of a given type from a bunch of typed values
    fn from_data_typed_values<V: AsRef<[Value]>>(values: V, data_type: &DataType) -> Result<Self>
    where
        Self: TryFrom<Value, Error = Error>,
    {
        let slice = values.as_ref();
        match data_type {
            DataType::Struct(structured) if slice.len() == structured.fields().len() => {
                let result: Result<Vec<(String, Rc<Value>)>> = structured
                    .fields()
                    .iter()
                    .zip(slice)
                    .map(|((field, data_type), value)| {
                        if value.data_type().is_subset_of(data_type) {
                            Ok((field.clone(), Rc::new(value.as_data_type(data_type)?)))
                        } else {
                            Err(Error::value(format!(
                                "{}, of type {} is not of type {data_type}",
                                value.clone(),
                                value.data_type()
                            )))
                        }
                    })
                    .collect();
                Ok(Struct::new(result?))
            }
            _ if slice.len() == 1 => Ok(TryFrom::try_from(slice[0].as_data_type(data_type)?)?),
            _ => Err(Error::conversion(
                "Cannot convert a slice of many values into a simple type",
            )),
        }
    }
}

impl_variant_conversions!(Struct);

// Index Structs
impl<P: Path> Index<P> for Struct {
    type Output = Value;

    fn index(&self, index: P) -> &Self::Output {
        self.hierarchy()[index]
    }
}

impl Index<usize> for Struct {
    type Output = Rc<Value>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.field_from_index(index).1
    }
}

/// Union value
#[derive(Clone, Hash, PartialEq, PartialOrd, Debug, Deserialize, Serialize)]
pub struct Union((String, Rc<Value>));

impl Union {
    /// Create a Struct from a rc slice of fields
    pub fn new(field: String, value: Rc<Value>) -> Union {
        Union((field, value))
    }
    /// Create from one field
    pub fn from_field<S: Into<String>, V: Into<Value>>(s: S, v: V) -> Union {
        Union::new(s.into(), Rc::new(v.into()))
    }
    /// Create from one datatype
    pub fn from_value(value: Value) -> Union {
        Union::new(namer::new_name(""), Rc::new(value))
    }
    pub fn hierarchy(&self) -> Hierarchy<&Value> {
        let h: Hierarchy<&Value> = [(self.0 .0.to_string(), self.0 .1.as_ref())]
            .into_iter()
            .collect();
        h.chain(
            self.0
                 .1
                .as_ref()
                .hierarchy()
                .prepend(&[self.0 .0.to_string()]),
        )
    }
}

impl DataTyped for Union {
    fn data_type(&self) -> DataType {
        DataType::from(data_type::Union::from(self.clone()))
    }
}

impl Deref for Union {
    type Target = (String, Rc<Value>);

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<S: Into<String>, V: Into<Value>> From<(S, V)> for Union {
    fn from(field: (S, V)) -> Self {
        Union::from_field(field.0, field.1)
    }
}

impl fmt::Display for Union {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{{}: {}}}", self.0 .0, self.0 .1)
    }
}

impl Variant for Union {
    type Wrapped = (String, Rc<Value>);
}

impl_variant_conversions!(Union);

/// Optional value
#[derive(Clone, Hash, PartialEq, PartialOrd, Debug, Deserialize, Serialize)]
pub struct Optional(Option<Rc<Value>>);

impl Optional {
    /// Create a Struct from a rc slice of fields
    pub fn new(value: Option<Rc<Value>>) -> Optional {
        Optional(value)
    }
    /// Create a none value
    pub fn none() -> Optional {
        Optional::new(None)
    }
    /// Create from a value
    pub fn some(value: Value) -> Optional {
        Optional::new(Some(Rc::new(value)))
    }
}

impl DataTyped for Optional {
    fn data_type(&self) -> DataType {
        DataType::from(data_type::Optional::from(self.clone()))
    }
}

impl Deref for Optional {
    type Target = Option<Rc<Value>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Display for Optional {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            self.0
                .as_deref()
                .map_or("none".to_string(), |v| format!("some({})", v))
        )
    }
}

impl Variant for Optional {
    type Wrapped = Option<Rc<Value>>;
}

impl_variant_conversions!(Optional);

/// List value
#[derive(Clone, Hash, PartialEq, PartialOrd, Debug, Deserialize, Serialize)]
pub struct List(Vec<Value>);

impl List {
    pub fn to_vec(&self) -> &Vec<Value> {
        &self.0
    }
}

impl DataTyped for List {
    fn data_type(&self) -> DataType {
        DataType::from(data_type::List::from(self.clone()))
    }
}

impl Deref for List {
    type Target = Vec<Value>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Display for List {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({})",
            self.0.iter().map(|v| format!("{}", v)).join(", ")
        )
    }
}

impl Variant for List {
    type Wrapped = Vec<Value>;
}

impl_variant_conversions!(List);

/// Build a List value out of many values
impl FromIterator<Value> for List {
    fn from_iter<T: IntoIterator<Item = Value>>(iter: T) -> Self {
        List(iter.into_iter().collect())
    }
}

/// Set value
#[derive(Clone, Hash, PartialEq, PartialOrd, Debug, Deserialize, Serialize)]
pub struct Set(BTreeSet<Value>);

impl DataTyped for Set {
    fn data_type(&self) -> DataType {
        DataType::from(data_type::Set::from(self.clone()))
    }
}

impl Deref for Set {
    type Target = BTreeSet<Value>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Display for Set {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{{{}}}",
            self.0.iter().map(|v| format!("{}", v)).join(", ")
        )
    }
}

impl Variant for Set {
    type Wrapped = BTreeSet<Value>;
}

impl_variant_conversions!(Set);

/// Build a Set value out of many values
impl FromIterator<Value> for Set {
    fn from_iter<T: IntoIterator<Item = Value>>(iter: T) -> Self {
        Set(iter.into_iter().collect())
    }
}

/// Array value
#[derive(Clone, Hash, PartialEq, PartialOrd, Debug, Deserialize, Serialize)]
pub struct Array((Vec<Value>, Vec<usize>));

impl DataTyped for Array {
    fn data_type(&self) -> DataType {
        DataType::from(data_type::Array::from(self.clone()))
    }
}

impl Deref for Array {
    type Target = (Vec<Value>, Vec<usize>);

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Display for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}]",
            self.0
                 .0
                .iter()
                .map(|v| format!("{}", v))
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}

impl Variant for Array {
    type Wrapped = (Vec<Value>, Vec<usize>);
}

impl_variant_conversions!(Array);

/// Date value
#[derive(Clone, Hash, PartialEq, PartialOrd, Debug, Deserialize, Serialize)]
pub struct Date(chrono::NaiveDate);

impl DataTyped for Date {
    fn data_type(&self) -> DataType {
        DataType::from(data_type::Date::from(self.clone()))
    }
}

impl Deref for Date {
    type Target = chrono::NaiveDate;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Display for Date {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Variant for Date {
    type Wrapped = chrono::NaiveDate;
}

impl_variant_conversions!(Date);

/// Time value
#[derive(Clone, Hash, PartialEq, PartialOrd, Debug, Deserialize, Serialize)]
pub struct Time(chrono::NaiveTime);

impl DataTyped for Time {
    fn data_type(&self) -> DataType {
        DataType::from(data_type::Time::from(self.clone()))
    }
}

impl Deref for Time {
    type Target = chrono::NaiveTime;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Display for Time {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Variant for Time {
    type Wrapped = chrono::NaiveTime;
}

impl_variant_conversions!(Time);

/// DateTime value
#[derive(Clone, Hash, PartialEq, PartialOrd, Debug, Deserialize, Serialize)]
pub struct DateTime(chrono::NaiveDateTime);

impl DataTyped for DateTime {
    fn data_type(&self) -> DataType {
        DataType::from(data_type::DateTime::from(self.clone()))
    }
}

impl Deref for DateTime {
    type Target = chrono::NaiveDateTime;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Display for DateTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Variant for DateTime {
    type Wrapped = chrono::NaiveDateTime;
}

impl_variant_conversions!(DateTime);

/// Duration value
#[derive(Clone, Hash, PartialEq, PartialOrd, Debug)]
pub struct Duration(chrono::Duration);

impl DataTyped for Duration {
    fn data_type(&self) -> DataType {
        DataType::from(data_type::Duration::from(self.clone()))
    }
}

impl Deref for Duration {
    type Target = chrono::Duration;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Display for Duration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Variant for Duration {
    type Wrapped = chrono::Duration;
}

impl<'de> Deserialize<'de> for Duration {
    fn deserialize<D>(deserializer: D) -> result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error;
        let duration = chrono::Duration::from_std(std::time::Duration::deserialize(deserializer)?)
            .map_err(Error::custom)?;
        Ok(Duration(duration))
    }
}

impl Serialize for Duration {
    fn serialize<S>(&self, serializer: S) -> result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::Error;
        self.0
            .to_std()
            .map_err(Error::custom)?
            .serialize(serializer)
    }
}

impl_variant_conversions!(Duration);

/// Id value
#[derive(Clone, Hash, PartialEq, PartialOrd, Debug, Deserialize, Serialize)]
pub struct Id(String);

impl DataTyped for Id {
    fn data_type(&self) -> DataType {
        DataType::from(data_type::Id::from(self.clone()))
    }
}

impl Deref for Id {
    type Target = String;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Display for Id {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Variant for Id {
    type Wrapped = String;
}

impl_wrapped_conversions!(Id);

/// Function value
#[derive(Clone)]
pub struct Function(Rc<dyn function::Function>);

impl DataTyped for Function {
    fn data_type(&self) -> DataType {
        self.0.data_type()
    }
}

impl Deref for Function {
    type Target = Rc<dyn function::Function>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl cmp::PartialEq for Function {
    fn eq(&self, other: &Self) -> bool {
        // Rc::ptr_eq(&self.0, &other.0)
        let s = &self.0 as *const _ as *const u8;
        let o = &other.0 as *const _ as *const u8;
        s == o
    }
}

impl cmp::PartialOrd for Function {
    fn partial_cmp(&self, _other: &Self) -> Option<cmp::Ordering> {
        None
    }
}

impl hash::Hash for Function {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0).hash(state)
    }
}

impl fmt::Debug for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} -> {}", self.0.domain(), self.0.co_domain())
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} -> {}", self.0.domain(), self.0.co_domain())
    }
}

impl Variant for Function {
    type Wrapped = Rc<dyn function::Function>;
}

// A dummy implementation of deserialization
impl<'de> Deserialize<'de> for Function {
    fn deserialize<D>(_deserializer: D) -> result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Defaults to id function
        Ok(Function(Rc::new(function::null())))
    }
}

// A dummy implementation of serialization
impl Serialize for Function {
    fn serialize<S>(&self, serializer: S) -> result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("Function", 1)?;
        state.serialize_field("function", "null")?;
        state.end()
    }
}

impl_variant_conversions!(Function);

/// A Value type containing any value sub-type
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum Value {
    Unit(Unit),
    Boolean(Boolean),
    Integer(Integer),
    Enum(Enum),
    Float(Float),
    Text(Text),
    Bytes(Bytes),
    Struct(Struct),
    Union(Union),
    Optional(Optional),
    List(List),
    Set(Set),
    Array(Array),
    Date(Date),
    Time(Time),
    DateTime(DateTime),
    Duration(Duration),
    Id(Id),
    Function(Function),
}

impl Value {
    // Some builders
    pub fn unit() -> Value {
        Value::from(())
    }

    pub fn boolean(b: bool) -> Value {
        Value::from(b)
    }

    pub fn integer(i: i64) -> Value {
        Value::from(i)
    }

    pub fn enumeration<E: Into<Rc<[(String, i64)]>>>(i: i64, e: E) -> Value {
        Value::from((i, e.into()))
    }

    pub fn float(f: f64) -> Value {
        Value::from(f)
    }

    pub fn text<S: Into<String>>(s: S) -> Value {
        Value::from(s.into())
    }

    pub fn bytes<B: Into<Vec<u8>>>(b: B) -> Value {
        Value::from(b.into())
    }

    pub fn structured<S: Clone + Into<String>, V: Clone + Into<Rc<Value>>, F: AsRef<[(S, V)]>>(
        f: F,
    ) -> Value {
        Value::Struct(Struct::new(
            f.as_ref()
                .iter()
                .map(|(f, v)| (f.clone().into(), v.clone().into()))
                .collect(),
        ))
    }

    pub fn structured_from_values<V: AsRef<[Value]>>(values: V) -> Value {
        Value::Struct(Struct::from_values(values.as_ref()))
    }

    pub fn union<V: Into<Rc<Value>>>(f: String, v: V) -> Value {
        Value::from((f, v.into()))
    }

    pub fn some<V: Into<Rc<Value>>>(v: V) -> Value {
        Value::from(Some(v.into()))
    }

    pub fn none() -> Value {
        Value::from(None)
    }

    pub fn list<L: IntoIterator<Item = Value>>(l: L) -> Value {
        Value::from(l.into_iter().collect::<Vec<Value>>())
    }

    pub fn set<S: IntoIterator<Item = Value>>(s: S) -> Value {
        Value::from(s.into_iter().collect::<BTreeSet<Value>>())
    }

    pub fn array<V: IntoIterator<Item = Value>, const K: usize>(v: V, s: [usize; K]) -> Value {
        Value::from((v.into_iter().collect::<Vec<Value>>(), s.to_vec()))
    }

    pub fn date(d: chrono::NaiveDate) -> Value {
        Value::from(d)
    }

    pub fn time(d: chrono::NaiveTime) -> Value {
        Value::from(d)
    }

    pub fn date_time(d: chrono::NaiveDateTime) -> Value {
        Value::from(d)
    }

    pub fn duration(d: chrono::Duration) -> Value {
        Value::from(d)
    }

    pub fn id<S: Into<String>>(s: S) -> Value {
        Value::from(s.into())
    }

    pub fn function<F: function::Function + 'static, T: Into<Rc<F>>>(f: T) -> Value {
        Value::Function(Function(f.into()))
    }

    pub fn hierarchy(&self) -> Hierarchy<&Value> {
        match self {
            Value::Struct(x) => x.hierarchy(),
            _ => Hierarchy::from([(Vec::<&str>::new(), self)]),
        }
    }
}

macro_rules! impl_conversions {
    ( $Variant:ident ) => {
        impl From<$Variant> for Value {
            fn from(v: $Variant) -> Self {
                Value::$Variant(v)
            }
        }

        impl TryFrom<Value> for $Variant {
            type Error = Error;

            fn try_from(value: Value) -> Result<Self> {
                if let Value::$Variant(v) = value {
                    Ok(v)
                } else {
                    Err(Error::value(stringify!($Variant)))
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
impl_conversions!(Optional);
impl_conversions!(List);
impl_conversions!(Set);
impl_conversions!(Array);
impl_conversions!(Date);
impl_conversions!(Time);
impl_conversions!(DateTime);
impl_conversions!(Duration);
impl_conversions!(Id);
impl_conversions!(Function);

impl DataTyped for Value {
    fn data_type(&self) -> DataType {
        for_all_variants!(
            self,
            x,
            x.data_type(),
            [
                Unit, Boolean, Integer, Enum, Float, Text, Bytes, Struct, Union, Optional, List,
                Set, Array, Date, Time, DateTime, Duration, Id, Function
            ]
        )
    }
}

impl cmp::PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        for_all_variant_pairs!(
            self,
            other,
            s,
            o,
            s.eq(o),
            [
                Boolean, Integer, Enum, Float, Text, Bytes, Struct, Union, Optional, List, Set,
                Array, Date, Time, DateTime, Duration, Id, Function
            ],
            { core::mem::discriminant(self) == core::mem::discriminant(other) }
        )
    }
}

impl cmp::Eq for Value {}

impl cmp::PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        for_all_variant_pairs!(
            self,
            other,
            s,
            o,
            s.partial_cmp(o),
            [
                Boolean, Integer, Enum, Float, Text, Bytes, Struct, Union, Optional, List, Set,
                Array, Date, Time, DateTime, Duration, Id, Function
            ],
            { None }
        )
    }
}

impl cmp::Ord for Value {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.partial_cmp(other)
            .unwrap_or_else(|| self.to_string().cmp(&other.to_string()))
    }
}

impl<P: Path> Index<P> for Value {
    type Output = Value;

    fn index(&self, index: P) -> &Self::Output {
        self.hierarchy()[index]
    }
}

/// Implement Expr traits
macro_rules! impl_traits{($($Variant:ident),*) => {
    impl hash::Hash for Value {
        fn hash<H: hash::Hasher>(&self, state: &mut H) {
            core::mem::discriminant(self).hash(state);
            match &self {
                $(Value::$Variant(variant) => variant.hash(state),)*
            }
        }
    }

    impl fmt::Display for Value {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                $(Value::$Variant(variant) => write!(f, "{}", variant),)*
            }
        }
    }
}}

impl_traits!(
    Unit, Boolean, Integer, Enum, Float, Text, Bytes, Struct, Union, Optional, List, Set, Array,
    Date, Time, DateTime, Duration, Id, Function
);

impl Variant for Value {
    type Wrapped = Value;
}

// Some more conversions

/// Value -> (A)
impl<A: Bound> TryFrom<Value> for (A,)
where
    Value: TryInto<A, Error = Error>,
{
    type Error = Error;
    fn try_from(value: Value) -> Result<Self> {
        let intervals: A = value.try_into()?;
        Ok((intervals,))
    }
}

/// Value -> (A, B)
impl<A: Bound, B: Bound> TryFrom<Value> for (A, B)
where
    Value: TryInto<A, Error = Error> + TryInto<B, Error = Error>,
{
    type Error = Error;
    fn try_from(value: Value) -> Result<Self> {
        let structured: Struct = value.try_into()?;
        let left: A = structured.value("0")?.as_ref().clone().try_into()?;
        let right: B = structured.value("1")?.as_ref().clone().try_into()?;
        Ok((left, right))
    }
}

/// Value -> (A, B, C)
impl<A: Bound, B: Bound, C: Bound> TryFrom<Value> for (A, B, C)
where
    Value: TryInto<A, Error = Error> + TryInto<B, Error = Error> + TryInto<C, Error = Error>,
{
    type Error = Error;
    fn try_from(value: Value) -> Result<Self> {
        let structured: Struct = value.try_into()?;
        let inter_a: A = structured.value("0")?.as_ref().clone().try_into()?;
        let inter_b: B = structured.value("1")?.as_ref().clone().try_into()?;
        let inter_c: C = structured.value("2")?.as_ref().clone().try_into()?;
        Ok((inter_a, inter_b, inter_c))
    }
}

/// (A) -> Value
impl<A: Bound> From<(A,)> for Value
where
    A: Into<Value>,
{
    fn from(value: (A,)) -> Self {
        value.0.into()
    }
}

/// (A, B) -> Value
impl<A: Bound, B: Bound> From<(A, B)> for Value
where
    A: Into<Value>,
    B: Into<Value>,
{
    fn from(value: (A, B)) -> Self {
        Struct::from_values(&[value.0.into(), value.1.into()]).into()
    }
}

/// (A, B, C) -> Value
impl<A: Bound, B: Bound, C: Bound> From<(A, B, C)> for Value
where
    A: Into<Value>,
    B: Into<Value>,
    C: Into<Value>,
{
    fn from(value: (A, B, C)) -> Self {
        Struct::from_values(&[value.0.into(), value.1.into(), value.2.into()]).into()
    }
}

// Value algebra

impl And<Value> for Value {
    type Product = Value;
    fn and(self, other: Value) -> Self::Product {
        // Simplify in the case of struct and Unit
        match self {
            Value::Unit(_u) => other,
            Value::Struct(s) => s.and(other).into(),
            s => Struct::from_value(s).and(other).into(),
        }
    }
}

impl<S: Into<String>, V: Into<Rc<Value>>> And<(S, V)> for Value {
    type Product = Value;
    fn and(self, other: (S, V)) -> Self::Product {
        self.and(Value::from(Struct::from(other)))
    }
}

impl<T> ops::BitAnd<T> for Value
where
    Self: And<T>,
{
    type Output = <Self as And<T>>::Product;

    fn bitand(self, rhs: T) -> Self::Output {
        self.and(rhs)
    }
}

/// Build a Value out of many Values
impl FromIterator<Value> for Value {
    fn from_iter<T: IntoIterator<Item = Value>>(iter: T) -> Self {
        Value::List(iter.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;
    use std::convert::TryInto;

    // TODO Write tests for all values

    #[test]
    fn test_build() {
        // TODO fix this test
        let list = [
            Value::structured([
                ("a", Value::from(true)),
                ("b", Value::from(5)),
                ("c", Value::from(5.5)),
            ]),
            Value::structured([
                ("a", Value::from(false)),
                ("b", Value::from(8)),
                ("c", Value::from(12.1)),
            ]),
        ];
        println!("list = {:?}", list);
        println!(
            "list type = {} = {}",
            list[0].data_type(),
            list[1].data_type()
        );
        let value = Value::list(list);
        println!("value = {}", value);
        println!("type = {}", value.data_type());
    }

    #[test]
    fn test_deref() {
        let f = Float::from(5.8);
        println!("f = {:?}", f);
        println!("*f + 3.1 = {:?}", *f + 3.1);
    }

    #[test]
    fn test_build_function() {
        let func = Value::function(function::exp());
        println!("value = {}", func);
        println!("type = {}", func.data_type());
    }

    #[test]
    fn test_display() {
        let value = Value::some(Value::structured([
            ("a", Value::boolean(true)),
            ("b", Value::integer(5)),
            ("c", Value::float(5.5)),
        ]));
        println!("value = {}", value);
        println!("type = {}", value.data_type());
    }

    #[test]
    fn test_struct() {
        let str_type =
            data_type::Struct::from_data_types(&[DataType::integer(), DataType::float()]);
        println!("{str_type}");
        // From typed values
        let value: Value =
            Struct::from_data_typed_values(&[1.into(), 1.0.into()], &str_type.into())
                .unwrap()
                .into();
        println!("value = {}", value);
        println!("type = {}", value.data_type());
        let structured: Struct = value.try_into().unwrap();
        let first: Integer = structured
            .value("0")
            .unwrap()
            .as_ref()
            .clone()
            .try_into()
            .unwrap();
        assert_eq!(first, 1.into());
        // From values
        let value: Value = Struct::from_values(&[1.into(), 1.0.into()]).into();
        println!("value = {}", value);
        println!("type = {}", value.data_type());
        let structured: Struct = value.try_into().unwrap();
        let second: Float = structured
            .value("1")
            .unwrap()
            .as_ref()
            .clone()
            .try_into()
            .unwrap();
        assert_eq!(second, 1.0.into());
    }

    #[test]
    fn test_struct_index() {
        let a = Value::unit() & Value::from(true) & Value::from(false) & Value::from(0.7);
        println!("a = {}", &a);
        let b = Value::unit()
            & ("a", Value::from(true))
            & Value::unit()
            & a
            & ("c", Value::from(12))
            & ("d", Value::from(3.2));
        let b = Struct::try_from(b).unwrap();
        println!("b = {b}");
        println!("b[4] = {}", b[4]);
        println!("b[c] = {}", b["c"]);
        assert_eq!(*b[4], Value::from(12));
    }

    #[test]
    fn test_json_serde() {
        let str_type =
            data_type::Struct::from_data_types(&[DataType::integer(), DataType::float()]);
        println!("{str_type}");
        // From typed values
        let value: Value =
            Struct::from_data_typed_values(&[1.into(), 1.0.into()], &str_type.into())
                .unwrap()
                .into();
        println!("value = {value}");
        let json = serde_json::to_string_pretty(&value).expect("json value");
        println!("value as json = {json}");
        let value_from: Value = serde_json::from_str(&json).expect("value");
        println!("value from json = {value_from}");
        assert_eq!(value, value_from);
        // From values
        let value: Value = Struct::from_values(&[
            1.into(),
            1.0.into(),
            chrono::NaiveDateTime::parse_from_str("2015-09-05 23:56:04", "%Y-%m-%d %H:%M:%S")
                .unwrap()
                .into(),
            Some(Rc::new(chrono::Duration::seconds(100).into())).into(),
        ])
        .into();
        println!("value = {value}");
        let json = serde_json::to_string_pretty(&value).expect("json value");
        println!("value as json = {json}");
        let value_from: Value = serde_json::from_str(&json).expect("value");
        println!("value from json = {value_from}");
        assert_eq!(value, value_from);
    }

    #[test]
    fn test_json_serde_value() {
        // From values
        let value: Value = Struct::from_values(&[
            1.into(),
            1.0.into(),
            chrono::NaiveDateTime::parse_from_str("2015-09-05 23:56:04", "%Y-%m-%d %H:%M:%S")
                .unwrap()
                .into(),
            Some(Rc::new(chrono::Duration::seconds(100).into())).into(),
        ])
        .into();
        println!("value = {value}");
        let json_value = serde_json::to_value(&value).expect("json value");
        println!("value as json = {json_value:#?}");
        let value_from: Value = serde_json::from_value(json_value).expect("value");
        println!("value from json = {value_from}");
        assert_eq!(value, value_from);
    }

    #[test]
    fn test_hierarchy() {
        let value = Value::structured([
            ("a", Value::boolean(true)),
            ("b", Value::integer(5)),
            ("c", Value::float(5.5)),
        ]);
        assert_eq!(
            value.hierarchy(),
            Hierarchy::from([
                ("a", &Value::boolean(true)),
                ("b", &Value::integer(5)),
                ("c", &Value::float(5.5)),
            ])
        );
    }

    #[test]
    fn test_index() {
        let a = Value::structured([("a_0", Value::from(-10)), ("a_1", Value::from(1))]);
        let b = Value::structured([("b_0", Value::from(0.1)), ("b_1", Value::from(10.0))]);
        let x = Value::structured([("a", a), ("b", b)]);
        println!("x = {}", x);
        println!("x['a'] = {}", x["a"]);
        println!("x['a.a_1'] = {}", x[["a", "a_0"]]);
        assert_eq!(x[["b", "b_1"]], 10.0.into());
    }
}
