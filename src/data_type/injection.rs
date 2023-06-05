//! # Inclusion maps
//!
//! Inclusion maps between types and values
//!
//! The type system enforces consistency between familly of types,
//! but dynamic check is performed for the details
//!
//! ## Main properties
//!
//! By default, each DataType Variant has its own notion of inclusion and set operations
//!
//! Cross-variant operations are implemented thanks to injective maps between types/sets in different variants
//!
//!
// Stability: ⭐️

use std::{convert, error, fmt, result};

use super::{
    super::data_type,
    intervals::{self, Values},
    *,
};

/// We produce an inclusion map function as a proof of the possibility of the conversion
/// This should be consistent with the into_data_type method
pub trait InjectInto<CoDomain: Variant>: Variant {
    type Injection: Injection<Domain = Self, CoDomain = CoDomain>;
    fn inject_into(&self, other: &CoDomain) -> Result<Self::Injection>;
}

/// The errors maps can lead to
#[derive(Debug)]
pub enum Error {
    ArgumentOutOfRange(String),
    SetOutOfRange(String),
    NoInjection(String),
}

impl Error {
    pub fn argument_out_of_range(arg: impl fmt::Display, range: impl fmt::Display) -> Error {
        Error::ArgumentOutOfRange(format!("{} not in {}", arg, range))
    }
    pub fn set_out_of_range(set: impl fmt::Display, range: impl fmt::Display) -> Error {
        Error::SetOutOfRange(format!("{} not in {}", set, range))
    }
    pub fn no_injection(domain: impl fmt::Display, co_domain: impl fmt::Display) -> Error {
        Error::NoInjection(format!(
            "No injection found from {} into {}",
            domain, co_domain
        ))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::ArgumentOutOfRange(arg) => writeln!(f, "ArgumentOutOfRange: {}", arg),
            Error::SetOutOfRange(set) => writeln!(f, "SetOutOfRange: {}", set),
            Error::NoInjection(set) => writeln!(f, "NoInjection: {}", set),
        }
    }
}

impl error::Error for Error {}

impl convert::From<convert::Infallible> for Error {
    fn from(err: convert::Infallible) -> Self {
        Error::NoInjection(err.to_string())
    }
}
impl convert::From<data_type::Error> for Error {
    fn from(err: data_type::Error) -> Self {
        Error::NoInjection(err.to_string())
    }
}
impl convert::From<value::Error> for Error {
    fn from(err: value::Error) -> Self {
        Error::NoInjection(err.to_string())
    }
}

pub type Result<T> = result::Result<T, Error>;

/// An inclusion map trait
pub trait Injection: fmt::Debug + fmt::Display {
    type Domain: Variant;
    type CoDomain: Variant;
    /// The domain of the injection
    fn domain(&self) -> Self::Domain;
    /// A super-image of the domain
    fn co_domain(&self) -> Self::CoDomain;
    /// A super-image of a set (a set containing the image of the set and included in the co-domain)
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain>;
    /// The actual implementation of the injection
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element>;
}

/// First step of injection builder
/// ```
/// # use qrlew::data_type::{*, injection::*};
/// let domain = Integer::from_interval(2, 3);
/// let co_domain = Float::from_interval(2., 3.);
/// // Create a map from ints to floats
/// let inj = From(domain).into(co_domain);
/// ```
pub struct From<Domain: Variant>(pub Domain);

impl<Domain: Variant> From<Domain> {
    /// Compose injections
    pub fn then<CoDomain: Variant>(
        self,
        then: CoDomain,
    ) -> Then<Domain, CoDomain, Base<Domain, CoDomain>>
    where
        Base<Domain, CoDomain>: Injection<Domain = Domain, CoDomain = CoDomain>,
    {
        Then {
            from: self.0.clone(),
            then: then.clone(),
            built_injection: Base::new(self.0, then),
        }
    }

    /// Compose injections with default
    pub fn then_default<CoDomain: Variant + Default>(
        self,
    ) -> Then<Domain, CoDomain, Base<Domain, CoDomain>>
    where
        Base<Domain, CoDomain>: Injection<Domain = Domain, CoDomain = CoDomain>,
    {
        self.then(CoDomain::default())
    }

    /// Build the injection
    pub fn into<CoDomain: Variant>(self, into: CoDomain) -> Result<Base<Domain, CoDomain>>
    where
        Base<Domain, CoDomain>: Injection<Domain = Domain, CoDomain = CoDomain>,
    {
        Base::new(self.0, into)
    }

    /// Build the injection with default
    pub fn into_default<CoDomain: Variant + Default>(self) -> Result<Base<Domain, CoDomain>>
    where
        Base<Domain, CoDomain>: Injection<Domain = Domain, CoDomain = CoDomain>,
    {
        self.into(CoDomain::default())
    }

    /// Build the injection and compute an image
    pub fn super_image<CoDomain: Variant + Default>(self) -> Result<CoDomain>
    where
        Base<Domain, CoDomain>: Injection<Domain = Domain, CoDomain = CoDomain>,
    {
        let set = self.0.clone();
        self.into_default::<CoDomain>()?.super_image(&set)
    }
}

/// Composed injection builder
/// ```
/// # use qrlew::data_type::{*, injection::*};
/// let domain = Boolean::default();
/// let intermediate = Integer::from_values([0, 1]);
/// let co_domain = Float::from_values([0., 1.]);
/// // Create an injection from bools to floats by chaining two injections
/// let inj = From(domain).then(intermediate).into(co_domain);
/// ```
pub struct Then<
    Domain: Variant,
    CoDomain: Variant,
    Inj: Injection<Domain = Domain, CoDomain = CoDomain>,
> {
    from: Domain,
    then: CoDomain,
    built_injection: Result<Inj>,
}

impl<
        LeftDomain: Variant,
        MidDomain: Variant,
        Inj: Injection<Domain = LeftDomain, CoDomain = MidDomain>,
    > Then<LeftDomain, MidDomain, Inj>
{
    /// Compose injections
    pub fn then<RightDomain: Variant>(
        self,
        then: RightDomain,
    ) -> Then<
        LeftDomain,
        RightDomain,
        Composed<LeftDomain, MidDomain, RightDomain, Inj, Base<MidDomain, RightDomain>>,
    >
    where
        Base<MidDomain, RightDomain>: Injection<Domain = MidDomain, CoDomain = RightDomain>,
    {
        // One check would be sufficient
        Then {
            from: self.from,
            then: then.clone(),
            built_injection: (|| {
                Ok(Composed::new(
                    self.built_injection?,
                    Base::new(self.then, then)?,
                ))
            })(),
        }
    }

    /// Compose injections with default
    pub fn then_default<RightDomain: Variant + Default>(
        self,
    ) -> Then<
        LeftDomain,
        RightDomain,
        Composed<LeftDomain, MidDomain, RightDomain, Inj, Base<MidDomain, RightDomain>>,
    >
    where
        Base<MidDomain, RightDomain>: Injection<Domain = MidDomain, CoDomain = RightDomain>,
    {
        self.then(RightDomain::default())
    }

    /// Build the injection
    pub fn into<RightDomain: Variant>(
        self,
        into: RightDomain,
    ) -> Result<Composed<LeftDomain, MidDomain, RightDomain, Inj, Base<MidDomain, RightDomain>>>
    where
        Base<MidDomain, RightDomain>: Injection<Domain = MidDomain, CoDomain = RightDomain>,
    {
        // One check would be sufficient
        Ok(Composed::new(
            self.built_injection?,
            Base::new(self.then, into)?,
        ))
    }

    /// Build the injection with default
    pub fn into_default<RightDomain: Variant + Default>(
        self,
    ) -> Result<Composed<LeftDomain, MidDomain, RightDomain, Inj, Base<MidDomain, RightDomain>>>
    where
        Base<MidDomain, RightDomain>: Injection<Domain = MidDomain, CoDomain = RightDomain>,
    {
        self.into(RightDomain::default())
    }

    /// Build the injection and compute an image
    pub fn super_image<RightDomain: Variant + Default>(self) -> Result<RightDomain>
    where
        Base<MidDomain, RightDomain>: Injection<Domain = MidDomain, CoDomain = RightDomain>,
    {
        let set = self.from.clone();
        self.into_default::<RightDomain>()?.super_image(&set)
    }
}

/// An actual base injection
#[derive(Debug)]
pub struct Base<Domain: Variant, CoDomain: Variant> {
    domain: Domain,
    co_domain: CoDomain,
}

impl<Domain: Variant, CoDomain: Variant> fmt::Display for Base<Domain, CoDomain> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} -> {}", self.domain, self.co_domain)
    }
}

/// Methods specific to the injection case
impl<Domain: Variant, CoDomain: Variant> Base<Domain, CoDomain>
where
    Base<Domain, CoDomain>: Injection<Domain = Domain, CoDomain = CoDomain>,
{
    /// Constructor for Base Maps
    pub fn new(domain: Domain, co_domain: CoDomain) -> Result<Base<Domain, CoDomain>> {
        let result = Base { domain, co_domain };
        // Conditional compilation
        if cfg!(feature = "checked_injections") {
            result.checked()
        } else {
            Ok(result)
        }
    }
    // Utility functions to check the consistency of values

    /// Check the image function
    fn checked_image(&self, set: &Domain, super_image: CoDomain) -> Result<CoDomain> {
        if !set.is_subset_of(&self.domain()) {
            Err(Error::set_out_of_range(set, self.domain()))
        } else if !super_image.is_subset_of(&self.co_domain()) {
            Err(Error::set_out_of_range(super_image, self.co_domain()))
        } else {
            Ok(super_image)
        }
    }

    /// Check the value function
    fn checked_value(
        &self,
        arg: &<Domain as Variant>::Element,
        value: <CoDomain as Variant>::Element,
    ) -> Result<<CoDomain as Variant>::Element> {
        if !self.domain().contains(arg) {
            Err(Error::argument_out_of_range(arg, self.domain()))
        } else if !self.co_domain().contains(&value) {
            Err(Error::argument_out_of_range(value, self.co_domain()))
        } else {
            Ok(value)
        }
    }

    /// Check the injection
    fn checked(self) -> Result<Self>
    where
        Self: Sized,
    {
        if self.super_image(&self.domain()).is_err() {
            Err(Error::no_injection(self.domain(), self.co_domain()))
        } else {
            Ok(self)
        }
    }

    /// Build an injection value function out of a simple value function on wrapped elements that cannot fail
    /// fn value(&self, arg: &<Self::Domain as Variant>::Element) -> Result<<Self::CoDomain as Variant>::Element>;
    fn value_map<
        F: Fn(
            &<Domain::Element as value::Variant>::Wrapped,
        ) -> <CoDomain::Element as value::Variant>::Wrapped,
    >(
        &self,
        value: F,
        arg: &Domain::Element,
    ) -> Result<CoDomain::Element> {
        self.checked_value(
            arg,
            value(&<Domain::Element as value::Variant>::Wrapped::from(
                arg.clone(),
            ))
            .into(),
        )
    }

    /// Build an injection value function out of a simple value function on wrapped elements
    /// fn value(&self, arg: &<Self::Domain as Variant>::Element) -> Result<<Self::CoDomain as Variant>::Element>;
    fn value_map_option<
        F: Fn(
            &<Domain::Element as value::Variant>::Wrapped,
        ) -> Option<<CoDomain::Element as value::Variant>::Wrapped>,
    >(
        &self,
        value: F,
        arg: &Domain::Element,
    ) -> Result<CoDomain::Element> {
        self.checked_value(
            arg,
            value(&<Domain::Element as value::Variant>::Wrapped::from(
                arg.clone(),
            ))
            .ok_or_else(|| Error::no_injection(self.domain(), self.co_domain()))?
            .into(),
        )
    }
}

/// Methods specific to the Intervals case
impl<B: intervals::Bound, C: intervals::Bound> Base<Intervals<B>, Intervals<C>>
where
    Intervals<B>: Variant,
    <Intervals<B> as Variant>::Element: convert::From<B>,
    Intervals<C>: Variant,
    <Intervals<C> as Variant>::Element: convert::Into<C>,
    Base<Intervals<B>, Intervals<C>>: Injection<Domain = Intervals<B>, CoDomain = Intervals<C>>,
{
    /// Build an injection value function out of a simple value function on wrapped elements that cannot fail
    /// fn value(&self, arg: &<Self::Domain as Variant>::Element) -> Result<<Self::CoDomain as Variant>::Element>;
    fn intervals_image(&self, set: &Intervals<B>) -> Result<Intervals<C>> {
        let image: Result<Intervals<C>> = set
            .iter()
            .map(|[min, max]| {
                let value_min: C = self.value(&min.clone().into())?.into();
                let value_max: C = self.value(&max.clone().into())?.into();
                if value_min < value_max {
                    Ok([value_min, value_max])
                } else {
                    Ok([value_max, value_min])
                }
            })
            .collect();
        self.checked_image(set, image?)
    }
}

/// Implement specific methods in the Intervals case

/// Composed injections
#[derive(Debug)]
pub struct Composed<
    LeftDomain: Variant,
    MidDomain: Variant,
    RightDomain: Variant,
    Left: Injection<Domain = LeftDomain, CoDomain = MidDomain>,
    Right: Injection<Domain = MidDomain, CoDomain = RightDomain>,
> {
    left: Left,
    right: Right,
}

impl<
        LeftDomain: Variant,
        MidDomain: Variant,
        RightDomain: Variant,
        Left: Injection<Domain = LeftDomain, CoDomain = MidDomain>,
        Right: Injection<Domain = MidDomain, CoDomain = RightDomain>,
    > fmt::Display for Composed<LeftDomain, MidDomain, RightDomain, Left, Right>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} -> {}", self.left.domain(), self.right.co_domain())
    }
}

impl<
        LeftDomain: Variant,
        MidDomain: Variant,
        RightDomain: Variant,
        Left: Injection<Domain = LeftDomain, CoDomain = MidDomain>,
        Right: Injection<Domain = MidDomain, CoDomain = RightDomain>,
    > Composed<LeftDomain, MidDomain, RightDomain, Left, Right>
{
    pub fn new(
        left: Left,
        right: Right,
    ) -> Composed<LeftDomain, MidDomain, RightDomain, Left, Right> {
        assert!(
            left.co_domain().is_subset_of(&right.domain()),
            "The domains should be adapted"
        );
        Composed { left, right }
    }
}

impl<
        LeftDomain: Variant,
        MidDomain: Variant,
        RightDomain: Variant,
        Left: Injection<Domain = LeftDomain, CoDomain = MidDomain>,
        Right: Injection<Domain = MidDomain, CoDomain = RightDomain>,
    > Injection for Composed<LeftDomain, MidDomain, RightDomain, Left, Right>
{
    type Domain = LeftDomain;
    type CoDomain = RightDomain;
    fn domain(&self) -> Self::Domain {
        self.left.domain()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.right.co_domain()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        self.right.super_image(&self.left.super_image(set)?)
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        self.right.value(&self.left.value(arg)?)
    }
}

// Specific injections

/// Boolean -> Integer
impl Injection for Base<Boolean, Integer> {
    type Domain = Boolean;
    type CoDomain = Integer;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        self.intervals_image(&set.clone().into_values())
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        self.value_map(|arg| *arg as i64, arg)
    }
}

/// Boolean -> Text
impl Injection for Base<Boolean, Text> {
    type Domain = Boolean;
    type CoDomain = Text;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        self.intervals_image(&set.clone().into_values())
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        self.value_map(|arg| format!("{arg}"), arg)
    }
}

/// Integer -> Boolean
impl Injection for Base<Integer, Boolean> {
    type Domain = Integer;
    type CoDomain = Boolean;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        self.intervals_image(&set.clone().into_values())
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        self.value_map_option(
            |arg| match arg {
                0 => Some(false),
                1 => Some(true),
                _ => None,
            },
            arg,
        )
    }
}

/// Integer -> Float
impl Injection for Base<Integer, Float> {
    type Domain = Integer;
    type CoDomain = Float;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        self.intervals_image(&set.clone().into_values())
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        self.value_map(|arg| *arg as f64, arg)
    }
}

/// Integer -> Text
impl Injection for Base<Integer, Text> {
    type Domain = Integer;
    type CoDomain = Text;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        let values = set.clone().into_values();
        if values.all_values() {
            self.intervals_image(&values)
        } else {
            Ok(Text::full())
        }
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        self.value_map(|arg| format!("{arg}"), arg)
    }
}

/// Float -> Integer
impl Injection for Base<Float, Integer> {
    type Domain = Float;
    type CoDomain = Integer;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        self.intervals_image(set)
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        self.value_map_option(
            |arg| {
                if (*arg as i64) as f64 == *arg {
                    Some(*arg as i64)
                } else {
                    None
                }
            },
            arg,
        )
    }
}

/// Float -> Text
impl Injection for Base<Float, Text> {
    type Domain = Float;
    type CoDomain = Text;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        if set.all_values() {
            self.intervals_image(set)
        } else {
            Ok(Text::full())
        }
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        self.value_map(|arg| format!("{arg}"), arg)
    }
}

/// Text -> Bytes
impl Injection for Base<Text, Bytes> {
    type Domain = Text;
    type CoDomain = Bytes;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        self.checked_image(set, Bytes)
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        self.checked_value(arg, value::Bytes::from(arg.as_bytes().to_vec()))
    }
}

/// Date -> DateTime
impl Injection for Base<Date, DateTime> {
    type Domain = Date;
    type CoDomain = DateTime;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        self.intervals_image(&set.clone().into_values())
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        self.value_map(|arg| arg.and_hms(0, 0, 0), arg)
    }
}

/// Date -> Text
impl Injection for Base<Date, Text> {
    type Domain = Date;
    type CoDomain = Text;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        let values = set.clone().into_values();
        if values.all_values() {
            self.intervals_image(&values)
        } else {
            Ok(Text::full())
        }
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        self.value_map(|arg| format!("{arg}"), arg)
    }
}

/// Time -> Text
impl Injection for Base<Time, Text> {
    type Domain = Time;
    type CoDomain = Text;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        if set.all_values() {
            self.intervals_image(set)
        } else {
            Ok(Text::full())
        }
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        self.value_map(|arg| format!("{arg}"), arg)
    }
}

/// DateTime -> Date
impl Injection for Base<DateTime, Date> {
    type Domain = DateTime;
    type CoDomain = Date;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        self.intervals_image(set)
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        self.value_map_option(
            |arg| {
                let date = arg.date();
                if *arg == date.and_hms(0, 0, 0) {
                    Some(date)
                } else {
                    None
                }
            },
            arg,
        )
    }
}

/// DateTime -> Text
impl Injection for Base<DateTime, Text> {
    type Domain = DateTime;
    type CoDomain = Text;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        if set.all_values() {
            self.intervals_image(set)
        } else {
            Ok(Text::full())
        }
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        self.value_map(|arg| format!("{arg}"), arg)
    }
}

/// Duration -> Text
impl Injection for Base<Duration, Text> {
    type Domain = Duration;
    type CoDomain = Text;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        if set.all_values() {
            self.intervals_image(set)
        } else {
            Ok(Text::full())
        }
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        self.value_map(|arg| format!("{arg}"), arg)
    }
}

// Composite injections

// Generic injections

/// Identity for Primitive Variants
///
/// When the variant is not primitive, the self-injection is not obvious
impl<Domain: Primitive> Injection for Base<Domain, Domain> {
    type Domain = Domain;
    type CoDomain = Domain;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        self.checked_image(set, set.clone())
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        self.checked_value(arg, arg.clone())
    }
}

/// Struct -> Struct
impl Injection for Base<Struct, Struct> {
    type Domain = Struct;
    type CoDomain = Struct;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    // Any value in a subtype (more fields with narrower types) maps to a value in its supertype (less fields with broader types)
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        self.co_domain
            .fields()
            .iter()
            .map(|(f, t)| {
                Ok((
                    f.clone(),
                    From(self.domain.data_type(f).as_ref().clone())
                        .into((t.as_ref().clone()).clone())?
                        .super_image(&set.data_type(f))?,
                ))
            })
            .collect()
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        let result: Result<Vec<(String, Rc<value::Value>)>> = arg
            .iter()
            .map(|(field, value)| {
                if self
                    .co_domain
                    .fields()
                    .iter()
                    .any(|(co_field, _)| field == co_field)
                {
                    Ok((
                        field.clone(),
                        Rc::new(
                            From(self.domain.data_type(field).as_ref().clone())
                                .into(self.co_domain.data_type(field).as_ref().clone())?
                                .value(value)?,
                        ),
                    ))
                } else {
                    Ok((field.clone(), value.clone()))
                }
            })
            .collect();
        self.checked_value(arg, value::Struct::from(result?))
    }
}

/// Union -> Union
impl Injection for Base<Union, Union> {
    type Domain = Union;
    type CoDomain = Union;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    // Any value in a subtype (less terms with narrower types) maps to a value in its supertype (more terms with broader types)
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        self.domain
            .fields()
            .iter()
            .map(|(f, t)| {
                Ok((
                    f.clone(),
                    From(self.domain.data_type(f).as_ref().clone())
                        .into(t.as_ref().clone())?
                        .super_image(&set.data_type(f))?,
                ))
            })
            .collect()
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        Ok(value::Union::from((
            arg.0.clone(),
            From(self.domain.data_type(&arg.0).as_ref().clone())
                .into(self.co_domain.data_type(&arg.0).as_ref().clone())?
                .value(&arg.1)?,
        )))
    }
}

/// Optional -> Optional
impl Injection for Base<Optional, Optional> {
    type Domain = Optional;
    type CoDomain = Optional;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        Ok(Optional::from(
            From(self.domain.data_type().clone())
                .into(self.co_domain.data_type().clone())?
                .super_image(set.data_type())?,
        ))
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        Ok(value::Optional::from(
            arg.as_ref()
                .map(|a| {
                    From(self.domain.data_type().clone())
                        .into(self.co_domain.data_type().clone())?
                        .value(a)
                })
                .map(|v| Rc::new(v.unwrap()))
                .clone(),
        ))
    }
}

/// List -> List
impl Injection for Base<List, List> {
    type Domain = List;
    type CoDomain = List;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        let data_type = From(self.domain.data_type().clone())
            .into(self.co_domain.data_type().clone())?
            .super_image(set.data_type())?;
        let size = From(self.domain.size().clone())
            .into(self.co_domain.size().clone())?
            .super_image(set.size())?;
        Ok(List::from_data_type_size(data_type, size))
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        Ok(value::List::from_iter(
            arg.iter()
                .map(|a| {
                    From(self.domain.data_type().clone())
                        .into(self.co_domain.data_type().clone())?
                        .value(a)
                })
                .map(|v| v.unwrap()),
        ))
    }
}

/// Set -> Set
impl Injection for Base<Set, Set> {
    type Domain = Set;
    type CoDomain = Set;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        let data_type = From(self.domain.data_type().clone())
            .into(self.co_domain.data_type().clone())?
            .super_image(set.data_type())?;
        let size = From(self.domain.size().clone())
            .into(self.co_domain.size().clone())?
            .super_image(set.size())?;
        Ok(Set::from_data_type_size(data_type, size))
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        Ok(value::Set::from_iter(
            arg.iter()
                .map(|a| {
                    From(self.domain.data_type().clone())
                        .into(self.co_domain.data_type().clone())?
                        .value(a)
                })
                .map(|v| v.unwrap()),
        ))
    }
}

/// Array -> Array
impl Injection for Base<Array, Array> {
    type Domain = Array;
    type CoDomain = Array;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        let data_type = From(self.domain.data_type().clone())
            .into(self.co_domain.data_type().clone())?
            .super_image(set.data_type())?;
        if set.shape() == self.domain.shape() {
            Ok(Array::from_data_type_shape(data_type, set.shape()))
        } else {
            Err(Error::set_out_of_range(set, &self.domain()))
        }
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        let (values, shape) = &**arg;
        Ok(value::Array::from((
            values
                .iter()
                .map(|a| {
                    From(self.domain.data_type().clone())
                        .into(self.co_domain.data_type().clone())?
                        .value(a)
                })
                .map(|v| v.unwrap())
                .collect(),
            shape.clone(),
        )))
    }
}

/// Generic DataType -> Variant injections

/// DataType -> Struct
impl Injection for Base<DataType, Struct> {
    type Domain = DataType;
    type CoDomain = Struct;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        match (set, self.domain()) {
            (DataType::Struct(set), DataType::Struct(domain)) => {
                From(domain).into(self.co_domain())?.super_image(set)
            }
            (set, _) => self.checked_image(set, Struct::from_data_type(set.clone())),
        }
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        match (arg, self.domain()) {
            (value::Value::Struct(arg), DataType::Struct(domain)) => {
                From(domain).into(self.co_domain())?.value(arg)
            }
            (arg, _) => self.checked_value(arg, value::Struct::from_value(arg.clone())),
        }
    }
}

/// DataType -> Union
/// This could be improved by selecting more carefuly the branch of the Union
impl Injection for Base<DataType, Union> {
    type Domain = DataType;
    type CoDomain = Union;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        self.co_domain
            .fields
            .iter()
            .fold(None, |best_image, (field, term)| {
                let term_image = From(set.clone())
                    .into(term.as_ref().clone())
                    .ok()
                    .and_then(|injection| injection.super_image(set).ok());
                best_image.map_or_else(
                    || Some(Union::from_field(field, term_image.clone()?)),
                    |best_image: Union| {
                        if term_image.clone().map_or(false, |term_image| {
                            term_image.is_subset_of(&best_image.field_from_index(0).1)
                        }) {
                            Some(Union::from_field(field, term_image.clone()?))
                        } else {
                            Some(best_image)
                        }
                    },
                )
            })
            .ok_or(Error::set_out_of_range(set, self.domain()))
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        let arg_value: value::Value = arg.clone().into();
        let field = self
            .co_domain
            .fields
            .iter()
            .find(|(_, t)| t.contains(&arg_value))
            .ok_or(Error::argument_out_of_range(arg, self.domain()))?;
        Ok(value::Union::from_field(&field.0, arg_value))
    }
}

/// DataType -> Optional
impl Injection for Base<DataType, Optional> {
    type Domain = DataType;
    type CoDomain = Optional;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        match (set, self.domain()) {
            (DataType::Optional(set), DataType::Optional(domain)) => {
                From(domain).into(self.co_domain())?.super_image(set)
            }
            (set, _) => self.checked_image(
                set,
                Optional::from(set.clone().into_data_type(self.co_domain().data_type())?),
            ),
        }
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        match (arg, self.domain()) {
            (value::Value::Optional(arg), DataType::Optional(domain)) => {
                From(domain).into(self.co_domain())?.value(arg)
            }
            (arg, _) => self.checked_value(arg, value::Optional::some(arg.clone())),
        }
    }
}

/// DataType -> List
impl Injection for Base<DataType, List> {
    type Domain = DataType;
    type CoDomain = List;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        match (set, self.domain()) {
            (DataType::List(set), DataType::List(domain)) => {
                From(domain).into(self.co_domain())?.super_image(set)
            }
            (set, _) => self.checked_image(
                set,
                List::from_data_type_size(
                    set.clone().into_data_type(self.co_domain().data_type())?,
                    Integer::from_value(1),
                ),
            ),
        }
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        match (arg, self.domain()) {
            (value::Value::List(arg), DataType::List(domain)) => {
                From(domain).into(self.co_domain())?.value(arg)
            }
            (arg, _) => self.checked_value(arg, value::List::from(vec![arg.clone()])),
        }
    }
}

/// Generic Variant -> DataType injection

/// Unit -> DataType
/// TODO add more conversions
impl Injection for Base<Unit, DataType> {
    type Domain = Unit;
    type CoDomain = DataType;
    fn domain(&self) -> Self::Domain {
        self.domain
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        match self.co_domain() {
            DataType::Unit(co_domain) => Ok(From(self.domain())
                .into(co_domain)?
                .super_image(set)?
                .into()),
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        match self.co_domain() {
            DataType::Unit(co_domain) => {
                Ok(From(self.domain()).into(co_domain)?.value(arg)?.into())
            }
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
}

/// Boolean -> DataType
/// TODO add more conversions
impl Injection for Base<Boolean, DataType> {
    type Domain = Boolean;
    type CoDomain = DataType;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        match self.co_domain() {
            DataType::Null if self.domain().is_empty() => Ok(DataType::Null),
            DataType::Boolean(co_domain) => Ok(From(self.domain())
                .into(co_domain)?
                .super_image(set)?
                .into()),
            DataType::Integer(co_domain) => Ok(From(self.domain())
                .into(co_domain)?
                .super_image(set)?
                .into()),
            DataType::Float(co_domain) => Ok(From(self.domain())
                .then_default::<Integer>()
                .into(co_domain)?
                .super_image(set)?
                .into()),
            DataType::Text(co_domain) => Ok(From(self.domain())
                .into(co_domain)?
                .super_image(set)?
                .into()),
            DataType::Bytes(co_domain) => Ok(From(self.domain())
                .then_default::<Text>()
                .into(co_domain)?
                .super_image(set)?
                .into()),
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        match self.co_domain() {
            DataType::Null => Err(Error::argument_out_of_range(arg, self.domain())),
            DataType::Boolean(co_domain) => {
                Ok(From(self.domain()).into(co_domain)?.value(arg)?.into())
            }
            DataType::Integer(co_domain) => {
                Ok(From(self.domain()).into(co_domain)?.value(arg)?.into())
            }
            DataType::Float(co_domain) => Ok(From(self.domain())
                .then_default::<Integer>()
                .into(co_domain)?
                .value(arg)?
                .into()),
            DataType::Text(co_domain) => {
                Ok(From(self.domain()).into(co_domain)?.value(arg)?.into())
            }
            DataType::Bytes(co_domain) => Ok(From(self.domain())
                .then_default::<Text>()
                .into(co_domain)?
                .value(arg)?
                .into()),
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
}

/// Integer -> DataType
/// TODO add more conversions
impl Injection for Base<Integer, DataType> {
    type Domain = Integer;
    type CoDomain = DataType;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        match self.co_domain() {
            DataType::Null if self.domain().is_empty() => Ok(DataType::Null),
            DataType::Integer(co_domain) => Ok(From(self.domain())
                .into(co_domain)?
                .super_image(set)?
                .into()),
            DataType::Float(co_domain) => Ok(From(self.domain())
                .into(co_domain)?
                .super_image(set)?
                .into()),
            DataType::Text(co_domain) => Ok(From(self.domain())
                .into(co_domain)?
                .super_image(set)?
                .into()),
            DataType::Bytes(co_domain) => Ok(From(self.domain())
                .then_default::<Text>()
                .into(co_domain)?
                .super_image(set)?
                .into()),
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        match self.co_domain() {
            DataType::Null => Err(Error::argument_out_of_range(arg, self.domain())),
            DataType::Integer(co_domain) => {
                Ok(From(self.domain()).into(co_domain)?.value(arg)?.into())
            }
            DataType::Float(co_domain) => {
                Ok(From(self.domain()).into(co_domain)?.value(arg)?.into())
            }
            DataType::Text(co_domain) => {
                Ok(From(self.domain()).into(co_domain)?.value(arg)?.into())
            }
            DataType::Bytes(co_domain) => Ok(From(self.domain())
                .then_default::<Text>()
                .into(co_domain)?
                .value(arg)?
                .into()),
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
}

/// Enum -> DataType
/// TODO add more conversions
impl Injection for Base<Enum, DataType> {
    type Domain = Enum;
    type CoDomain = DataType;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        match self.co_domain() {
            DataType::Enum(co_domain) => Ok(From(self.domain())
                .into(co_domain)?
                .super_image(set)?
                .into()),
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        match self.co_domain() {
            DataType::Enum(co_domain) => {
                Ok(From(self.domain()).into(co_domain)?.value(arg)?.into())
            }
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
}

/// Float -> DataType
/// TODO add more conversions
impl Injection for Base<Float, DataType> {
    type Domain = Float;
    type CoDomain = DataType;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        match self.co_domain() {
            DataType::Null if self.domain().is_empty() => Ok(DataType::Null),
            DataType::Float(co_domain) => Ok(From(self.domain())
                .into(co_domain)?
                .super_image(set)?
                .into()),
            DataType::Text(co_domain) => Ok(From(self.domain())
                .into(co_domain)?
                .super_image(set)?
                .into()),
            DataType::Bytes(co_domain) => Ok(From(self.domain())
                .then_default::<Text>()
                .into(co_domain)?
                .super_image(set)?
                .into()),
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        match self.co_domain() {
            DataType::Null => Err(Error::argument_out_of_range(arg, self.domain())),
            DataType::Float(co_domain) => {
                Ok(From(self.domain()).into(co_domain)?.value(arg)?.into())
            }
            DataType::Text(co_domain) => {
                Ok(From(self.domain()).into(co_domain)?.value(arg)?.into())
            }
            DataType::Bytes(co_domain) => Ok(From(self.domain())
                .then_default::<Text>()
                .into(co_domain)?
                .value(arg)?
                .into()),
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
}

/// Text -> DataType
/// TODO add more conversions
impl Injection for Base<Text, DataType> {
    type Domain = Text;
    type CoDomain = DataType;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        match self.co_domain() {
            DataType::Text(co_domain) => Ok(From(self.domain())
                .into(co_domain)?
                .super_image(set)?
                .into()),
            DataType::Bytes(co_domain) => Ok(From(self.domain())
                .into(co_domain)?
                .super_image(set)?
                .into()),
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        match self.co_domain() {
            DataType::Text(co_domain) => {
                Ok(From(self.domain()).into(co_domain)?.value(arg)?.into())
            }
            DataType::Bytes(co_domain) => Ok(From(self.domain())
                .then_default::<Text>()
                .into(co_domain)?
                .value(arg)?
                .into()),
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
}

/// Bytes -> DataType
/// TODO add more conversions
impl Injection for Base<Bytes, DataType> {
    type Domain = Bytes;
    type CoDomain = DataType;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        match self.co_domain() {
            DataType::Bytes(co_domain) => Ok(From(self.domain())
                .into(co_domain)?
                .super_image(set)?
                .into()),
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        match self.co_domain() {
            DataType::Bytes(co_domain) => {
                Ok(From(self.domain()).into(co_domain)?.value(arg)?.into())
            }
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
}

/// Struct -> DataType
/// TODO add more conversions
impl Injection for Base<Struct, DataType> {
    type Domain = Struct;
    type CoDomain = DataType;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        match self.co_domain() {
            DataType::Struct(co_domain) => Ok(From(self.domain())
                .into(co_domain)?
                .super_image(set)?
                .into()),
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        match self.co_domain() {
            DataType::Struct(co_domain) => {
                Ok(From(self.domain()).into(co_domain)?.value(arg)?.into())
            }
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
}

/// Union -> DataType
impl Injection for Base<Union, DataType> {
    type Domain = Union;
    type CoDomain = DataType;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        match self.co_domain() {
            DataType::Union(co_domain) => Ok(From(self.domain())
                .into(co_domain)?
                .super_image(set)?
                .into()),
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        match self.co_domain() {
            DataType::Union(co_domain) => {
                Ok(From(self.domain()).into(co_domain)?.value(arg)?.into())
            }
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
}

/// Optional -> DataType
impl Injection for Base<Optional, DataType> {
    type Domain = Optional;
    type CoDomain = DataType;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        match self.co_domain() {
            DataType::Optional(co_domain) => Ok(From(self.domain())
                .into(co_domain)?
                .super_image(set)?
                .into()),
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        match self.co_domain() {
            DataType::Optional(co_domain) => {
                Ok(From(self.domain()).into(co_domain)?.value(arg)?.into())
            }
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
}

/// List -> DataType
impl Injection for Base<List, DataType> {
    type Domain = List;
    type CoDomain = DataType;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        match self.co_domain() {
            DataType::List(co_domain) => Ok(From(self.domain())
                .into(co_domain)?
                .super_image(set)?
                .into()),
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        match self.co_domain() {
            DataType::List(co_domain) => {
                Ok(From(self.domain()).into(co_domain)?.value(arg)?.into())
            }
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
}

/// Set -> DataType
impl Injection for Base<Set, DataType> {
    type Domain = Set;
    type CoDomain = DataType;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        match self.co_domain() {
            DataType::Set(co_domain) => Ok(From(self.domain())
                .into(co_domain)?
                .super_image(set)?
                .into()),
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        match self.co_domain() {
            DataType::Set(co_domain) => Ok(From(self.domain()).into(co_domain)?.value(arg)?.into()),
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
}

/// Array -> DataType
impl Injection for Base<Array, DataType> {
    type Domain = Array;
    type CoDomain = DataType;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        match self.co_domain() {
            DataType::Array(co_domain) => Ok(From(self.domain())
                .into(co_domain)?
                .super_image(set)?
                .into()),
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        match self.co_domain() {
            DataType::Array(co_domain) => {
                Ok(From(self.domain()).into(co_domain)?.value(arg)?.into())
            }
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
}

/// Date -> DataType
/// TODO add more conversions
impl Injection for Base<Date, DataType> {
    type Domain = Date;
    type CoDomain = DataType;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        match self.co_domain() {
            DataType::Null if self.domain().is_empty() => Ok(DataType::Null),
            DataType::Date(co_domain) => Ok(From(self.domain())
                .into(co_domain)?
                .super_image(set)?
                .into()),
            DataType::DateTime(co_domain) => Ok(From(self.domain())
                .into(co_domain)?
                .super_image(set)?
                .into()),
            DataType::Text(co_domain) => Ok(From(self.domain())
                .into(co_domain)?
                .super_image(set)?
                .into()),
            DataType::Bytes(co_domain) => Ok(From(self.domain())
                .then_default::<Text>()
                .into(co_domain)?
                .super_image(set)?
                .into()),
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        match self.co_domain() {
            DataType::Null => Err(Error::argument_out_of_range(arg, self.domain())),
            DataType::Date(co_domain) => {
                Ok(From(self.domain()).into(co_domain)?.value(arg)?.into())
            }
            DataType::DateTime(co_domain) => {
                Ok(From(self.domain()).into(co_domain)?.value(arg)?.into())
            }
            DataType::Text(co_domain) => {
                Ok(From(self.domain()).into(co_domain)?.value(arg)?.into())
            }
            DataType::Bytes(co_domain) => Ok(From(self.domain())
                .then_default::<Text>()
                .into(co_domain)?
                .value(arg)?
                .into()),
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
}

/// Time -> DataType
/// TODO add more conversions
impl Injection for Base<Time, DataType> {
    type Domain = Time;
    type CoDomain = DataType;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        match self.co_domain() {
            DataType::Null if self.domain().is_empty() => Ok(DataType::Null),
            DataType::Time(co_domain) => Ok(From(self.domain())
                .into(co_domain)?
                .super_image(set)?
                .into()),
            DataType::Text(co_domain) => Ok(From(self.domain())
                .into(co_domain)?
                .super_image(set)?
                .into()),
            DataType::Bytes(co_domain) => Ok(From(self.domain())
                .then_default::<Text>()
                .into(co_domain)?
                .super_image(set)?
                .into()),
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        match self.co_domain() {
            DataType::Null => Err(Error::argument_out_of_range(arg, self.domain())),
            DataType::Time(co_domain) => {
                Ok(From(self.domain()).into(co_domain)?.value(arg)?.into())
            }
            DataType::Text(co_domain) => {
                Ok(From(self.domain()).into(co_domain)?.value(arg)?.into())
            }
            DataType::Bytes(co_domain) => Ok(From(self.domain())
                .then_default::<Text>()
                .into(co_domain)?
                .value(arg)?
                .into()),
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
}

/// DateTime -> DataType
/// TODO add more conversions
impl Injection for Base<DateTime, DataType> {
    type Domain = DateTime;
    type CoDomain = DataType;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        match self.co_domain() {
            DataType::Null if self.domain().is_empty() => Ok(DataType::Null),
            DataType::DateTime(co_domain) => Ok(From(self.domain())
                .into(co_domain)?
                .super_image(set)?
                .into()),
            DataType::Text(co_domain) => Ok(From(self.domain())
                .into(co_domain)?
                .super_image(set)?
                .into()),
            DataType::Bytes(co_domain) => Ok(From(self.domain())
                .then_default::<Text>()
                .into(co_domain)?
                .super_image(set)?
                .into()),
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        match self.co_domain() {
            DataType::Null => Err(Error::argument_out_of_range(arg, self.domain())),
            DataType::DateTime(co_domain) => {
                Ok(From(self.domain()).into(co_domain)?.value(arg)?.into())
            }
            DataType::Text(co_domain) => {
                Ok(From(self.domain()).into(co_domain)?.value(arg)?.into())
            }
            DataType::Bytes(co_domain) => Ok(From(self.domain())
                .then_default::<Text>()
                .into(co_domain)?
                .value(arg)?
                .into()),
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
}

/// Duration -> DataType
/// TODO add more conversions
impl Injection for Base<Duration, DataType> {
    type Domain = Duration;
    type CoDomain = DataType;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        match self.co_domain() {
            DataType::Null if self.domain().is_empty() => Ok(DataType::Null),
            DataType::Duration(co_domain) => Ok(From(self.domain())
                .into(co_domain)?
                .super_image(set)?
                .into()),
            DataType::Text(co_domain) => Ok(From(self.domain())
                .into(co_domain)?
                .super_image(set)?
                .into()),
            DataType::Bytes(co_domain) => Ok(From(self.domain())
                .then_default::<Text>()
                .into(co_domain)?
                .super_image(set)?
                .into()),
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        match self.co_domain() {
            DataType::Null => Err(Error::argument_out_of_range(arg, self.domain())),
            DataType::Duration(co_domain) => {
                Ok(From(self.domain()).into(co_domain)?.value(arg)?.into())
            }
            DataType::Text(co_domain) => {
                Ok(From(self.domain()).into(co_domain)?.value(arg)?.into())
            }
            DataType::Bytes(co_domain) => Ok(From(self.domain())
                .then_default::<Text>()
                .into(co_domain)?
                .value(arg)?
                .into()),
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
}

/// Id -> DataType
/// TODO add more conversions
impl Injection for Base<Id, DataType> {
    type Domain = Id;
    type CoDomain = DataType;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        match self.co_domain() {
            DataType::Id(co_domain) => Ok(From(self.domain())
                .into(co_domain)?
                .super_image(set)?
                .into()),
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        match self.co_domain() {
            DataType::Id(co_domain) => Ok(From(self.domain()).into(co_domain)?.value(arg)?.into()),
            co_domain => Err(Error::no_injection(self.domain(), co_domain)),
        }
    }
}

// TODO Degenerate Composite Variant -> DataType
impl Injection for Base<Function, DataType> {
    type Domain = Function;
    type CoDomain = DataType;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        self.checked_image(set, set.clone().into())
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        self.checked_value(arg, arg.clone().into())
    }
}

// Empty DataTypes for injections from Null
fn null_super_image(co_domain: DataType) -> Result<DataType> {
    match co_domain {
        DataType::Null => Ok(DataType::Null),
        DataType::Boolean(_) => Ok(Boolean::empty().into()),
        DataType::Integer(_) => Ok(Integer::empty().into()),
        DataType::Float(_) => Ok(Float::empty().into()),
        DataType::Date(_) => Ok(Date::empty().into()),
        DataType::Time(_) => Ok(Time::empty().into()),
        DataType::DateTime(_) => Ok(DateTime::empty().into()),
        DataType::Duration(_) => Ok(Duration::empty().into()),
        co_domain => Err(Error::no_injection(DataType::Null, co_domain)),
    }
}

// Generic DataType -> DataType injection

/// DataType -> DataType
impl Injection for Base<DataType, DataType> {
    type Domain = DataType;
    type CoDomain = DataType;
    fn domain(&self) -> Self::Domain {
        self.domain.clone()
    }
    fn co_domain(&self) -> Self::CoDomain {
        self.co_domain.clone()
    }
    fn super_image(&self, set: &Self::Domain) -> Result<Self::CoDomain> {
        match (set, self.domain(), self.co_domain()) {
            // Some composite types first
            (set, domain, DataType::Struct(co_domain)) => {
                Ok(From(domain).into(co_domain)?.super_image(set)?.into())
            }
            (set, domain, DataType::Union(co_domain)) => {
                Ok(From(domain).into(co_domain)?.super_image(set)?.into())
            }
            (set, domain, DataType::Optional(co_domain)) => {
                Ok(From(domain).into(co_domain)?.super_image(set)?.into())
            }
            (set, domain, DataType::List(co_domain)) => {
                Ok(From(domain).into(co_domain)?.super_image(set)?.into())
            }
            // Any
            (set, _, DataType::Any) => self.checked_image(set, set.clone()),
            // Null
            (DataType::Null, DataType::Null, co_domain) => null_super_image(co_domain),
            (DataType::Boolean(set), DataType::Null, co_domain) if set.is_empty() => {
                null_super_image(co_domain)
            }
            (DataType::Integer(set), DataType::Null, co_domain) if set.is_empty() => {
                null_super_image(co_domain)
            }
            (DataType::Float(set), DataType::Null, co_domain) if set.is_empty() => {
                null_super_image(co_domain)
            }
            (DataType::Text(set), DataType::Null, co_domain) if set.is_empty() => {
                null_super_image(co_domain)
            }
            (DataType::Date(set), DataType::Null, co_domain) if set.is_empty() => {
                null_super_image(co_domain)
            }
            (DataType::Time(set), DataType::Null, co_domain) if set.is_empty() => {
                null_super_image(co_domain)
            }
            (DataType::DateTime(set), DataType::Null, co_domain) if set.is_empty() => {
                null_super_image(co_domain)
            }
            (DataType::Duration(set), DataType::Null, co_domain) if set.is_empty() => {
                null_super_image(co_domain)
            }
            // Unit
            (DataType::Unit(set), DataType::Unit(domain), co_domain) => {
                Ok(From(domain).into(co_domain)?.super_image(set)?)
            }
            // Boolean
            (DataType::Boolean(set), DataType::Boolean(domain), co_domain) => {
                Ok(From(domain).into(co_domain)?.super_image(set)?)
            }
            // Integer
            (DataType::Integer(set), DataType::Integer(domain), co_domain) => {
                Ok(From(domain).into(co_domain)?.super_image(set)?)
            }
            // Enum
            (DataType::Enum(set), DataType::Enum(domain), co_domain) => {
                Ok(From(domain).into(co_domain)?.super_image(set)?)
            }
            // Float
            (DataType::Float(set), DataType::Float(domain), co_domain) => {
                Ok(From(domain).into(co_domain)?.super_image(set)?)
            }
            // Text
            (DataType::Text(set), DataType::Text(domain), co_domain) => {
                Ok(From(domain).into(co_domain)?.super_image(set)?)
            }
            // Bytes
            (DataType::Bytes(set), DataType::Bytes(domain), co_domain) => {
                Ok(From(domain).into(co_domain)?.super_image(set)?)
            }
            // Date
            (DataType::Date(set), DataType::Date(domain), co_domain) => {
                Ok(From(domain).into(co_domain)?.super_image(set)?)
            }
            // Time
            (DataType::Time(set), DataType::Time(domain), co_domain) => {
                Ok(From(domain).into(co_domain)?.super_image(set)?)
            }
            // DateTime
            (DataType::DateTime(set), DataType::DateTime(domain), co_domain) => {
                Ok(From(domain).into(co_domain)?.super_image(set)?)
            }
            // Duration
            (DataType::Duration(set), DataType::Duration(domain), co_domain) => {
                Ok(From(domain).into(co_domain)?.super_image(set)?)
            }
            // Id
            (DataType::Id(set), DataType::Id(domain), co_domain) => {
                Ok(From(domain).into(co_domain)?.super_image(set)?)
            }
            // Anything else
            (_, domain, co_domain) => Err(Error::no_injection(domain, co_domain)),
        }
    }
    fn value(
        &self,
        arg: &<Self::Domain as Variant>::Element,
    ) -> Result<<Self::CoDomain as Variant>::Element> {
        match (arg, self.domain(), self.co_domain()) {
            // Some composite types first
            (arg, domain, DataType::Struct(co_domain)) => {
                Ok(From(domain).into(co_domain)?.value(arg)?.into())
            }
            (arg, domain, DataType::Union(co_domain)) => {
                Ok(From(domain).into(co_domain)?.value(arg)?.into())
            }
            (arg, domain, DataType::Optional(co_domain)) => {
                Ok(From(domain).into(co_domain)?.value(arg)?.into())
            }
            (arg, domain, DataType::List(co_domain)) => {
                Ok(From(domain).into(co_domain)?.value(arg)?.into())
            }
            // Any
            (arg, _, DataType::Any) => self.checked_value(arg, arg.clone()),
            // Null
            (arg, DataType::Null, _) => Err(Error::argument_out_of_range(arg, self.domain())),
            // Unit
            (value::Value::Unit(arg), DataType::Unit(domain), co_domain) => {
                Ok(From(domain).into(co_domain)?.value(arg)?)
            }
            // Boolean
            (value::Value::Boolean(arg), DataType::Boolean(domain), co_domain) => {
                Ok(From(domain).into(co_domain)?.value(arg)?)
            }
            // Integer
            (value::Value::Integer(arg), DataType::Integer(domain), co_domain) => {
                Ok(From(domain).into(co_domain)?.value(arg)?)
            }
            // Enum
            (value::Value::Enum(arg), DataType::Enum(domain), co_domain) => {
                Ok(From(domain).into(co_domain)?.value(arg)?)
            }
            // Float
            (value::Value::Float(arg), DataType::Float(domain), co_domain) => {
                Ok(From(domain).into(co_domain)?.value(arg)?)
            }
            // Text
            (value::Value::Text(arg), DataType::Text(domain), co_domain) => {
                Ok(From(domain).into(co_domain)?.value(arg)?)
            }
            // Bytes
            (value::Value::Bytes(arg), DataType::Bytes(domain), co_domain) => {
                Ok(From(domain).into(co_domain)?.value(arg)?)
            }
            // Date
            (value::Value::Date(arg), DataType::Date(domain), co_domain) => {
                Ok(From(domain).into(co_domain)?.value(arg)?)
            }
            // Time
            (value::Value::Time(arg), DataType::Time(domain), co_domain) => {
                Ok(From(domain).into(co_domain)?.value(arg)?)
            }
            // DateTime
            (value::Value::DateTime(arg), DataType::DateTime(domain), co_domain) => {
                Ok(From(domain).into(co_domain)?.value(arg)?)
            }
            // Duration
            (value::Value::Duration(arg), DataType::Duration(domain), co_domain) => {
                Ok(From(domain).into(co_domain)?.value(arg)?)
            }
            // Id
            (value::Value::Id(arg), DataType::Id(domain), co_domain) => {
                Ok(From(domain).into(co_domain)?.value(arg)?)
            }
            // Anything else
            (_, domain, co_domain) => Err(Error::no_injection(domain, co_domain)),
        }
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

// TODO Write tests for all types
#[cfg(test)]
mod tests {
    use super::{value::Value, *};
    #[cfg(feature = "checked_injections")]
    use std::panic::catch_unwind;

    fn test_injection<
        Dom: Variant,
        CoDom: Variant,
        Inj: Injection<Domain = Dom, CoDomain = CoDom>,
    >(
        inj: Inj,
        value: Dom::Element,
        expected_result: CoDom::Element,
        set: Dom,
        expected_image: CoDom,
    ) where
        CoDom: fmt::Debug + fmt::Display,
        CoDom::Element: fmt::Debug + fmt::Display,
    {
        println!("{} -> {}", inj.domain(), inj.co_domain());
        // Call
        let result = inj.value(&value).unwrap();
        println!("{}", result);
        assert_eq!(result, expected_result);
        // Image
        let image = inj.super_image(&set).unwrap();
        println!("{}", image);
        assert_eq!(image, expected_image);
    }

    #[test]
    #[cfg(feature = "checked_injections")]
    fn test_injection_building() {
        // null-null
        let null_null = From(DataType::Null).into(DataType::Null);
        println!("{:?}", null_null);
        assert!(null_null.is_ok());
        // interval-null
        let interval_null = From(Float::from_interval(0., 1.)).into(DataType::Null);
        println!("{:?}", interval_null);
        assert!(interval_null.is_err());
        // empty-interval-null
        let empty_interval_null = From(Float::empty()).into(DataType::Null);
        println!("{:?}", empty_interval_null);
        assert!(empty_interval_null.is_ok());
        // null-interval
        let null_empty_interval = From(DataType::Null).into(DataType::from(Float::empty()));
        println!("{:?}", null_empty_interval);
        assert!(null_empty_interval.is_ok());
        // null-empty-interval
        let null_interval = From(DataType::Null).into(DataType::from(Float::from_interval(0., 1.)));
        println!("{:?}", null_interval);
        assert!(null_interval.is_ok());
    }

    #[test]
    fn test_injection_bool_int() {
        test_injection(
            From(Boolean::default()).into(Integer::default()).unwrap(),
            true.into(),
            value::Integer::from(1),
            [true, false].iter().collect(),
            [1, 0].iter().collect(),
        );
    }

    #[test]
    fn test_injection_int_bool() {
        test_injection(
            From(Integer::from_interval(0, 1))
                .into(Boolean::default())
                .unwrap(),
            1.into(),
            true.into(),
            [1, 0].iter().collect(),
            [true, false].iter().collect(),
        );
    }

    #[test]
    fn test_injection_int_float() {
        test_injection(
            From(Integer::from_interval(0, 10))
                .into(Float::from_interval(0., 10.))
                .unwrap(),
            2.into(),
            value::Float::from(2.),
            [1, 2].iter().collect(),
            [1., 2.].iter().collect(),
        );
    }

    #[test]
    #[cfg(feature = "checked_injections")]
    fn test_injection_int_float_too_small() {
        // This would fail because of the panicking check() in the builder
        assert!(catch_unwind(|| From(Integer::from_interval(0, 10))
            .into(Float::from_interval(0., 5.))
            .unwrap())
        .is_err());
        let inj = From(Integer::from_interval(0, 5))
            .into(Float::from_interval(0., 5.))
            .unwrap();
        assert!(catch_unwind(|| inj.value(&2.into()).unwrap())
            .map(|res| res == value::Float::from(2.))
            .unwrap());
        assert!(catch_unwind(|| inj.value(&8.into()).unwrap()).is_err());
        assert!(
            catch_unwind(|| inj.super_image(&[1, 3].iter().collect()).unwrap())
                .map(|im| im == [1., 3.].iter().collect())
                .unwrap()
        );
        assert!(catch_unwind(|| inj.super_image(&[1, 20].iter().collect()).unwrap()).is_err());
    }

    #[test]
    fn test_injection_int_text() {
        test_injection(
            From(Integer::from_interval(0, 10))
                .into(Text::default())
                .unwrap(),
            2.into(),
            value::Text::from("2".to_string()),
            [1, 2].iter().collect(),
            ["1".to_string(), "2".to_string()].iter().collect(),
        );
    }

    #[test]
    fn test_injection_float_text() {
        test_injection(
            From(Float::from_interval(0., 10.))
                .into(Text::default())
                .unwrap(),
            2.0.into(),
            value::Text::from("2".to_string()),
            [1., 2.].iter().collect(),
            ["1".to_string(), "2".to_string()].iter().collect(),
        );
    }

    #[test]
    fn test_injection_bool_float() {
        test_injection(
            From(Boolean::default())
                .then(Integer::from_interval(0, 10))
                .into(Float::default())
                .unwrap(),
            true.into(),
            value::Float::from(1.),
            [false, true].iter().collect(),
            [0., 1.].iter().collect(),
        );
    }

    #[test]
    fn test_injection_struct_struct() {
        let inj = From(Struct::from_data_types(&[
            DataType::from(Value::from(1.0)),
            DataType::from(Value::from(0.5)),
        ]))
        .into(Struct::from_data_types(&[
            DataType::float(),
            DataType::float(),
        ]))
        .unwrap();
        println!("{inj}");
        let inj = From(Struct::from_data_types(&[
            DataType::integer(),
            DataType::from(Value::from(0.5)),
            DataType::float_interval(0., 1.),
        ]))
        .into(Struct::from_data_types(&[
            DataType::float(),
            DataType::float(),
        ]))
        .unwrap();
        println!("{inj}");
        let val = value::Struct::from_values(&[1.into(), 0.5.into(), 0.5.into()]);
        println!("{} -> {}", val, inj.value(&val).unwrap());
    }

    #[test]
    fn test_injection_union_union() {
        let inj = From(Union::from_data_types(&[
            DataType::integer(),
            DataType::from(Value::from(0.5)),
        ]))
        .into(Union::from_data_types(&[
            DataType::float(),
            DataType::float(),
            DataType::float_interval(0., 1.),
        ]))
        .unwrap();
        println!("{inj}");
        let val = value::Union::from_field("1", 0.5);
        println!("{} -> {}", val, inj.value(&val).unwrap());
    }

    #[test]
    fn test_injection_optional_optional() {
        let inj = From(Optional::from_data_type(DataType::integer()))
            .into(Optional::from_data_type(DataType::float()))
            .unwrap();
        println!("{inj}");
    }

    #[test]
    fn test_injection_into_optional() {
        let inj = From(DataType::float_interval(0., 10.))
            .into(Optional::from(DataType::float()))
            .unwrap();
        println!("{inj}");
        test_injection(
            From(DataType::float_interval(0., 10.))
                .into(Optional::from(DataType::float()))
                .unwrap(),
            0.5.into(),
            value::Optional::some(Value::float(0.5)),
            DataType::float_values([0.1, 0.5]),
            Optional::from(DataType::float_values([0.1, 0.5])),
        );
    }

    #[test]
    fn test_injection_into_union() {
        let inj = From(DataType::float_interval(0., 10.))
            .into(
                Union::null()
                    .or(DataType::float())
                    .or(DataType::integer_interval(0, 10)),
            )
            .unwrap();
        println!("{inj}");
        test_injection(
            From(DataType::integer_interval(0, 6))
                .into(Union::null().or(DataType::float().or(DataType::integer_interval(0, 10))))
                .unwrap(),
            5.into(),
            value::Union::from_field("1", Value::integer(5)),
            DataType::integer_values([1, 5]),
            Union::from_field("1", DataType::integer_values([1, 5])),
        );
    }

    #[test]
    fn test_complex_injection_into_union() {
        let f = DataType::float();
        println!("\nf = {f}");
        let t = DataType::Union(Union::from_data_types(
            vec![DataType::float(), DataType::float()].as_slice(),
        ));
        println!("t = {t}");
        let ft = f.into_data_type(&t).unwrap();
        println!("ft = {ft}");
        assert_eq!(
            ft,
            DataType::Union(Union::from_field("1", DataType::float()))
        );

        let f = DataType::Any;
        println!("\nf = {f}");
        let t = DataType::Union(Union::from_data_types(
            vec![DataType::Any, DataType::Any].as_slice(),
        ));
        println!("t = {t}");
        let ft = f.into_data_type(&t).unwrap();
        println!("ft = {ft}");
        assert_eq!(ft, DataType::Union(Union::from_field("1", DataType::Any)));

        let f = DataType::list(DataType::Any, 1, 100);
        println!("\nf = {f}");
        let t = DataType::list(DataType::optional(DataType::Any), 1, 200)
            | DataType::list(DataType::Any, 1, 200);
        println!("t = {t}");
        let ft = f.into_data_type(&t).unwrap();
        println!("ft = {ft}");
        assert_eq!(
            &ft.to_string(),
            "union{0: list(option(any), size ∈ int[1 100])}"
        );

        let f = DataType::float();
        println!("\nf = {f}");
        let t = DataType::optional(DataType::Union(Union::from_data_types(
            vec![DataType::float(), DataType::float()].as_slice(),
        )));
        println!("t = {t}");
        let ft = f.into_data_type(&t).unwrap();
        println!("ft = {ft}");
        assert_eq!(
            ft,
            DataType::optional(DataType::Union(Union::from_field("1", DataType::float())))
        );

        let f = DataType::float();
        println!("\nf = {f}");
        let t = DataType::optional(DataType::float()) | DataType::float();
        println!("t = {t}");
        let ft = f.into_data_type(&t).unwrap();
        println!("ft = {ft}");
        assert_eq!(
            ft,
            DataType::optional(DataType::Union(Union::from_field("1", DataType::float())))
        );

        let f = DataType::Any;
        println!("\nf = {f}");
        let t = DataType::optional(DataType::Any) | DataType::Any;
        println!("t = {t}");

        let ft = f.into_data_type(&t).unwrap();
        println!("ft = {ft}");
        assert_eq!(
            ft,
            DataType::optional(DataType::Union(Union::from_field("1", DataType::Any)))
        );
    }

    #[test]
    fn test_injection_into_list() {
        let inj = From(DataType::float_interval(0., 10.))
            .into(List::from_data_type_size(
                DataType::float(),
                Integer::from_interval(1, 100),
            ))
            .unwrap();
        println!("{inj}");
        test_injection(
            From(DataType::float_interval(0., 10.))
                .into(List::from_data_type_size(
                    DataType::float(),
                    Integer::from_interval(1, 100),
                ))
                .unwrap(),
            5.0.into(),
            value::List::from(vec![Value::float(5.0)]),
            DataType::integer_values([1, 5]),
            List::from_data_type_size(DataType::float_values([1., 5.]), Integer::from_value(1)),
        );
    }

    #[test]
    fn test_is_subset() {
        let dom = DataType::float();
        println!("dom = {dom}");
        let co_dom = DataType::optional(DataType::float());
        println!("co_dom = {co_dom}");
        println!("dom ⊂ co_dom = {}", dom.is_subset_of(&co_dom));
    }

    #[test]
    fn test_complex_inclusions() {
        let dom = DataType::integer_range(3..=190);
        println!("dom = {dom}");
        let co_dom = DataType::list(DataType::integer(), 0, i64::MAX as usize);
        println!("co_dom = {co_dom}");
        println!("dom ⊂ co_dom = {}", dom.is_subset_of(&co_dom));
        let union_co_dom = co_dom.clone() | DataType::list(DataType::float(), 0, i64::MAX as usize);
        println!("union_co_dom = {union_co_dom}");
        println!(
            "co_dom ⊂ union_co_dom = {}",
            co_dom.is_subset_of(&union_co_dom)
        );
        println!("dom ⊂ union_co_dom = {}", dom.is_subset_of(&union_co_dom));
    }
}
