//! A simple Intervals definition

use chrono::{self, NaiveDate};
use itertools::Itertools;
use std::{
    cmp,
    collections::BTreeSet,
    fmt, hash,
    iter::Iterator,
    ops::{self, Deref},
    vec,
};

/// A trait for types to be used as a base for `Intervals`
pub trait Bound: PartialEq + PartialOrd + Clone + fmt::Debug + fmt::Display {
    fn name() -> String;
    fn min() -> Self;
    fn max() -> Self;
    fn hash<H: hash::Hasher>(&self, state: &mut H);
}

// Some implementations

/// Bool
impl Bound for bool {
    fn name() -> String {
        "bool".to_string()
    }
    fn min() -> Self {
        false
    }
    fn max() -> Self {
        true
    }
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        hash::Hash::hash(self, state)
    }
}

/// Integer
impl Bound for i64 {
    fn name() -> String {
        "int".to_string()
    }
    fn min() -> Self {
        i64::MIN
    }
    fn max() -> Self {
        i64::MAX
    }
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        hash::Hash::hash(self, state)
    }
}

/// Float
impl Bound for f64 {
    fn name() -> String {
        "float".to_string()
    }
    fn min() -> Self {
        f64::MIN
    }
    fn max() -> Self {
        f64::MAX
    }
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        hash::Hash::hash(&self.to_be_bytes(), state)
    }
}

/// String
/// Order for strings is lexicographic order byte-wise.
/// We use a very high UTF-8 character to code the
impl Bound for String {
    fn name() -> String {
        "str".to_string()
    }
    fn min() -> Self {
        "\u{00}".to_string()
    }
    fn max() -> Self {
        "\u{10FFFF}".to_string()
    }
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        hash::Hash::hash(self, state)
    }
}

/// Date
impl Bound for chrono::NaiveDate {
    fn name() -> String {
        "date".to_string()
    }
    fn min() -> Self {
        chrono::naive::MIN_DATE
    }
    fn max() -> Self {
        chrono::naive::MAX_DATE
    }
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        hash::Hash::hash(self, state)
    }
}

/// Time
impl Bound for chrono::NaiveTime {
    fn name() -> String {
        "time".to_string()
    }
    fn min() -> Self {
        chrono::NaiveTime::from_num_seconds_from_midnight(0, 0)
    }
    fn max() -> Self {
        chrono::NaiveTime::from_num_seconds_from_midnight(86399, 1_999_999_999)
    }
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        hash::Hash::hash(self, state)
    }
}

/// DateTime
impl Bound for chrono::NaiveDateTime {
    fn name() -> String {
        "datetime".to_string()
    }
    fn min() -> Self {
        chrono::naive::MIN_DATETIME
    }
    fn max() -> Self {
        chrono::naive::MAX_DATETIME
    }
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        hash::Hash::hash(self, state)
    }
}

/// Duration
impl Bound for chrono::Duration {
    fn name() -> String {
        "duration".to_string()
    }
    fn min() -> Self {
        chrono::Duration::min_value()
    }
    fn max() -> Self {
        chrono::Duration::max_value()
    }
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        hash::Hash::hash(self, state)
    }
}
/// A bunch of intervals
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Intervals<B: Bound> {
    capacity: usize,
    intervals: Vec<[B; 2]>,
}

const CAPACITY: usize = 1 << 7; // Above 128 we shorten the structure

impl<B: Bound> Intervals<B> {
    /// The only way to build `Intervals`
    pub fn new() -> Intervals<B> {
        let result = Intervals {
            capacity: CAPACITY,
            intervals: Vec::new(),
        };
        // Intervals are shortened to enforce capacity
        result.to_simple_superset()
    }

    /// Build an empty `Intervals` set
    pub fn empty() -> Intervals<B> {
        Intervals::new()
    }

    /// Build a full `Intervals` set
    pub fn full() -> Intervals<B> {
        Intervals::from_interval(B::min(), B::max())
    }

    // Basic boolean operations

    /// Union with a single interval
    pub fn union_interval(mut self, min: B, max: B) -> Self {
        // Make sure min and max are in the right order
        assert!(min <= max);
        // Find the insertion points of the new interval
        // min_index will be the first such that self.bounds[min_index-1][1] < min <= self.bounds[min_index][1]
        let min_index: usize = self
            .intervals
            .iter()
            .position(|[_, iter_max]| &min <= iter_max)
            .unwrap_or(self.intervals.len());
        // max_index will be the first such that self.bounds[max_index-1][0] <= max < self.bounds[max_index][0]
        let max_index: usize = self
            .intervals
            .iter()
            .position(|[iter_min, _]| &max < iter_min)
            .unwrap_or(self.intervals.len());
        // Compute new min and max
        let min = if min_index < self.intervals.len() && self.intervals[min_index][0] < min {
            self.intervals[min_index][0].clone()
        } else {
            min
        };
        let max = if max_index > 0 && max < self.intervals[max_index - 1][1] {
            self.intervals[max_index - 1][1].clone()
        } else {
            max
        };
        // Insert the new interval and move the existing values
        self.intervals.drain(min_index..max_index);
        self.intervals.insert(min_index, [min, max]);
        // Set the new length
        self
    }

    /// Union with a single value
    pub fn union_value(self, value: B) -> Self {
        self.union_interval(value.clone(), value)
    }

    /// Union with a lower-bounded interval
    pub fn union_min(self, min: B) -> Self {
        self.union_interval(min, B::max())
    }

    /// Union with an upper-bounded interval
    pub fn union_max(self, max: B) -> Self {
        self.union_interval(B::min(), max)
    }

    /// Union with a range
    pub fn union_range<R: ops::RangeBounds<B>>(self, range: R) -> Self {
        match (range.start_bound(), range.end_bound()) {
            (ops::Bound::Included(min), ops::Bound::Included(max)) => {
                self.union_interval(min.clone(), max.clone())
            }
            (ops::Bound::Included(min), ops::Bound::Unbounded) => {
                self.union_interval(min.clone(), B::max())
            }
            (ops::Bound::Unbounded, ops::Bound::Included(max)) => {
                self.union_interval(B::min(), max.clone())
            }
            (ops::Bound::Unbounded, ops::Bound::Unbounded) => {
                self.union_interval(B::min(), B::max())
            }
            _ => panic!("Only closed ranges are admitted in an `Intervals` object."),
        }
    }

    /// A union of 2 `Intervals` objects
    pub fn union(self, other: Intervals<B>) -> Self {
        if other.len() <= self.len() {
            other
                .intervals
                .into_iter()
                .fold(self, |result, [min, max]| result.union_interval(min, max))
        } else {
            other.union(self)
        }
    }

    /// Intersection with a single interval
    pub fn intersection_interval(mut self, min: B, max: B) -> Self {
        // Make sure min and max are in the right order
        assert!(min <= max);
        // Find the insertion points of the new interval
        // min_index will be the first such that self.bounds[min_index-1][1] < min <= self.bounds[min_index][1]
        let min_index: usize = self
            .intervals
            .iter()
            .position(|[_, iter_max]| &min <= iter_max)
            .unwrap_or(self.intervals.len());
        // max_index will be the first such that self.bounds[max_index-1][0] <= max < self.bounds[max_index][0]
        let max_index: usize = self
            .intervals
            .iter()
            .position(|[iter_min, _]| &max < iter_min)
            .unwrap_or(self.intervals.len());
        // Compute new min and max
        let min = if min_index < self.intervals.len() && min < self.intervals[min_index][0] {
            self.intervals[min_index][0].clone()
        } else {
            min
        };
        let max = if max_index > 0 && self.intervals[max_index - 1][1] < max {
            self.intervals[max_index - 1][1].clone()
        } else {
            max
        };
        // Insert the new interval
        if min_index < self.intervals.len() {
            self.intervals[min_index][0] = min;
        }
        if max_index > 0 {
            self.intervals[max_index - 1][1] = max;
        }
        // Drop some values starting by the end
        if max_index < self.intervals.len() {
            self.intervals.drain(max_index..self.intervals.len());
        }
        if min_index > 0 {
            self.intervals.drain(0..min_index);
        }
        // Set the new length
        self
    }

    /// Intersection with a single value
    pub fn intersection_value(self, value: B) -> Self {
        self.intersection_interval(value.clone(), value)
    }

    /// Intersection with a lower-bounded interval
    pub fn intersection_min(self, min: B) -> Self {
        self.intersection_interval(min, B::max())
    }

    /// Intersection with an upper-bounded interval
    pub fn intersection_max(self, max: B) -> Self {
        self.intersection_interval(B::min(), max)
    }

    /// Intersection with a range
    pub fn intersection_range<R: ops::RangeBounds<B>>(self, range: R) -> Self {
        match (range.start_bound(), range.end_bound()) {
            (ops::Bound::Included(min), ops::Bound::Included(max)) => {
                self.intersection_interval(min.clone(), max.clone())
            }
            (ops::Bound::Included(min), ops::Bound::Unbounded) => {
                self.intersection_interval(min.clone(), B::max())
            }
            (ops::Bound::Unbounded, ops::Bound::Included(max)) => {
                self.intersection_interval(B::min(), max.clone())
            }
            (ops::Bound::Unbounded, ops::Bound::Unbounded) => {
                self.intersection_interval(B::min(), B::max())
            }
            _ => panic!("Only closed ranges are admitted in an `Intervals` object."),
        }
    }

    /// An intersection of 2 `Intervals` objects
    pub fn intersection(self, other: Intervals<B>) -> Self {
        if other.len() <= self.len() {
            other
                .intervals
                .into_iter()
                .map(|[min, max]| self.clone().intersection_interval(min, max))
                .fold(Intervals::empty(), |result, intervals| {
                    result.union(intervals)
                })
        } else {
            other.intersection(self)
        }
    }

    // conversion builders

    /// Create from interval
    pub fn from_interval(min: B, max: B) -> Intervals<B> {
        Intervals::new().union_interval(min, max)
    }

    /// Create from lower bound
    pub fn from_min(min: B) -> Intervals<B> {
        Intervals::new().union_min(min)
    }

    /// Create from upper bound
    pub fn from_max(max: B) -> Intervals<B> {
        Intervals::new().union_max(max)
    }

    /// Create a singleton value
    pub fn from_value(value: B) -> Intervals<B> {
        Intervals::new().union_value(value)
    }

    /// Create from range
    pub fn from_range<R: ops::RangeBounds<B>>(range: R) -> Intervals<B> {
        Intervals::new().union_range(range)
    }

    /// Create from multiple values
    pub fn from_values<A: AsRef<[B]>>(values: A) -> Intervals<B> {
        values
            .as_ref()
            .iter()
            .fold(Intervals::new(), |intervals, value| {
                intervals.union_value(value.clone())
            })
    }

    /// Create from multiple intervals
    pub fn from_intervals<A: AsRef<[[B; 2]]>>(intervals: A) -> Intervals<B> {
        intervals
            .as_ref()
            .iter()
            .fold(Intervals::new(), |intervals, [min, max]| {
                intervals.union_interval(min.clone(), max.clone())
            })
    }

    // Accessors

    pub fn len(&self) -> usize {
        self.intervals.len()
    }

    pub fn min(&self) -> Option<&B> {
        Some(&self.intervals.first()?[0])
    }

    pub fn max(&self) -> Option<&B> {
        Some(&self.intervals.last()?[1])
    }

    pub fn is_empty(&self) -> bool {
        self.intervals.is_empty()
    }

    pub fn all_values(&self) -> bool {
        self.intervals.iter().all(|[min, max]| min == max)
    }

    /// Test if the interval is a singleton
    pub fn is_value(&self) -> bool {
        self.intervals.len() == 1 && self.intervals[0][0] == self.intervals[0][1]
    }

    /// Convert an `Intervals` into a simple `Intervals`
    pub fn into_interval(self) -> Intervals<B> {
        match (self.min(), self.max()) {
            (Some(min), Some(max)) => Intervals::from_interval(min.clone(), max.clone()),
            _ => Intervals::empty(),
        }
    }

    /// When the structure is too large, merge into a super-interval
    pub fn to_simple_superset(self) -> Intervals<B> {
        if self.intervals.len() < self.capacity {
            self
        } else {
            self.into_interval()
        }
    }

    /// `self` is a subset of `other`
    pub fn is_subset_of(&self, other: &Intervals<B>) -> bool {
        &self.clone().intersection(other.clone()) == self
    }

    /// `self` is a superset of `other`
    pub fn is_superset_of(&self, other: &Intervals<B>) -> bool {
        other.is_subset_of(self)
    }

    /// `self` contains `value`
    pub fn contains(&self, value: &B) -> bool {
        self.is_superset_of(&Intervals::from(value.clone()))
    }

    /// Apply a function to each interval
    pub fn map<C: Bound, F: Fn(B, B) -> [C; 2]>(self, f: F) -> Intervals<C> {
        self.into_iter().map(|[min, max]| f(min, max)).collect()
    }

    /// Apply a function to each bound
    pub fn map_bounds<C: Bound, F: Fn(B) -> C>(self, f: F) -> Intervals<C> {
        self.into_iter()
            .map(|[min, max]| [f(min), f(max)])
            .collect()
    }

    pub fn flat_map<C: Bound, F: Fn(B, B) -> Intervals<C>>(self, f: F) -> Intervals<C> {
        self.into_iter()
            .map(|[min, max]| f(min, max))
            .fold(Intervals::empty(), |result, intervals| result | intervals)
    }

    pub fn for_each<F: FnMut(B, B)>(self, mut f: F) {
        self.into_iter().for_each(|[min, max]| f(min, max));
    }
}

impl<B: Bound> Default for Intervals<B> {
    fn default() -> Self {
        Intervals::full()
    }
}

#[allow(clippy::derive_hash_xor_eq)]
impl<B: Bound> hash::Hash for Intervals<B> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.intervals.iter().for_each(|[min, max]| {
            min.hash(state);
            max.hash(state);
        })
    }
}

impl<B: Bound> Deref for Intervals<B> {
    type Target = [[B; 2]];

    fn deref(&self) -> &Self::Target {
        self.intervals.deref()
    }
}

// Some conversions

/// From a value
impl<B: Bound> From<B> for Intervals<B> {
    fn from(from: B) -> Intervals<B> {
        Intervals::from_value(from)
    }
}

/// From an interval
impl<B: Bound> From<[B; 2]> for Intervals<B> {
    fn from(from: [B; 2]) -> Intervals<B> {
        let [min, max] = from;
        Intervals::from_interval(min, max)
    }
}

/// From a range
impl<B: Bound> From<ops::RangeInclusive<B>> for Intervals<B> {
    fn from(from: ops::RangeInclusive<B>) -> Intervals<B> {
        Intervals::from_range(from)
    }
}

impl<B: Bound> From<ops::RangeFrom<B>> for Intervals<B> {
    fn from(from: ops::RangeFrom<B>) -> Intervals<B> {
        Intervals::from_range(from)
    }
}

impl<B: Bound> From<ops::RangeToInclusive<B>> for Intervals<B> {
    fn from(from: ops::RangeToInclusive<B>) -> Intervals<B> {
        Intervals::from_range(from)
    }
}

impl<B: Bound> From<ops::RangeFull> for Intervals<B> {
    fn from(_: ops::RangeFull) -> Intervals<B> {
        Intervals::full()
    }
}

/// From an `Intervals` object
impl<B: Bound> From<Intervals<B>> for Vec<[B; 2]> {
    fn from(from: Intervals<B>) -> Vec<[B; 2]> {
        from.intervals
    }
}

/// From an iterator of intervals
impl<'a, B: Bound + 'a> FromIterator<[&'a B; 2]> for Intervals<B> {
    fn from_iter<I: IntoIterator<Item = [&'a B; 2]>>(iter: I) -> Self {
        iter.into_iter()
            .fold(Intervals::new(), |intervals, [min, max]| {
                intervals.union_interval(min.clone(), max.clone())
            })
    }
}

/// From an iterator of values
impl<'a, B: Bound + 'a> FromIterator<&'a B> for Intervals<B> {
    fn from_iter<I: IntoIterator<Item = &'a B>>(iter: I) -> Self {
        iter.into_iter().fold(Intervals::new(), |intervals, value| {
            intervals.union_value(value.clone())
        })
    }
}

/// From an iterator of intervals
impl<B: Bound> FromIterator<[B; 2]> for Intervals<B> {
    fn from_iter<I: IntoIterator<Item = [B; 2]>>(iter: I) -> Self {
        iter.into_iter()
            .fold(Intervals::new(), |intervals, [min, max]| {
                intervals.union_interval(min, max)
            })
    }
}

/// From an iterator of values
impl<B: Bound> FromIterator<B> for Intervals<B> {
    fn from_iter<I: IntoIterator<Item = B>>(iter: I) -> Self {
        iter.into_iter().fold(Intervals::new(), |intervals, value| {
            intervals.union_value(value)
        })
    }
}

/// Into an iterator of intervals
impl<B: Bound> IntoIterator for Intervals<B> {
    type Item = [B; 2];
    type IntoIter = vec::IntoIter<[B; 2]>;

    fn into_iter(self) -> Self::IntoIter {
        self.intervals.into_iter()
    }
}

/// From a slice of intervals
impl<B: Bound> From<&[[B; 2]]> for Intervals<B> {
    fn from(from: &[[B; 2]]) -> Intervals<B> {
        from.iter()
            .map(|[min, max]| [min.clone(), max.clone()])
            .collect()
    }
}

/// From a slice of values
impl<B: Bound> From<&[B]> for Intervals<B> {
    fn from(from: &[B]) -> Intervals<B> {
        from.iter().collect()
    }
}

// Unions and intersections

/// Union
impl<B: Bound> ops::BitOr for Intervals<B> {
    type Output = Intervals<B>;

    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}

/// Union with an interval
impl<B: Bound> ops::BitOr<[B; 2]> for Intervals<B> {
    type Output = Intervals<B>;

    fn bitor(self, rhs: [B; 2]) -> Self::Output {
        let [min, max] = rhs;
        self.union_interval(min, max)
    }
}

/// Union with a value
impl<B: Bound> ops::BitOr<B> for Intervals<B> {
    type Output = Intervals<B>;

    fn bitor(self, rhs: B) -> Self::Output {
        self.union_value(rhs)
    }
}

/// Union with a range
impl<B: Bound> ops::BitOr<ops::RangeInclusive<B>> for Intervals<B> {
    type Output = Intervals<B>;

    fn bitor(self, rhs: ops::RangeInclusive<B>) -> Self::Output {
        self.union_range(rhs)
    }
}

/// Union with a range
impl<B: Bound> ops::BitOr<ops::RangeFrom<B>> for Intervals<B> {
    type Output = Intervals<B>;

    fn bitor(self, rhs: ops::RangeFrom<B>) -> Self::Output {
        self.union_range(rhs)
    }
}

/// Union with a range
impl<B: Bound> ops::BitOr<ops::RangeToInclusive<B>> for Intervals<B> {
    type Output = Intervals<B>;

    fn bitor(self, rhs: ops::RangeToInclusive<B>) -> Self::Output {
        self.union_range(rhs)
    }
}

/// Union with a range
impl<B: Bound> ops::BitOr<ops::RangeFull> for Intervals<B> {
    type Output = Intervals<B>;

    fn bitor(self, rhs: ops::RangeFull) -> Self::Output {
        self.union_range(rhs)
    }
}

/// Intersection
impl<B: Bound> ops::BitAnd for Intervals<B> {
    type Output = Intervals<B>;

    fn bitand(self, rhs: Self) -> Self::Output {
        self.intersection(rhs)
    }
}

/// Intersection with an interval
impl<B: Bound> ops::BitAnd<[B; 2]> for Intervals<B> {
    type Output = Intervals<B>;

    fn bitand(self, rhs: [B; 2]) -> Self::Output {
        let [min, max] = rhs;
        self.intersection_interval(min, max)
    }
}

/// Intersection with a value
impl<B: Bound> ops::BitAnd<B> for Intervals<B> {
    type Output = Intervals<B>;

    fn bitand(self, rhs: B) -> Self::Output {
        self.intersection_value(rhs)
    }
}

/// Intersection with a range
impl<B: Bound> ops::BitAnd<ops::RangeInclusive<B>> for Intervals<B> {
    type Output = Intervals<B>;

    fn bitand(self, rhs: ops::RangeInclusive<B>) -> Self::Output {
        self.intersection_range(rhs)
    }
}

/// Intersection with a range
impl<B: Bound> ops::BitAnd<ops::RangeFrom<B>> for Intervals<B> {
    type Output = Intervals<B>;

    fn bitand(self, rhs: ops::RangeFrom<B>) -> Self::Output {
        self.intersection_range(rhs)
    }
}

/// Intersection with a range
impl<B: Bound> ops::BitAnd<ops::RangeToInclusive<B>> for Intervals<B> {
    type Output = Intervals<B>;

    fn bitand(self, rhs: ops::RangeToInclusive<B>) -> Self::Output {
        self.intersection_range(rhs)
    }
}

/// Intersection with a range
impl<B: Bound> ops::BitAnd<ops::RangeFull> for Intervals<B> {
    type Output = Intervals<B>;

    fn bitand(self, rhs: ops::RangeFull) -> Self::Output {
        self.intersection_range(rhs)
    }
}

/// Implement `PartialOrd` for `Intervals<B>`
impl<B: Bound> cmp::PartialOrd for Intervals<B> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        match (
            self.capacity == other.capacity,
            self.is_subset_of(other),
            other.is_subset_of(self),
        ) {
            (true, true, true) => Some(cmp::Ordering::Equal),
            (true, true, false) => Some(cmp::Ordering::Less),
            (true, false, true) => Some(cmp::Ordering::Greater),
            _ => None,
        }
    }
}

impl<B: Bound> fmt::Display for Intervals<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            write!(f, "∅")
        } else if self.all_values() {
            write!(
                f,
                "{}{{{}}}",
                B::name(),
                self.intervals
                    .iter()
                    .map(|interval| format!("{}", interval[0]))
                    .join(", ")
            )
        } else {
            write!(
                f,
                "{}{}",
                B::name(),
                self.intervals
                    .iter()
                    .map(|[s, e]| {
                        match (s == e, s == &B::min(), e == &B::max()) {
                            (true, _, _) => format!("{{{}}}", s),
                            (_, true, true) => String::new(),
                            (_, true, false) => format!("(-∞, {}]", e),
                            (_, false, true) => format!("[{}, +∞)", s),
                            (_, false, false) => format!("[{} {}]", s, e),
                        }
                    })
                    .join("∪")
            )
        }
    }
}

pub trait Values<B: Bound>: Sized {
    fn values_len(&self) -> Option<usize>;
    fn max_value_len(&self) -> usize;
    fn values(&self) -> Vec<B>;
    fn into_values(self) -> Intervals<B>;
}

// Specific implementations
impl Values<bool> for Intervals<bool> {
    fn values_len(&self) -> Option<usize> {
        Some(if self.max()? == self.min()? { 1 } else { 2 })
    }
    fn max_value_len(&self) -> usize {
        self.capacity
    }
    fn values(&self) -> Vec<bool> {
        self.intervals
            .clone()
            .into_iter()
            .flat_map(|[a, b]| BTreeSet::from([a, b]).into_iter())
            .collect()
    }
    fn into_values(self) -> Intervals<bool> {
        if self
            .values_len()
            .map(|l| l < self.max_value_len())
            .unwrap_or(false)
        {
            self.values().into_iter().collect()
        } else {
            self
        }
    }
}

impl Values<i64> for Intervals<i64> {
    fn values_len(&self) -> Option<usize> {
        let min = (*self.min()?).clamp(-(self.capacity as i64), self.capacity as i64);
        let max = (*self.max()?).clamp(-(self.capacity as i64), self.capacity as i64);
        Some((max - min) as usize)
    }
    fn max_value_len(&self) -> usize {
        self.capacity
    }
    fn values(&self) -> Vec<i64> {
        self.intervals
            .clone()
            .into_iter()
            .flat_map(|[a, b]| a..=b)
            .collect()
    }
    fn into_values(self) -> Intervals<i64> {
        if self
            .values_len()
            .map(|l| l < self.max_value_len())
            .unwrap_or(false)
        {
            self.values().into_iter().collect()
        } else {
            self
        }
    }
}

impl Values<NaiveDate> for Intervals<NaiveDate> {
    fn values_len(&self) -> Option<usize> {
        Some(self.max()?.signed_duration_since(*self.min()?).num_days() as usize)
    }
    fn max_value_len(&self) -> usize {
        self.capacity
    }
    fn values(&self) -> Vec<NaiveDate> {
        self.intervals
            .clone()
            .into_iter()
            .flat_map(|[a, b]| a.iter_days().take_while(move |d| d <= &b))
            .collect()
    }
    fn into_values(self) -> Intervals<NaiveDate> {
        if self
            .values_len()
            .map(|l| l < self.max_value_len())
            .unwrap_or(false)
        {
            self.values().into_iter().collect()
        } else {
            self
        }
    }
}

#[cfg(test)]
mod tests {
    use chrono::NaiveDate;

    use super::*;

    #[test]
    fn test_ordered() {
        let min_str: String = Bound::min();
        let max_str: String = Bound::max();
        println!(r#""{min_str}"" < "some text" < "{max_str}""#);
        assert!(min_str.as_str() < "some text");
        assert!("some text" < max_str.as_str());
    }

    #[test]
    fn test_intervals() {
        let mut intervals = Intervals::new().union_interval(5, 5);
        println!("intervals = {}", intervals);
        intervals = intervals.union_interval(5, 10);
        println!("intervals = {}", intervals);
        intervals = intervals.union_interval(8, 15);
        println!("intervals = {}", intervals);
        intervals = intervals.union_interval(18, 25);
        println!("intervals = {}", intervals);
        intervals = intervals.union_interval(30, Bound::max());
        println!("intervals = {}", intervals);
        intervals = intervals.union_interval(Bound::min(), 1);
        println!("intervals = {}", intervals);
        intervals = intervals.union_interval(-1, 14);
        println!("intervals = {}", intervals);
        intervals = intervals.union_interval(10, 27);
        println!("intervals = {}", intervals);
        assert_eq!(
            intervals,
            Intervals::new()
                .union_interval(Bound::min(), 27)
                .union_interval(30, Bound::max())
        );
        intervals = intervals.intersection_interval(0, 100);
        println!("intervals = {}", intervals);
        assert_eq!(
            intervals,
            Intervals::new()
                .union_interval(0, 27)
                .union_interval(30, 100)
        );
    }

    #[test]
    fn test_ops() {
        let intervals = Intervals::empty() | (5.0..=7.0) | 1.0 | (..=-2.) | (6.0..=20.);
        println!("intervals = {}", intervals);
        assert_eq!(
            intervals,
            Intervals::new()
                .union_max(-2.)
                .union_interval(5., 20.)
                .union_value(1.)
        );
    }

    #[test]
    fn test_conversion() {
        let mut intervals: Intervals<f64> = 5.0.into();
        println!("intervals = {}", intervals);
        intervals = (0..10).map(|i| i as f64).collect();
        println!("intervals = {}", intervals);
        assert!(intervals.all_values());
    }

    #[test]
    fn test_union() {
        let left: Intervals<f64> = Intervals::from_interval(1., 2.) | [5., 8.];
        let right = Intervals::from_intervals([[-1., 1.5], [15., 18.]]);
        let result = left.clone() | right.clone();
        println!("{} ∪ {} = {}", left, right, result);
        assert!(Intervals::from_intervals([[-1., 2.], [5., 8.], [15., 18.]]) == result);
    }

    #[test]
    fn test_intersection_interval() {
        let a = Intervals::from_interval(1.5, 2.);
        let b = a.intersection_interval(-1., 1.);
        assert_eq!(b, Intervals::empty());

        let a = Intervals::from_interval(1., 2.);
        let b = a.intersection_interval(-1., 1.5);
        assert_eq!(b, Intervals::from_interval(1., 1.5));

        let a = Intervals::from_interval(-1., 2.);
        let b = a.intersection_interval(1., 1.5);
        assert_eq!(b, Intervals::from_interval(1., 1.5));

        let a = Intervals::from_interval(-1., 2.);
        let b = a.intersection_interval(1., 1.5);
        assert_eq!(b, Intervals::from_interval(1., 1.5));

        let a = Intervals::from_values([1., 2., 5., 8.]);
        let b = a.intersection_value(2.);
        assert_eq!(b, Intervals::from_value(2.));
    }

    #[test]
    fn test_intersection() {
        let left = Intervals::from_intervals([[1., 2.], [5., 8.]]);
        let right = Intervals::from_intervals([[-1., 1.5], [15., 18.]]);
        let result = left.clone() & right.clone();
        println!("{} ∩ {} = {}", left, right, result);
        assert!(Intervals::from_interval(1., 1.5) == result);
    }

    #[test]
    fn test_union_values() {
        let left = Intervals::from_values([1., 2.]);
        let right = Intervals::from_values([5., 8.]);
        let result = left.clone() | right.clone();
        println!("{} ∪ {} = {}", left, right, result);
        assert!(Intervals::from_values([1., 2., 5., 8.]) == result);
    }

    #[test]
    fn test_intersection_values() {
        let left = Intervals::from_values([0., 1., 2.]);
        let right = Intervals::from_values([1., 2., 5., 8.]);
        let result = left.clone() & right.clone();
        println!("{} ∩ {} = {}", left, right, result);
        assert!(Intervals::from_values([1., 2.]) == result);
        let left = Intervals::from_values([1., 2.]);
        let right = Intervals::from_values([1., 2., 5., 8.]);
        let result = left.clone() & right.clone();
        println!("{} ∩ {} = {}", left, right, result);
        assert!(Intervals::from_values([1., 2.]) == result);
    }

    #[test]
    fn test_inclusion() {
        let left = Intervals::from_intervals([[1, 5], [4, 6], [-5, 0]]);
        let right = Intervals::from_intervals([[1, 5], [4, 6], [-5, 0]]);
        println!("{left} < {right} = {:?}", left < right);
        println!("{left} > {right} = {:?}", left > right);
        assert_eq!(left.partial_cmp(&right), Some(cmp::Ordering::Equal));
        println!("{left} <= {right} = {:?}", left <= right);
        assert!(left <= right);
        let left = Intervals::from_values([1., 2.]);
        let right = Intervals::from_values([1., 2., 5., 8.]);
        println!("{left} < {right} = {:?}", left < right);
        println!("left.is_subset_of(&right) = {}", left.is_subset_of(&right));
        println!(
            "left.clone().intersection(right.clone()) = {}",
            left.clone().intersection(right.clone())
        );
        println!("{left}.cmp({right}) = {:?}", left.partial_cmp(&right));
        assert!(left < right);
    }

    #[test]
    fn test_display() {
        println!("{}", Intervals::from_interval(2, 5));
        println!("{}", Intervals::from_values([24.0, 1.2, 5.1, 10.0]));
        println!("{}", Intervals::from_intervals([[1, 5], [4, 6], [-5, 0]]));
        let intervals = Intervals::from_intervals([[1, 5], [4, 6], [-5, 0]])
            | Intervals::from_values([2, 4, 8, 20]);
        println!("intervals = {intervals}");
        assert_eq!(format!("{intervals}"), "int[-5 0]∪[1 6]∪{8}∪{20}");
        let intervals = Intervals::from_intervals([[1, 5], [4, 6], [-5, 0]])
            & Intervals::from_values([-20, -1, 2, 4, 8, 20]);
        println!("intervals = {intervals}");
        assert_eq!(format!("{intervals}"), "int{-1, 2, 4}");
    }

    #[test]
    #[should_panic]
    fn test_invalid() {
        let _i = Intervals::from_interval(5, 2); // Should fail
        let _v = Intervals::from_values([1.2, 5.1, 0.0]); // Should work
    }

    #[test]
    fn test_empty_inter() {
        let u = Intervals::from_interval(1.0, 5.0);
        let v: Intervals<f64> = Intervals::from_interval(-5.0, 3.0);
        let w: Intervals<f64> = Intervals::from_interval(-5.0, -1.0);
        println!("{} ∩ {} = {}", u, v, u.clone().intersection(v.clone()));
        assert_eq!(
            u.clone().intersection(v),
            Intervals::from_interval(1.0, 3.0)
        );
        println!("{} ∩ {} = {}", u, w, u.clone().intersection(w.clone()));
        assert!(u.intersection(w).is_empty());
    }

    #[test]
    fn test_strings() {
        let u = Intervals::from_values([
            String::from("Hello"),
            String::from("world"),
            String::from("ancient"),
            String::from("World"),
        ]);
        let v = Intervals::from_values([
            String::from("Hello"),
            String::from("new"),
            String::from("world"),
        ]);
        let w = Intervals::from_interval(String::from("Hello"), String::from("Z"));
        println!("{} ∩ {} = {}", u, v, u.clone().intersection(v.clone()));
        assert_eq!(
            u.clone().intersection(v),
            Intervals::from_values([String::from("Hello"), String::from("world")])
        );
        println!("{} ∩ {} = {}", u, w, u.clone().intersection(w.clone()));
        assert_eq!(
            u.intersection(w),
            Intervals::from_values([String::from("Hello"), String::from("World")])
        );
    }

    #[test]
    fn test_values() {
        let bools: Intervals<bool> = [[false, true]].into_iter().collect();
        println!("{bools}");
        for b in bools.values() {
            println!("{b}");
        }
        let bools: Intervals<bool> = [[false, false], [true, true]].into_iter().collect();
        println!("{bools}");
        for b in bools.values() {
            println!("{b}");
        }
        let ints: Intervals<i64> = [[-5, -2], [2, 5], [10, 20], [100, 100]]
            .into_iter()
            .collect();
        println!("{ints}");
        for i in ints.values() {
            println!("{i}");
        }
        let dates: Intervals<NaiveDate> = [
            [
                NaiveDate::from_ymd(2022, 12, 1),
                NaiveDate::from_ymd(2022, 12, 25),
            ],
            [
                NaiveDate::from_ymd(1980, 12, 1),
                NaiveDate::from_ymd(1980, 12, 25),
            ],
        ]
        .into_iter()
        .collect();
        println!("{dates}");
        for d in dates.values() {
            println!("{d}");
        }
    }
}
