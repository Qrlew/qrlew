use crate::visitor::{self, Acceptor};
use chrono;
use rand::{
    distributions::{Alphanumeric, DistString, Distribution, Standard},
    seq::IteratorRandom,
    Rng,
};
use std::{cell::RefCell, rc::Rc};

use super::{
    intervals,
    value::{self, Value},
    DataType,
};

pub trait Generator {
    type Generated;

    fn generate<R: Rng>(&self, rng: &mut R) -> Self::Generated;
}

// Implement BoundGenerator for all bounds
trait Bound: intervals::Bound {
    fn generate_between<R: Rng>(rng: &mut R, interval: &[Self; 2]) -> Self;
}

impl Bound for bool {
    fn generate_between<R: Rng>(rng: &mut R, interval: &[Self; 2]) -> Self {
        rng.gen_range((interval[0] as i64)..=(interval[1] as i64)) == 1
    }
}

impl Bound for i64 {
    fn generate_between<R: Rng>(rng: &mut R, interval: &[Self; 2]) -> Self {
        rng.gen_range(interval[0]..=interval[1])
    }
}

impl Bound for f64 {
    fn generate_between<R: Rng>(rng: &mut R, interval: &[Self; 2]) -> Self {
        rng.gen_range(interval[0]..=interval[1])
    }
}

impl Bound for String {
    // TODO make this cleaner someday
    /// A rather inefficient implementation for some corner cases
    fn generate_between<R: Rng>(rng: &mut R, interval: &[Self; 2]) -> Self {
        if interval[0] == interval[1] {
            interval[0].clone()
        } else {
            const MAX_LEN: usize = 64;
            let len = rng.gen_range(0..=MAX_LEN);
            let mut result: String = Alphanumeric.sample_string(rng, len);
            // Resample until ok
            for _ in 0..64 {
                if result >= interval[0] && result <= interval[1] {
                    return result;
                }
                result = Alphanumeric.sample_string(rng, len);
            }
            result
        }
    }
}

impl Bound for chrono::NaiveDate {
    fn generate_between<R: Rng>(rng: &mut R, interval: &[Self; 2]) -> Self {
        let duration = interval[1] - interval[0];
        let days = duration.num_days();
        interval[0] + chrono::Duration::days(rng.gen_range(0..=days))
    }
}

impl Bound for chrono::NaiveTime {
    fn generate_between<R: Rng>(rng: &mut R, interval: &[Self; 2]) -> Self {
        let duration = interval[1] - interval[0];
        let seconds = duration.num_seconds();
        interval[0] + chrono::Duration::seconds(rng.gen_range(0..=seconds))
    }
}

impl Bound for chrono::NaiveDateTime {
    fn generate_between<R: Rng>(rng: &mut R, interval: &[Self; 2]) -> Self {
        let duration = interval[1] - interval[0];
        let seconds = duration.num_seconds();
        interval[0] + chrono::Duration::seconds(rng.gen_range(0..=seconds))
    }
}

impl Bound for chrono::Duration {
    fn generate_between<R: Rng>(rng: &mut R, interval: &[Self; 2]) -> Self {
        let duration = interval[1] - interval[0];
        let seconds = duration.num_seconds();
        interval[0] + chrono::Duration::seconds(rng.gen_range(0..=seconds))
    }
}

impl<B: Bound> Generator for intervals::Intervals<B> {
    type Generated = B;

    fn generate<R: Rng>(&self, rng: &mut R) -> Self::Generated {
        let index = rng.gen_range(0..self.len());
        B::generate_between(rng, &self[index])
    }
}

// Visit a DataType for value generation
struct Visitor<R: Rng>(RefCell<R>);

impl<R: Rng> Visitor<R> {
    const ID_LEN: usize = 8;

    fn new(rng: R) -> Self {
        Visitor(RefCell::new(rng))
    }
}

impl<'a, R: Rng> visitor::Visitor<'a, DataType, Value> for Visitor<R> {
    fn visit(
        &self,
        acceptor: &'a DataType,
        dependencies: visitor::Visited<'a, DataType, Value>,
    ) -> Value {
        match acceptor {
            DataType::Unit(_) => Value::unit(),
            DataType::Boolean(b) => b.generate(&mut *self.0.borrow_mut()).into(),
            DataType::Integer(i) => i.generate(&mut *self.0.borrow_mut()).into(),
            DataType::Float(f) => f.generate(&mut *self.0.borrow_mut()).into(),
            DataType::Text(t) => t.generate(&mut *self.0.borrow_mut()).into(),
            DataType::Struct(s) => {
                value::Struct::from_iter(s.iter().map(|(s, t)| (s, dependencies.get(&**t).clone())))
                    .into()
            }
            DataType::Union(u) => {
                let (s, v) = u
                    .iter()
                    .map(|(s, t)| (s, dependencies.get(&**t)))
                    .choose(&mut *self.0.borrow_mut())
                    .unwrap();
                (s.clone(), Rc::new(v.clone())).into()
            }
            DataType::Optional(o) => {
                let is_some: bool = Standard.sample(&mut *self.0.borrow_mut());
                is_some
                    .then(|| Rc::new(dependencies.get(o.data_type()).clone()))
                    .into()
            }
            DataType::Date(d) => d.generate(&mut *self.0.borrow_mut()).into(),
            DataType::Time(t) => t.generate(&mut *self.0.borrow_mut()).into(),
            DataType::DateTime(dt) => dt.generate(&mut *self.0.borrow_mut()).into(),
            DataType::Duration(d) => d.generate(&mut *self.0.borrow_mut()).into(),
            DataType::Id(_) => Alphanumeric
                .sample_string(&mut *self.0.borrow_mut(), Self::ID_LEN)
                .into(),
            _ => todo!(),
        }
    }
}

impl Generator for DataType {
    type Generated = Value;

    fn generate<R: Rng>(&self, rng: &mut R) -> Self::Generated {
        self.accept(Visitor::new(rng))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_type::{self, And, Or};
    use rand::thread_rng;

    #[test]
    fn generate_integers() {
        let mut rng = thread_rng();
        println!("{:?}", i64::generate_between(&mut rng, &[0, 12]));
        println!("{:?}", i64::generate_between(&mut rng, &[7, 7]));
    }

    #[test]
    fn generate_strings() {
        let mut rng = thread_rng();
        println!(
            "{:?}",
            String::generate_between(&mut rng, &["hello".to_string(), "hello".to_string()])
        );
        println!(
            "{:?}",
            String::generate_between(&mut rng, &["hello".to_string(), "world".to_string()])
        );
    }

    #[test]
    fn generate_dates() {
        let mut rng = thread_rng();
        println!(
            "{:?}",
            chrono::NaiveDate::generate_between(
                &mut rng,
                &[
                    chrono::NaiveDate::from_ymd_opt(2012, 05, 01).unwrap(),
                    chrono::NaiveDate::from_ymd_opt(2012, 05, 01).unwrap()
                ]
            )
        );
        println!(
            "{:?}",
            chrono::NaiveDate::generate_between(
                &mut rng,
                &[
                    chrono::NaiveDate::from_ymd_opt(2012, 05, 01).unwrap(),
                    chrono::NaiveDate::from_ymd_opt(2022, 05, 01).unwrap()
                ]
            )
        );
    }

    #[test]
    fn generate_datetimes() {
        let mut rng = thread_rng();
        println!(
            "{:?}",
            chrono::NaiveDateTime::generate_between(
                &mut rng,
                &[
                    chrono::NaiveDateTime::from_timestamp_opt(1662921288, 0).unwrap(),
                    chrono::NaiveDateTime::from_timestamp_opt(1662921288, 0).unwrap()
                ]
            )
        );
        println!(
            "{:?}",
            chrono::NaiveDateTime::generate_between(
                &mut rng,
                &[
                    chrono::NaiveDateTime::from_timestamp_opt(1662921288, 0).unwrap(),
                    chrono::NaiveDateTime::from_timestamp_opt(1693921288, 0).unwrap()
                ]
            )
        );
    }

    #[test]
    fn generate_float_intervals() {
        let intervals = data_type::Float::from_intervals([[-10., 0.], [1., 1.], [2., 2.]]);
        for _ in 0..100 {
            println!("{:?}", intervals.generate(&mut thread_rng()));
        }
    }

    #[test]
    fn generate_data_type() {
        let table_1 = DataType::unit()
            .and(("a", DataType::float_interval(0., 10.)))
            .and(("b", DataType::optional(DataType::float_interval(-1., 1.))))
            .and((
                "c",
                DataType::date_interval(
                    chrono::NaiveDate::from_ymd_opt(1980, 12, 06).unwrap(),
                    chrono::NaiveDate::from_ymd_opt(2023, 12, 06).unwrap(),
                ),
            ));
        let table_2 = DataType::unit()
            .and(("x", DataType::integer_interval(0, 100)))
            .and(("y", DataType::optional(DataType::text())))
            .and(("z", DataType::text_values(["foo".into(), "bar".into()])));
        let db = DataType::Null.or(("tab_1", table_1)).or(("tab_2", table_2));
        for _ in 0..1000 {
            println!("{}", db.generate(&mut thread_rng()));
        }
    }
}
