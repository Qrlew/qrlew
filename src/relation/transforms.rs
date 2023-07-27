//! A few transforms for relations
//!

use super::{Join, Map, Reduce, Relation, Set, Table, Values, Variant as _};
use crate::display::Dot;
use crate::namer;
use crate::{
    builder::{Ready, With, WithIterator},
    data_type::{
        self,
        intervals::{Bound, Intervals},
        DataTyped
    },
    expr::{aggregate, Aggregate, Expr, Value},
    hierarchy::Hierarchy,
    relation, DataType,
};
use itertools::Itertools;
use sqlparser::test_utils::join;
use std::collections::HashMap;
use std::{
    convert::Infallible,
    error, fmt,
    num::ParseFloatError,
    ops::{self, Deref},
    rc::Rc,
    result,
};

#[derive(Debug, PartialEq)]
pub enum Error {
    InvalidRelation(String),
    InvalidArguments(String),
    NoPublicValuesError(String),
    Other(String),
}

impl Error {
    pub fn invalid_relation(relation: impl fmt::Display) -> Error {
        Error::InvalidRelation(format!("{} is invalid", relation))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidRelation(desc) => writeln!(f, "InvalidRelation: {}", desc),
            Error::InvalidArguments(desc) => writeln!(f, "InvalidArguments: {}", desc),
            Error::NoPublicValuesError(desc) => {writeln!(f, "NoPublicValuesError: {}", desc)}
            Error::Other(err) => writeln!(f, "{}", err),
        }
    }
}

impl error::Error for Error {}

impl From<relation::Error> for Error {
    fn from(err: relation::Error) -> Self {
        Error::Other(err.to_string())
    }
}
impl From<crate::expr::Error> for Error {
    fn from(err: crate::expr::Error) -> Self {
        Error::Other(err.to_string())
    }
}
impl From<crate::io::Error> for Error {
    fn from(err: crate::io::Error) -> Self {
        Error::Other(err.to_string())
    }
}

impl From<ParseFloatError> for Error {
    fn from(err: ParseFloatError) -> Self {
        Error::Other(err.to_string())
    }
}

impl From<Infallible> for Error {
    fn from(err: Infallible) -> Self {
        Error::Other(err.to_string())
    }
}

pub type Result<T> = result::Result<T, Error>;

/* Reduce
 */

impl Table {
    /// Rename a Table
    pub fn with_name(mut self, name: String) -> Table {
        self.name = name;
        self
    }
}

/* Map
 */

impl Map {
    /// Rename a Map
    pub fn with_name(mut self, name: String) -> Map {
        self.name = name;
        self
    }
    /// Prepend a field to a Map
    pub fn with_field(self, name: &str, expr: Expr) -> Map {
        Relation::map().with((name, expr)).with(self).build()
    }
    /// Insert a field in a Map at position index
    pub fn insert_field(self, index: usize, inserted_name: &str, inserted_expr: Expr) -> Map {
        let Map {
            name,
            projection,
            filter,
            order_by,
            limit,
            schema,
            input,
            ..
        } = self;
        let mut builder = Map::builder().name(name);
        let field_exprs: Vec<_> = schema.into_iter().zip(projection).collect();
        for (f, e) in &field_exprs[0..index] {
            builder = builder.with((f.name().to_string(), e.clone()));
        }
        builder = builder.with((inserted_name, inserted_expr));
        for (f, e) in &field_exprs[index..field_exprs.len()] {
            builder = builder.with((f.name().to_string(), e.clone()));
        }
        // Filter
        builder = filter.into_iter().fold(builder, |b, f| b.filter(f));
        // Order by
        builder = order_by
            .into_iter()
            .fold(builder, |b, o| b.order_by(o.expr, o.asc));
        // Limit
        builder = limit.into_iter().fold(builder, |b, l| b.limit(l));
        builder.input(input).build()
    }
    /// Filter fields
    pub fn filter_fields<P: Fn(&str) -> bool>(self, predicate: P) -> Map {
        Relation::map().filter_fields_with(self, predicate).build()
    }
    /// Map fields
    pub fn map_fields<F: Fn(&str, Expr) -> Expr>(self, f: F) -> Map {
        Relation::map().map_with(self, f).build()
    }
    /// Rename fields
    pub fn rename_fields<F: Fn(&str, Expr) -> String>(self, f: F) -> Map {
        Relation::map().rename_with(self, f).build()
    }
}

// A few utility objects
#[derive(Clone, Debug)]
pub struct Step<'a> {
    pub referring_id: &'a str,
    pub referred_relation: &'a str,
    pub referred_id: &'a str,
}

impl<'a> From<(&'a str, &'a str, &'a str)> for Step<'a> {
    fn from((referring_id, referred_relation, referred_id): (&'a str, &'a str, &'a str)) -> Self {
        Step {
            referring_id,
            referred_relation,
            referred_id,
        }
    }
}

/* Reduce
 */

impl Reduce {
    /// Rename a Reduce
    pub fn with_name(mut self, name: String) -> Reduce {
        self.name = name;
        self
    }

    pub fn clip_aggregates(
        self,
        vectors: &str,
        clipping_values: Vec<(&str, f64)>,
    ) -> Result<Relation> {
        let (map_names, out_vectors, base, coordinates): (
            Vec<(String, String)>,
            Option<String>,
            Vec<String>,
            Vec<String>,
        ) = self
            .schema()
            .clone()
            .iter()
            .zip(self.aggregate.into_iter())
            .fold((vec![], None, vec![], vec![]), |(mn, v, b, c), (f, x)| {
                if let (name, Expr::Aggregate(agg)) = (f.name(), x) {
                    let argname = agg.argument_name().unwrap().clone();
                    let mut mn = mn;
                    mn.push((argname.clone(), name.to_string()));
                    match agg.aggregate() {
                        aggregate::Aggregate::Sum => {
                            let mut c = c;
                            c.push(argname);
                            (mn, v, b, c)
                        }
                        aggregate::Aggregate::First => {
                            if name == vectors {
                                let v = Some(argname);
                                (mn, v, b, c)
                            } else {
                                let mut b = b;
                                b.push(argname);
                                (mn, v, b, c)
                            }
                        }
                        _ => (mn, v, b, c),
                    }
                } else {
                    (mn, v, b, c)
                }
            });

        let vectors = if let Some(v) = out_vectors {
            Ok(v)
        } else {
            Err(Error::InvalidArguments(format!(
                "{vectors} should be in the input `Relation`"
            )))
        };
        let len_clipping_values = clipping_values.len();
        let len_coordinates = coordinates.len();
        if len_clipping_values != len_coordinates {
            return Err(Error::InvalidArguments(format!(
                "You must provide one clipping_value for each output field. \n \
                Got {len_clipping_values} clipping values for {len_coordinates} output fields"
            )));
        }
        let clipped_relation = self.input.as_ref().clone().clipped_sum(
            vectors?.as_str(),
            base.iter().map(|s| s.as_str()).collect(),
            coordinates.iter().map(|s| s.as_str()).collect(),
            clipping_values,
        );
        let map_names: HashMap<String, String> = map_names.into_iter().collect();
        Ok(clipped_relation.rename_fields(|n, _| map_names[n].to_string()))
    }

    /// Rename fields
    pub fn rename_fields<F: Fn(&str, Expr) -> String>(self, f: F) -> Reduce {
        Relation::reduce().rename_with(self, f).build()
    }
}

/* Join
 */

impl Join {
    /// Rename a Join
    pub fn with_name(mut self, name: String) -> Join {
        self.name = name;
        self
    }
}

/* Set
 */

impl Set {
    /// Rename a Join
    pub fn with_name(mut self, name: String) -> Set {
        self.name = name;
        self
    }
}

/* Values
 */

impl Values {
    /// Rename a Values
    pub fn with_name(mut self, name: String) -> Values {
        self.name = name;
        self
    }
}

#[derive(Clone, Debug)]
pub struct Path<'a>(pub Vec<Step<'a>>);

impl<'a> Deref for Path<'a> {
    type Target = Vec<Step<'a>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> FromIterator<&'a (&'a str, &'a str, &'a str)> for Path<'a> {
    fn from_iter<T: IntoIterator<Item = &'a (&'a str, &'a str, &'a str)>>(iter: T) -> Self {
        Path(
            iter.into_iter()
                .map(|(referring_id, referred_relation, referred_id)| Step {
                    referring_id,
                    referred_relation,
                    referred_id,
                })
                .collect(),
        )
    }
}

impl<'a> IntoIterator for Path<'a> {
    type Item = Step<'a>;
    type IntoIter = <Vec<Step<'a>> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

/// A link to a relation and a field to keep with a new name
#[derive(Clone, Debug)]
pub struct ReferredField<'a> {
    pub referring_id: &'a str,
    pub referred_relation: &'a str,
    pub referred_id: &'a str,
    pub referred_field: &'a str,
    pub referred_field_name: &'a str,
}

#[derive(Clone, Debug)]
pub struct FieldPath<'a>(pub Vec<ReferredField<'a>>);

impl<'a> FieldPath<'a> {
    pub fn from_path(
        path: Path<'a>,
        referred_field: &'a str,
        referred_field_name: &'a str,
    ) -> Self {
        let mut field_path = FieldPath(vec![]);
        let mut last_step: Option<Step> = None;
        // Fill the vec
        for step in path {
            if let Some(last_step) = &mut last_step {
                field_path.0.push(ReferredField {
                    referring_id: last_step.referring_id,
                    referred_relation: last_step.referred_relation,
                    referred_id: last_step.referred_id,
                    referred_field: step.referring_id,
                    referred_field_name,
                });
                *last_step = Step {
                    referring_id: referred_field_name,
                    referred_relation: step.referred_relation,
                    referred_id: step.referred_id,
                };
            } else {
                last_step = Some(step);
            }
        }
        if let Some(last_step) = last_step {
            field_path.0.push(ReferredField {
                referring_id: last_step.referring_id,
                referred_relation: last_step.referred_relation,
                referred_id: last_step.referred_id,
                referred_field,
                referred_field_name,
            });
        }
        field_path
    }
}

impl<'a> Deref for FieldPath<'a> {
    type Target = Vec<ReferredField<'a>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> IntoIterator for FieldPath<'a> {
    type Item = ReferredField<'a>;
    type IntoIter = <Vec<ReferredField<'a>> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl Relation {
    /// Rename a Relation
    pub fn with_name(self, name: String) -> Relation {
        match self {
            Relation::Table(t) => t.with_name(name).into(),
            Relation::Map(m) => m.with_name(name).into(),
            Relation::Reduce(r) => r.with_name(name).into(),
            Relation::Join(j) => j.with_name(name).into(),
            Relation::Set(s) => s.with_name(name).into(),
            Relation::Values(v) => v.with_name(name).into(),
        }
    }
    /// Add a field that derives from existing fields
    pub fn identity_with_field(self, name: &str, expr: Expr) -> Relation {
        Relation::map()
            .with((name, expr))
            .with_iter(
                self.schema()
                    .iter()
                    .map(|f| (f.name(), Expr::col(f.name()))),
            )
            .input(self)
            .build()
    }
    /// Insert a field that derives from existing fields
    pub fn identity_insert_field(
        self,
        index: usize,
        inserted_name: &str,
        inserted_expr: Expr,
    ) -> Relation {
        let mut builder = Relation::map();
        let named_exprs: Vec<_> = self
            .schema()
            .iter()
            .map(|f| (f.name(), Expr::col(f.name())))
            .collect();
        for (n, e) in &named_exprs[0..index] {
            builder = builder.with((n.to_string(), e.clone()));
        }
        builder = builder.with((inserted_name, inserted_expr));
        for (n, e) in &named_exprs[index..named_exprs.len()] {
            builder = builder.with((n.to_string(), e.clone()));
        }
        builder.input(self).build()
    }
    /// Add a field that derives from input fields
    pub fn with_field(self, name: &str, expr: Expr) -> Relation {
        match self {
            // Simply add a column on Maps
            Relation::Map(map) => map.with_field(name, expr).into(),
            relation => relation.identity_with_field(name, expr),
        }
    }
    /// Insert a field that derives from input fields
    pub fn insert_field(self, index: usize, inserted_name: &str, inserted_expr: Expr) -> Relation {
        match self {
            // Simply add a column on Maps
            Relation::Map(map) => map.insert_field(index, inserted_name, inserted_expr).into(),
            relation => relation.identity_insert_field(index, inserted_name, inserted_expr),
        }
    }
    /// Add a field designated with a foreign relation and a field
    pub fn with_referred_field<'a>(
        self,
        referring_id: &'a str,
        referred_relation: Rc<Relation>,
        referred_id: &'a str,
        referred_field: &'a str,
        referred_field_name: &'a str,
    ) -> Relation {
        let left_size = referred_relation.schema().len();
        let names: Vec<String> = self
            .schema()
            .iter()
            .map(|f| f.name().to_string())
            .filter(|name| name != referred_field_name)
            .collect();
        let join: Relation = Relation::join()
            .inner()
            .on(Expr::eq(
                Expr::qcol(self.name(), referring_id),
                Expr::qcol(referred_relation.name(), referred_id),
            ))
            .left(referred_relation)
            .right(self)
            .build();
        let left: Vec<_> = join
            .schema()
            .iter()
            .zip(join.input_fields())
            .take(left_size)
            .collect();
        let right: Vec<_> = join
            .schema()
            .iter()
            .zip(join.input_fields())
            .skip(left_size)
            .collect();
        Relation::map()
            .with_iter(left.into_iter().find_map(|(o, i)| {
                (referred_field == i.name()).then_some((referred_field_name, Expr::col(o.name())))
            }))
            .with_iter(right.into_iter().filter_map(|(o, i)| {
                names
                    .contains(&i.name().to_string())
                    .then_some((i.name(), Expr::col(o.name())))
            }))
            .input(join)
            .build()
    }

    /// Add a field designated with a "fiald path"
    pub fn with_field_path<'a>(
        self,
        relations: &'a Hierarchy<Rc<Relation>>,
        path: &'a [(&'a str, &'a str, &'a str)],
        referred_field: &'a str,
        referred_field_name: &'a str,
    ) -> Relation {
        if path.is_empty() {
            self.identity_with_field(referred_field_name, Expr::col(referred_field))
        } else {
            let path = Path::from_iter(path);
            let field_path = FieldPath::from_path(path, referred_field, referred_field_name);
            // Build the relation following the path to compute the new field
            field_path.into_iter().fold(
                self,
                |relation,
                 ReferredField {
                     referring_id,
                     referred_relation,
                     referred_id,
                     referred_field,
                     referred_field_name,
                 }| {
                    relation.with_referred_field(
                        referring_id,
                        relations
                            .get(&[referred_relation.to_string()])
                            .unwrap()
                            .clone(),
                        referred_id,
                        referred_field,
                        referred_field_name,
                    )
                },
            )
        }
    }

    pub fn filter_fields<P: Fn(&str) -> bool>(self, predicate: P) -> Relation {
        match self {
            Relation::Map(map) => map.filter_fields(predicate).into(),
            relation => {
                Relation::map()
                    .with_iter(relation.schema().iter().filter_map(|f| {
                        predicate(f.name()).then_some((f.name(), Expr::col(f.name())))
                    }))
                    .input(relation)
                    .build()
            }
        }
    }

    pub fn map_fields<F: Fn(&str, Expr) -> Expr>(self, f: F) -> Relation {
        match self {
            Relation::Map(map) => map.map_fields(f).into(),
            relation => Relation::map()
                .with_iter(
                    relation
                        .schema()
                        .iter()
                        .map(|field| (field.name(), f(field.name(), Expr::col(field.name())))),
                )
                .input(relation)
                .build(),
        }
    }

    pub fn rename_fields<F: Fn(&str, Expr) -> String>(self, f: F) -> Relation {
        match self {
            Relation::Map(map) => map.rename_fields(f).into(),
            Relation::Reduce(red) => red.rename_fields(f).into(),
            relation => Relation::map()
                .with_iter(relation.schema().iter().map(|field| {
                    (
                        f(field.name(), Expr::col(field.name())),
                        Expr::col(field.name()),
                    )
                }))
                .input(relation)
                .build(),
        }
    }

    pub fn sum_by(self, base: Vec<&str>, coordinates: Vec<&str>) -> Self {
        let mut reduce = Relation::reduce().input(self.clone());
        reduce = base
            .iter()
            .fold(reduce, |acc, s| acc.with_group_by_column(s.to_string()));
        reduce = reduce.with_iter(
            coordinates
                .iter()
                .map(|c| (*c, Expr::sum(Expr::col(c.to_string())))),
        );
        reduce.build()
    }

    pub fn l1_norm(self, vectors: &str, base: Vec<&str>, coordinates: Vec<&str>) -> Self {
        let mut vectors_base = vec![vectors];
        vectors_base.extend(base.clone());
        let first = self.sum_by(vectors_base, coordinates.clone());

        let map_rel = first.map_fields(|n, e| {
            if coordinates.contains(&n) {
                Expr::abs(e)
            } else {
                e
            }
        });

        if base.is_empty() {
            map_rel
        } else {
            map_rel.sum_by(vec![vectors], coordinates)
        }
    }

    pub fn l2_norm(self, vectors: &str, base: Vec<&str>, coordinates: Vec<&str>) -> Self {
        if base.is_empty() {
            self.l1_norm(vectors, base, coordinates)
        } else {
            let mut vectors_base = vec![vectors];
            vectors_base.extend(base.clone());
            let first = self.sum_by(vectors_base, coordinates.clone());

            let map_rel = first.map_fields(|n, e| {
                if coordinates.contains(&n) {
                    Expr::pow(e, Expr::val(2))
                } else {
                    e
                }
            });
            let reduce_rel = map_rel.sum_by(vec![vectors], coordinates.clone());
            reduce_rel.map_fields(|n, e| {
                if coordinates.contains(&n) {
                    Expr::sqrt(e)
                } else {
                    e
                }
            })
        }
    }

    /// This transform multiplies the coordinates in `self` relation by their corresponding weights in `weight_relation`.
    /// `weight_relation` contains the coordinates weights and the vectors columns
    /// `self` contains the coordinates, the base and vectors columns
    pub fn renormalize(
        self,
        weight_relation: Self,
        vectors: &str,
        base: Vec<&str>,
        coordinates: Vec<&str>,
    ) -> Self {
        // Join the two relations on the peid column
        let join: Relation = Relation::join()
            .left(self.clone())
            .right(weight_relation.clone())
            .inner()
            .on(Expr::eq(
                Expr::qcol(self.name(), vectors),
                Expr::qcol(weight_relation.name(), vectors),
            ))
            .build();

        // Multiply by weights
        let mut grouping_cols: Vec<Expr> = vec![];
        let mut weighted_agg: Vec<Expr> = vec![];
        let left_len = if base.is_empty() {
            self.schema().len() + 1
        } else {
            self.schema().len()
        };
        let join_len = join.schema().len();
        let out_fields = join.schema().fields();
        let in_fields = join.input_fields();
        for i in 0..left_len {
            // length + 1
            if coordinates.contains(&in_fields[i].name()) {
                let mut pos = i + 1;
                while &in_fields[i].name() != &in_fields[pos].name() {
                    pos += 1;
                    if pos > join_len {
                        panic!()
                    }
                }

                weighted_agg.push(Expr::multiply(
                    Expr::col(out_fields[i].name()),
                    Expr::col(out_fields[pos].name()),
                ));
            } else {
                grouping_cols.push(Expr::col(out_fields[i].name()));
            }
        }

        let mut vectors_base = vec![vectors];
        vectors_base.extend(base.clone());
        Relation::map()
            .input(join)
            .with_iter(
                vectors_base
                    .iter()
                    .zip(grouping_cols.iter())
                    .map(|(s, e)| (s.to_string(), e.clone())),
            )
            .with_iter(
                coordinates
                    .iter()
                    .zip(weighted_agg.iter())
                    .map(|(s, e)| (s.to_string(), e.clone())),
            )
            .build()
    }

    /// For each coordinate, rescale the columns by 1 / max(c, norm_l2(coordinate))
    /// where the l2 norm is computed for each elecment of `vectors`
    /// The `self` relation must contain the vectors, base and coordinates columns
    pub fn clipped_sum(
        self,
        vectors: &str,
        base: Vec<&str>,
        coordinates: Vec<&str>,
        clipping_values: Vec<(&str, f64)>,
    ) -> Self {
        let norm = self
            .clone()
            .l2_norm(vectors.clone(), base.clone(), coordinates.clone());

        let map_clipping_values: HashMap<&str, f64> = clipping_values.into_iter().collect();

        let weights = norm.map_fields(|n, e| {
            if coordinates.contains(&n) {
                Expr::divide(
                    Expr::val(2),
                    Expr::plus(
                        Expr::abs(Expr::minus(
                            Expr::divide(e.clone(), Expr::val(map_clipping_values[&n])),
                            Expr::val(1),
                        )),
                        Expr::plus(
                            Expr::divide(e, Expr::val(map_clipping_values[&n])),
                            Expr::val(1),
                        ),
                    ),
                )
            } else {
                Expr::col(n)
            }
        });

        let aggregated_relation: Relation = if base.is_empty() {
            Relation::map()
                .input(self)
                .with((vectors, Expr::col(vectors)))
                .with_iter(
                    coordinates
                        .iter()
                        .map(|s| (s.to_string(), Expr::col(s.to_string()))),
                )
                .build()
        } else {
            let mut vectors_base = vec![vectors];
            vectors_base.extend(base.clone());
            self.sum_by(vectors_base, coordinates.clone())
        };

        let weighted_relation =
            aggregated_relation.renormalize(weights, vectors, base.clone(), coordinates.clone());

        weighted_relation.sum_by(base, coordinates)
    }

    pub fn clip_aggregates(self, vectors: &str, clipping_values: Vec<(&str, f64)>) -> Result<Self> {
        match self {
            Relation::Reduce(reduce) => reduce.clip_aggregates(vectors, clipping_values),
            _ => todo!(),
        }
    }

    /// Add gaussian noise of a given standard deviation to the given columns
    pub fn add_gaussian_noise(self, name_sigmas: Vec<(&str, f64)>) -> Relation {
        let name_sigmas: HashMap<&str, f64> = name_sigmas.into_iter().collect();
        Relation::map()
            // .with_iter(name_sigmas.into_iter().map(|(name, sigma)| (name, Expr::col(name).add_gaussian_noise(sigma))))
            .with_iter(self.schema().iter().map(|f| {
                if name_sigmas.contains_key(&f.name()) {
                    (
                        f.name(),
                        Expr::col(f.name()).add_gaussian_noise(name_sigmas[f.name()]),
                    )
                } else {
                    (f.name(), Expr::col(f.name()))
                }
            }))
            .input(self)
            .build()
    }

    /// Returns a `Relation::Map` that inputs `self` and filter by `predicate`
    pub fn filter(self, predicate: Expr) -> Relation {
        Relation::map()
            .with_iter(
                self.schema()
                    .iter()
                    .map(|f| (f.name(), Expr::col(f.name()))),
            )
            .filter(predicate)
            .input(self)
            .build()
    }

    /// Returns a filtered `Relation`
    ///
    /// # Arguments
    /// - `columns`: `Vec<(column_name, minimal_value, maximal_value, possible_values)>`
    ///
    /// For example,
    /// `filter_columns(vec![("my_col", Value::float(2.), Value::float(10.), vec![Value::integer(4), Value::integer(9)])])`
    /// returns a filtered `Relation` whose `filter` is equivalent to `(my_col > 2.) and (my_col < 10) and (my_col in (4, 9)`
    pub fn filter_columns(
        self,
        columns: Vec<(
            &str,
            Option<data_type::value::Value>,
            Option<data_type::value::Value>,
            Vec<data_type::value::Value>,
        )>,
    ) -> Relation {
        let predicate = Expr::filter(columns);
        self.filter(predicate)
    }

    /// Poisson sampling of a relation. It samples each line with probability 0 <= proba <= 1
    pub fn poisson_sampling(self, proba: f64) -> Relation {
        //make sure proba is between 0 and 1.
        assert!(0.0 <= proba && proba <= 1.0);

        let sampled_relation: Relation = Relation::map()
            .with_iter(
                self.schema()
                    .iter()
                    .map(|f| (f.name(), Expr::col(f.name()))),
            )
            .filter(Expr::lt(
                Expr::random(namer::new_id("POISSON_SAMPLING")),
                Expr::val(proba),
            ))
            .input(self)
            .build();
        sampled_relation
    }

    /// sampling without replacemnts.
    /// It creates a Map using self as an imput which applies
    /// WHERE RANDOM() < rate_multiplier * rate ORDER BY RANDOM() LIMIT rate*size
    /// and preserves the input schema fields.
    /// WHERE RANDOM() < rate_multiplier * rate is for optimization purposes
    pub fn sampling_without_replacements(self, rate: f64, rate_multiplier: f64) -> Relation {
        //make sure rate is between 0 and 1.
        assert!(0.0 <= rate && rate <= 1.0);

        let size = self.size().max().map_or(0, |v| (*v as f64 * rate) as usize);

        let sampled_relation: Relation = Relation::map()
            .with_iter(
                self.schema()
                    .iter()
                    .map(|f| (f.name(), Expr::col(f.name()))),
            )
            .filter(Expr::lt(
                Expr::random(namer::new_id("SAMPLING_WITHOUT_REPLACEMENT")),
                Expr::val(rate_multiplier * rate),
            ))
            .order_by(
                Expr::random(namer::new_id("SAMPLING_WITHOUT_REPLACEMENT")),
                false,
            )
            .limit(size)
            .input(self)
            .build();
        sampled_relation
    }

    /// Returns a Relation whose fields have unique values
    fn unique(self, columns: Vec<&str>) -> Relation {
        let named_columns: Vec<(&str, Expr)> =
            columns.into_iter().map(|c| (c, Expr::col(c))).collect();

        Relation::reduce()
            .group_by_iter(named_columns.iter().cloned().map(|(_, col)| col))
            .with_iter(
                named_columns
                    .into_iter()
                    .map(|(name, col)| (name, Expr::first(col))),
            )
            .input(self)
            .build()
    }

    /// Returns a `Relation` whose output fields correspond to the `aggregates`
    /// grouped by the expressions in `grouping_exprs`.
    /// If `grouping_exprs` is not empty, we order by the grouping expressions.
    fn build_ordered_reduce(
        self,
        grouping_exprs: Vec<Expr>,
        aggregates: Vec<(&str, Expr)>,
    ) -> Relation {
        let red: Relation = Relation::reduce()
            .with_iter(aggregates.clone())
            .group_by_iter(grouping_exprs.clone())
            .input(self)
            .build();

        if grouping_exprs.is_empty() {
            red
        } else {
            Relation::map()
                .with_iter(aggregates.into_iter().map(|(f, _)| (f, Expr::col(f))))
                .order_by_iter(grouping_exprs.into_iter().map(|x| (x, true)).collect())
                .input(red)
                .build()
        }
    }

    /// Build a relation whose output fields are to the aggregations in `aggregates`
    /// applied on the UNIQUE values of the column `column` and grouped by the columns in `group_by`.
    /// If `grouping_by` is not empty, we order by the grouping expressions.
    pub fn distinct_aggregates(
        self,
        column: &str,
        group_by: Vec<&str>,
        aggregates: Vec<(&str, aggregate::Aggregate)>,
    ) -> Relation {
        let mut columns = vec![column];
        columns.extend(group_by.iter());
        let red = self.unique(columns);

        // Build the second reduce
        let mut aggregates_exprs: Vec<(&str, Expr)> = vec![];
        let mut grouping_exprs: Vec<Expr> = vec![];
        group_by.into_iter().for_each(|c| {
            let col = Expr::col(c);
            aggregates_exprs.push((c, Expr::first(col.clone())));
            grouping_exprs.push(col);
        });
        aggregates.into_iter().for_each(|(c, agg)| {
            aggregates_exprs.push((
                c,
                Expr::Aggregate(Aggregate::new(agg, Rc::new(Expr::col(column)))),
            ))
        });

        // Add order by
        red.build_ordered_reduce(grouping_exprs, aggregates_exprs)
    }

    pub fn all_values_column(&self, colname: &str) -> Result<Relation> {
        let datatype = self.schema().field(colname).unwrap().data_type();
        if let Some(values) = datatype.possible_values() {
            let rel: Relation = Relation::values().name(colname).values(values).build();
            Ok(rel)
        } else {
            Err(Error::NoPublicValuesError(colname.to_string()))
        }
    }

    pub fn all_values(&self) -> Result<Relation> {
        let vec_of_rel: Vec<Relation> = self.schema()
        .iter()
        .map(|c| self.all_values_column(c.name()))
        .collect()?;

        Ok(
            vec_of_rel.iter()
            .reduce(|l, r| l.cross_join(r)?)
        )
    }

    // Returns the cross join between `self` and `right` where
    // the output names of the fields are conserved.
    // This fails if one column name is contained in both relations
    pub fn cross_join(self, right: Self) -> Result<Relation> {
        let left_names: Vec<String> = self.schema().iter().map(|f| f.name().to_string()).collect();
        let right_names: Vec<String> = right
            .schema()
            .iter()
            .map(|f| f.name().to_string())
            .collect();

        if left_names.iter().any(|item| right_names.contains(item)) {
            return Err(
                Error::InvalidArguments(
                    "Cannot use `cross_join` method for joining two relations containing fields with the same names.".to_string()
                )
            );
        }
        Ok(Relation::join()
            .left(self.clone())
            .right(right.clone())
            .cross()
            .left_names(left_names)
            .right_names(right_names)
            .build())
    }

    pub fn left_join(self, right: Self, on: Vec<(&str, &str)>) -> Result<Relation> {
        if on.is_empty() {
            return Err(Error::InvalidArguments(
                "Vector `on` cannot be empty.".into(),
            ));
        }
        let left_names: Vec<String> = self.schema().iter().map(|f| f.name().to_string()).collect();
        let right_names: Vec<String> = right
            .schema()
            .iter()
            .map(|f| f.name().to_string())
            .collect();
        let on: Vec<Expr> = on
            .into_iter()
            .map(|(l, r)| Expr::eq(Expr::qcol(self.name(), l), Expr::qcol(right.name(), r)))
            .collect();
        if left_names.iter().any(|item| right_names.contains(item)) {
            return Err(
                Error::InvalidArguments(
                    "Cannot use `left_join` method for joining two relations containing fields with the same names.".to_string()
                )
            );
        }
        Ok(Relation::join()
            .left(self.clone())
            .right(right.clone())
            .left_outer()
            .on_iter(on)
            .left_names(left_names)
            .right_names(right_names)
            .build())
    }
}

impl With<(&str, Expr)> for Relation {
    fn with(self, (name, expr): (&str, Expr)) -> Self {
        self.identity_with_field(name, expr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ast,
        data_type::{value::List, DataTyped},
        display::Dot,
        io::{postgresql, Database},
        relation::schema::Schema,
        sql::parse,
    };
    use colored::Colorize;
    use itertools::Itertools;
    use sqlparser::keywords::RIGHT;

    #[test]
    fn test_with_computed_field() {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        let table = relations.get(&["table_1".into()]).unwrap().as_ref().clone();
        let relation =
            Relation::try_from(parse("SELECT * FROM table_1").unwrap().with(&relations)).unwrap();
        // Table
        assert!(table.schema()[0].name() != "peid");
        let table = table.identity_with_field("peid", expr!(a + b));
        assert!(table.schema()[0].name() == "peid");
        // Relation
        relation.display_dot().unwrap();
        assert!(relation.schema()[0].name() != "peid");
        let relation = relation.identity_with_field("peid", expr!(cos(a)));
        relation.display_dot().unwrap();
        assert!(relation.schema()[0].name() == "peid");
    }

    #[test]
    fn test_filter_fields() {
        let database = postgresql::test_database();
        let relations = database.relations();
        let relation =
            Relation::try_from(parse("SELECT * FROM table_1").unwrap().with(&relations)).unwrap();
        let relation = relation.with_field("peid", expr!(cos(a)));
        assert!(relation.schema()[0].name() == "peid");
        let relation = relation.filter_fields(|n| n != "peid");
        assert!(relation.schema()[0].name() != "peid");
    }

    #[test]
    fn test_referred_field() {
        let database = postgresql::test_database();
        let relations = database.relations();
        let orders =
            Relation::try_from(parse("SELECT * FROM order_table").unwrap().with(&relations))
                .unwrap();
        let user = relations.get(&["user_table".to_string()]).unwrap().as_ref();
        let relation =
            orders.with_referred_field("user_id", Rc::new(user.clone()), "id", "id", "peid");
        assert!(relation.schema()[0].name() == "peid");
        let relation = relation.filter_fields(|n| n != "peid");
        assert!(relation.schema()[0].name() != "peid");
    }

    #[test]
    fn test_field_path() {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        // Link orders to users
        let orders = relations
            .get(&["order_table".to_string()])
            .unwrap()
            .as_ref();
        let relation = orders.clone().with_field_path(
            &relations,
            &[("user_id", "user_table", "id")],
            "id",
            "peid",
        );
        assert!(relation.schema()[0].name() == "peid");
        // Link items to orders
        let items = relations.get(&["item_table".to_string()]).unwrap().as_ref();
        let relation = items.clone().with_field_path(
            &relations,
            &[
                ("order_id", "order_table", "id"),
                ("user_id", "user_table", "id"),
            ],
            "name",
            "peid",
        );
        assert!(relation.schema()[0].name() == "peid");
        // Produce the query
        relation.display_dot();
        let query: &str = &ast::Query::from(&relation).to_string();
        println!("{query}");
        println!(
            "{}\n{}",
            format!("{query}").yellow(),
            database
                .query(query)
                .unwrap()
                .iter()
                .map(ToString::to_string)
                .join("\n")
        );
        let relation = relation.filter_fields(|n| n != "peid");
        assert!(relation.schema()[0].name() != "peid");
    }

    fn refacto_results(results: Vec<List>, size: usize) -> Vec<Vec<String>> {
        let mut sorted_results: Vec<Vec<String>> = vec![];
        for row in results {
            let mut str_row = vec![];
            for i in 0..size {
                str_row.push(match row[i].to_string().parse::<f64>() {
                    Ok(f) => ((f * 1000.).round() / 1000.).to_string(),
                    Err(_) => row[i].to_string(),
                })
            }
            sorted_results.push(str_row)
        }
        sorted_results.sort();
        sorted_results
    }

    #[test]
    fn test_compute_norm_for_table() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        let table = relations
            .get(&["item_table".into()])
            .unwrap()
            .as_ref()
            .clone();
        // L1 Norm
        let amount_norm = table
            .clone()
            .l1_norm("order_id", vec!["item"], vec!["price"]);
        // amount_norm.display_dot().unwrap();
        let query: &str = &ast::Query::from(&amount_norm).to_string();
        println!("Query = {}", query);
        let valid_query = "SELECT order_id, SUM(sum_by_group) FROM (SELECT order_id, item, SUM(ABS(price)) AS sum_by_group FROM item_table GROUP BY order_id, item) AS subquery GROUP BY order_id";
        assert_eq!(
            database.query(query).unwrap(),
            database.query(valid_query).unwrap()
        );
        // L2 Norm
        let amount_norm = table.l2_norm("order_id", vec!["item"], vec!["price"]);
        amount_norm.display_dot().unwrap();
        let query: &str = &ast::Query::from(&amount_norm).to_string();
        let valid_query = "SELECT order_id, SQRT(SUM(sum_by_group)) FROM (SELECT order_id, item, POWER(SUM(price), 2) AS sum_by_group FROM item_table GROUP BY order_id, item) AS subquery GROUP BY order_id";
        assert_eq!(
            database.query(query).unwrap(),
            database.query(valid_query).unwrap()
        );
    }

    #[test]
    fn test_compute_norm_for_empty_base() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        let table = relations
            .get(&["item_table".into()])
            .unwrap()
            .as_ref()
            .clone();
        // L1 Norm
        let amount_norm = table.clone().l1_norm("order_id", vec![], vec!["price"]);
        amount_norm.display_dot().unwrap();
        let query: &str = &ast::Query::from(&amount_norm).to_string();
        println!("Query = {}", query);
        let valid_query = "SELECT order_id, ABS(SUM(price)) FROM item_table GROUP BY order_id";
        database.query(query).unwrap();
        assert_eq!(
            database.query(query).unwrap(),
            database.query(valid_query).unwrap()
        );

        // L2 Norm
        let amount_norm = table.l2_norm("order_id", vec![], vec!["price"]);
        amount_norm.display_dot().unwrap();
        let query: &str = &ast::Query::from(&amount_norm).to_string();
        let valid_query =
            "SELECT order_id, SQRT(POWER(SUM(price), 2)) FROM item_table GROUP BY order_id";
        database.query(query).unwrap();
        assert_eq!(
            database.query(query).unwrap(),
            database.query(valid_query).unwrap()
        );
    }

    #[test]
    fn test_compute_norm_for_map() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        let relation = Relation::try_from(
            parse("SELECT price - 25 AS std_price, * FROM item_table")
                .unwrap()
                .with(&relations),
        )
        .unwrap();
        relation.display_dot().unwrap();
        // L1 Norm
        let relation_norm =
            relation
                .clone()
                .l1_norm("order_id", vec!["item"], vec!["price", "std_price"]);
        relation_norm.display_dot().unwrap();
        let query: &str = &ast::Query::from(&relation_norm).to_string();
        //println!("Query = {}", query);
        let valid_query = "SELECT order_id, SUM(sum_1), SUM(sum_2) FROM (SELECT order_id, item, ABS(SUM(price)) AS sum_1, ABS(SUM(std_price)) AS sum_2 FROM ( SELECT price - 25 AS std_price, * FROM item_table ) AS intermediate_table GROUP BY order_id, item) AS subquery GROUP BY order_id";
        assert_eq!(
            database.query(query).unwrap(),
            database.query(valid_query).unwrap()
        );
        // L2 Norm
        let relation_norm = relation.l2_norm("order_id", vec!["item"], vec!["price", "std_price"]);
        relation_norm.display_dot().unwrap();
        let query: &str = &ast::Query::from(&relation_norm).to_string();
        let valid_query = "SELECT order_id, SQRT(SUM(sum_1)), SQRT(SUM(sum_2)) FROM (SELECT order_id, item, POWER(SUM(price), 2) AS sum_1, POWER(SUM(std_price), 2) AS sum_2 FROM ( SELECT price - 25 AS std_price, * FROM item_table ) AS intermediate_table GROUP BY order_id, item) AS subquery GROUP BY order_id";
        assert_eq!(
            database.query(query).unwrap(),
            database.query(valid_query).unwrap()
        );
    }

    #[test]
    fn test_compute_norm_for_join() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        let left: Relation = relations
            .get(&["item_table".into()])
            .unwrap()
            .as_ref()
            .clone();
        let right: Relation = relations
            .get(&["order_table".into()])
            .unwrap()
            .as_ref()
            .clone();
        let relation: Relation = Relation::join()
            .left(left)
            .right(right)
            .on(Expr::eq(
                Expr::qcol("item_table", "order_id"),
                Expr::qcol("order_table", "id"),
            ))
            .build();
        let schema = relation.schema().clone();
        let item = schema.field_from_index(1).unwrap().name();
        let price = schema.field_from_index(2).unwrap().name();
        let user_id = schema.field_from_index(4).unwrap().name();
        let date = schema.field_from_index(6).unwrap().name();

        // L1 Norm
        let relation_norm = relation
            .clone()
            .l1_norm(user_id, vec![item, date], vec![price]);
        relation_norm.display_dot().unwrap();
        let query: &str = &ast::Query::from(&relation_norm).to_string();
        println!("Query = {}", query);

        let valid_query = "SELECT user_id, SUM(sum_1) FROM (SELECT user_id, item, date, ABS(SUM(price)) AS sum_1 FROM item_table JOIN order_table ON item_table.order_id = order_table.id GROUP BY user_id, item, date) AS subquery GROUP BY user_id";
        assert_eq!(
            database.query(query).unwrap(),
            database.query(valid_query).unwrap()
        );
        // L2 Norm
        let relation_norm = relation.l2_norm(user_id, vec![item, date], vec![price]);
        relation_norm.display_dot().unwrap();
        let query: &str = &ast::Query::from(&relation_norm).to_string();
        let valid_query = "SELECT user_id, SQRT(SUM(sum_1)) FROM (SELECT user_id, item, date, POWER(SUM(price), 2) AS sum_1 FROM item_table JOIN order_table ON item_table.order_id = order_table.id GROUP BY user_id, item, date) AS subquery GROUP BY user_id";
        assert_eq!(
            database.query(query).unwrap(),
            database.query(valid_query).unwrap()
        );
        // DEBUG
        for row in database.query(query).unwrap() {
            println!("{row}")
        }
    }

    #[test]
    fn test_clipped_sum_for_table() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        let table = relations
            .get(&["item_table".into()])
            .unwrap()
            .as_ref()
            .clone();
        let clipped_relation = table.clone().clipped_sum(
            "order_id",
            vec!["item"],
            vec!["price"],
            vec![("price", 45.)],
        );
        clipped_relation.display_dot().unwrap();
        let query: &str = &ast::Query::from(&clipped_relation).to_string();
        let valid_query = r#"
        WITH norms AS (
            SELECT order_id, SQRT(SUM(sum_by_group)) AS norm FROM (
                SELECT order_id, item, POWER(SUM(price), 2) AS sum_by_group FROM item_table GROUP BY order_id, item
              ) AS subquery GROUP BY order_id
          ), weights AS (SELECT order_id, CASE WHEN 45 / norm < 1 THEN 45 / norm ELSE 1 END AS weight FROM norms)
          SELECT item, SUM(price*weight) FROM item_table LEFT JOIN weights USING (order_id) GROUP BY item;
        "#;
        let my_res = database.query(query).unwrap();
        let true_res = database.query(valid_query).unwrap();
        assert_eq!(refacto_results(my_res, 2), refacto_results(true_res, 2));
    }

    #[test]
    fn test_clipped_sum_with_empty_base() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        let table = relations
            .get(&["item_table".into()])
            .unwrap()
            .as_ref()
            .clone();
        let clipped_relation =
            table
                .clone()
                .clipped_sum("order_id", vec![], vec!["price"], vec![("price", 45.)]);
        clipped_relation.display_dot().unwrap();
        let query: &str = &ast::Query::from(&clipped_relation).to_string();
        println!("Query: {}", query);
        let valid_query = r#"
            WITH norms AS (
                SELECT order_id, ABS(SUM(price)) AS norm FROM item_table GROUP BY order_id
            ), weights AS (
                SELECT order_id, CASE WHEN 45 / norm < 1 THEN 45 / norm ELSE 1 END AS weight FROM norms
            )
            SELECT SUM(price*weight) FROM item_table LEFT JOIN weights USING (order_id);
        "#;
        let my_res = refacto_results(database.query(query).unwrap(), 1);
        let true_res = refacto_results(database.query(valid_query).unwrap(), 1);
        assert_eq!(my_res, true_res);
    }

    #[test]
    fn test_clipped_sum_for_map() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        let relation = Relation::try_from(
            parse("SELECT price * 25 AS std_price, * FROM item_table")
                .unwrap()
                .with(&relations),
        )
        .unwrap();
        relation.display_dot().unwrap();

        // L2 Norm
        let clipped_relation = relation.clone().clipped_sum(
            "order_id",
            vec!["item"],
            vec!["price", "std_price"],
            vec![("std_price", 45.), ("price", 50.)],
        );
        clipped_relation.display_dot().unwrap();

        let query: &str = &ast::Query::from(&clipped_relation).to_string();
        let valid_query = r#"
        WITH my_table AS (
            SELECT price * 25 AS std_price, * FROM item_table
          ), norms AS (
            SELECT order_id, SQRT(SUM(sum_by_group)) AS norm1, SQRT(SUM(sum_by_group2)) AS norm2 FROM (
              SELECT order_id, item, POWER(SUM(price), 2) AS sum_by_group, POWER(SUM(std_price), 2) AS sum_by_group2 FROM my_table GROUP BY order_id, item
            ) AS subquery GROUP BY order_id
          ), weights AS (SELECT order_id, CASE WHEN 50 / norm1 < 1 THEN 50 / norm1 ELSE 1 END AS weight1, CASE WHEN 45 / norm2 < 1 THEN 45 / norm2 ELSE 1 END AS weight2 FROM norms)
          SELECT item, SUM(price*weight1), SUM(std_price*weight2) FROM my_table LEFT JOIN weights USING (order_id) GROUP BY item;
        "#;

        let my_res = refacto_results(database.query(query).unwrap(), 3);
        let true_res = refacto_results(database.query(valid_query).unwrap(), 3);
        assert_eq!(my_res, true_res);
    }

    #[test]
    fn test_clipped_sum_for_join() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        let left: Relation = relations
            .get(&["item_table".into()])
            .unwrap()
            .as_ref()
            .clone();
        let right: Relation = relations
            .get(&["order_table".into()])
            .unwrap()
            .as_ref()
            .clone();
        let relation: Relation = Relation::join()
            .left(left)
            .right(right)
            .on(Expr::eq(
                Expr::qcol("item_table", "order_id"),
                Expr::qcol("order_table", "id"),
            ))
            .build();
        relation.display_dot().unwrap();
        let schema = relation.schema().clone();
        let item = schema.field_from_index(1).unwrap().name();
        let price = schema.field_from_index(2).unwrap().name();
        let user_id = schema.field_from_index(4).unwrap().name();
        let date = schema.field_from_index(6).unwrap().name();

        let clipped_relation =
            relation.clipped_sum(user_id, vec![item, date], vec![price], vec![(price, 50.)]);
        clipped_relation.display_dot().unwrap();
        let query: &str = &ast::Query::from(&clipped_relation).to_string();
        let valid_query = r#"
        WITH join_table AS (
            SELECT * FROM item_table JOIN order_table ON item_table.order_id = order_table.id
           ), norms AS (
            SELECT user_id, SQRT(SUM(sum_1)) AS norm FROM (SELECT user_id, item, date, POWER(SUM(price), 2) AS sum_1 FROM join_table  GROUP BY user_id, item, date) As subq GROUP BY user_id
           ), weights AS (
             SELECT user_id, CASE WHEN 50 / norm < 1 THEN 50 / norm ELSE 1 END AS weight FROM norms
           ) SELECT item, date, SUM(price*weight)  FROM join_table LEFT JOIN weights USING (user_id) GROUP BY item, date;
        "#;

        let my_res = refacto_results(database.query(query).unwrap(), 3);
        let true_res = refacto_results(database.query(valid_query).unwrap(), 3);
        assert_eq!(my_res, true_res);
    }

    #[test]
    fn test_clip_aggregates_reduce() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        let table = relations
            .get(&["item_table".into()])
            .unwrap()
            .as_ref()
            .clone();

        // with GROUP BY
        let my_relation: Relation = Relation::reduce()
            .input(table.clone())
            .with(("sum_price", Expr::sum(Expr::col("price"))))
            .with_group_by_column("item")
            .with_group_by_column("order_id")
            .build();

        let schema = my_relation.inputs()[0].schema().clone();
        let price = schema.field_from_index(0).unwrap().name();
        let clipped_relation = my_relation
            .clip_aggregates("order_id", vec![(price, 45.)])
            .unwrap();
        let name_fields: Vec<&str> = clipped_relation.schema().iter().map(|f| f.name()).collect();
        assert_eq!(name_fields, vec!["item", "sum_price"]);
        clipped_relation.display_dot();

        let query: &str = &ast::Query::from(&clipped_relation).to_string();
        println!("Query: {}", query);
        let valid_query = r#"
        WITH norms AS (
            SELECT order_id, SQRT(SUM(sum_by_group)) AS norm FROM (
                SELECT order_id, item, POWER(SUM(price), 2) AS sum_by_group FROM item_table GROUP BY order_id, item
              ) AS subquery GROUP BY order_id
          ), weights AS (SELECT order_id, CASE WHEN 45 / norm < 1 THEN 45 / norm ELSE 1 END AS weight FROM norms)
          SELECT item, SUM(price*weight) FROM item_table LEFT JOIN weights USING (order_id) GROUP BY item;
        "#;
        let my_res = refacto_results(database.query(query).unwrap(), 2);
        let true_res = refacto_results(database.query(valid_query).unwrap(), 2);
        assert_eq!(my_res, true_res);

        // without GROUP BY
        let my_relation: Relation = Relation::reduce()
            .input(table)
            .with(("sum_price", Expr::sum(Expr::col("price"))))
            .with_group_by_column("order_id")
            .build();

        let schema = my_relation.inputs()[0].schema().clone();
        let price = schema.field_from_index(0).unwrap().name();
        let clipped_relation = my_relation
            .clip_aggregates("order_id", vec![(price, 45.)])
            .unwrap();
        let name_fields: Vec<&str> = clipped_relation.schema().iter().map(|f| f.name()).collect();
        assert_eq!(name_fields, vec!["sum_price"]);
        clipped_relation.display_dot();

        let query: &str = &ast::Query::from(&clipped_relation).to_string();
        println!("Query: {}", query);
        let valid_query = r#"
            WITH norms AS (
                SELECT order_id, ABS(SUM(price)) AS norm FROM item_table GROUP BY order_id
            ), weights AS (
                SELECT order_id, CASE WHEN 45 / norm < 1 THEN 45 / norm ELSE 1 END AS weight FROM norms
            )
            SELECT SUM(price*weight) FROM item_table LEFT JOIN weights USING (order_id);
        "#;
        let my_res = refacto_results(database.query(query).unwrap(), 1);
        let true_res = refacto_results(database.query(valid_query).unwrap(), 1);
        assert_eq!(my_res, true_res);
    }

    #[test]
    fn test_clip_aggregates_complex_reduce() {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        let initial_query = r#"
        SELECT user_id AS user_id, item AS item, 5 * price AS std_price, price AS price, date AS date
        FROM item_table LEFT JOIN order_table ON item_table.order_id = order_table.id
        "#;
        let relation = Relation::try_from(parse(initial_query).unwrap().with(&relations)).unwrap();
        let relation: Relation = Relation::reduce()
            .input(relation)
            .with_group_by_column("user_id")
            .with_group_by_column("item")
            .with(("sum1", Expr::sum(Expr::col("price"))))
            .with(("sum2", Expr::sum(Expr::col("std_price"))))
            .build();
        relation.display_dot();

        let schema = relation.inputs()[0].schema().clone();
        let price = schema.field_from_index(2).unwrap().name();
        let std_price = schema.field_from_index(3).unwrap().name();
        let clipped_relation = relation
            .clip_aggregates("user_id", vec![(price, 45.), (std_price, 50.)])
            .unwrap();
        clipped_relation.display_dot();
        let name_fields: Vec<&str> = clipped_relation.schema().iter().map(|f| f.name()).collect();
        assert_eq!(name_fields, vec!["item", "sum1", "sum2"]);

        let query: &str = &ast::Query::from(&clipped_relation).to_string();
        println!("Query: {}", query);
        let valid_query = r#"
        WITH my_table AS (
            SELECT user_id AS user_id, item AS item, 5 * price AS std_price, price AS price
            FROM item_table LEFT JOIN order_table ON item_table.order_id = order_table.id
        ),norms AS (
            SELECT user_id, SQRT(SUM(sum_1)) AS norm, SQRT(SUM(sum_2)) AS norm2 FROM (SELECT user_id, item, POWER(SUM(price), 2) AS sum_1, POWER(SUM(std_price), 2) AS sum_2 FROM my_table GROUP BY user_id, item) As subq GROUP BY user_id
        ), weights AS (
            SELECT user_id, CASE WHEN 45 / norm < 1 THEN 45 / norm ELSE 1 END AS weight, CASE WHEN 50 / norm2 < 1 THEN 50 / norm2 ELSE 1 END AS weight2 FROM norms
        )
        SELECT my_table.item, SUM(price*weight) AS sum1, SUM(std_price*weight2) As sum2 FROM my_table LEFT JOIN weights USING (user_id) GROUP BY item;
        "#;
        let my_res: Vec<Vec<String>> = refacto_results(database.query(query).unwrap(), 3);
        let true_res = refacto_results(database.query(valid_query).unwrap(), 3);
        // for (r1, r2) in my_res.iter().zip(true_res.iter()) {
        //     if r1!=r2 {
        //         println!("{:?} != {:?}", r1, r2);
        //     }
        // }
        // assert_eq!(my_res, true_res); // todo: fix that
    }

    #[test]
    fn test_add_noise() {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        // CReate a relation to add noise to
        let relation = Relation::try_from(
            parse("SELECT 0.0 as z, sum(price) as a, sum(price) as b FROM item_table GROUP BY order_id")
                .unwrap()
                .with(&relations),
        )
        .unwrap();
        let relation_with_noise = relation.add_gaussian_noise(vec![("z", 1.)]);
        println!("Schema = {}", relation_with_noise.schema());
        relation_with_noise.display_dot().unwrap();

        // Add noise directly
        for row in database
            .query(
                &ast::Query::try_from(&relation_with_noise)
                    .unwrap()
                    .to_string(),
            )
            .unwrap()
        {
            println!("Row = {row}");
        }
    }

    #[test]
    fn test_rename_fields() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        let table = relations
            .get(&["item_table".into()])
            .unwrap()
            .as_ref()
            .clone();

        // with GROUP BY
        let my_relation: Relation = Relation::reduce()
            .input(table.clone())
            .with(("sum_price", Expr::sum(Expr::col("price"))))
            .with_group_by_column("item")
            .with_group_by_column("order_id")
            .build();
        my_relation.display_dot();

        let renamed_relation = my_relation.clone().rename_fields(|n, _| {
            if n == "sum_price" {
                "SumPrice".to_string()
            } else if n == "item" {
                "ITEM".to_string()
            } else {
                "unknown".to_string()
            }
        });
        renamed_relation.display_dot();
    }

    #[test]
    fn test_filter() {
        let database = postgresql::test_database();
        let relations = database.relations();

        let relation = Relation::try_from(
            parse("SELECT exp(a) AS my_a, b As my_b FROM table_1")
                .unwrap()
                .with(&relations),
        )
        .unwrap();
        let filtered_relation = relation.filter(Expr::and(
            Expr::and(
                Expr::gt(Expr::col("my_a"), Expr::val(5.)),
                Expr::lt(Expr::col("my_b"), Expr::val(0.)),
            ),
            Expr::lt(Expr::col("my_a"), Expr::val(100.)),
        ));
        _ = filtered_relation.display_dot();
        assert_eq!(
            filtered_relation
                .schema()
                .field("my_a")
                .unwrap()
                .data_type(),
            DataType::float_interval(5., 100.)
        );
        assert_eq!(
            filtered_relation
                .schema()
                .field("my_b")
                .unwrap()
                .data_type(),
            DataType::optional(DataType::float_interval(-1., 0.))
        );
        if let Relation::Map(m) = filtered_relation {
            assert_eq!(
                m.filter.unwrap(),
                Expr::and(
                    Expr::and(
                        Expr::gt(Expr::col("my_a"), Expr::val(5.)),
                        Expr::lt(Expr::col("my_b"), Expr::val(0.))
                    ),
                    Expr::lt(Expr::col("my_a"), Expr::val(100.))
                )
            )
        }

        let relation =
            Relation::try_from(parse("SELECT * FROM table_1").unwrap().with(&relations)).unwrap();
        let filtered_relation = relation.filter(Expr::and(
            Expr::gt(Expr::col("a"), Expr::val(5.)),
            Expr::lt(Expr::col("b"), Expr::val(0.5)),
        ));
        _ = filtered_relation.display_dot();
        assert_eq!(
            filtered_relation.schema().field("a").unwrap().data_type(),
            DataType::float_interval(5., 10.)
        );
        assert_eq!(
            filtered_relation.schema().field("b").unwrap().data_type(),
            DataType::optional(DataType::float_interval(-1., 0.5))
        );
        if let Relation::Map(m) = filtered_relation {
            assert_eq!(
                m.filter.unwrap(),
                Expr::and(
                    Expr::gt(Expr::col("a"), Expr::val(5.)),
                    Expr::lt(Expr::col("b"), Expr::val(0.5))
                )
            )
        }

        let relation = Relation::try_from(
            parse("SELECT a, Sum(d) AS sum_d FROM table_1 GROUP BY a")
                .unwrap()
                .with(&relations),
        )
        .unwrap();
        let filtered_relation = relation.filter(Expr::and(
            Expr::gt(Expr::col("a"), Expr::val(5.)),
            Expr::lt(Expr::col("sum_d"), Expr::val(15)),
        ));
        _ = filtered_relation.display_dot();
        assert_eq!(
            filtered_relation.schema().field("a").unwrap().data_type(),
            DataType::float_interval(5., 10.)
        );
        assert_eq!(
            filtered_relation
                .schema()
                .field("sum_d")
                .unwrap()
                .data_type(),
            DataType::integer_interval(0, 15)
        );
    }

    fn test_possion_sampling() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        let proba = 0.5;

        let table = relations
            .get(&["item_table".into()])
            .unwrap()
            .as_ref()
            .clone();

        let reduce: Relation = Relation::reduce()
            .input(table.clone())
            .with(("sum_price", Expr::sum(Expr::col("price"))))
            .with_group_by_column("item")
            .with_group_by_column("order_id")
            .build();

        let map: Relation = Relation::map()
            .with(Expr::abs(Expr::col("order_id")))
            .input(table.clone())
            .build();

        let join: Relation = Relation::join()
            .left(relations.get(&["order_table".into()]).unwrap().clone())
            .right(table.clone())
            .on(Expr::eq(Expr::col("id"), Expr::col("order_id")))
            .build();

        let sampled_table = table.clone().poisson_sampling(proba);
        namer::reset();
        let expected_sampled_table: Relation = Relation::map()
            .with_iter(
                table
                    .clone()
                    .schema()
                    .iter()
                    .map(|f| (f.name(), Expr::col(f.name()))),
            )
            .filter(Expr::lt(
                Expr::random(namer::new_id("POISSON_SAMPLING")),
                Expr::val(proba),
            ))
            .input(table.clone())
            .build();
        namer::reset();
        let sampled_reduce = reduce.clone().poisson_sampling(proba);
        namer::reset();
        let expected_sampled_reduce: Relation = Relation::map()
            .with_iter(
                reduce
                    .clone()
                    .schema()
                    .iter()
                    .map(|f| (f.name(), Expr::col(f.name()))),
            )
            .filter(Expr::lt(
                Expr::random(namer::new_id("POISSON_SAMPLING")),
                Expr::val(proba),
            ))
            .input(reduce.clone())
            .build();
        namer::reset();
        let sampled_map: Relation = map.clone().poisson_sampling(proba);
        namer::reset();
        let expected_sampled_map: Relation = Relation::map()
            .with_iter(
                map.clone()
                    .schema()
                    .iter()
                    .map(|f| (f.name(), Expr::col(f.name()))),
            )
            .filter(Expr::lt(
                Expr::random(namer::new_id("POISSON_SAMPLING")),
                Expr::val(proba),
            ))
            .input(map.clone())
            .build();
        namer::reset();
        let sampled_join: Relation = join.clone().poisson_sampling(proba);
        namer::reset();
        let expected_sampled_join: Relation = Relation::map()
            .with_iter(
                join.clone()
                    .schema()
                    .iter()
                    .map(|f| (f.name(), Expr::col(f.name()))),
            )
            .filter(Expr::lt(
                Expr::random(namer::new_id("POISSON_SAMPLING")),
                Expr::val(proba),
            ))
            .input(join.clone())
            .build();

        sampled_table.display_dot().unwrap();
        sampled_reduce.display_dot().unwrap();
        sampled_map.display_dot().unwrap();
        sampled_join.display_dot().unwrap();

        assert_eq!(expected_sampled_table, sampled_table);
        assert_eq!(expected_sampled_reduce, sampled_reduce);
        assert_eq!(expected_sampled_map, sampled_map);
        assert_eq!(expected_sampled_join, sampled_join);
    }

    #[ignore]
    #[test]
    fn test_sampling_query() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        // relation with reduce
        let relation = Relation::try_from(
            parse("SELECT 0.0 as z, sum(price) as a, sum(price) as b FROM item_table GROUP BY order_id")
                .unwrap()
                .with(&relations),
        )
        .unwrap();

        let proba = 0.5;
        namer::reset();
        let sampled_relation = relation.poisson_sampling(proba);

        let query_sampled_relation = &ast::Query::try_from(&sampled_relation).unwrap().to_string();

        let expected_query = r#"WITH
        map_qcqr (field_z650, field_08wv) AS (SELECT price AS field_z650, order_id AS field_08wv FROM item_table),
        reduce_8knj (field_glfp) AS (SELECT sum(field_z650) AS field_glfp FROM map_qcqr GROUP BY field_08wv),
        map_xyv8 (z, a, b) AS (SELECT 0 AS z, field_glfp AS a, field_glfp AS b FROM reduce_8knj),
        map_bfzk (z, a, b) AS (SELECT z AS z, a AS a, b AS b FROM map_xyv8 WHERE (random()) < (0.5))
        SELECT * FROM map_bfzk"#;

        assert_eq!(
            expected_query.replace('\n', " ").replace(' ', ""),
            (&query_sampled_relation[..]).replace(' ', "")
        );
        print!("{}\n", query_sampled_relation);

        // relation with map
        let relation = Relation::try_from(
            parse("SELECT LOG(price) FROM item_table")
                .unwrap()
                .with(&relations),
        )
        .unwrap();

        let proba = 0.5;
        namer::reset();
        let sampled_relation = relation.poisson_sampling(proba);

        let query_sampled_relation = &ast::Query::try_from(&sampled_relation).unwrap().to_string();

        let expected_query = r#"WITH map_gj2u (field_uy24) AS (SELECT log(price) AS field_uy24 FROM item_table),
        map_upop (field_uy24) AS (SELECT field_uy24 AS field_uy24 FROM map_gj2u WHERE (random()) < (0.5))
        SELECT * FROM map_upop"#;

        assert_eq!(
            expected_query.replace('\n', " ").replace(' ', ""),
            (&query_sampled_relation[..]).replace(' ', "")
        );
        print!("{}\n", query_sampled_relation);

        // relation with join
        let relation = Relation::try_from(
            parse("SELECT * FROM order_table JOIN item_table ON(id=order_id)")
                .unwrap()
                .with(&relations),
        )
        .unwrap();

        let proba = 0.5;
        namer::reset();
        let sampled_relation = relation.poisson_sampling(proba);

        let query_sampled_relation = &ast::Query::try_from(&sampled_relation).unwrap().to_string();

        let expected_query = r#"WITH
        join__e_y (field_eygr, field_0wjz, field_cg0j, field_idxm, field_0eqn, field_3ned, field_gwco) AS (
            SELECT * FROM order_table JOIN item_table ON (order_table.id) = (item_table.order_id)
        ), map_8r2s (field_eygr, field_0wjz, field_cg0j, field_idxm, field_0eqn, field_3ned, field_gwco) AS (
            SELECT field_eygr AS field_eygr, field_0wjz AS field_0wjz, field_cg0j AS field_cg0j,
                field_idxm AS field_idxm, field_0eqn AS field_0eqn, field_3ned AS field_3ned, field_gwco AS field_gwco
            FROM join__e_y
        ), map_yko1 (field_eygr, field_0wjz, field_cg0j, field_idxm, field_0eqn, field_3ned, field_gwco) AS (
            SELECT field_eygr AS field_eygr, field_0wjz AS field_0wjz, field_cg0j AS field_cg0j,
                field_idxm AS field_idxm, field_0eqn AS field_0eqn, field_3ned AS field_3ned, field_gwco AS field_gwco
            FROM map_8r2s WHERE (random()) < (0.5)
        ) SELECT * FROM map_yko1"#;

        assert_eq!(
            expected_query.replace('\n', " ").replace(' ', ""),
            (&query_sampled_relation[..]).replace(' ', "")
        );
        print!("{}\n", query_sampled_relation)
    }

    #[test]
    fn test_unique() {
        let table: Relation = Relation::table()
            .name("table")
            .schema(
                Schema::builder()
                    .with(("a", DataType::integer_range(1..=10)))
                    .with(("b", DataType::integer_values([1, 2, 5, 6, 7, 8])))
                    .with(("c", DataType::integer_range(5..=20)))
                    .build(),
            )
            .build();

        // Without group by
        let unique_rel = table.unique(vec!["a", "b"]);
        println!("{}", unique_rel);
        _ = unique_rel.display_dot();
    }

    #[test]
    fn test_build_ordered_reduce() {
        let table: Relation = Relation::table()
            .name("table")
            .schema(
                Schema::builder()
                    .with(("a", DataType::integer_range(1..=10)))
                    .with(("b", DataType::integer_values([1, 2, 5, 6, 7, 8])))
                    .with(("c", DataType::integer_range(5..=20)))
                    .build(),
            )
            .build();

        // Without group by
        let grouping_exprs = vec![];
        let aggregates = vec![
            ("sum_a", Expr::sum(Expr::col("a"))),
            ("count_b", Expr::count(Expr::col("a"))),
        ];
        let rel = table
            .clone()
            .build_ordered_reduce(grouping_exprs, aggregates);
        println!("{}", rel);
        _ = rel.display_dot();

        // With group by
        let grouping_exprs = vec![Expr::col("c")];
        let aggregates = vec![
            ("sum_a", Expr::sum(Expr::col("a"))),
            ("count_b", Expr::count(Expr::col("a"))),
        ];
        let rel = table.build_ordered_reduce(grouping_exprs, aggregates);
        println!("{}", rel);
        _ = rel.display_dot();
    }

    #[test]
    fn test_distinct_aggregates() {
        let table: Relation = Relation::table()
            .name("table")
            .schema(
                Schema::builder()
                    .with(("a", DataType::integer_range(1..=10)))
                    .with(("b", DataType::integer_values([1, 2, 5, 6, 7, 8])))
                    .with(("c", DataType::integer_range(5..=20)))
                    .build(),
            )
            .build();

        // Without group by
        let column = "a";
        let group_by = vec![];
        let aggregates = vec![
            ("sum_distinct_a", aggregate::Aggregate::Sum),
            ("count_distinct_a", aggregate::Aggregate::Count),
        ];
        let distinct_rel = table
            .clone()
            .distinct_aggregates(column, group_by, aggregates);
        println!("{}", distinct_rel);
        _ = distinct_rel.display_dot();

        // With group by
        let column = "a";
        let group_by = vec!["b", "c"];
        let aggregates = vec![
            ("sum_distinct_a", aggregate::Aggregate::Sum),
            ("count_distinct_a", aggregate::Aggregate::Count),
        ];
        let distinct_rel = table
            .clone()
            .distinct_aggregates(column, group_by, aggregates);
        println!("{}", distinct_rel);
        _ = distinct_rel.display_dot();
    }

    #[test]
    fn test_public_values_column() {
        let table: Relation = Relation::table()
            .name("table")
            .schema(
                Schema::builder()
                    .with(("a", DataType::float_range(1.0..=10.0)))
                    .with(("b", DataType::integer_values([1, 2, 5])))
            )
            .build();

        // table
        let rel = table.public_values_column("b").unwrap();
        let rel_values: Relation = Relation::values().name("b").values([1, 2, 5]).build();
        rel.display_dot();
        assert_eq!(rel, rel_values);
        assert!(table.public_values_column("a").is_err());

        // map
        let map: Relation = Relation::map()
        .name("map_1")
        .with(("exp_a", Expr::exp(Expr::col("a"))))
        .input(table.clone())
        .with(("exp_b",  Expr::exp(Expr::col("b"))))
        .build();
        let rel = map.public_values_column("exp_b").unwrap();
        rel.display_dot();
        assert!(map.public_values_column("exp_a").is_err());
    }

    #[test]
    fn test_left_join() {
        let table1: Relation = Relation::table()
            .name("table")
            .schema(
                Schema::builder()
                    .with(("a", DataType::integer_range(1..=10)))
                    .with(("b", DataType::integer_values([1, 2, 5, 6, 7, 8])))
                    .build(),
            )
            .build();

        let table2: Relation = Relation::table()
            .name("table")
            .schema(
                Schema::builder()
                    .with(("c", DataType::integer_range(5..=20)))
                    .with(("d", DataType::integer_range(1..=100)))
                    .build(),
            )
            .build();

        let joined_rel = table1
            .clone()
            .left_join(table2.clone(), vec![("a", "c")])
            .unwrap();
        _ = joined_rel.display_dot();
    }

    #[test]
    fn test_cross_join() {
        let table1: Relation = Relation::table()
            .name("table")
            .schema(
                Schema::builder()
                    .with(("a", DataType::integer_range(1..=10)))
                    .with(("b", DataType::integer_values([1, 2, 5, 6, 7, 8])))
                    .build(),
            )
            .build();

        let table2: Relation = Relation::table()
            .name("table")
            .schema(
                Schema::builder()
                    .with(("c", DataType::integer_range(5..=20)))
                    .with(("d", DataType::integer_range(1..=100)))
                    .build(),
            )
            .build();

        let joined_rel = table1.clone().cross_join(table2.clone()).unwrap();
        _ = joined_rel.display_dot();
    }
}
