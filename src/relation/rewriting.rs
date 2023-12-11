//! A few transforms for relations
//!

use super::{Join, Map, Reduce, Relation, Set, Table, Values, Variant as _};
use crate::{
    builder::{Ready, With, WithIterator},
    data_type::{self, function::Function, DataType, DataTyped, Variant as _},
    expr::{self, aggregate, Aggregate, Expr, Value},
    io, namer, relation,
};
use std::{
    collections::{BTreeMap, HashMap},
    convert::Infallible,
    error, fmt,
    num::ParseFloatError,
    result,
    sync::Arc,
};

#[derive(Debug, PartialEq)]
pub enum Error {
    InvalidRelation(String),
    InvalidArguments(String),
    NoPublicValues(String),
    Other(String),
}

impl Error {
    pub fn invalid_relation(relation: impl fmt::Display) -> Error {
        Error::InvalidRelation(format!("{} is invalid", relation))
    }
    pub fn invalid_arguments(relation: impl fmt::Display) -> Error {
        Error::InvalidArguments(format!("{} is invalid", relation))
    }
    pub fn no_public_values(relation: impl fmt::Display) -> Error {
        Error::NoPublicValues(format!("{} is invalid", relation))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidRelation(desc) => writeln!(f, "InvalidRelation: {}", desc),
            Error::InvalidArguments(desc) => writeln!(f, "InvalidArguments: {}", desc),
            Error::NoPublicValues(desc) => {
                writeln!(f, "NoPublicValues: {}", desc)
            }
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
impl From<relation::Error> for Error {
    fn from(err: relation::Error) -> Self {
        Error::Other(err.to_string())
    }
}
impl From<expr::Error> for Error {
    fn from(err: crate::expr::Error) -> Self {
        Error::Other(err.to_string())
    }
}
impl From<io::Error> for Error {
    fn from(err: crate::io::Error) -> Self {
        Error::Other(err.to_string())
    }
}
impl From<data_type::Error> for Error {
    fn from(err: data_type::Error) -> Self {
        Error::Other(err.to_string())
    }
}

impl From<ParseFloatError> for Error {
    fn from(err: ParseFloatError) -> Self {
        Error::Other(err.to_string())
    }
}

pub type Result<T> = result::Result<T, Error>;

/* Reduce
 */

impl Table {
    /// Create a new Table with a new name
    pub fn with_name(mut self, name: String) -> Table {
        self.name = name;
        self
    }
}

/* Map
 */

impl Map {
    /// Create a new Map with a new name
    pub fn with_name(mut self, name: String) -> Map {
        self.name = name;
        self
    }
    /// Create a new Map with a prepended field
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
    pub fn rename_fields<F: Fn(&str, &Expr) -> String>(self, f: F) -> Map {
        Relation::map().rename_with(self, f).build()
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

    /// Clip all sums in the `Reduce`
    pub fn l2_clipped_all_sums(&self, entities: &str) -> Result<Relation> {
        let mut input_entities: Option<&str> = None;
        let input_groups: Vec<&str> = self.group_by_names();
        let mut clipping_values: Vec<(&str, f64)> = vec![];
        let mut names: HashMap<&str, &str> = HashMap::new();
        for (name, aggregate) in self.named_aggregates() {
            if name == entities {
                input_entities = aggregate.column().last().ok();
            } else if aggregate.aggregate() == &aggregate::Aggregate::Sum {
                let value_name = aggregate.column().last()?;
                let value_data_type = self.input().schema()[value_name].data_type();
                let absolute_bound = value_data_type.absolute_upper_bound().unwrap_or(1.0);
                names.insert(value_name, name);
                clipping_values.push((value_name, absolute_bound)) // TODO Set a better clipping value
            }
        }
        let input_entities = input_entities.ok_or(Error::invalid_arguments(entities))?;
        Ok(self
            .input()
            .clone()
            .l2_clipped_sums(input_entities, input_groups, clipping_values)
            .rename_fields(|s, _| names.get(s).unwrap_or(&s).to_string()))
    }

    /// Rename fields
    pub fn rename_fields<F: Fn(&str, &Expr) -> String>(self, f: F) -> Reduce {
        Relation::reduce().rename_with(self, f).build()
    }

    /// In the current `Reduce` contains grouping columns,
    /// adds them to the `aggregate` field.
    pub fn with_grouping_columns(self) -> Reduce {
        if self.group_by().is_empty() {
            self
        } else {
            Reduce::builder()
                .with(self.clone()) // Must be first in order to conserve the order
                .with_iter(
                    self.group_by_names()
                        .into_iter()
                        .map(|s| (s, Expr::first(Expr::col(s)))),
                )
                .build()
        }
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
    /// Filter fields
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
    /// Map fields
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
    /// Rename fields
    pub fn rename_fields<F: Fn(&str, &Expr) -> String>(self, f: F) -> Relation {
        match self {
            Relation::Map(map) => map.rename_fields(f).into(),
            Relation::Reduce(red) => red.rename_fields(f).into(),
            relation => Relation::map()
                .with_iter(relation.schema().iter().map(|field| {
                    (
                        f(field.name(), &Expr::col(field.name())),
                        Expr::col(field.name()),
                    )
                }))
                .input(relation)
                .build(),
        }
    }
    /// Sum values for each group.
    /// Groups form the basis of a vector space, the sums are the coordinates.
    pub fn sums_by_group(self, groups: Vec<&str>, values: Vec<&str>) -> Self {
        let mut reduce = Relation::reduce().input(self.clone());
        reduce = groups
            .into_iter()
            .fold(reduce, |acc, s| acc.with_group_by_column(s));
        reduce = reduce.with_iter(
            values
                .into_iter()
                .map(|c| (c, Expr::sum(Expr::col(c.to_string())))),
        );
        reduce.build()
    }
    /// Compute L1 norms of the vectors formed by the group values for each entities
    pub fn l1_norms(self, entities: &str, groups: Vec<&str>, values: Vec<&str>) -> Self {
        let mut entities_groups = vec![entities];
        entities_groups.extend(groups.clone());
        self.sums_by_group(entities_groups, values.clone())
            .map_fields(|field, expr| {
                if values.contains(&field) {
                    Expr::abs(expr)
                } else {
                    expr
                }
            })
            .sums_by_group(vec![entities], values)
    }
    /// Compute L2 norms of the vectors formed by the group values for each entities
    pub fn l2_norms(self, entities: &str, groups: Vec<&str>, values: Vec<&str>) -> Self {
        let mut entities_groups = vec![entities];
        entities_groups.extend(groups.clone());
        self.sums_by_group(entities_groups, values.clone())
            .map_fields(|field_name, expr| {
                if values.contains(&field_name) {
                    // TODO Remove abs
                    // Abs is here to signal a positive number
                    Expr::abs(Expr::multiply(expr.clone(), expr))
                } else {
                    expr
                }
            })
            .sums_by_group(vec![entities], values.clone())
            .map_fields(|field_name, expr| {
                if values.contains(&field_name) {
                    Expr::sqrt(expr)
                } else {
                    expr
                }
            })
    }

    /// This transform multiplies the values in `self` relation by their corresponding `scale_factors`.
    /// `scale_factors` contains the entities scaling factors and the vectors columns
    /// `self` contains the coordinates, the base and vectors columns
    pub fn scale(self, entities: &str, values: Vec<&str>, scale_factors: Relation) -> Self {
        // TODO fix this
        // Join the two relations on the entity column
        let join: Relation = Relation::join()
            .inner()
            .on_eq(entities, entities)
            .left_names(
                self.fields()
                    .into_iter()
                    .map(|field| field.name())
                    .collect(),
            )
            .right_names(
                scale_factors
                    .fields()
                    .into_iter()
                    .map(|field| format!("_SCALE_FACTOR_{}", field.name()))
                    .collect(),
            )
            .left(self)
            .right(scale_factors)
            .build();
        // Multiply the values by the factors
        join.map_fields(|field_name, expr| {
            if values.contains(&field_name) {
                Expr::multiply(expr, Expr::col(format!("_SCALE_FACTOR_{}", field_name)))
            } else {
                expr
            }
        })
    }

    /// For each coordinate, rescale the columns by 1 / greatest(1, norm_l2/C)
    /// where the l2 norm is computed for each elecment of `vectors`
    /// The `self` relation must contain the vectors, base and coordinates columns
    pub fn l2_clipped_sums(
        self,
        entities: &str,
        groups: Vec<&str>,
        value_clippings: Vec<(&str, f64)>,
    ) -> Self {
        // Arrange the values
        let value_clippings: HashMap<&str, f64> = value_clippings.into_iter().collect();
        // Compute the norm
        let norms = self.clone().l2_norms(
            entities,
            groups.clone(),
            value_clippings.keys().cloned().collect(),
        );
        // Compute the scaling factors
        let scaling_factors = norms.map_fields(|field_name, expr| {
            if value_clippings.contains_key(&field_name) {
                let value_clipping = value_clippings[&field_name];
                if value_clipping == 0.0 {
                    Expr::val(value_clipping)
                } else {
                    Expr::divide(
                        Expr::val(1),
                        Expr::greatest(
                            Expr::val(1),
                            Expr::divide(expr.clone(), Expr::val(value_clipping)),
                        ),
                    )
                }
            } else {
                expr
            }
        });
        let clipped_relation = self.scale(
            entities,
            value_clippings.keys().cloned().collect(),
            scaling_factors,
        );
        clipped_relation.sums_by_group(groups, value_clippings.keys().cloned().collect())
    }

    /// Clip sums in the first `Reduce`s found
    pub fn l2_clipped_all_sums(&self, entities: &str) -> Result<Self> {
        match self {
            Relation::Reduce(reduce) => reduce.l2_clipped_all_sums(entities),
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

    /// Add gaussian noise of a given standard deviation to the given columns, while keeping the column min and max
    pub fn add_clipped_gaussian_noise(self, name_sigmas: Vec<(&str, f64)>) -> Relation {
        let name_sigmas: HashMap<&str, f64> = name_sigmas.into_iter().collect();
        Relation::map()
            // .with_iter(name_sigmas.into_iter().map(|(name, sigma)| (name, Expr::col(name).add_gaussian_noise(sigma))))
            .with_iter(self.schema().iter().map(|f| {
                if name_sigmas.contains_key(&f.name()) {
                    let x = Expr::coalesce(Expr::col(f.name()), Expr::val(0.));
                    let float_data_type: data_type::Float = x
                        .super_image(&f.data_type())
                        .unwrap()
                        .into_data_type(&DataType::float())
                        .unwrap()
                        .try_into()
                        .unwrap();
                    (
                        f.name(),
                        Expr::least(
                            Expr::val(*float_data_type.max().unwrap()),
                            Expr::greatest(
                                Expr::val(*float_data_type.min().unwrap()),
                                x.add_gaussian_noise(name_sigmas[f.name()]),
                            ),
                        ),
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
        columns: BTreeMap<
            &str,
            (
                Option<data_type::value::Value>,
                Option<data_type::value::Value>,
                Vec<data_type::value::Value>,
            ),
        >,
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
    fn ordered_reduce(self, grouping_exprs: Vec<Expr>, aggregates: Vec<(&str, Expr)>) -> Relation {
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

    /// GROUP BY all the fields. This mimicks the sql `DISTINCT` in the
    /// `SELECT` clause.
    pub fn distinct(self) -> Relation {
        let fields = self.schema()
            .iter()
            .map(|f| f.name().to_string())
            .collect::<Vec<_>>();
        Relation::reduce()
            .input(self)
            .with_iter(fields.iter().map(|f| (f, Expr::first(Expr::col(f)))))
            .group_by_iter(fields.iter().map(|f| Expr::col(f)))
            .build()
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
                Expr::Aggregate(Aggregate::new(agg, Arc::new(Expr::col(column)))),
            ))
        });
        red.ordered_reduce(grouping_exprs, aggregates_exprs)
    }

    pub fn public_values_column(&self, col_name: &str) -> Result<Relation> {
        let data_type = self.schema().field(col_name).unwrap().data_type();
        let values: Vec<Value> = data_type.try_into()?;
        Ok(Relation::values().name(col_name).values(values).build())
    }

    pub fn public_values(&self) -> Result<Relation> {
        let vec_of_rel: Result<Vec<Relation>> = self
            .schema()
            .iter()
            .map(|c| self.public_values_column(c.name()))
            .collect();

        Ok(vec_of_rel?
            .into_iter()
            .reduce(|l, r| l.cross_join(r).unwrap())
            .unwrap())
    }

    /// Returns the cross join between `self` and `right` where
    /// the output names of the fields are conserved.
    /// This fails if one column name is contained in both relations
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
                    format!(
                        "Cannot use `cross_join` method for joining two relations containing fields with the same names.\
                        left: {:?}\nright: {:?}", left_names, right_names
                    )
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

    /// Returns the outer join between `self` and `right` where
    /// the output names of the fields are conserved.
    /// The joining criteria is the equality of columns with the same name
    pub fn natural_inner_join(self, right: Self) -> Relation {
        let mut left_names: Vec<String> = vec![];
        let mut right_names: Vec<String> = vec![];
        let mut names: Vec<(String, Expr)> = vec![];
        for f in self.fields() {
            let col = f.name().to_string();
            left_names.push(col.clone());
            names.push((col.clone(), Expr::col(col)));
        }
        for f in right.fields() {
            let col = f.name().to_string();
            if left_names.contains(&col) {
                right_names.push(format!("right_{}", col));
            } else {
                right_names.push(col.clone());
                names.push((col.clone(), Expr::col(col)));
            }
        }
        let join: Relation = Relation::join()
            .left(self.clone())
            .right(right.clone())
            .inner()
            .left_names(left_names)
            .right_names(right_names)
            .build();
        Relation::map()
            .input(join)
            .with_iter(names)
            .build()
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
        data_type::{value::List, DataType, DataTyped},
        display::Dot,
        expr::AggregateColumn,
        io::{postgresql, Database},
        relation::schema::Schema,
        sql::parse,
    };
    use colored::Colorize;
    use itertools::Itertools;

    #[test]
    fn test_with_computed_field() {
        let database = postgresql::test_database();
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
        let relation = orders.with_referred_field(
            "user_id".into(),
            Arc::new(user.clone()),
            "id".into(),
            "id".into(),
            "peid".into(),
        );
        assert!(relation.schema()[0].name() == "peid");
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
    fn test_sums_by_group() {
        let database = postgresql::test_database();
        let relations = database.relations();
        let mut relation = relations
            .get(&["item_table".into()])
            .unwrap()
            .as_ref()
            .clone();
        // Print query before
        println!("Before: {}", &ast::Query::from(&relation));
        relation.display_dot().unwrap();
        // Sum by group
        relation = relation.sums_by_group(vec!["order_id"], vec!["price"]);
        // Print query after
        println!("After: {}", &ast::Query::from(&relation));
        relation.display_dot().unwrap();
        assert_eq!(
            relation.data_type(),
            DataType::structured(vec![
                ("order_id", DataType::integer_interval(0, 100)),
                ("price", DataType::float_interval(0., 15000.)),
            ])
        );
    }

    #[test]
    fn test_l1_norms() {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        let mut relation = relations
            .get(&["user_table".into()])
            .unwrap()
            .as_ref()
            .clone();
        // Compute l1 norm
        relation = relation.l1_norms("id", vec!["city"], vec!["age"]);
        // Print query
        let query = &ast::Query::from(&relation);
        println!("After: {}", query);
        relation.display_dot().unwrap();
        let expected_query = "SELECT id, SUM(ABS(age)) FROM (SELECT id, city, SUM(age) AS age FROM user_table GROUP BY id, city) AS sums GROUP BY id";
        assert_eq!(
            database.query(&query.to_string()).unwrap(),
            database.query(expected_query).unwrap()
        );
        // To double check
        for row in database.query("SELECT id, SUM(ABS(age)) FROM (SELECT id, city, SUM(age) AS age FROM user_table GROUP BY id, city) AS sums GROUP BY id ORDER BY id").unwrap() {
            println!("{row}");
        }
        for row in database
            .query("SELECT id, count(id) FROM user_table GROUP BY id ORDER BY id")
            .unwrap()
        {
            println!("{row}");
        }
        for row in database
            .query("SELECT id, age FROM user_table ORDER BY id")
            .unwrap()
        {
            println!("{row}");
        }
    }

    #[test]
    fn test_l2_norms() {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        let mut relation = relations
            .get(&["user_table".into()])
            .unwrap()
            .as_ref()
            .clone();
        // Compute l2 norm
        relation = relation.l2_norms("id", vec!["city"], vec!["age"]);
        // Print query
        let query = &ast::Query::from(&relation);
        println!("After: {}", query);
        relation.display_dot().unwrap();
        let expected_query = "SELECT id, SQRT(SUM(age*age)) FROM (SELECT id, city, SUM(age) AS age FROM user_table GROUP BY id, city) AS sums GROUP BY id";
        assert_eq!(
            database.query(&query.to_string()).unwrap(),
            database.query(expected_query).unwrap()
        );
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
            .l1_norms("order_id", vec!["item"], vec!["price"]);
        // amount_norm.display_dot().unwrap();
        let query: &str = &ast::Query::from(&amount_norm).to_string();
        println!("Query = {}", query);
        let valid_query = "SELECT order_id, SUM(sum_by_group) FROM (SELECT order_id, item, SUM(ABS(price)) AS sum_by_group FROM item_table GROUP BY order_id, item) AS subquery GROUP BY order_id";
        assert_eq!(
            database.query(query).unwrap(),
            database.query(valid_query).unwrap()
        );
        // L2 Norm
        let amount_norm = table.l2_norms("order_id", vec!["item"], vec!["price"]);
        amount_norm.display_dot().unwrap();
        let query: &str = &ast::Query::from(&amount_norm).to_string();
        let valid_query = "SELECT order_id, SQRT(SUM(sum_by_group)) FROM (SELECT order_id, item, POWER(SUM(price), 2) AS sum_by_group FROM item_table GROUP BY order_id, item) AS subquery GROUP BY order_id";
        assert_eq!(
            database.query(query).unwrap(),
            database.query(valid_query).unwrap()
        );
    }

    #[test]
    fn test_compute_norm_for_empty_groups() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        let table = relations
            .get(&["item_table".into()])
            .unwrap()
            .as_ref()
            .clone();
        // L1 Norm
        let amount_norm = table.clone().l1_norms("order_id", vec![], vec!["price"]);
        amount_norm.display_dot().unwrap();
        let query: &str = &format!("{} ORDER BY order_id", ast::Query::from(&amount_norm));
        println!("Query = {}", query);
        let valid_query =
            "SELECT order_id, ABS(SUM(price)) FROM item_table GROUP BY order_id ORDER BY order_id";
        assert_eq!(
            database.query(query).unwrap(),
            database.query(valid_query).unwrap()
        );

        // L2 Norm
        let amount_norm = table.l2_norms("order_id", vec![], vec!["price"]);
        amount_norm.display_dot().unwrap();
        let query: &str = &format!("{} ORDER BY order_id", ast::Query::from(&amount_norm));
        let valid_query =
            "SELECT order_id, SQRT(POWER(SUM(price), 2)) FROM item_table GROUP BY order_id ORDER BY order_id";
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
                .l1_norms("order_id", vec!["item"], vec!["price", "std_price"]);
        relation_norm.display_dot().unwrap();
        let query: &str = &ast::Query::from(&relation_norm).to_string();
        //println!("Query = {}", query);
        let valid_query = "SELECT order_id, SUM(sum_1), SUM(sum_2) FROM (SELECT order_id, item, ABS(SUM(price)) AS sum_1, ABS(SUM(std_price)) AS sum_2 FROM ( SELECT price - 25 AS std_price, * FROM item_table ) AS intermediate_table GROUP BY order_id, item) AS subquery GROUP BY order_id";
        assert_eq!(
            database.query(query).unwrap(),
            database.query(valid_query).unwrap()
        );
        // L2 Norm
        let relation_norm = relation.l2_norms("order_id", vec!["item"], vec!["price", "std_price"]);
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
            .on_eq("order_id", "id")
            .build();
        let schema = relation.schema().clone();
        let item = schema.field_from_index(1).unwrap().name();
        let price = schema.field_from_index(2).unwrap().name();
        let user_id = schema.field_from_index(4).unwrap().name();
        let date = schema.field_from_index(6).unwrap().name();

        // L1 Norm
        let relation_norm = relation
            .clone()
            .l1_norms(user_id, vec![item, date], vec![price]);
        relation_norm.display_dot().unwrap();
        let query: &str = &ast::Query::from(&relation_norm).to_string();
        println!("Query = {}", query);

        let valid_query = "SELECT user_id, SUM(sum_1) FROM (SELECT user_id, item, date, ABS(SUM(price)) AS sum_1 FROM item_table JOIN order_table ON item_table.order_id = order_table.id GROUP BY user_id, item, date) AS subquery GROUP BY user_id";
        assert_eq!(
            database.query(query).unwrap(),
            database.query(valid_query).unwrap()
        );
        // L2 Norm
        let relation_norm = relation.l2_norms(user_id, vec![item, date], vec![price]);
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
    fn test_l2_clipped_sums() {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        let mut relation = relations
            .get(&["user_table".into()])
            .unwrap()
            .as_ref()
            .clone();

        // Compute l2 norm
        let clipped_relation =
            relation
                .clone()
                .l2_clipped_sums("id", vec!["city"], vec![("age", 20.)]);
        clipped_relation.display_dot().unwrap();
        // Print query
        let query = &ast::Query::from(&clipped_relation).to_string();
        println!("After: {}", query);
        for row in database.query(query).unwrap() {
            println!("{row}");
        }

        // 100
        let norm = 100.;
        let clipped_relation_100 =
            relation
                .clone()
                .l2_clipped_sums("id", vec!["city"], vec![("age", norm)]);
        for row in database
            .query(&ast::Query::from(&clipped_relation_100).to_string())
            .unwrap()
        {
            println!("{row}");
        }

        // 1000
        let norm = 1000.;
        let clipped_relation_1000 =
            relation
                .clone()
                .l2_clipped_sums("id", vec!["city"], vec![("age", norm)]);
        for row in database
            .query(&ast::Query::from(&clipped_relation_1000).to_string())
            .unwrap()
        {
            println!("{row}");
        }
        assert!(
            database
                .query(&ast::Query::from(&clipped_relation_100).to_string())
                .unwrap()
                != database
                    .query(&ast::Query::from(&clipped_relation_1000).to_string())
                    .unwrap()
        );

        // 10000
        let norm = 10000.;
        let clipped_relation_10000 =
            relation
                .clone()
                .l2_clipped_sums("id", vec!["city"], vec![("age", norm)]);
        for row in database
            .query(&ast::Query::from(&clipped_relation_10000).to_string())
            .unwrap()
        {
            println!("{row}");
        }
        assert!(
            database
                .query(&ast::Query::from(&clipped_relation_1000).to_string())
                .unwrap()
                == database
                    .query(&ast::Query::from(&clipped_relation_10000).to_string())
                    .unwrap()
        );
        for row in database
            .query("SELECT city, sum(age) FROM user_table GROUP BY city")
            .unwrap()
        {
            println!("{row}");
        }
        assert!(
            database
                .query(&ast::Query::from(&clipped_relation_1000).to_string())
                .unwrap()
                == database
                    .query("SELECT city, sum(age) FROM user_table GROUP BY city")
                    .unwrap()
        );
    }

    #[test]
    fn test_l2_clipped_all_sums_reduce() {
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
            .group_by(Expr::col("item"))
            .with_group_by_column("order_id")
            .build();

        my_relation.display_dot().unwrap();
        let clipped_relation = my_relation.l2_clipped_all_sums("order_id").unwrap();
        clipped_relation.display_dot().unwrap();
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
    fn test_filter_on_map() {
        let database = postgresql::test_database();
        let relations = database.relations();

        let relation = Relation::try_from(
            parse("SELECT exp(a) AS my_a, b As my_b FROM table_1")
                .unwrap()
                .with(&relations),
        )
        .unwrap();
        let filtering_expr = Expr::and(
            Expr::and(
                Expr::gt(Expr::col("my_a"), Expr::val(5.)),
                Expr::lt(Expr::col("my_b"), Expr::val(0.)),
            ),
            Expr::lt(Expr::col("my_a"), Expr::val(100.)),
        );
        println!("{}", filtering_expr);
        let filtered_relation = relation.filter(filtering_expr);
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
            DataType::float_interval(-1., 0.)
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
    }

    #[test]
    fn test_filter_on_wildcard() {
        let database = postgresql::test_database();
        let relations = database.relations();

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
            DataType::float_interval(-1., 0.5)
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
    }

    #[test]
    fn test_filter_on_reduce() {
        let database = postgresql::test_database();
        let relations = database.relations();

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

    #[ignore]
    #[test]
    fn test_poisson_sampling() {
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
    fn test_ordered_reduce() {
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
        let rel = table.clone().ordered_reduce(grouping_exprs, aggregates);
        println!("{}", rel);
        _ = rel.display_dot();

        // With group by
        let grouping_exprs = vec![Expr::col("c")];
        let aggregates = vec![
            ("sum_a", Expr::sum(Expr::col("a"))),
            ("count_b", Expr::count(Expr::col("a"))),
        ];
        let rel = table.ordered_reduce(grouping_exprs, aggregates);
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
                    .build(),
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
            .with(("exp_b", Expr::exp(Expr::col("b"))))
            .build();
        let rel = map.public_values_column("exp_b").unwrap();
        rel.display_dot();
        assert!(map.public_values_column("exp_a").is_err());
    }

    #[test]
    fn test_public_values() {
        // table
        let table: Relation = Relation::table()
            .name("table")
            .schema(
                Schema::builder()
                    .with(("a", DataType::float_values([1.0, 10.0])))
                    .with(("b", DataType::integer_values([1, 2, 5])))
                    .build(),
            )
            .build();
        let rel = table.public_values().unwrap();
        rel.display_dot();

        let table: Relation = Relation::table()
            .name("table")
            .schema(
                Schema::builder()
                    .with(("a", DataType::float_interval(1.0, 10.0)))
                    .with(("b", DataType::integer_interval(1, 2)))
                    .build(),
            )
            .build();
        let rel = table.public_values();
        assert!(rel.is_err());

        // map
        let table: Relation = Relation::table()
            .name("table")
            .schema(
                Schema::builder()
                    .with(("a", DataType::float_values([1.0, 10.0])))
                    .with(("b", DataType::integer_values([1, 2, 5])))
                    .build(),
            )
            .build();
        let map: Relation = Relation::map()
            .name("map_1")
            .with(("a", Expr::col("a")))
            .with(("b", Expr::col("b")))
            .input(table)
            .build();
        let rel = map.public_values().unwrap();
        rel.display_dot();

        // map
        let table: Relation = Relation::table()
            .name("table")
            .schema(
                Schema::builder()
                    .with(("a", DataType::float_interval(1.0, 10.0)))
                    .with(("b", DataType::integer_values([1, 2, 5])))
                    .build(),
            )
            .build();
        let map: Relation = Relation::map()
            .name("map_1")
            .with(("a", Expr::col("a")))
            .with(("b", Expr::col("b")))
            .filter(Expr::in_list(
                Expr::col("a"),
                Expr::list([1., 2., 3.5, 4.5]),
            ))
            .input(table)
            .build();
        let rel = map.public_values().unwrap();
        rel.display_dot();
    }

    #[test]
    fn test_cross_join() {
        let table_1: Relation = Relation::table()
            .name("table_1")
            .schema(
                Schema::builder()
                    .with(("a", DataType::integer_range(1..=10)))
                    .with(("b", DataType::integer_values([1, 2, 5, 6, 7, 8])))
                    .build(),
            )
            .build();

        let table_2: Relation = Relation::table()
            .name("table_2")
            .schema(
                Schema::builder()
                    .with(("c", DataType::integer_range(5..=20)))
                    .with(("d", DataType::integer_range(1..=100)))
                    .build(),
            )
            .build();

        let joined_rel = table_1.clone().cross_join(table_2.clone()).unwrap();
        joined_rel.display_dot();
    }

    #[test]
    fn test_with_grouping_columns() {
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

        // no GROUP BY
        let red = Reduce::new(
            "reduce_relation".to_string(),
            vec![("sum_a".to_string(), AggregateColumn::sum("a"))],
            vec![],
            Arc::new(table.clone()),
        );
        let red_with_grouping_columns = red.clone().with_grouping_columns();
        assert_eq!(red, red_with_grouping_columns);

        // grouping columns are already in `aggregate`
        let red = Reduce::new(
            "reduce_relation".to_string(),
            vec![
                ("sum_a".to_string(), AggregateColumn::sum("a")),
                ("b".to_string(), AggregateColumn::first("b")),
            ],
            vec!["b".into()],
            Arc::new(table.clone()),
        );
        let red_with_grouping_columns = red.clone().with_grouping_columns();
        assert_eq!(red_with_grouping_columns.aggregate().len(), 2);
        let names_aggs = vec!["sum_a", "b"];
        assert_eq!(
            red_with_grouping_columns
                .named_aggregates()
                .iter()
                .map(|(s, _)| *s)
                .collect::<Vec<_>>(),
            names_aggs
        );

        // grouping columns are not in `aggregate`
        let red = Reduce::new(
            "reduce_relation".to_string(),
            vec![("sum_a".to_string(), AggregateColumn::sum("a"))],
            vec!["b".into()],
            Arc::new(table.clone()),
        );
        let red_with_grouping_columns = red.clone().with_grouping_columns();
        assert_eq!(red_with_grouping_columns.aggregate().len(), 2);
        let names_aggs = vec!["sum_a", "b"];
        assert_eq!(
            red_with_grouping_columns
                .named_aggregates()
                .iter()
                .map(|(s, _)| *s)
                .collect::<Vec<_>>(),
            names_aggs
        );

        // grouping columns are not in `aggregate`
        let red = Reduce::new(
            "reduce_relation".to_string(),
            vec![
                ("b".to_string(), AggregateColumn::first("b")),
                ("sum_a".to_string(), AggregateColumn::sum("a")),
            ],
            vec!["b".into(), "c".into()],
            Arc::new(table.clone()),
        );
        let red_with_grouping_columns = red.clone().with_grouping_columns();
        let names_aggs = vec!["b", "sum_a", "c"];
        assert_eq!(
            red_with_grouping_columns
                .named_aggregates()
                .iter()
                .map(|(s, _)| *s)
                .collect::<Vec<_>>(),
            names_aggs
        );

        // not the same order
        let red = Reduce::new(
            "reduce_relation".to_string(),
            vec![
                ("b".to_string(), AggregateColumn::first("b")),
                ("c".to_string(), AggregateColumn::first("c")),
                ("sum_a".to_string(), AggregateColumn::sum("a")),
            ],
            vec!["b".into(), "c".into()],
            Arc::new(table.clone()),
        );
        let red_with_grouping_columns = red.clone().with_grouping_columns();
        let names_aggs = vec!["b", "c", "sum_a"];
        assert_eq!(
            red_with_grouping_columns
                .named_aggregates()
                .iter()
                .map(|(s, _)| *s)
                .collect::<Vec<_>>(),
            names_aggs
        );
    }

    #[test]
    fn test_distinct() {
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

        // Table
        let distinct_relation = table.clone().distinct();
        assert_eq!(distinct_relation.schema(), table.schema());
        assert!(matches!(distinct_relation, Relation::Reduce(_)));
        if let Relation::Reduce(red) = distinct_relation {
            assert_eq!(red.group_by.len(), table.schema().len())
        }

        // Map
        let relation: Relation = Relation::map()
            .input(table.clone())
            .with(expr!(a * b))
            .with(("my_c", expr!(c)))
            .build();
        let distinct_relation = relation.clone().distinct();
        assert_eq!(distinct_relation.schema(), relation.schema());
        assert!(matches!(distinct_relation, Relation::Reduce(_)));
        if let Relation::Reduce(red) = distinct_relation {
            assert_eq!(red.group_by.len(), relation.schema().len())
        }

        // Reduce
        let relation: Relation = Relation::reduce()
            .input(table.clone())
            .with(expr!(count(a)))
            //.with_group_by_column("c")
            .with(("twice_c", expr!(first(2*c))))
            .group_by(expr!(c))
            .build();
        let distinct_relation = relation.clone().distinct();
        distinct_relation.display_dot();
        assert_eq!(distinct_relation.schema(), relation.schema());
        assert!(matches!(distinct_relation, Relation::Reduce(_)));
        if let Relation::Reduce(red) = distinct_relation {
            assert_eq!(red.group_by.len(), relation.schema().len())
        }
    }
}
