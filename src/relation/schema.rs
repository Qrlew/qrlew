use itertools::Itertools;
use std::{
    collections::HashSet,
    fmt::Display,
    ops::{BitAnd, Deref, Index},
};

use super::{field::Field, Error, Result};
use crate::{
    builder::{Ready, With},
    data_type::{
        function::{bivariate_max, bivariate_min, Function as _},
        DataType, DataTyped, Variant,
    },
    expr::{function, identifier::Identifier, Expr, Function, Value},
};

/// A struct holding Fields as in https://github.com/apache/arrow-datafusion/blob/5b23180cf75ea7155d7c35a40f224ce4d5ad7fb8/datafusion/src/logical_plan/dfschema.rs#L36
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Schema {
    fields: Vec<Field>,
}

impl Schema {
    /// Schema constructor, checking for name collisions
    pub fn new(fields: Vec<Field>) -> Self {
        // The fields must all be different
        let mut field_names = HashSet::new();
        assert!(
            fields
                .iter()
                .all(move |field| field_names.insert(field.name())),
            "You cannot create a schema with ambiguous column names"
        );
        Schema { fields }
    }

    /// Create an empty schema
    pub fn empty() -> Self {
        Schema::new(vec![])
    }

    /// Create a schema with a new field from an existing Shema
    pub fn from_field<F: Into<Field>>(field: F) -> Self {
        Schema::new(vec![field.into()])
    }

    /// Create a schema with a new field from an existing Shema
    pub fn with<F: Into<Field>>(self, field: F) -> Self {
        let mut fields = self.fields;
        fields.push(field.into());
        Schema::new(fields)
    }

    /// Builder
    pub fn builder() -> Builder {
        Builder::new()
    }

    // Accessors

    /// Get a list of fields
    pub fn fields(&self) -> &[Field] {
        &self.fields
    }

    /// Access a field by name
    pub fn field(&self, name: &str) -> Result<&Field> {
        if let Some(index) = self.fields.iter().position(|f| f.name() == name) {
            Ok(&self.fields[index])
        } else {
            Err(Error::invalid_name(name))
        }
    }

    /// Find the index of the field with the given name
    pub fn index_from_name(&self, name: &str) -> Result<usize> {
        if let Some(index) = self.fields.iter().position(|f| f.name() == name) {
            Ok(index)
        } else {
            Err(Error::invalid_name(name))
        }
    }

    /// Access a field by index
    pub fn field_from_index(&self, index: usize) -> Result<&Field> {
        Ok(self
            .fields
            .get(index)
            .ok_or_else(|| Error::invalid_index(index))?)
    }

    /// Access a field by identifier
    pub fn field_from_identifier(&self, identifier: &Identifier) -> Result<&Field> {
        assert_eq!(identifier.len(), 1);
        self.field(&identifier.head()?)
    }

    /// Iter over the Schema
    pub fn iter(&self) -> impl Iterator<Item = &Field> {
        self.fields.iter()
    }

    /// Returns a new instance of `Schema` where the `data_type` of the
    /// field whose name is `name` is set to `datatype`. All the other fields
    /// are unchanged.
    fn with_name_datatype(self, name: String, datatype: DataType) -> Schema {
        let new_fields: Vec<Field> = self
            .into_iter()
            .map(|f| {
                if f.name() == name {
                    Field::new(name.clone(), datatype.clone(), f.constraint())
                } else {
                    f
                }
            })
            .collect();
        Schema::new(new_fields)
    }

    /// Returns a new `Schema` where the `fields` of this `Schema`
    /// has been filtered by predicate `Expr`
    pub fn filter(&self, predicate: &Expr) -> Result<Self> {
        match predicate {
            Expr::Function(func) => self.filter_by_function(&func),
            _ => Ok(self.clone()),
        }
    }

    /// Returns a new `Schema` where the `fields` of this `Schema`
    /// has been filtered by predicate `Expr::Function`
    ///
    /// Note: for the moment, we support only `Function` made of the composition of:
    /// - `Gt`, `GtEq`, `Lt`, `LtEq` functions comparing a column to a float or an integer value,
    /// - `Eq` function comparing a column to any value,
    /// - `And` function between two supported Expr::Function,
    /// - 'InList` test if a column value belongs to a list
    // TODO : OR
    pub fn filter_by_function(&self, predicate: &Function) -> Result<Self> {
        {
            let datatypes: Vec<(&str, DataType)> = self
                .fields()
                .iter()
                .map(|f| (f.name(), f.data_type()))
                .collect();
            let datatype = DataType::structured(datatypes);
            let mut new_schema = self.clone();

            match (predicate.function(), predicate.arguments().as_slice()) {
                (function::Function::And, [left, right]) => {
                    let schema1 = self.filter(right)?.filter(left)?;
                    let schema2 = self.filter(left)?.filter(right)?;
                    new_schema = Schema::new(
                        schema1
                            .iter()
                            .zip(schema2)
                            .map(|(f1, f2)| {
                                Ok(Field::from_name_data_type(
                                    f1.name(),
                                    f1.data_type().super_intersection(&f2.data_type())?,
                                ))
                            })
                            .collect::<Result<Vec<Field>>>()?,
                    )
                }
                // Set min or max
                (function::Function::Gt, [left, right])
                | (function::Function::GtEq, [left, right])
                | (function::Function::Lt, [right, left])
                | (function::Function::LtEq, [right, left]) => {
                    let left_dt = left.super_image(&datatype).unwrap();
                    let right_dt = right.super_image(&datatype).unwrap();

                    let left_dt = if let DataType::Optional(o) = left_dt {
                        o.data_type().clone()
                    } else {
                        left_dt
                    };

                    let right_dt = if let DataType::Optional(o) = right_dt {
                        o.data_type().clone()
                    } else {
                        right_dt
                    };

                    let set =
                        DataType::structured_from_data_types([left_dt.clone(), right_dt.clone()]);
                    if let Expr::Column(col) = left {
                        let dt = bivariate_max()
                            .super_image(&set)
                            .unwrap()
                            .super_intersection(&left_dt)?;
                        new_schema = new_schema.with_name_datatype(col.head().unwrap(), dt)
                    }
                    if let Expr::Column(col) = right {
                        let dt = bivariate_min()
                            .super_image(&set)
                            .unwrap()
                            .super_intersection(&right_dt)?;
                        new_schema = new_schema.with_name_datatype(col.head().unwrap(), dt)
                    }
                }
                (function::Function::Eq, [left, right]) => {
                    let left_dt = left.super_image(&datatype)?;
                    let right_dt = right.super_image(&datatype)?;
                    let dt = left_dt.super_intersection(&right_dt)?;
                    if let Expr::Column(col) = left {
                        new_schema = new_schema.with_name_datatype(col.head().unwrap(), dt.clone())
                    }
                    if let Expr::Column(col) = right {
                        new_schema = new_schema.with_name_datatype(col.head().unwrap(), dt)
                    }
                }
                (function::Function::InList, [Expr::Column(col), Expr::Value(Value::List(l))]) => {
                    let dt = DataType::from_iter(l.to_vec().clone())
                        .super_intersection(&new_schema.field(&col.head()?).unwrap().data_type())?;
                    new_schema = new_schema.with_name_datatype(col.head().unwrap(), dt)
                }
                _ => (),
            }
            Ok(new_schema)
        }
    }
}

impl Default for Schema {
    fn default() -> Self {
        Schema::empty()
    }
}

impl Display for Schema {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{{}}}",
            self.iter().map(|field| format!("{field}")).join(", ")
        )
    }
}

impl Deref for Schema {
    type Target = [Field];

    fn deref(&self) -> &Self::Target {
        self.fields.deref()
    }
}

impl Index<&str> for Schema {
    type Output = Field;

    fn index(&self, name: &str) -> &Self::Output {
        self.field(name).unwrap()
    }
}

impl Index<usize> for Schema {
    type Output = Field;

    fn index(&self, index: usize) -> &Self::Output {
        self.field_from_index(index).unwrap()
    }
}

impl Index<&Identifier> for Schema {
    type Output = Field;

    fn index(&self, identifier: &Identifier) -> &Self::Output {
        self.field_from_identifier(identifier).unwrap()
    }
}

impl<F: Into<Field>> BitAnd<F> for Schema {
    type Output = Schema;

    fn bitand(self, rhs: F) -> Self::Output {
        self.with(rhs)
    }
}

impl From<Field> for Schema {
    fn from(field: Field) -> Self {
        Schema::from_field(field)
    }
}

impl<S: Into<String>, T: Into<DataType>> From<(S, T)> for Schema {
    fn from(name_data_type: (S, T)) -> Self {
        Schema::from_field(Field::from(name_data_type))
    }
}

/// A conversion from a fixed size array
impl<F: Into<Field>, const N: usize> From<[F; N]> for Schema {
    fn from(fields: [F; N]) -> Self {
        fields.into_iter().collect()
    }
}

impl From<DataType> for Schema {
    fn from(data_type: DataType) -> Self {
        Schema::from_field(Field::from_data_type(data_type))
    }
}

impl<F: Into<Field>> FromIterator<F> for Schema {
    fn from_iter<T: IntoIterator<Item = F>>(iter: T) -> Self {
        let fields: Vec<Field> = iter.into_iter().map(|field| field.into()).collect();
        Schema::new(fields)
    }
}

impl IntoIterator for Schema {
    type Item = Field;
    type IntoIter = <Vec<Field> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.fields.into_iter()
    }
}

impl Extend<Field> for Schema {
    fn extend<T: IntoIterator<Item = Field>>(&mut self, iter: T) {
        self.fields.extend(iter);
    }
}

impl DataTyped for Schema {
    fn data_type(&self) -> DataType {
        let fields: Vec<(&str, DataType)> = self
            .iter()
            .map(|field| (field.name(), field.data_type()))
            .collect();
        DataType::structured(&fields)
    }
}

#[derive(Debug, Default)]
pub struct Builder {
    /// Schema fields
    fields: Vec<Field>,
}

impl Builder {
    pub fn new() -> Builder {
        Builder { fields: vec![] }
    }
}

impl<S: Into<String>, T: Into<DataType>> With<(S, T)> for Builder {
    fn with(mut self, name_data_type: (S, T)) -> Self {
        self.fields.push(Field::from_name_data_type(
            name_data_type.0,
            name_data_type.1,
        ));
        self
    }
}

impl With<DataType> for Builder {
    fn with(mut self, data_type: DataType) -> Self {
        self.fields.push(Field::from_data_type(data_type));
        self
    }
}

impl Ready<Schema> for Builder {
    type Error = Error;

    fn try_build(self) -> Result<Schema> {
        Ok(self.fields.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_type::DataType;
    use std::panic::catch_unwind;

    #[test]
    fn test_new() {
        let schema = Schema::empty() & ("a", DataType::float()) & ("b", DataType::text());
        println!("schema = {}", schema);
        // Add a new field
        let schema = schema.with(("c", DataType::unit()));
        println!("schema = {}", schema);
        // Fail on adding existing field
        assert!(catch_unwind(|| { schema.with(("a", DataType::unit())) }).is_err())
    }

    #[test]
    fn test_from_iter() {
        let schema: Schema = "Abcd"
            .chars()
            .map(|c| (c.to_string(), DataType::date_time()))
            .collect();
        println!("schema = {}", schema);
        assert!(catch_unwind(|| { schema.with(("a", DataType::unit())) }).is_ok())
    }

    #[test]
    fn test_from_data_type_iter() {
        let schema: Schema = vec![
            DataType::float(),
            DataType::integer(),
            DataType::date(),
            DataType::text(),
            DataType::boolean(),
            DataType::float_range(0.0..=12.),
        ]
        .into_iter()
        .collect();
        println!("schema = {}", schema);
    }

    #[test]
    fn test_data_typed() {
        let schema = Schema::empty()
            & ("Float", DataType::float())
            & ("N", DataType::integer_min(0))
            & ("Z", DataType::integer())
            & ("Text", DataType::text())
            & ("Date", DataType::date());
        println!("schema = {}", schema);
        println!("schema data-type = {}", schema.data_type());
    }

    #[test]
    fn test_data_typed_array() {
        let schema = Schema::from([
            ("Float", DataType::float()),
            ("N", DataType::integer_min(0)),
            ("Z", DataType::integer()),
            ("Text", DataType::text()),
            ("Date", DataType::date()),
        ]);
        println!("schema = {}", schema);
        println!("schema data-type = {}", schema.data_type());
    }

    #[test]
    fn test_data_index() {
        let schema = Schema::empty()
            & ("Float", DataType::float())
            & ("N", DataType::integer_min(0))
            & ("Z", DataType::integer())
            & ("Text", DataType::text())
            & ("Date", DataType::date());
        println!("schema = {}", schema);
        println!(
            "schema[{}] = {}",
            Identifier::from(["Text"]),
            schema[&Identifier::from(["Text"])]
        );
        println!("schema[3] = {}", schema[3]);
        println!(r#"schema["Text"] = {}"#, schema["Text"]);
        assert_eq!(schema["Text"], schema[3]);
        assert_eq!(schema["Text"], schema[&Identifier::from(["Text"])]);
    }

    #[test]
    fn test_builder() {
        let schema = Schema::builder()
            .with(DataType::integer_min(0))
            .with(("Z", DataType::integer()))
            .with(("Text", DataType::text()))
            .with(("N", DataType::date()))
            .build();
        println!("schema = {}", schema);
        println!(
            "schema[{}] = {}",
            Identifier::from(["Text"]),
            schema[&Identifier::from(["Text"])]
        );
        println!("schema[3] = {}", schema[2]);
        println!(r#"schema["Text"] = {}"#, schema["Text"]);
        assert_eq!(schema["Text"], schema[2]);
        assert_eq!(schema["Text"], schema[&Identifier::from(["Text"])]);
    }

    #[test]
    fn test_filter() {
        let schema = Schema::builder()
            .with(("a", DataType::integer_max(20)))
            .with(("b", DataType::integer_max(100)))
            .with(("c", DataType::text()))
            .with(("d", DataType::float()))
            .build();
        let filtered_schema = Schema::builder()
            .with(("a", DataType::integer_interval(5, 20)))
            .with(("b", DataType::integer_interval(3, 9)))
            .with(("c", DataType::text()))
            .with(("d", DataType::float()))
            .build();
        let expression = expr!(and(and(and(gt(a, 5), gt(b, 3)), lt_eq(b, 9)), lt_eq(a, 90)));
        assert_eq!(schema.filter(&expression).unwrap(), filtered_schema);

        let schema = Schema::builder()
            .with(("a", DataType::integer_max(20)))
            .with(("b", DataType::integer_max(100)))
            .with(("c", DataType::text()))
            .with(("d", DataType::float()))
            .build();
        let filtered_schema = Schema::builder()
            .with(("a", DataType::integer_max(20)))
            .with(("b", DataType::integer_max(100)))
            .with(("c", DataType::text_value("a".to_string())))
            .with(("d", DataType::float()))
            .build();
        let expression = Expr::eq(Expr::col("c"), Expr::val("a".to_string()));
        assert_eq!(schema.filter(&expression).unwrap(), filtered_schema);
    }

    #[test]
    fn test_filter_simple() {
        let schema = Schema::from([
            ("a", DataType::float_interval(-10., 10.)),
            ("b", DataType::integer_interval(0, 8)),
            ("c", DataType::float()),
        ]);

        // ((((a > 5) and (b < 4)) and ((9 >= a) and (2 <= b))) and (c = 0.99))
        let x = expr!(and(
            and(and(gt(a, 5), lt(b, 4.)), and(gt_eq(9., a), lt_eq(2, b))),
            eq(c, 0.99)
        ));
        println!("{}", x);
        let filtered_schema = schema.filter(&x).unwrap();
        let true_schema = Schema::from([
            ("a", DataType::float_interval(5., 9.)),
            ("b", DataType::integer_interval(2, 4)),
            ("c", DataType::float_value(0.99)),
        ]);
        assert_eq!(filtered_schema, true_schema);

        // ((a = 45) and (b = 3.5) and (0 = c))
        let x = expr!(and(eq(a, 45), and(eq(b, 3.5), eq(0, c))));
        println!("{}", x);
        let filtered_schema = schema.filter(&x).unwrap();
        let true_schema = Schema::from([
            ("a", DataType::Null),
            ("b", DataType::Null),
            ("c", DataType::float_value(0.)),
        ]);
        assert_eq!(filtered_schema, true_schema);

        // (b in (1, 3, 4.5))
        let val = Expr::list([1., 3., 4.5]);
        let a = Expr::in_list(Expr::col("a"), val.clone());
        let b = Expr::in_list(Expr::col("b"), val.clone());
        let x = Expr::and(a, b);
        println!("{}", x);
        let filtered_schema = schema.filter(&x).unwrap();
        println!("{}", filtered_schema);
        let true_schema = Schema::from([
            ("a", DataType::float_values([1., 3., 4.5])),
            ("b", DataType::integer_values([1, 3])),
            ("c", DataType::float()),
        ]);
        assert_eq!(filtered_schema, true_schema);

        // (b = exp(a))
        let x = expr!(eq(b, exp(a)));
        let schema = Schema::from([
            ("a", DataType::float_interval(-1., 1.)),
            ("b", DataType::float()),
        ]);
        let filtered_schema = schema.filter(&x).unwrap();
        println!("{} -> {}", x, filtered_schema);
        let true_schema = Schema::from([
            ("a", DataType::float_interval(-1., 1.)),
            (
                "b",
                DataType::float_interval((-1. as f64).exp(), (1. as f64).exp()),
            ),
        ]);
        assert_eq!(filtered_schema, true_schema);
    }

    #[test]
    fn test_filter_with_simple_column_deps() {
        let schema = Schema::from([
            ("a", DataType::float_interval(-10., 10.)),
            ("b", DataType::integer_interval(0, 20)),
        ]);
        // (b < a)
        let x = expr!(lt(b, a));
        let filtered_schema = schema.filter(&x).unwrap();
        println!("{} -> {}", x, filtered_schema);
        let true_schema = Schema::from([
            ("a", DataType::float_interval(-0., 10.)),
            ("b", DataType::integer_interval(0, 10)),
        ]);
        assert_eq!(filtered_schema, true_schema);
        // (a > b)
        let x = expr!(gt(a, b));
        let filtered_schema = schema.filter(&x).unwrap();
        println!("{} -> {}", x, filtered_schema);
        assert_eq!(filtered_schema, true_schema);

        // (b = a)
        let x = expr!(eq(b, a));
        let filtered_schema = schema.filter(&x).unwrap();
        println!("{} -> {}", x, filtered_schema);
        let true_schema = Schema::from([
            ("a", DataType::integer_interval(0, 10)),
            ("b", DataType::integer_interval(0, 10)),
        ]);
        assert_eq!(filtered_schema, true_schema);
        // (a = b)
        let x = expr!(eq(a, b));
        let filtered_schema = schema.filter(&x).unwrap();
        println!("{} -> {}", x, filtered_schema);
        assert_eq!(filtered_schema, true_schema);
    }

    #[test]
    fn test_filter_with_column_deps() {
        let schema = Schema::from([
            ("a", DataType::float_interval(-10., 10.)),
            ("b", DataType::integer_interval(0, 18)),
            ("c", DataType::float()),
        ]);

        // ((b < 2) and (b = c))
        let x = expr!(and(lt(b, 2), eq(b, c)));
        let filtered_schema = schema.filter(&x).unwrap();
        println!("{} -> {}", x, filtered_schema);
        let true_schema = Schema::from([
            ("a", DataType::float_interval(-10., 10.)),
            ("b", DataType::integer_values([0, 1, 2])), // TODO != DataType::integer_interval([0, 2])) ?
            ("c", DataType::float_values([0., 1., 2.])),
        ]);
        assert_eq!(filtered_schema, true_schema);

        // ((b = c) and (b < 2))
        let x = expr!(and(lt(b, 2), eq(b, c)));
        let filtered_schema = schema.filter(&x).unwrap();
        println!("{} -> {}", x, filtered_schema);
        let true_schema = Schema::from([
            ("a", DataType::float_interval(-10., 10.)),
            ("b", DataType::integer_values([0, 1, 2])),
            ("c", DataType::float_values([0., 1., 2.])),
        ]);
        assert_eq!(filtered_schema, true_schema);

        // ((((a > 5) and (b < 14)) and ((b >= a) and (2 <= b))) and (a = c))
        let x = expr!(and(
            and(and(gt(a, 5), lt(b, 14.)), and(gt_eq(b, a), lt_eq(2, b))),
            eq(a, c)
        ));
        let filtered_schema = schema.filter(&x).unwrap();
        println!("{} -> {}", x, filtered_schema);
        let true_schema = Schema::from([
            ("a", DataType::float_interval(5., 10.)),
            (
                "b",
                DataType::integer_values([5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
            ),
            ("c", DataType::float_interval(5., 10.)),
        ]);
        assert_eq!(filtered_schema, true_schema);

        // (a >= (2 * b))
        let schema = Schema::from([
            ("a", DataType::float_interval(-10., 10.)),
            ("b", DataType::integer_interval(0, 8)),
        ]);
        let x = expr!(gt_eq(a, 2 * b));
        let filtered_schema = schema.filter(&x).unwrap();
        println!("{} -> {}", x, filtered_schema);
        let true_schema = Schema::from([
            ("a", DataType::float_interval(0., 10.)),
            ("b", DataType::integer_interval(0, 8)),
        ]);
        assert_eq!(filtered_schema, true_schema);

        // (a <= (2 * b))
        let schema = Schema::from([
            ("a", DataType::float_interval(-10., 10.)),
            ("b", DataType::integer_interval(0, 2)),
        ]);
        let x = expr!(lt_eq(a, 2 * b));
        let filtered_schema = schema.filter(&x).unwrap();
        println!("{} -> {}", x, filtered_schema);
        let true_schema = Schema::from([
            ("a", DataType::float_interval(-10., 4.)),
            ("b", DataType::integer_interval(0, 2)),
        ]);
        assert_eq!(filtered_schema, true_schema);
    }

    #[test]
    fn test_filter_composed() {
        let schema = Schema::from([
            ("a", DataType::float_interval(-10., 10.)),
            ("b", DataType::integer_interval(0, 8)),
        ]);

        let x1 = expr!(lt(a, (3 * 5 - 8)));
        let x2 = expr!(gt(b, ((5 / 2 - 1) + 2)));
        let x3 = expr!(gt_eq(a, 2 * b));
        println!("x1 = {}, x2 = {}, x3 = {}", x1, x2, x3);

        // (a < ((3 * 5) - 8))
        let filtered_schema = schema.filter(&x1).unwrap();
        println!("x1 = {} -> {}", x1, filtered_schema);
        let true_schema = Schema::from([
            ("a", DataType::float_interval(-10., 7.)),
            ("b", DataType::integer_interval(0, 8)),
        ]);
        assert_eq!(filtered_schema, true_schema);

        // (b > (((5 / 2) - 1) + 2))
        let filtered_schema = schema.filter(&x2).unwrap();
        println!("x2 = {} -> {}", x2, filtered_schema);
        let true_schema = Schema::from([
            ("a", DataType::float_interval(-10., 10.)),
            ("b", DataType::integer_interval(3, 8)),
        ]);
        assert_eq!(filtered_schema, true_schema);

        // (a >= (2 * b))
        let filtered_schema = schema.filter(&x3).unwrap();
        println!("x3 = {} -> {}", x3, filtered_schema);
        let true_schema = Schema::from([
            ("a", DataType::float_interval(0., 10.)),
            ("b", DataType::integer_interval(0, 8)),
        ]);
        assert_eq!(filtered_schema, true_schema);

        let true_schema = Schema::from([
            ("a", DataType::float_interval(6.0, 7.)),
            ("b", DataType::integer_interval(3, 8)),
        ]);

        //  (x1 and (x2 and x3))
        let x = Expr::and(x1.clone(), Expr::and(x2.clone(), x3.clone()));
        let filtered_schema = schema.filter(&x).unwrap();
        println!("{} -> {}", x, filtered_schema);
        assert_eq!(filtered_schema, true_schema);

        //  (x3 and (x1 and x2))
        let x = Expr::and(x3.clone(), Expr::and(x1.clone(), x2.clone()));
        let filtered_schema = schema.filter(&x).unwrap();
        println!("{} -> {}", x, filtered_schema);
        assert_eq!(filtered_schema, true_schema);

        //  (x2 and (x3 and x1))
        let x = Expr::and(x2.clone(), Expr::and(x3.clone(), x1.clone()));
        let filtered_schema = schema.filter(&x).unwrap();
        println!("{} -> {}", x, filtered_schema);
        assert_eq!(filtered_schema, true_schema);

        // ((x1 and (x3 and x2))
        let x = Expr::and(x1.clone(), Expr::and(x3.clone(), x2.clone()));
        let filtered_schema = schema.filter(&x).unwrap();
        println!("{} -> {}", x, filtered_schema);
        assert_eq!(filtered_schema, true_schema);

        // ((x2 and (x1 and x3))
        let x = Expr::and(x2.clone(), Expr::and(x1.clone(), x3.clone()));
        let filtered_schema = schema.filter(&x).unwrap();
        println!("{} -> {}", x, filtered_schema);
        assert_eq!(filtered_schema, true_schema);

        // ((x3 and (x2 and x1))
        let x = Expr::and(x2.clone(), Expr::and(x1.clone(), x3.clone()));
        let filtered_schema = schema.filter(&x).unwrap();
        println!("{} -> {}", x, filtered_schema);
        assert_eq!(filtered_schema, true_schema);
    }

    #[test]
    fn test_filter_optional() {
        let schema = Schema::from([("a", DataType::optional(DataType::float_interval(-10., 10.)))]);

        // (a > 1)
        let x = expr!(gt(a, 1));
        let filtered_schema = schema.filter(&x).unwrap();
        println!("{} -> {}", x, filtered_schema);
        let true_schema = Schema::from([("a", DataType::float_interval(1., 10.))]);
        assert_eq!(filtered_schema, true_schema);

        // (a < 1)
        let x = expr!(lt(a, 1));
        let filtered_schema = schema.filter(&x).unwrap();
        println!("{} -> {}", x, filtered_schema);
        let true_schema = Schema::from([("a", DataType::float_interval(-10., 1.))]);
        assert_eq!(filtered_schema, true_schema);
    }
}
