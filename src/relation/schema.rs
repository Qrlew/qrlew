use itertools::Itertools;
use std::{
    collections::HashSet,
    fmt::Display,
    ops::{BitAnd, Deref, Index},
};

use super::{field::Field, Error, Result};
use crate::{
    builder::{Ready, With},
    data_type::{self, DataType, DataTyped},
    expr::{identifier::Identifier, Expr},
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

    /// Returns a new `Schema` where the `fields` of this `Schema`
    /// has been filtered by predicate `Expr`
    pub fn filter(&self, predicate: &Expr) -> Self {
        self.data_type().filter(predicate).into()
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
        match data_type {
            DataType::Struct(s) => Self::new(
                s.iter()
                    .map(|(name, t)| Field::from_name_data_type(name, t.as_ref().clone()))
                    .collect::<Vec<_>>(),
            ),
            DataType::Union(_) => todo!(),
            _ => Schema::from_field(Field::from_data_type(data_type)),
        }
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
    use crate::data_type::{DataType, Variant};
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
        assert_eq!(schema.filter(&expression), filtered_schema);

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
        assert_eq!(schema.filter(&expression), filtered_schema);

        let schema = Schema::builder()
            .with(("a", DataType::integer_interval(-10, 2)))
            .with(("b", DataType::integer_interval(-2, 5)))
            .with(("c", DataType::text()))
            .with(("d", DataType::float_interval(-100., 100.)))
            .build();
        let filtered_schema = Schema::builder()
            .with(("a", DataType::integer_interval(-2, 2)))
            .with((
                "b",
                DataType::from(data_type::Integer::from_intervals([
                    [-2, -2],
                    [-1, -1],
                    [0, 0],
                    [1, 1],
                    [2, 2],
                ])),
            ))
            .with(("c", DataType::text()))
            .with((
                "d",
                DataType::integer_interval(-2, 2)
                    .into_variant(&DataType::float())
                    .unwrap(),
            ))
            .build();
        let expression = Expr::and(
            Expr::lt(Expr::col("b"), Expr::col("a")),
            Expr::eq(Expr::col("d"), Expr::col("b")),
        );
        assert_eq!(schema.filter(&expression), filtered_schema);
    }
}
