use std::fmt;

use crate::{
    data_type::{DataType, DataTyped},
    namer,
};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Constraint {
    Unique,
    PrimaryKey,
    ForeignKey,
}

/// A Field as in https://github.com/apache/arrow-datafusion/blob/5b23180cf75ea7155d7c35a40f224ce4d5ad7fb8/datafusion/src/logical_plan/dfschema.rs#L413
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Field {
    name: String,
    data_type: DataType,
    constraint: Option<Constraint>,
}

impl Field {
    /// Constructor
    pub fn new(name: String, data_type: DataType, constraint: Option<Constraint>) -> Field {
        Field {
            name,
            data_type,
            constraint,
        }
    }

    pub fn from_name_data_type<S: Into<String>, T: Into<DataType>>(name: S, data_type: T) -> Field {
        Field::new(name.into(), data_type.into(), None)
    }

    pub fn from_data_type<T: Into<DataType>>(data_type: T) -> Field {
        Field::new(namer::new_name("field"), data_type.into(), None)
    }

    /// Return the `Field`'s name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the `Field`'s constraint
    /// Constraint is a Copy type
    pub fn constraint(&self) -> Option<Constraint> {
        self.constraint
    }

    /// Check if there is a constraint
    pub fn has_constraint(&self) -> bool {
        self.constraint.is_some()
    }

    /// Create a new Field with name
    pub fn with_name<S: Into<String>>(self, name: S) -> Field {
        Field::new(name.into(), self.data_type, self.constraint)
    }

    /// Create a new Field with type
    pub fn with_data_type<T: Into<DataType>>(self, data_type: T) -> Field {
        Field::new(self.name, data_type.into(), self.constraint)
    }

    /// Create a new Field with contraint
    pub fn with_constraint(self, constraint: Constraint) -> Field {
        Field::new(self.name, self.data_type, Some(constraint))
    }
}

impl fmt::Display for Field {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(constraint) = self.constraint {
            write!(f, "{}: {} ({:?})", self.name, self.data_type, constraint)
        } else {
            write!(f, "{}: {}", self.name, self.data_type)
        }
    }
}

impl<S: Into<String>, T: Into<DataType>> From<(S, T)> for Field {
    fn from(name_data_type: (S, T)) -> Self {
        Field::from_name_data_type(name_data_type.0, name_data_type.1)
    }
}

impl From<DataType> for Field {
    fn from(data_type: DataType) -> Self {
        Field::from_data_type(data_type)
    }
}

impl DataTyped for Field {
    fn data_type(&self) -> DataType {
        self.data_type.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let field: Field = ("test", DataType::float_range(0.0..=1.0)).into();
        println!("field = {field}");
        assert!(!field.has_constraint());
        let field = field
            .with_name("with-constraint")
            .with_constraint(Constraint::PrimaryKey);
        println!("field = {field}");
        assert!(field.has_constraint());
    }

    #[test]
    fn test_new_from_type() {
        let field: Field = DataType::float_range(0.0..=1.0).into();
        println!("field = {field}");
        assert!(!field.has_constraint());
        let field = field
            .with_name("with-constraint")
            .with_constraint(Constraint::PrimaryKey);
        println!("field = {field}");
        assert!(field.has_constraint());
    }
}
