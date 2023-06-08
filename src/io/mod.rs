pub mod postgresql;
#[cfg(feature = "sqlite")]
pub mod sqlite;

use crate::{
    builder::{Ready, With},
    data_type::{
        self,
        value::{self},
        DataType,
    },
    hierarchy::Hierarchy,
    relation::{builder::TableBuilder, schema::Schema, Relation, Table, Variant as _},
};
use std::{convert::Infallible, error, fmt, io, num, rc::Rc, result};

const DATA_GENERATION_SEED: u64 = 1234;

// Error management
#[derive(Debug)]
pub enum Error {
    Database(String),
    Dataset(String),
    Other(String),
}

impl Error {
    pub fn database(database: impl fmt::Display) -> Error {
        Error::Database(format!("Database error {}", database))
    }
    pub fn dataset(dataset: impl fmt::Display) -> Error {
        Error::Dataset(format!("Dataset error {}", dataset))
    }
    pub fn other(desc: impl fmt::Display) -> Error {
        Error::Other(format!("{}", desc))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Database(database) => writeln!(f, "Database: {}", database),
            Error::Dataset(dataset) => writeln!(f, "Dataset: {}", dataset),
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
impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Error::Other(err.to_string())
    }
}
impl From<num::ParseIntError> for Error {
    fn from(err: num::ParseIntError) -> Self {
        Error::Other(err.to_string())
    }
}
impl From<data_type::Error> for Error {
    fn from(err: data_type::Error) -> Self {
        Error::Other(err.to_string())
    }
}
impl From<data_type::value::Error> for Error {
    fn from(err: data_type::value::Error) -> Self {
        Error::Other(err.to_string())
    }
}

pub type Result<T> = result::Result<T, Error>;

pub trait Database: Sized {
    const MAX_SIZE: usize = 1 << 14;
    /// Create a database
    fn new(name: String, tables: Vec<Table>) -> Result<Self>;
    /// Get the name
    fn name(&self) -> &str;
    /// Get the tables
    fn tables(&self) -> &[Table];
    /// Get a mutable reference to the tables
    fn tables_mut(&mut self) -> &mut Vec<Table>;
    /// Get a dictionary of relations
    fn relations(&self) -> Hierarchy<Rc<Relation>> {
        self.tables()
            .iter()
            .map(|t| {
                (
                    [self.name().clone(), t.name().clone()],
                    Rc::new(t.clone().into()),
                )
            })
            .collect()
    }
    /// Create an empty db
    fn empty(name: String) -> Result<Self> {
        Self::new(name, Vec::new())
    }
    /// Create a table from a table object
    fn create_table(&mut self, table: &Table) -> Result<usize>;
    /// Insert data in the tables
    fn insert_data(&mut self, table: &Table) -> Result<()>;
    /// Execute a query
    fn query(&mut self, query: &str) -> Result<Vec<value::List>>;
    /// Test the equivalence of queries regarding the output with this DB
    fn eq(&mut self, left: &str, right: &str) -> bool {
        if let (Ok(left), Ok(right)) = (self.query(left), self.query(right)) {
            left == right
        } else {
            false
        }
    }
    /// A basic test DB
    fn test_tables() -> Vec<Table> {
        vec![
            TableBuilder::new()
                .name("table_1")
                .size(10)
                .schema(
                    Schema::empty()
                        .with(("a", DataType::float_interval(0., 10.)))
                        .with(("b", DataType::optional(DataType::float_interval(-1., 1.))))
                        .with((
                            "c",
                            DataType::date_interval(
                                chrono::NaiveDate::from_ymd_opt(1980, 12, 06).unwrap(),
                                chrono::NaiveDate::from_ymd_opt(2023, 12, 06).unwrap(),
                            ),
                        ))
                        .with(("d", DataType::integer_interval(0, 10))),
                )
                .build(),
            TableBuilder::new()
                .name("table_2")
                .size(200)
                .schema(
                    Schema::empty()
                        .with(("x", DataType::integer_interval(0, 100)))
                        .with(("y", DataType::optional(DataType::text())))
                        .with(("z", DataType::text_values(["Foo".into(), "Bar".into()]))),
                )
                .build(),
            TableBuilder::new()
                .name("user_table")
                .size(100)
                .schema(
                    Schema::empty()
                        .with(("id", DataType::integer_interval(0, 100)))
                        .with(("name", DataType::text()))
                        .with((
                            "age",
                            DataType::optional(DataType::float_interval(0., 200.)),
                        ))
                        .with((
                            "city",
                            DataType::text_values(["Paris".into(), "New-York".into()]),
                        )),
                )
                .build(),
            TableBuilder::new()
                .name("order_table")
                .size(200)
                .schema(
                    Schema::empty()
                        .with(("id", DataType::integer_interval(0, 100)))
                        .with(("user_id", DataType::integer_interval(0, 101)))
                        .with(("description", DataType::text()))
                        .with((
                            "date",
                            DataType::date_interval(
                                chrono::NaiveDate::from_ymd_opt(2020, 12, 06).unwrap(),
                                chrono::NaiveDate::from_ymd_opt(2023, 12, 06).unwrap(),
                            ),
                        )),
                )
                .build(),
            TableBuilder::new()
                .name("item_table")
                .size(300)
                .schema(
                    Schema::empty()
                        .with(("order_id", DataType::integer_interval(0, 100)))
                        .with(("item", DataType::text()))
                        .with(("price", DataType::float_interval(0., 50.))),
                )
                .build(),
        ]
    }

    /// Add a vec of tables
    fn with_tables(self, tables: Vec<Table>) -> Result<Self> {
        tables.into_iter().fold(Ok(self), |db, t| db?.with(t))
    }

    /// A basic test DB
    fn with_test_tables(self) -> Result<Self> {
        self.with_tables(Self::test_tables())
    }
}

impl<D: Database> With<Table, Result<Self>> for D {
    fn with(mut self, input: Table) -> Result<Self> {
        self.create_table(&input)?;
        self.insert_data(&input)?;
        self.tables_mut().push(input);
        Ok(self)
    }
}
