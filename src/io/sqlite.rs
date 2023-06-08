use super::{Database as DatabaseTrait, Error, Result};
use crate::{
    data_type::{
        generator::Generator,
        value::{self, Value},
        DataTyped,
    },
    relation::{Table, Variant as _},
};
use rand::{rngs::StdRng, SeedableRng};
use rusqlite::{
    self, params_from_iter,
    types::{FromSql, FromSqlResult, Null, ToSql, ToSqlOutput},
    Connection,
};
use std::result;

const DB: &str = "qrlew-test";

/// Converts sqlite errors to io errors
impl From<rusqlite::Error> for Error {
    fn from(err: rusqlite::Error) -> Self {
        Error::Other(err.to_string())
    }
}

/// Create a database for tests
#[derive(Debug)]
pub struct Database {
    name: String,
    tables: Vec<Table>,
    connection: Connection,
}

impl DatabaseTrait for Database {
    fn new(name: String, tables: Vec<Table>) -> Result<Self> {
        let connection = Connection::open_in_memory()?;
        let mut database = Database {
            name,
            tables,
            connection,
        };
        for table in database.tables.clone() {
            database.create_table(&table)?;
            database.insert_data(&table)?;
        }
        Ok(database)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn tables(&self) -> &[Table] {
        &self.tables
    }

    fn tables_mut(&mut self) -> &mut Vec<Table> {
        &mut self.tables
    }

    fn create_table(&mut self, table: &Table) -> Result<usize> {
        Ok(self.connection.execute(&table.create().to_string(), ())?)
    }

    fn insert_data(&mut self, table: &Table) -> Result<()> {
        let seed: u64 = 1234;
        let mut rng = StdRng::seed_from_u64(seed);
        let size = Database::MAX_SIZE.min(table.size().generate(&mut rng) as usize);
        let mut statement = self.connection.prepare(&table.insert('?').to_string())?;
        for _ in 0..size {
            let structured: value::Struct =
                table.schema().data_type().generate(&mut rng).try_into()?;
            statement.execute(params_from_iter(structured.iter().map(|(_, v)| &**v)))?;
        }
        Ok(())
    }

    fn query(&mut self, query: &str) -> Result<Vec<value::List>> {
        let mut statement = self.connection.prepare(query)?;
        let result: result::Result<Vec<value::List>, rusqlite::Error> = statement
            .query_map([], |row| {
                (0..row.as_ref().column_count())
                    .map(|i| row.get(i))
                    .collect()
            })?
            .collect();
        Ok(result?)
    }
}

/// Implement the conversion of a Value to ToSqlOutput
impl ToSql for Value {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        match self {
            Value::Boolean(b) => b.to_sql(),
            Value::Integer(i) => i.to_sql(),
            Value::Float(f) => f.to_sql(),
            Value::Text(t) => t.to_sql(),
            Value::Optional(o) => o
                .as_ref()
                .map(|v| (&**v).to_sql())
                .unwrap_or_else(|| Null.to_sql()),
            Value::Date(d) => d.to_sql(),
            Value::Time(t) => t.to_sql(),
            Value::DateTime(dt) => dt.to_sql(),
            Value::Id(i) => i.to_sql(),
            _ => todo!(),
        }
    }
}

/// Read sql results as value
impl FromSql for Value {
    fn column_result(value: rusqlite::types::ValueRef<'_>) -> FromSqlResult<Self> {
        Ok(match value {
            rusqlite::types::ValueRef::Null => Value::unit(),
            rusqlite::types::ValueRef::Integer(i) => Value::integer(i),
            rusqlite::types::ValueRef::Real(f) => Value::float(f),
            rusqlite::types::ValueRef::Text(s) => Value::text(String::from_utf8_lossy(s)),
            rusqlite::types::ValueRef::Blob(b) => Value::bytes(b.clone()),
        })
    }
}

pub fn test_database() -> Database {
    Database::empty(DB.into())
        .expect("Database")
        .with_test_tables()
        .expect("Database with tables")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn database_hierarchy() -> Result<()> {
        let database = test_database();
        println!("{}", database.relations());
        Ok(())
    }

    #[test]
    fn database_display() -> Result<()> {
        let mut database = test_database();
        for query in [
            "SELECT * FROM table_1",
            "WITH cte AS (SELECT * FROM table_1) SELECT * FROM cte",
            "SELECT * FROM table_2",
        ] {
            println!("\n{query}");
            for row in database.query(query)? {
                println!("{}", row);
            }
        }
        Ok(())
    }

    #[test]
    fn database_test() -> Result<()> {
        let mut database = test_database();
        assert!(!database.eq("SELECT * FROM table_1", "SELECT * FROM table_2"));
        assert!(database.eq(
            "SELECT * FROM table_1",
            "WITH cte AS (SELECT * FROM table_1) SELECT * FROM cte"
        ));
        Ok(())
    }
}
