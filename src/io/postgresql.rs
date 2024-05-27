//! An object creating a docker container and releasing it after use
//!

use super::{Database as DatabaseTrait, Error, Result, DATA_GENERATION_SEED};
use crate::{
    data_type::{
        generator::Generator,
        value::{self, Value},
        DataTyped,
    },
    dialect_translation::postgresql::PostgreSqlTranslator,
    namer,
    relation::{Table, Variant as _},
};
use std::{env, fmt, process::Command, str::FromStr, sync::Arc, sync::Mutex, thread, time};

use colored::Colorize;
use postgres::{
    self,
    types::{FromSql, ToSql, Type},
};
use r2d2::Pool;
use r2d2_postgres::{postgres::NoTls, PostgresConnectionManager};
use rand::{rngs::StdRng, SeedableRng};
use rust_decimal::{prelude::ToPrimitive, Decimal};

const DB: &str = "qrlew-test";
const PORT: usize = 5432;
const USER: &str = "postgres";
const PASSWORD: &str = "qrlew-test";

/// Converts sqlite errors to io errors
impl From<postgres::Error> for Error {
    fn from(err: postgres::Error) -> Self {
        Error::Other(err.to_string())
    }
}

pub struct Database {
    name: String,
    tables: Vec<Table>,
    pool: Pool<PostgresConnectionManager<NoTls>>,
    drop: bool,
}

/// Only one pool
pub static POSTGRES_POOL: Mutex<Option<Pool<PostgresConnectionManager<NoTls>>>> = Mutex::new(None);
/// Only one thread start a container
pub static POSTGRES_CONTAINER: Mutex<bool> = Mutex::new(false);

impl Database {
    // fn db() -> String {
    //     env::var("POSTGRES_DB").unwrap_or(DB.into())
    // }

    fn port() -> usize {
        match env::var("POSTGRES_PORT") {
            Ok(port) => usize::from_str(&port).unwrap_or(PORT),
            Err(_) => PORT,
        }
    }

    fn user() -> String {
        env::var("POSTGRES_USER").unwrap_or(USER.into())
    }

    fn password() -> String {
        env::var("POSTGRES_PASSWORD").unwrap_or(PASSWORD.into())
    }

    /// Try to build a pool from an existing DB
    /// A postgresql instance must exist
    /// `docker run --name qrlew-test -p 5432:5432 -e POSTGRES_PASSWORD=qrlew-test -d postgres`
    fn build_pool_from_existing() -> Result<Pool<PostgresConnectionManager<NoTls>>> {
        let manager = PostgresConnectionManager::new(
            format!(
                "host=localhost port={} user={} password={}",
                Database::port(),
                Database::user(),
                Database::password()
            )
            .parse()?,
            NoTls,
        );
        Ok(r2d2::Pool::builder().max_size(10).build(manager)?)
    }

    /// Try to build a pool from a DB in a container
    fn build_pool_from_container(name: String) -> Result<Pool<PostgresConnectionManager<NoTls>>> {
        let mut postgres_container = POSTGRES_CONTAINER.lock().unwrap();
        if *postgres_container == false {
            // A new container will be started
            *postgres_container = true;
            // Other threads will wait for this to be ready
            let name = namer::new_name(name);
            let port = PORT + namer::new_id("pg-port");
            // Test the connexion and launch a test instance if necessary
            if !Command::new("docker")
                .arg("start")
                .arg(&name)
                .status()?
                .success()
            {
                log::debug!("Starting the DB");
                // If the container does not exist
                // Start a new container
                // Run: `docker run --name test-db -e POSTGRES_PASSWORD=test -d postgres`
                let output = Command::new("docker")
                    .arg("run")
                    .arg("--name")
                    .arg(&name)
                    .arg("-d")
                    .arg("--rm")
                    .arg("-e")
                    .arg(format!("POSTGRES_PASSWORD={PASSWORD}"))
                    .arg("-p")
                    .arg(format!("{}:5432", port))
                    .arg("postgres")
                    .output()?;
                log::info!("{:?}", output);
                log::info!("Waiting for the DB to start");
                while !Command::new("docker")
                    .arg("exec")
                    .arg(&name)
                    .arg("pg_isready")
                    .status()?
                    .success()
                {
                    thread::sleep(time::Duration::from_millis(200));
                    log::info!("Waiting...");
                }
                log::info!("{}", "DB ready".red());
            }
            let manager = PostgresConnectionManager::new(
                format!("host=localhost port={port} user={USER} password={PASSWORD}").parse()?,
                NoTls,
            );
            Ok(r2d2::Pool::builder().max_size(10).build(manager)?)
        } else {
            Database::build_pool_from_existing()
        }
    }
}

impl fmt::Debug for Database {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Database")
            .field("name", &self.name)
            .field("tables", &self.tables)
            .finish()
    }
}

impl DatabaseTrait for Database {
    fn new(name: String, tables: Vec<Table>) -> Result<Self> {
        let mut postgres_pool = POSTGRES_POOL.lock().unwrap();
        if let None = *postgres_pool {
            *postgres_pool = Some(
                Database::build_pool_from_existing()
                    .or_else(|_| Database::build_pool_from_container(name.clone()))?,
            );
        }
        let pool = postgres_pool.as_ref().unwrap().clone();
        let table_names: Vec<String> = pool
            .get()?
            .query(
                "SELECT * FROM pg_catalog.pg_tables WHERE schemaname='public'",
                &[],
            )?
            .into_iter()
            .map(|row| row.get("tablename"))
            .collect();
        if table_names.is_empty() {
            Database {
                name,
                tables: vec![],
                pool,
                drop: false,
            }
            .with_tables(tables)
        } else {
            Ok(Database {
                name,
                tables,
                pool,
                drop: false,
            })
        }
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
        let mut connection = self.pool.get()?;
        let _qq = table.create(PostgreSqlTranslator).to_string();
        Ok(connection.execute(&table.create(PostgreSqlTranslator).to_string(), &[])? as usize)
    }

    fn insert_data(&mut self, table: &Table) -> Result<()> {
        let mut rng = StdRng::seed_from_u64(DATA_GENERATION_SEED);
        let size = Database::MAX_SIZE.min(table.size().generate(&mut rng) as usize);
        let mut connection = self.pool.get()?;
        let statement = connection.prepare(&table.insert("$", PostgreSqlTranslator).to_string())?;
        for _ in 0..size {
            let structured: value::Struct =
                table.schema().data_type().generate(&mut rng).try_into()?;
            let values: Result<Vec<SqlValue>> = structured
                .into_iter()
                .map(|(_, v)| (**v).clone().try_into())
                .collect();
            let values = values?;
            let params: Vec<&(dyn ToSql + Sync)> =
                values.iter().map(|v| v as &(dyn ToSql + Sync)).collect();
            connection.execute(&statement, &params)?;
        }
        Ok(())
    }

    fn query(&mut self, query: &str) -> Result<Vec<value::List>> {
        let rows: Vec<_>;
        {
            let mut connection = self.pool.get()?;
            let statement = connection.prepare(query)?;
            rows = connection.query(&statement, &[])?;
        }
        Ok(rows
            .into_iter()
            .map(|r| {
                let values: Vec<SqlValue> = (0..r.len()).into_iter().map(|i| r.get(i)).collect();
                value::List::from_iter(values.into_iter().map(|v| v.try_into().expect("Convert")))
            })
            .collect())
    }
}

impl Drop for Database {
    fn drop(&mut self) {
        if self.drop {
            Command::new("docker")
                .arg("rm")
                .arg("--force")
                .arg(self.name())
                .status()
                .expect("Deleted container");
        }
    }
}

#[derive(Debug, Clone)]
enum SqlValue {
    Boolean(value::Boolean),
    Integer(value::Integer),
    Float(value::Float),
    Text(value::Text),
    Optional(Option<Box<SqlValue>>),
    Date(value::Date),
    Time(value::Time),
    DateTime(value::DateTime),
    Id(value::Id),
}

impl TryFrom<Value> for SqlValue {
    type Error = Error;

    fn try_from(value: Value) -> Result<Self> {
        match value {
            Value::Boolean(b) => Ok(SqlValue::Boolean(b)),
            Value::Integer(i) => Ok(SqlValue::Integer(i)),
            Value::Float(f) => Ok(SqlValue::Float(f)),
            Value::Text(t) => Ok(SqlValue::Text(t)),
            Value::Optional(o) => o
                .as_deref()
                .map(|v| SqlValue::try_from(v.clone()))
                .map_or(Ok(None), |r| r.map(|v| Some(Box::new(v))))
                .map(|o| SqlValue::Optional(o)),
            Value::Date(d) => Ok(SqlValue::Date(d)),
            Value::Time(t) => Ok(SqlValue::Time(t)),
            Value::DateTime(d) => Ok(SqlValue::DateTime(d)),
            Value::Id(i) => Ok(SqlValue::Id(i)),
            _ => Err(Error::other(value)),
        }
    }
}

impl TryFrom<SqlValue> for Value {
    type Error = Error;

    fn try_from(value: SqlValue) -> Result<Self> {
        match value {
            SqlValue::Boolean(b) => Ok(Value::Boolean(b)),
            SqlValue::Integer(i) => Ok(Value::Integer(i)),
            SqlValue::Float(f) => Ok(Value::Float(f)),
            SqlValue::Text(t) => Ok(Value::Text(t)),
            SqlValue::Optional(o) => o
                .map(|v| Value::try_from(*v))
                .map_or(Ok(None), |r| r.map(|v| Some(Arc::new(v))))
                .map(|o| Value::from(o)),
            SqlValue::Date(d) => Ok(Value::Date(d)),
            SqlValue::Time(t) => Ok(Value::Time(t)),
            SqlValue::DateTime(d) => Ok(Value::DateTime(d)),
            SqlValue::Id(i) => Ok(Value::Id(i)),
        }
    }
}

impl ToSql for SqlValue {
    fn to_sql(
        &self,
        ty: &Type,
        out: &mut postgres::types::private::BytesMut,
    ) -> std::result::Result<postgres::types::IsNull, Box<dyn std::error::Error + Sync + Send>>
    where
        Self: Sized,
    {
        match self {
            SqlValue::Boolean(b) => b.to_sql(ty, out),
            SqlValue::Integer(i) => i.to_sql(ty, out),
            SqlValue::Float(f) => f.to_sql(ty, out),
            SqlValue::Text(t) => t.to_sql(ty, out),
            SqlValue::Optional(o) => o.as_deref().to_sql(ty, out),
            SqlValue::Date(d) => d.to_sql(ty, out),
            SqlValue::Time(t) => t.to_sql(ty, out),
            SqlValue::DateTime(d) => d.to_sql(ty, out),
            SqlValue::Id(i) => i.to_sql(ty, out),
        }
    }

    postgres::types::accepts!(
        BOOL, INT2, INT4, INT8, NUMERIC, FLOAT4, FLOAT8, NUMERIC, VARCHAR, TEXT, DATE, TIME,
        TIMESTAMP
    );

    postgres::types::to_sql_checked!();
}

impl<'a> FromSql<'a> for SqlValue {
    fn from_sql(
        ty: &Type,
        raw: &'a [u8],
    ) -> std::result::Result<Self, Box<dyn std::error::Error + Sync + Send>> {
        match ty {
            &Type::BOOL => bool::from_sql(ty, raw).map(|b| SqlValue::Boolean(b.into())),
            // &Type::INT4 | &Type::INT8 => {
            //     i64::from_sql(ty, raw).map(|i| SqlValue::Integer(i.into()))
            // }
            &Type::INT4 => i32::from_sql(ty, raw).map(|i| SqlValue::Integer((i as i64).into())),
            &Type::INT8 => i64::from_sql(ty, raw).map(|i| SqlValue::Integer(i.into())),
            &Type::FLOAT4 | &Type::FLOAT8 => {
                f64::from_sql(ty, raw).map(|f| SqlValue::Float(f.into()))
            }
            &Type::NUMERIC => Decimal::from_sql(ty, raw)
                .map(|d| SqlValue::Float(d.to_f64().unwrap_or_default().into())),
            &Type::VARCHAR | &Type::TEXT => {
                String::from_sql(ty, raw).map(|s| SqlValue::Text(s.into()))
            }
            &Type::DATE => chrono::NaiveDate::from_sql(ty, raw).map(|d| SqlValue::Date(d.into())),
            &Type::TIME => chrono::NaiveTime::from_sql(ty, raw).map(|t| SqlValue::Time(t.into())),
            &Type::TIMESTAMP => {
                chrono::NaiveDateTime::from_sql(ty, raw).map(|d| SqlValue::DateTime(d.into()))
            }
            _ => todo!(),
        }
    }

    fn from_sql_null(
        _ty: &Type,
    ) -> std::result::Result<Self, Box<dyn std::error::Error + Sync + Send>> {
        Ok(SqlValue::Optional(None))
    }

    postgres::types::accepts!(
        BOOL, INT2, INT4, INT8, FLOAT4, FLOAT8, NUMERIC, VARCHAR, TEXT, DATE, TIME, TIMESTAMP
    );
}

pub fn test_database() -> Database {
    // Database::test()
    Database::new(DB.into(), Database::test_tables()).expect("Database")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn database_display() -> Result<()> {
        let mut database = test_database();
        for query in [
            "SELECT count(a), 1+sum(a), d FROM table_1 group by d",
            "SELECT AVG(x) as a FROM table_2",
            "SELECT 1+count(y) as a, sum(1+x) as b FROM table_2",
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
        println!("Pool {}", database.pool.max_size());
        assert!(!database.eq("SELECT * FROM table_1", "SELECT * FROM table_2"));
        assert!(database.eq(
            "SELECT * FROM table_1",
            "WITH cte AS (SELECT * FROM table_1) SELECT * FROM cte"
        ));
        Ok(())
    }
}
