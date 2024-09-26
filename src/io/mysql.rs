//! An object creating a docker container and releasing it after use
//!

use super::{Database as DatabaseTrait, Error, Result, DATA_GENERATION_SEED};
use crate::{
    data_type::{
        self,
        generator::Generator,
        value::{self, Value},
        DataTyped,
    },
    dialect_translation::mysql::MySqlTranslator,
    namer,
    relation::{Table, Variant as _},
};
use std::{
    env, fmt, ops::Deref as _, process::Command, str::FromStr, string::FromUtf8Error, sync::Mutex,
    thread, time,
};

use chrono::{Datelike, Timelike as _};
use colored::Colorize;
use mysql::{prelude::*, OptsBuilder, Value as MySqlValue};
use r2d2::Pool;
use r2d2_mysql::MySqlConnectionManager;
use rand::{rngs::StdRng, SeedableRng};

const DB: &str = "qrlew_mysql_test";
const PORT: usize = 3306;
const USER: &str = "root";
const PASSWORD: &str = "qrlew_test";

/// Converts mysql errors to io errors
impl From<mysql::Error> for Error {
    fn from(err: mysql::Error) -> Self {
        Error::Other(err.to_string())
    }
}

impl From<FromUtf8Error> for Error {
    fn from(value: FromUtf8Error) -> Self {
        Error::Other(value.to_string())
    }
}

pub struct Database {
    name: String,
    tables: Vec<Table>,
    pool: Pool<MySqlConnectionManager>,
    drop: bool,
}

/// Only one pool
pub static MYSQL_POOL: Mutex<Option<Pool<MySqlConnectionManager>>> = Mutex::new(None);
/// Only one thread starts a container
pub static MYSQL_CONTAINER: Mutex<bool> = Mutex::new(false);

impl Database {
    fn port() -> usize {
        match env::var("MYSQL_PORT") {
            Ok(port) => usize::from_str(&port).unwrap_or(PORT),
            Err(_) => PORT,
        }
    }

    fn user() -> String {
        env::var("MYSQL_USER").unwrap_or_else(|_| USER.into())
    }

    fn password() -> String {
        env::var("MYSQL_PASSWORD").unwrap_or_else(|_| PASSWORD.into())
    }

    /// Try to build a pool from an existing DB
    /// A MySQL instance must exist
    /// `docker run --name qrlew-test -p 3306:3306 -e MYSQL_ROOT_PASSWORD=qrlew_test -d mysql`
    fn build_pool_from_existing() -> Result<Pool<MySqlConnectionManager>> {
        let opts = OptsBuilder::new()
            .ip_or_hostname(Some("localhost"))
            .tcp_port(Database::port() as u16)
            .user(Some(&Database::user()))
            .pass(Some(&Database::password()))
            .db_name(Some(DB));
        let manager = MySqlConnectionManager::new(OptsBuilder::from(opts));
        Ok(r2d2::Pool::builder().max_size(10).build(manager)?)
    }

    /// Try to build a pool from a DB in a container
    fn build_pool_from_container(name: String) -> Result<Pool<MySqlConnectionManager>> {
        let mut mysql_container = MYSQL_CONTAINER.lock().unwrap();
        if !*mysql_container {
            // A new container will be started
            *mysql_container = true;
            // Other threads will wait for this to be ready
            let name = namer::new_name(name);
            let port = PORT + namer::new_id("mysql-port");

            // Test the connection and launch a test instance if necessary
            if !Command::new("docker")
                .arg("start")
                .arg(&name)
                .status()?
                .success()
            {
                log::debug!("Starting the DB");
                // If the container does not exist, start a new container
                // Run: `docker run --name test-db -e MYSQL_ROOT_PASSWORD=test -d mysql`
                let output = Command::new("docker")
                    .arg("run")
                    .arg("--name")
                    .arg(&name)
                    .arg("-d")
                    .arg("--rm")
                    .arg("-e")
                    .arg(format!("MYSQL_ROOT_PASSWORD={PASSWORD}"))
                    .arg("-p")
                    .arg(format!("{}:3306", port))
                    .arg("mysql")
                    .output()?;
                log::info!("{:?}", output);
                log::info!("Waiting for the DB to start");
                // Wait for the DB to be ready
                loop {
                    let output = Command::new("docker")
                        .arg("exec")
                        .arg(&name)
                        .arg("mysqladmin")
                        .arg("--user=root")
                        .arg(format!("--password={}", PASSWORD))
                        .arg("ping")
                        .output()?;
                    if output.status.success()
                        && String::from_utf8_lossy(&output.stdout).contains("mysqld is alive")
                    {
                        break;
                    }
                    thread::sleep(time::Duration::from_millis(200));
                    log::info!("Waiting...");
                }
                log::info!("{}", "DB ready".red());
            }
            let opts = OptsBuilder::new()
                .ip_or_hostname(Some("localhost"))
                .tcp_port(port as u16)
                .user(Some(&Database::user()))
                .pass(Some(&Database::password()))
                .db_name(Some(DB));
            let manager = MySqlConnectionManager::new(OptsBuilder::from(opts));
            let pool = r2d2::Pool::builder().max_size(10).build(manager)?;

            // Ensure database exists
            let mut conn = pool.get()?;
            conn.query_drop(format!("CREATE DATABASE IF NOT EXISTS `{}`;", DB))?;
            Ok(pool)
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
        let mut mysql_pool = MYSQL_POOL.lock().unwrap();
        if mysql_pool.is_none() {
            *mysql_pool = Some(
                Database::build_pool_from_existing()
                    .or_else(|_| Database::build_pool_from_container(name.clone()))?,
            );
        }
        let pool = mysql_pool.as_ref().unwrap().clone();
        let mut conn = pool.get()?;
        conn.query_drop(format!("CREATE DATABASE IF NOT EXISTS `{}`", DB))?;
        conn.query_drop(format!("USE `{}`", DB))?;
        let table_names: Vec<String> = conn.query("SHOW TABLES")?;
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
        let mut conn = self.pool.get()?;
        let query = table.create(MySqlTranslator).to_string();
        conn.query_drop(&query)?;
        Ok(0)
    }

    fn insert_data(&mut self, table: &Table) -> Result<()> {
        let mut rng = StdRng::seed_from_u64(DATA_GENERATION_SEED);
        let size = Database::MAX_SIZE.min(table.size().generate(&mut rng) as usize);
        let mut conn = self.pool.get()?;
        let query = table.insert("?", MySqlTranslator).to_string();
        for _ in 0..size {
            let structured: value::Struct =
                table.schema().data_type().generate(&mut rng).try_into()?;
            let values: Result<Vec<MySqlValue>> = structured
                .into_iter()
                .map(|(_, v)| MySqlValue::try_from((**v).clone()))
                .collect();
            let values = values?;
            conn.exec_drop(&query, values)?;
        }
        Ok(())
    }

    fn query(&mut self, query: &str) -> Result<Vec<value::List>> {
        let mut conn = self.pool.get()?;
        let result: Vec<mysql::Row> = conn.query(query)?;
        let rows: Result<Vec<value::List>> = result
            .into_iter()
            .map(|row| {
                let values: Result<Vec<Value>> = row
                    .unwrap()
                    .into_iter()
                    .map(|v| Value::try_from(v))
                    .collect();
                Ok(value::List::from_iter(values?))
            })
            .collect();
        Ok(rows?)
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

impl TryFrom<Value> for MySqlValue {
    type Error = Error;

    fn try_from(value: Value) -> Result<Self> {
        match value {
            Value::Boolean(b) => Ok(MySqlValue::from(b.deref())),
            Value::Integer(i) => Ok(MySqlValue::from(i.deref())),
            Value::Float(f) => Ok(MySqlValue::from(f.deref())),
            Value::Text(t) => Ok(MySqlValue::from(t.deref())),
            Value::Optional(o) => o
                .as_ref()
                .map(|v| MySqlValue::try_from((**v).clone()))
                .transpose()
                .map(|o| o.unwrap_or(MySqlValue::NULL)),
            Value::Date(d) => Ok(MySqlValue::Date(
                d.year() as u16,
                d.month() as u8,
                d.day() as u8,
                0,
                0,
                0,
                0,
            )),
            Value::Time(t) => Ok(MySqlValue::Time(
                false,
                0,
                t.hour() as u8,
                t.minute() as u8,
                t.second() as u8,
                0,
            )),
            Value::DateTime(dt) => {
                let dt_naive = dt.deref();
                let date = dt_naive.date();
                let time = dt_naive.time();
                Ok(MySqlValue::Date(
                    date.year() as u16,
                    date.month() as u8,
                    date.day() as u8,
                    time.hour() as u8,
                    time.minute() as u8,
                    time.second() as u8,
                    0,
                ))
            }
            Value::Id(i) => Ok(MySqlValue::from(i.deref())),
            _ => Err(Error::other(value)),
        }
    }
}

impl TryFrom<MySqlValue> for Value {
    type Error = Error;

    fn try_from(value: MySqlValue) -> Result<Self> {
        match value {
            MySqlValue::NULL => Ok(Value::Optional(data_type::value::Optional::new(None))),
            MySqlValue::Int(i) => Ok(Value::Integer(i.into())),
            MySqlValue::UInt(u) => Ok(Value::Integer((u as i64).into())),
            MySqlValue::Float(f) => Ok(Value::Float((f as f64).into())),
            MySqlValue::Double(d) => Ok(Value::Float(d.into())),
            MySqlValue::Bytes(bytes) => {
                let s = String::from_utf8(bytes)?;
                Ok(Value::Text(s.into()))
            }
            MySqlValue::Date(year, month, day, hour, min, sec, _) => {
                let dt = chrono::NaiveDate::from_ymd_opt(year as i32, month as u32, day as u32)
                    .ok_or_else(|| Error::other("Invalid date"))?;
                if hour == 0 && min == 0 && sec == 0 {
                    Ok(Value::Date(dt.into()))
                } else {
                    let time = chrono::NaiveTime::from_hms_opt(hour as u32, min as u32, sec as u32)
                        .ok_or_else(|| Error::other("Invalid time"))?;
                    Ok(Value::DateTime(chrono::NaiveDateTime::new(dt, time).into()))
                }
            }
            MySqlValue::Time(neg, days, hours, mins, secs, _) => {
                let total_secs = (((((days * 24) + u32::from(hours)) * 60 + u32::from(mins)) * 60
                    + u32::from(secs)) as i64)
                    * if neg { -1 } else { 1 };
                let time = chrono::NaiveTime::from_num_seconds_from_midnight_opt(
                    total_secs.abs() as u32,
                    0,
                )
                .ok_or_else(|| Error::other("Invalid time"))?;
                Ok(Value::Time(time.into()))
            }
        }
    }
}

pub fn test_database() -> Database {
    Database::new(DB.into(), Database::test_tables()).expect("Database")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn database_display() -> Result<()> {
        let mut database = test_database();
        for query in [
            "SELECT count(a), 1+sum(a), d FROM table_1 GROUP BY d",
            "SELECT AVG(x) as a FROM table_2",
            "SELECT 1+count(y) as a, sum(1+x) as b FROM table_2",
            "SELECT * FROM (SELECT * FROM table_1) as cte",
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
            "SELECT * FROM (SELECT * FROM table_1) as cte"
        ));
        Ok(())
    }
}
