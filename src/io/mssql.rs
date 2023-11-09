// mssql.rs

use super::{Database as DatabaseTrait, Error, Result, DATA_GENERATION_SEED};
use crate::{
    data_type::{
        generator::Generator,
        value::{self, Value, Variant},
        DataTyped, List,
    },
    namer,
    relation::{Table, Variant as _},
};
use sqlx::MssqlConnection;
use sqlx::{
    self,
    any::{Any, AnyTypeInfo},
    mssql::{self, Mssql, MssqlRow, MssqlTypeInfo, MssqlValueRef},
    Decode, Encode, FromRow, Row, Type, TypeInfo, ValueRef as _,
};
use std::{
    env, fmt, ops::Deref, process::Command, str::FromStr, sync::Arc, sync::Mutex, thread, time,
};

use sqlx::{mssql::MssqlConnectOptions, Connection};

const DB: &str = "qrlew-mssql-test";
const PORT: u16 = 1433;
const USER: &str = "SA";
const PASSWORD: &str = "MyPass@word";

impl From<sqlx::Error> for Error {
    fn from(err: sqlx::Error) -> Self {
        Error::Other(err.to_string())
    }
}

pub struct Database {
    name: String,
    tables: Vec<Table>,
    connection: MssqlConnection,
    drop: bool,
}

/// Only one thread start a container
pub static MSSQL_CONTAINER: Mutex<bool> = Mutex::new(false);

impl Database {
    fn db() -> String {
        env::var("MSSQL_DB").unwrap_or(DB.into())
    }

    fn port() -> u16 {
        match env::var("MSSQL_PORT") {
            Ok(port) => u16::from_str(&port).unwrap_or(PORT),
            Err(_) => PORT,
        }
    }

    fn user() -> String {
        env::var("MSSQL_USER").unwrap_or(USER.into())
    }

    fn password() -> String {
        env::var("MSSQL_PASSWORD").unwrap_or(PASSWORD.into())
    }

    /// A postgresql instance must exist
    /// `docker run --name qrlew-test -p 5432:5432 -e POSTGRES_PASSWORD=qrlew-test -d postgres`
    fn try_get_existing(name: String, tables: Vec<Table>) -> Result<Self> {
        log::info!("Try to get an existing DB");
        todo!()
    }

    // /// Get a Database from a container
    fn try_get_container(name: String, tables: Vec<Table>) -> Result<Self> {
        todo!()
    }
    //     let mut mssql_container = MSSQL_CONTAINER.lock().unwrap();
    //     if !*mssql_container {
    //         // A new container will be started
    //         *mssql_container = true;

    //         // Other threads will wait for this to be ready
    //         let name = namer::new_name(name);
    //         let port = PORT + namer::new_id("mssql-port") as u16;

    //         // Test the connection and launch a test instance if necessary
    //         if !Command::new("docker")
    //             .arg("start")
    //             .arg(&name)
    //             .status()?
    //             .success()
    //         {
    //             log::debug!("Starting the DB");
    //             // If the container does not exist, start a new container
    //             // Run: `docker run -e "ACCEPT_EULA=1" -e "MSSQL_SA_PASSWORD=MyPass@word" -e "MSSQL_USER=SA" -p 1433:1433 -d --name=qrlew-mssql-test mcr.microsoft.com/azure-sql-edge`
    //             let output = Command::new("docker")
    //                 .arg("run")
    //                 .arg("--name")
    //                 .arg(&name)
    //                 .arg("-d")
    //                 .arg("--rm")
    //                 .arg("-e")
    //                 .arg("ACCEPT_EULA=1")
    //                 .arg("-e")
    //                 .arg(format!("MSSQL_SA_PASSWORD={PASSWORD}"))
    //                 .arg("-e")
    //                 .arg("MSSQL_USER=SA")
    //                 .arg("-p")
    //                 .arg(format!("{}:1433", port))
    //                 .arg("mcr.microsoft.com/azure-sql-edge")
    //                 .output()?;
    //             log::info!("{:?}", output);
    //             log::info!("Waiting for the DB to start");
    //             while !Command::new("docker")
    //                 .arg("exec")
    //                 .arg(&name)
    //                 .arg("sqlcmd")
    //                 .arg("-S")
    //                 .arg(format!("localhost,{port}"))
    //                 .arg("-U")
    //                 .arg("SA")
    //                 .arg("-P")
    //                 .arg("{PASSWORD}")
    //                 .arg("-Q")
    //                 .arg("SELECT 1")
    //                 .status()?
    //                 .success()
    //             {
    //                 thread::sleep(time::Duration::from_millis(200));
    //                 log::info!("Waiting...");
    //             }
    //             log::info!("{}", "DB ready".red());
    //         }

    //         let env = Environment::new()?;

    //         let mut conn = env.connect(
    //             "YourDatabase", "SA", "My@Test@Password1",
    //             ConnectionOptions::default()
    //         )?;
    //         Ok(Database {
    //             name,
    //             tables: vec![],
    //             conn,
    //             drop: false,
    //         }
    //         .with_tables(tables)?)
    //     } else {
    //         Database::try_get_existing(name, tables)
    //     }
    // }
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
        Database::try_get_existing(name.clone(), tables.clone())
            .or_else(|_| Database::try_get_container(name, tables))
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
        todo!()
    }

    fn insert_data(&mut self, table: &Table) -> Result<()> {
        todo!()
    }

    fn query(&mut self, query: &str) -> Result<Vec<value::List>> {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async_query(query, &mut self.connection))
    }
    // Implement necessary methods as per your trait definition...
}

async fn async_query(query: &str, connection: &mut MssqlConnection) -> Result<Vec<value::List>> {
    let rows = sqlx::query(query).fetch_all(connection).await?;

    Ok(rows
        .iter()
        .map(|row: &MssqlRow| {
            let values: Vec<SqlValue> = (0..row.len())
                .map(|i| {
                    let val: SqlValue = row.get(i);
                    val
                })
                .collect();
            value::List::from_iter(values.into_iter().map(|v| v.try_into().expect("Convert")))
        })
        .collect())
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

impl Decode<'_, mssql::Mssql> for SqlValue {
    fn decode(value: MssqlValueRef<'_>) -> std::result::Result<Self, sqlx::error::BoxDynError> {
        let binding = value.type_info();
        let type_info = binding.as_ref();
        print!("\nDECODE\n");
        match type_info.name() {
            "BIT" => Ok(Value::from(<bool as Decode<Mssql>>::decode(value)?).try_into()?),
            "INT" => Ok(Value::from((<i32 as Decode<Mssql>>::decode(value)?) as i64).try_into()?),
            "BIGINT" => Ok(Value::from(<i64 as Decode<Mssql>>::decode(value)?).try_into()?),
            "BINARY" => todo!(),
            "CHAR" => Ok(Value::from(<String as Decode<Mssql>>::decode(value)?).try_into()?),
            "DATE" => todo!(),
            "DATETIME" => todo!(),
            "DATETIME2" => todo!(),
            "DATETIMEOFFSET" => todo!(),
            "DECIMAL" => Ok(Value::from(<f64 as Decode<Mssql>>::decode(value)?).try_into()?),
            "FLOAT" => Ok(Value::from(<f64 as Decode<Mssql>>::decode(value)?).try_into()?),
            "IMAGE" => todo!(),
            "MONEY" => todo!(),
            "NCHAR" => Ok(Value::from(<String as Decode<Mssql>>::decode(value)?).try_into()?),
            "NTEXT" => Ok(Value::from(<String as Decode<Mssql>>::decode(value)?).try_into()?),
            "NUMERIC" => Ok(Value::from(<f64 as Decode<Mssql>>::decode(value)?).try_into()?),
            "NVARCHAR" => Ok(Value::from(<String as Decode<Mssql>>::decode(value)?).try_into()?),
            "REAL" => todo!(),
            "SMALLDATETIME" => todo!(),
            "SMALLINT" => todo!(),
            "SMALLMONEY" => todo!(),
            "SQL_VARIANT" => todo!(),
            "TEXT" => Ok(Value::from(<String as Decode<Mssql>>::decode(value)?).try_into()?),
            "TIME" => todo!(),
            "TIMESTAMP" => todo!(),
            "TINYINT" => todo!(),
            "UNIQUEIDENTIFIER" => todo!(),
            "VARBINARY" => todo!(),
            "VARCHAR" => Ok(Value::from(<String as Decode<Mssql>>::decode(value)?).try_into()?),
            "XML" => todo!(),
            _ => Err(Box::new(sqlx::Error::Decode(
                format!("Unhandled type: {}", type_info.name()).into(),
            ))),
        }
    }
}

impl Encode<'_, mssql::Mssql> for SqlValue {
    fn encode_by_ref(
        &self,
        buf: &mut <mssql::Mssql as sqlx::database::HasArguments<'_>>::ArgumentBuffer,
    ) -> sqlx::encode::IsNull {
        print!("\nencode\n");
        match self {
            SqlValue::Boolean(b) => {
                buf.push(if *b.deref() { 1 } else { 0 });
                sqlx::encode::IsNull::No
            }
            SqlValue::Integer(i) => {
                buf.extend(i.deref().to_le_bytes());
                sqlx::encode::IsNull::No
            }
            SqlValue::Float(f) => {
                buf.extend(f.deref().to_le_bytes());
                sqlx::encode::IsNull::No
            }
            SqlValue::Text(t) => <&str as Encode<Mssql>>::encode_by_ref(&&t.deref()[..], buf),
            SqlValue::Optional(o) => {
                let value = o.clone().map(|v| v.as_ref().clone());
                <&Option<SqlValue> as Encode<Mssql>>::encode_by_ref(&&value, buf)
            }
            SqlValue::Date(_) => todo!(),
            SqlValue::Time(_) => todo!(),
            SqlValue::DateTime(_) => todo!(),
            SqlValue::Id(_) => todo!(),
        }
    }

    fn produces(&self) -> Option<<mssql::Mssql as sqlx::Database>::TypeInfo> {
        print!("\nproduces\n");
        match self {
            SqlValue::Boolean(_) => Some(<bool as Type<Mssql>>::type_info()),
            SqlValue::Integer(_) => Some(<i64 as Type<Mssql>>::type_info()),
            SqlValue::Float(_) => Some(<f64 as Type<Mssql>>::type_info()),
            SqlValue::Text(_) => Some(<String as Type<Mssql>>::type_info()),
            SqlValue::Optional(_) => todo!(),
            SqlValue::Date(_) => todo!(),
            SqlValue::Time(_) => todo!(),
            SqlValue::DateTime(_) => todo!(),
            SqlValue::Id(_) => todo!(),
        }
    }
}

impl Type<mssql::Mssql> for SqlValue {
    fn type_info() -> <mssql::Mssql as sqlx::Database>::TypeInfo {
        println!("\ntype_info for Value\n");
        <String as Type<Mssql>>::type_info()
    }

    fn compatible(ty: &<mssql::Mssql as sqlx::Database>::TypeInfo) -> bool {
        println!("\ncompatible for Value\n {:?}", ty.name());
        true
    }
}

use tiberius::{self, AuthMethod, Client, Config};
use tokio::net::TcpStream;
use tokio_util::compat::TokioAsyncWriteCompatExt;
// use async_std::net::TcpStream;

impl From<tiberius::error::Error> for Error {
    fn from(err: tiberius::error::Error) -> Self {
        Error::Other(err.to_string())
    }
}

pub fn test_database() -> Database {
    Database::new(DB.into(), Database::test_tables()).expect("Database")
}

#[cfg(test)]
mod tests {
    use super::*;

    // This attribute is necessary to use async code in tests
    #[tokio::test]
    async fn test_mssql_connection() -> Result<()> {
        let connection_string = "mssql://SA:MyPass@word@localhost:1433/master?encrypt=false";
        //let connection_options = MssqlConnectOptions::new(po);
        let connection_options = MssqlConnectOptions::from_str(connection_string)?;

        // Establish a connection to the database.
        let mut connection = MssqlConnection::connect_with(&connection_options).await?;

        let sql_query =
            "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'dbo'";
        // Execute a query.
        let rows = sqlx::query(sql_query).fetch_all(&mut connection).await?;

        let rows_as_vec: Vec<value::List> = rows
            .iter()
            .map(|row: &MssqlRow| {
                let values: Vec<SqlValue> = (0..row.len())
                    .map(|i| {
                        let val: SqlValue = row.get(i);
                        val
                    })
                    .collect();
                value::List::from_iter(values.into_iter().map(|v| v.try_into().expect("Convert")))
            })
            .collect();

        //let b2: Value = rows[0].try_get(0)?;
        //let b2: Value = rows[0].get(0);
        // let val = Value::from(b);

        println!("{:#?}", rows_as_vec);
        // let coverted_rows: Result<Vec<value::List>> = rows
        // .into_iter()
        // .map(|r| {
        //     let values: Vec<SqlValue> = (0..r.len()).into_iter().map(|i| r.get(i)).collect();
        //     value::List::from_iter(values.into_iter().map(|v| v.try_into().expect("Convert")))
        // })
        // .collect();
        // let cols = row.columns();
        // let tab_name: String = row.get(0);
        // println!("{:?}", tab_name);
        // Here you can assert the expected results
        // For example, check that the row is not null or has some expected value
        //assert!(!row.is_empty(), "The query result should not be empty");

        Ok(())
    }

    // // This attribute is necessary to use async code in tests
    // #[tokio::test]
    // async fn test_mssql_with_tiberius() -> Result<()> {

    //     let mut config = Config::new();

    //     config.host("localhost");
    //     config.port(1433);
    //     config.authentication(AuthMethod::sql_server("SA", "MyPass@word"));
    //     config.trust_cert();
    //     config.encryption(tiberius::EncryptionLevel::Off);

    //     let tcp = TcpStream::connect(config.get_addr()).await?;
    //     tcp.set_nodelay(true)?;

    //     let mut client = Client::connect(config, tcp.compat_write()).await?;
    //     Ok(())
    // }

    // #[test]
    // fn database_display() -> Result<()> {
    //     let mut database = test_database();
    //     for query in [
    //         "SELECT count(a), 1+sum(a), d FROM table_1 group by d",
    //         "SELECT AVG(x) as a FROM table_2",
    //         "SELECT 1+count(y) as a, sum(1+x) as b FROM table_2",
    //         "WITH cte AS (SELECT * FROM table_1) SELECT * FROM cte",
    //         "SELECT * FROM table_2",
    //     ] {
    //         println!("\n{query}");
    //         for row in database.query(query)? {
    //             println!("{}", row);
    //         }
    //     }
    //     Ok(())
    // }

    // #[test]
    // fn database_test() -> Result<()> {
    //     let mut database = test_database();
    //     assert!(!database.eq("SELECT * FROM table_1", "SELECT * FROM table_2"));
    //     assert!(database.eq(
    //         "SELECT * FROM table_1",
    //         "WITH cte AS (SELECT * FROM table_1) SELECT * FROM cte"
    //     ));
    //     Ok(())
    // }
}
