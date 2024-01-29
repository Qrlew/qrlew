/// Creating an mssql connector. This is experimental.
/// There are some types that are not supported
/// Date, Datetime and Interval are not yes supported
use super::{Database as DatabaseTrait, Error, Result, DATA_GENERATION_SEED};
use crate::{
    data_type::{
        generator::Generator,
        value::{self, Value, Variant},
        DataTyped, List,
    },
    namer,
    relation::{Schema, Table, TableBuilder, Variant as _},
    DataType, Ready as _,
};
use colored::Colorize;
use rand::{rngs::StdRng, SeedableRng};
use sqlx::{
    self,
    mssql::{
        self, Mssql, MssqlArguments, MssqlConnectOptions, MssqlPoolOptions, MssqlQueryResult,
        MssqlRow, MssqlValueRef,
    },
    query::Query,
    Connection, Decode, Encode, MssqlConnection, MssqlPool, Pool, Row, Type, TypeInfo,
    ValueRef as _,
};
use std::{
    env, fmt, ops::Deref, process::Command, str::FromStr, sync::Arc, sync::Mutex, thread, time,
};

use crate::dialect_translation::mssql::MsSqlTranslator;

const DB: &str = "qrlew-mssql-test";
const PORT: u16 = 1433;
const USER: &str = "SA";
const PASSWORD: &str = "Strong@Passw0rd";

impl From<sqlx::Error> for Error {
    fn from(err: sqlx::Error) -> Self {
        Error::Other(err.to_string())
    }
}

pub struct Database {
    name: String,
    tables: Vec<Table>,
    pool: MssqlPool,
    drop: bool,
}

pub static MSSQL_POOL: Mutex<Option<Pool<Mssql>>> = Mutex::new(None);
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

    /// A mssql instance must exist
    /// `docker run --name qrlew-mssql-test -p 1433:1433 -d mssql`
    fn build_pool_from_existing() -> Result<Pool<Mssql>> {
        log::info!("Try to get an existing DB");
        let rt = tokio::runtime::Runtime::new().unwrap();
        let pool = rt.block_on(async_connect(&format!(
            "mssql://{}:{}@localhost:{}/master?encrypt=false",
            Database::user(),
            Database::password(),
            Database::port(),
        )))?;
        Ok(pool)
    }

    /// Get a Database from a container
    fn build_pool_from_container(name: String) -> Result<Pool<Mssql>> {
        let mut mssql_container = MSSQL_CONTAINER.lock().unwrap();

        if *mssql_container == false {
            // A new container will be started
            *mssql_container = true;

            // Other threads will wait for this to be ready
            let name = namer::new_name(name);
            let port = PORT + namer::new_id("mssql-port") as u16;

            // Test the connection and launch a test instance if necessary
            if !Command::new("docker")
                .arg("start")
                .arg(&name)
                .status()?
                .success()
            {
                log::debug!("Starting the DB");
                // If the container does not exist, start a new container
                // Run: `docker run -e "ACCEPT_EULA=1" -e "MSSQL_SA_PASSWORD=Strong@Passw0rd" -e "MSSQL_USER=SA" -p 1433:1433 -d --name=qrlew-mssql-test mcr.microsoft.com/azure-sql-edge`
                // docker run -e 'ACCEPT_EULA=Y' -e 'SA_PASSWORD=Strong@Passw0rd' -p 1433:1433 --name mssql -d mcr.microsoft.com/mssql/server:2019-latest
                let output = Command::new("docker")
                    .arg("run")
                    .arg("--name")
                    .arg(&name)
                    .arg("-d")
                    .arg("--rm")
                    .arg("-e")
                    .arg("ACCEPT_EULA=1")
                    .arg("-e")
                    .arg(format!("MSSQL_SA_PASSWORD={PASSWORD}"))
                    .arg("-e")
                    .arg("MSSQL_USER=SA")
                    .arg("-p")
                    .arg(format!("{}:1433", port))
                    .arg("mcr.microsoft.com/azure-sql-edge")
                    .output()?;
                log::info!("{:?}", output);
                log::info!("Waiting for the DB to start");
                // execute "SELECT 1" to check that the server is up and running
                while !Command::new("docker")
                    .arg("exec")
                    .arg(&name)
                    .arg("/opt/mssql-tools/bin/sqlcmd")
                    .arg("-S")
                    .arg(format!("localhost,{port}"))
                    .arg("-U")
                    .arg("SA")
                    .arg("-P")
                    .arg(format!("{PASSWORD}"))
                    .arg("-Q")
                    .arg("SELECT 1 AS connection_successful")
                    .status()?
                    .success()
                {
                    thread::sleep(time::Duration::from_millis(2000));
                    log::info!("Waiting...");
                }
                //thread::sleep(time::Duration::from_millis(5000));
                log::info!("{}", "DB ready".red());
            }

            let rt = tokio::runtime::Runtime::new().unwrap();
            let pool = rt.block_on(async_connect(&format!(
                "mssql://{}:{}@localhost:{}/master?encrypt=false",
                Database::user(),
                Database::password(),
                Database::port(),
            )))?;
            Ok(pool)
        } else {
            Database::build_pool_from_existing()
        }
    }

    // Overriding test_tables. We don't support Date, Datetime yet so we are pushing tables without these types
    fn test_tables() -> Vec<Table> {
        vec![
            TableBuilder::new()
                .path(["table_1"])
                .name("table_1")
                .size(10)
                .schema(
                    Schema::empty()
                        .with(("a", DataType::float_interval(0., 10.)))
                        .with(("b", DataType::optional(DataType::float_interval(-1., 1.))))
                        .with(("d", DataType::integer_interval(0, 10))),
                )
                .build(),
            TableBuilder::new()
                .path(["table_2"])
                .name("table_2")
                .size(200)
                .schema(
                    Schema::empty()
                        .with(("x", DataType::integer_interval(0, 100)))
                        .with(("y", DataType::optional(DataType::text())))
                        .with(("z", DataType::text_values(["Foo".into(), "Bar".into()]))), //Can't push these? why?
                )
                .build(),
            TableBuilder::new()
                .path(["user_table"])
                .name("users")
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
                .path(["order_table"])
                .name("orders")
                .size(200)
                .schema(
                    Schema::empty()
                        .with(("id", DataType::integer_interval(0, 100)))
                        .with(("user_id", DataType::integer_interval(0, 101)))
                        .with(("description", DataType::text())),
                )
                .build(),
            TableBuilder::new()
                .path(["item_table"])
                .name("items")
                .size(300)
                .schema(
                    Schema::empty()
                        .with(("order_id", DataType::integer_interval(0, 100)))
                        .with(("item", DataType::text()))
                        .with(("price", DataType::float_interval(0., 50.))),
                )
                .build(),
            TableBuilder::new()
                .path(["large_user_table"])
                .name("more_users")
                .size(100000)
                .schema(
                    Schema::empty()
                        .with(("id", DataType::integer_interval(0, 1000)))
                        .with(("name", DataType::text()))
                        .with((
                            "age",
                            DataType::optional(DataType::float_interval(0., 200.)),
                        ))
                        .with((
                            "city",
                            DataType::text_values([
                                "Paris".into(),
                                "New-York".into(),
                                "Rome".into(),
                            ]),
                        ))
                        .with(("income", DataType::float_interval(100.0, 200000.0))),
                )
                .build(),
            // TODO: create table with names that need to be quoted
            TableBuilder::new()
                .path(["MY_SPECIAL_TABLE"])
                .name("my_table")
                .size(100)
                .schema(
                    Schema::empty()
                        .with(("Id", DataType::integer_interval(0, 1000)))
                        .with(("Na.Me", DataType::text()))
                        .with(("inc&ome", DataType::float_interval(100.0, 200000.0)))
                        .with(("normal_col", DataType::text())),
                )
                .build(),
        ]
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
        let mut mssql_pool = MSSQL_POOL.lock().unwrap();
        if let None = *mssql_pool {
            *mssql_pool = Some(
                Database::build_pool_from_existing()
                    .or_else(|_| Database::build_pool_from_container(name.clone()))?,
            );
        }
        let rt = tokio::runtime::Runtime::new().unwrap();
        let pool = mssql_pool.as_ref().unwrap().clone();
        let find_tables_query =
            "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'dbo'";
        let table_names_in_db: Vec<String> = rt
            .block_on(async_query(find_tables_query, &pool))?
            .iter()
            .map(|r| {
                let val_as_str: String = r.to_vec()[0].clone().try_into().unwrap();
                val_as_str
            })
            .collect();
        let tables_to_be_created: Vec<Table> = tables
            .iter()
            .filter(|tab| !table_names_in_db.contains(&tab.path().head().unwrap().to_string()))
            .cloned()
            .collect();
        if !tables_to_be_created.is_empty() {
            Database {
                name,
                tables: vec![],
                pool,
                drop: false,
            }
            .with_tables(tables_to_be_created)
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
        let rt = tokio::runtime::Runtime::new().unwrap();
        let translator = MsSqlTranslator;
        let create_table_query = &table.create(translator).to_string();

        let query = sqlx::query(&create_table_query[..]);
        let res = rt.block_on(async_execute(query, &self.pool))?;
        Ok(res.rows_affected() as usize)
    }

    fn insert_data(&mut self, table: &Table) -> Result<()> {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let mut rng = StdRng::seed_from_u64(DATA_GENERATION_SEED);
        let size = Database::MAX_SIZE.min(table.size().generate(&mut rng) as usize);
        let ins_stat = &table.insert("@p", MsSqlTranslator).to_string();

        for _ in 1..size {
            let structured: value::Struct =
                table.schema().data_type().generate(&mut rng).try_into()?;
            let values: Result<Vec<SqlValue>> = structured
                .into_iter()
                .map(|(_, v)| (**v).clone().try_into())
                .collect();
            let values = values?;
            let mut insert_query = sqlx::query(&ins_stat[..]);
            for value in &values {
                insert_query = insert_query.bind(value);
            }
            
            rt.block_on(async_execute(insert_query, &self.pool))?;
        }
        Ok(())
    }

    fn query(&mut self, query: &str) -> Result<Vec<value::List>> {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async_query(query, &self.pool))
    }
}

async fn async_query(query_str: &str, pool: &Pool<Mssql>) -> Result<Vec<value::List>> {
    let rows = sqlx::query(query_str).fetch_all(pool).await?;
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

async fn async_execute(
    query: Query<'_, mssql::Mssql, MssqlArguments>,
    pool: &Pool<Mssql>,
) -> Result<MssqlQueryResult> {
    Ok(query.execute(pool).await?)
}

async fn async_connect(connection_string: &str) -> Result<MssqlPool> {
    Ok(MssqlPoolOptions::new()
        .max_connections(10)
        .connect(connection_string)
        .await?)
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
        if value.is_null() {
            Ok(Value::from(None).try_into()?)
        } else {
            match type_info.name() {
                "BIT" => Ok(Value::from(<bool as Decode<'_, Mssql>>::decode(value)?).try_into()?),
                "INT" => Ok(
                    Value::from((<i32 as Decode<'_, Mssql>>::decode(value)?) as i64).try_into()?,
                ),
                "BIGINT" => Ok(Value::from(<i64 as Decode<'_, Mssql>>::decode(value)?).try_into()?),
                "BINARY" => todo!(),
                "CHAR" => {
                    Ok(Value::from(<String as Decode<'_, Mssql>>::decode(value)?).try_into()?)
                }
                "DATE" => todo!(),
                "DATETIME" => todo!(),
                "DATETIME2" => todo!(),
                "DATETIMEOFFSET" => todo!(),
                "DECIMAL" => {
                    Ok(Value::from(<f64 as Decode<'_, Mssql>>::decode(value)?).try_into()?)
                }
                "FLOAT" => Ok(Value::from(<f64 as Decode<'_, Mssql>>::decode(value)?).try_into()?),
                "IMAGE" => todo!(),
                "MONEY" => todo!(),
                "NCHAR" => {
                    Ok(Value::from(<String as Decode<'_, Mssql>>::decode(value)?).try_into()?)
                }
                "NTEXT" => {
                    Ok(Value::from(<String as Decode<'_, Mssql>>::decode(value)?).try_into()?)
                }
                "NUMERIC" => {
                    Ok(Value::from(<f64 as Decode<'_, Mssql>>::decode(value)?).try_into()?)
                }
                "NVARCHAR" => {
                    Ok(Value::from(<String as Decode<'_, Mssql>>::decode(value)?).try_into()?)
                }
                "REAL" => Ok(Value::from(<f64 as Decode<'_, Mssql>>::decode(value)?).try_into()?),
                "SMALLDATETIME" => todo!(),
                "SMALLINT" => Ok(
                    Value::from((<i16 as Decode<'_, Mssql>>::decode(value)?) as i64).try_into()?,
                ),
                "SMALLMONEY" => todo!(),
                "SQL_VARIANT" => todo!(),
                "TEXT" => {
                    Ok(Value::from(<String as Decode<'_, Mssql>>::decode(value)?).try_into()?)
                }
                "TIME" => todo!(),
                "TIMESTAMP" => todo!(),
                "TINYINT" => todo!(),
                "UNIQUEIDENTIFIER" => todo!(),
                "VARBINARY" => todo!(),
                "VARCHAR" => {
                    Ok(Value::from(<String as Decode<'_, Mssql>>::decode(value)?).try_into()?)
                }
                "XML" => todo!(),
                _ => Err(Box::new(sqlx::Error::Decode(
                    format!("Unhandled type: {}", type_info.name()).into(),
                ))),
            }
        }
    }
}

impl Encode<'_, mssql::Mssql> for SqlValue {
    fn encode_by_ref(
        &self,
        buf: &mut <mssql::Mssql as sqlx::database::HasArguments<'_>>::ArgumentBuffer,
    ) -> sqlx::encode::IsNull {
        match self {
            SqlValue::Boolean(b) => {
                buf.push(if *b.deref() { 1 } else { 0 });
                sqlx::encode::IsNull::No
            }
            SqlValue::Integer(i) => <i64 as Encode<'_, Mssql>>::encode_by_ref(i.deref(), buf),
            SqlValue::Float(f) => {
                buf.extend(f.deref().to_le_bytes());
                sqlx::encode::IsNull::No
            }
            SqlValue::Text(t) => <String as Encode<'_, Mssql>>::encode_by_ref(t.deref(), buf),
            SqlValue::Optional(o) => o
                .as_ref()
                .map(|v| <&SqlValue as Encode<'_, Mssql>>::encode_by_ref(&&**v, buf))
                .unwrap_or(sqlx::encode::IsNull::Yes),
            SqlValue::Date(_) => todo!(),
            SqlValue::Time(_) => todo!(),
            SqlValue::DateTime(_) => todo!(),
            SqlValue::Id(_) => todo!(),
        }
    }

    fn produces(&self) -> Option<<mssql::Mssql as sqlx::Database>::TypeInfo> {
        match self {
            SqlValue::Boolean(b) => Some(<bool as Type<Mssql>>::type_info()),
            SqlValue::Integer(i) => Some(<i64 as Type<Mssql>>::type_info()),
            SqlValue::Float(f) => Some(<f64 as Type<Mssql>>::type_info()),
            SqlValue::Text(t) => <String as Encode<'_, mssql::Mssql>>::produces(t.deref()),
            SqlValue::Optional(o) => {
                let value = o.clone().map(|v| v.as_ref().clone());
                <&Option<SqlValue> as Encode<'_, Mssql>>::produces(&&value)
            }
            SqlValue::Date(_) => todo!(),
            SqlValue::Time(_) => todo!(),
            SqlValue::DateTime(_) => todo!(),
            SqlValue::Id(_) => todo!(),
        }
    }
}

// implementing this is needed in order order to use .get of the row object
impl Type<mssql::Mssql> for SqlValue {
    fn type_info() -> <mssql::Mssql as sqlx::Database>::TypeInfo {
        <String as Type<Mssql>>::type_info()
    }

    fn compatible(ty: &<mssql::Mssql as sqlx::Database>::TypeInfo) -> bool {
        true
    }
}

pub fn test_database() -> Database {
    Database::new(DB.into(), Database::test_tables()).expect("Database")
}

#[cfg(test)]
mod tests {
    use sqlx::Executor;

    use crate::{
        relation::{Schema, TableBuilder},
        DataType, Ready as _,
    };

    use super::*;

    #[tokio::test]
    async fn test_insert_table_with_pool() -> Result<()> {
        let connection_string = "mssql://SA:Strong@Passw0rd@localhost:1433/master?encrypt=false";
        let pool = MssqlPoolOptions::new()
            .test_before_acquire(true)
            .connect(connection_string)
            .await?;

        let table_name = "table_5";

        let _ = Command::new("docker")
            .arg("exec")
            .arg("qrlew-mssql-test_0")
            .arg("/opt/mssql-tools/bin/sqlcmd")
            .arg("-S")
            .arg(format!("localhost,1433"))
            .arg("-U")
            .arg("SA")
            .arg("-P")
            .arg(format!("Strong@Passw0rd"))
            .arg("-Q")
            .arg(format!("DROP TABLE {table_name};  "))
            .status()?
            .success();

        let table: Table = TableBuilder::new()
            .path([table_name])
            .name(table_name)
            .size(10)
            .schema(
                Schema::empty()
                    // .with(("f", DataType::float_interval(0.0, 10.0)))
                    .with(("z", DataType::text_values(["Foo".into(), "Bar".into()]))), // .with(("x", DataType::integer_interval(0, 100)))
                                                                                       // .with(("y", DataType::optional(DataType::text()))), // .with(("z", DataType::text_values(["Foo".into(), "Bar".into()])))
            )
            .build();
        let mut rng = StdRng::seed_from_u64(DATA_GENERATION_SEED);

        let ins_stat = &table.insert("@p", MsSqlTranslator).to_string();
        let create_stat = &table.create(MsSqlTranslator).to_string();

        //let new_ins = "INSERT INTO table_2 (x) VALUES (@p1)".to_string();
        println!("{}\n", create_stat);
        println!("{}\n", ins_stat);

        let _ = pool.execute(&create_stat[..]).await?;

        for _ in 1..100 {
            let structured: value::Struct =
                table.schema().data_type().generate(&mut rng).try_into()?;
            let values: Result<Vec<SqlValue>> = structured
                .into_iter()
                .map(|(_, v)| (**v).clone().try_into())
                .collect();
            let values = values?;
            println!("{:?}", values);
            let mut insert_query = sqlx::query(&ins_stat[..]);
            for value in &values {
                insert_query = insert_query.bind(value);
            }
            println!("before insert");
            let _ = pool.execute(insert_query).await?;
            //rt.block_on(async_execute(insert_query, &pool))?;
            println!("after insert");

            let r = pool
                .execute(&format!("SELECT TOP(5) z from {table_name}")[..])
                .await?;
            println!("after execution");
            println!("results: {:?}", r);
        }
        Ok(())
    }

    #[test]
    fn database_display() -> Result<()> {
        let mut database = test_database();
        let query = "SELECT TOP (10) * FROM table_1";
        println!("\n{query}");
        for row in database.query(query)? {
            println!("{}", row);
        }

        let query = "SELECT TOP (10) * FROM table_2";
        println!("\n{query}");
        for row in database.query(query)? {
            println!("{}", row);
        }

        let query = "SELECT TOP (10) * FROM user_table";
        println!("\n{query}");
        for row in database.query(query)? {
            println!("{}", row);
        }

        let query = "SELECT TOP (10) * FROM large_user_table";
        println!("\n{query}");
        for row in database.query(query)? {
            println!("{}", row);
        }

        let query = "SELECT TOP (10) * FROM order_table";
        println!("\n{query}");
        for row in database.query(query)? {
            println!("{}", row);
        }

        let query = "SELECT TOP (10) * FROM item_table";
        println!("\n{query}");
        for row in database.query(query)? {
            println!("{}", row);
        }

        let query = "SELECT TOP (10) * FROM MY_SPECIAL_TABLE";
        println!("\n{query}");
        for row in database.query(query)? {
            println!("{}", row);
        }

        Ok(())
    }
}
