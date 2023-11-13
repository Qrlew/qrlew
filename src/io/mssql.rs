// Creating an mssql connector. Careful this is still in beta:
// TODO:
// MSSQL doesn't support IF NOT EXISTS in create table. If the table exists a runtime error will be rised
// In MSSQL the boolean type is create with the BIT keyword but in the sqlparser is not possible to define a DataType BIT

use super::{Database as DatabaseTrait, Error, Result, DATA_GENERATION_SEED};
use crate::{
    data_type::{
        generator::Generator,
        value::{self, Value, Variant},
        DataTyped, List,
    },
    namer,
    relation::{Table, Variant as _}, dialect_translation::mssql::MSSQLTranslator,
};
use colored::Colorize;
use rand::{rngs::StdRng, SeedableRng};
use sqlx::{MssqlConnection, Connection, mssql::{MssqlArguments, MssqlQueryResult}};
use sqlx::{
    self,
    mssql::{self, Mssql, MssqlRow, MssqlValueRef, MssqlConnectOptions},
    Decode, Encode, Row, Type, TypeInfo, ValueRef as _,
    query::Query
};
use std::{
    env, fmt, ops::Deref, process::Command, str::FromStr, sync::Arc, sync::Mutex, thread, time,
};

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

    /// A mssql instance must exist
    /// `docker run --name qrlew-mssql-test -p 1433:1433 -d mssql`
    fn try_get_existing(name: String, tables: Vec<Table>) -> Result<Self> {
        log::info!("Try to get an existing DB");
        let rt = tokio::runtime::Runtime::new().unwrap();
        let mut connection = rt.block_on(async_connect(
            &format!(
                "mssql://{}:{}@localhost:{}/master?encrypt=false",
                Database::user(),
                Database::password(),
                Database::port(),
            ),
        ))?;

        let find_tables_query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'public'";
        let table_names: Vec<String> = rt.block_on(async_query(find_tables_query, &mut connection))?
            .iter()
            .map(|r| {
                let val_as_str: String = r.to_vec()[0].clone().try_into().unwrap();
                val_as_str
            })
            .collect();
        if table_names.is_empty() {
            Database {
                name,
                tables: vec![],
                connection,
                drop: false,
            }.with_tables(tables)
        } else {
            Ok(Database {
                name,
                tables,
                connection,
                drop: false,
            })
        }
    }

    // /// Get a Database from a container
    fn try_get_container(name: String, tables: Vec<Table>) -> Result<Self> {
        let mut mssql_container = MSSQL_CONTAINER.lock().unwrap();
    
        if !*mssql_container {
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
                // Run: `docker run -e "ACCEPT_EULA=1" -e "MSSQL_SA_PASSWORD=MyPass@word" -e "MSSQL_USER=SA" -p 1433:1433 -d --name=qrlew-mssql-test mcr.microsoft.com/azure-sql-edge`
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
                while !Command::new("docker")
                    .arg("exec")
                    .arg(&name)
                    .arg("sqlcmd")
                    .arg("-S")
                    .arg(format!("localhost,{port}"))
                    .arg("-U")
                    .arg("SA")
                    .arg("-P")
                    .arg("{PASSWORD}")
                    .arg("-Q")
                    .arg("SELECT 1")
                    .status()?
                    .success()
                {
                    thread::sleep(time::Duration::from_millis(200));
                    log::info!("Waiting...");
                }
                log::info!("{}", "DB ready".red());
            }

            let rt = tokio::runtime::Runtime::new().unwrap();
            let connection = rt.block_on(async_connect(
                &format!(
                    "mssql://{}:{}@localhost:{}/master?encrypt=false",
                    Database::user(),
                    Database::password(),
                    Database::port(),
                ),
            ))?;
            Ok(Database {
                name,
                tables: vec![],
                connection,
                drop: false,
            }
            .with_tables(tables)?)
        } else {
            Database::try_get_existing(name, tables)
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
        let rt = tokio::runtime::Runtime::new().unwrap();
        let translator = MSSQLTranslator;
        let create_table_query = &table.create(translator).to_string();
        let query = sqlx::query(&create_table_query[..]);
        rt.block_on(async_execute(query, &mut self.connection))?;
        Ok(1 as usize)
    }

    fn insert_data(&mut self, table: &Table) -> Result<()> {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let mut rng = StdRng::seed_from_u64(DATA_GENERATION_SEED);
        let size = Database::MAX_SIZE.min(table.size().generate(&mut rng) as usize);
        let ins_stat = &table.insert("@p", MSSQLTranslator).to_string();
    
        for _ in 1..size {
            let structured: value::Struct = table.schema().data_type().generate(&mut rng).try_into()?;
            let values: Result<Vec<SqlValue>> = structured
                .into_iter()
                .map(|(_, v)| (**v).clone().try_into())
                .collect();
            let values = values?; 
            let mut insert_query = sqlx::query(&ins_stat[..]);
            for value in &values {
                insert_query = insert_query.bind(value);
            }
            rt.block_on(async_execute(insert_query, &mut self.connection))?;
        }
        Ok(())
    }

    fn query(&mut self, query: &str) -> Result<Vec<value::List>> {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async_query(query, &mut self.connection))
    }
}

async fn async_query(query_str: &str, connection: &mut MssqlConnection) -> Result<Vec<value::List>> {
    let rows = sqlx::query(query_str).fetch_all(connection).await?;

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

async fn async_execute(query: Query<'_, mssql::Mssql, MssqlArguments>, connection: &mut MssqlConnection) -> Result<MssqlQueryResult> {
    Ok(query.execute(connection).await?)
}

async fn async_connect(connection_string: &str) -> Result<MssqlConnection> {
    let connection_options = MssqlConnectOptions::from_str(connection_string)?;
    Ok(MssqlConnection::connect_with(&connection_options).await?)
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
        match type_info.name() {
            "BIT" => Ok(Value::from(<bool as Decode<'_, Mssql>>::decode(value)?).try_into()?),
            "INT" => Ok(Value::from((<i32 as Decode<'_, Mssql>>::decode(value)?) as i64).try_into()?),
            "BIGINT" => Ok(Value::from(<i64 as Decode<'_, Mssql>>::decode(value)?).try_into()?),
            "BINARY" => todo!(),
            "CHAR" => Ok(Value::from(<String as Decode<'_, Mssql>>::decode(value)?).try_into()?),
            "DATE" => todo!(),
            "DATETIME" => todo!(),
            "DATETIME2" => todo!(),
            "DATETIMEOFFSET" => todo!(),
            "DECIMAL" => Ok(Value::from(<f64 as Decode<'_, Mssql>>::decode(value)?).try_into()?),
            "FLOAT" => Ok(Value::from(<f64 as Decode<'_, Mssql>>::decode(value)?).try_into()?),
            "IMAGE" => todo!(),
            "MONEY" => todo!(),
            "NCHAR" => Ok(Value::from(<String as Decode<'_, Mssql>>::decode(value)?).try_into()?),
            "NTEXT" => Ok(Value::from(<String as Decode<'_, Mssql>>::decode(value)?).try_into()?),
            "NUMERIC" => Ok(Value::from(<f64 as Decode<'_, Mssql>>::decode(value)?).try_into()?),
            "NVARCHAR" => Ok(Value::from(<String as Decode<'_, Mssql>>::decode(value)?).try_into()?),
            "REAL" => todo!(),
            "SMALLDATETIME" => todo!(),
            "SMALLINT" => todo!(),
            "SMALLMONEY" => todo!(),
            "SQL_VARIANT" => todo!(),
            "TEXT" => Ok(Value::from(<String as Decode<'_, Mssql>>::decode(value)?).try_into()?),
            "TIME" => todo!(),
            "TIMESTAMP" => todo!(),
            "TINYINT" => todo!(),
            "UNIQUEIDENTIFIER" => todo!(),
            "VARBINARY" => todo!(),
            "VARCHAR" => Ok(Value::from(<String as Decode<'_, Mssql>>::decode(value)?).try_into()?),
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
        println!("encode_by_ref");
        match self {
            SqlValue::Boolean(b) => {
                buf.push(if *b.deref() { 1 } else { 0 });
                sqlx::encode::IsNull::No
            }
            SqlValue::Integer(i) => {
                <i64 as Encode<'_, Mssql>>::encode_by_ref(i.deref(), buf)
            }
            SqlValue::Float(f) => {
                println!("SqlValue::Float encode_by_ref");
                buf.extend(f.deref().to_le_bytes());
                sqlx::encode::IsNull::No
            }
            SqlValue::Text(t) => {
                println!("SqlValue::Text encode_by_ref");
                <String as Encode<'_, Mssql>>::encode_by_ref(t.deref(), buf)
            },
            SqlValue::Optional(o) => {
                o
                .as_ref()
                .map(|v| <&SqlValue as Encode<'_, Mssql>>::encode_by_ref(&&**v, buf))
                .unwrap_or( sqlx::encode::IsNull::Yes)
            }
            SqlValue::Date(_) => todo!(),
            SqlValue::Time(_) => todo!(),
            SqlValue::DateTime(_) => todo!(),
            SqlValue::Id(_) => todo!(),
        }
    }

    fn produces(&self) -> Option<<mssql::Mssql as sqlx::Database>::TypeInfo> {
        println!("produces");
        match self {
            SqlValue::Boolean(b) => Some(<bool as Type<Mssql>>::type_info()),
            SqlValue::Integer(i) => Some(<i64 as Type<Mssql>>::type_info()),
            SqlValue::Float(f) => Some(<f64 as Type<Mssql>>::type_info()),
            SqlValue::Text(t) => <String as Encode<'_, mssql::Mssql>>::produces(t.deref()),
            SqlValue::Optional(o) =>{
                let value = o.clone().map(|v| v.as_ref().clone());
                <&Option<SqlValue> as Encode<'_, Mssql>>::produces(&&value)
            },
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
        println!("type_info");
        <String as Type<Mssql>>::type_info()
    }

    fn compatible(ty: &<mssql::Mssql as sqlx::Database>::TypeInfo) -> bool {
        println!("compatible");
        true
    }
}

pub fn test_database() -> Database {
    Database::new(DB.into(), Database::test_tables()).expect("Database")
}


#[cfg(test)]
mod tests {
    use sqlx::Executor;

    use crate::{relation::{TableBuilder, Schema}, DataType, Ready as _};

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
        println!("{:#?}", rows_as_vec);
        Ok(())
    }

    #[tokio::test]
    async fn test_insert_table() -> Result<()> {
        let connection_string = "mssql://SA:MyPass@word@localhost:1433/master?encrypt=false";
        let connection_options = MssqlConnectOptions::from_str(connection_string)?;
        let mut conn = MssqlConnection::connect_with(&connection_options).await?;

        let mut rng = StdRng::seed_from_u64(DATA_GENERATION_SEED);

        let table: Table = TableBuilder::new()
            .path(["table_2"])
            .name("table_2")
            .size(5)
            .schema(Schema::empty()
                .with(("f", DataType::float_interval(0.0, 10.0)))
                .with(("z", DataType::text_values(["Foo".into(), "Bar".into()])))
                .with(("x", DataType::integer_interval(0, 100)))
                .with(("y", DataType::optional(DataType::text())))
                // .with(("z", DataType::text_values(["Foo".into(), "Bar".into()])))
            )
            .build();

        let ins_stat = &table.insert("@p", MSSQLTranslator).to_string();
        let create_stat = &table.create(MSSQLTranslator).to_string();

        //let new_ins = "INSERT INTO table_2 (x) VALUES (@p1)".to_string();
        println!("{}\n", create_stat);
        println!("{}\n", ins_stat);

        let _ = sqlx::query(&create_stat[..]).execute(&mut conn).await?;

        for _ in 1..100 {
            let structured: value::Struct = table.schema().data_type().generate(&mut rng).try_into()?;
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
            println!("before execution");
            insert_query.execute(&mut conn).await?;
            println!("after execution");
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_insert_strings() -> Result<()> {
        let connection_string = "mssql://SA:MyPass@word@localhost:1433/master?encrypt=false";
        let connection_options = MssqlConnectOptions::from_str(connection_string)?;
        let mut conn = MssqlConnection::connect_with(&connection_options).await?;
        let _ = conn.execute("CREATE TABLE users (id FLOAT);",).await?;

        for index in 1..=2_i32 {
            let done = sqlx::query("INSERT INTO users (id) VALUES (@p1)")
                .bind(index as f64)
                .execute(&mut conn)
                .await?;

            assert_eq!(done.rows_affected(), 1);
        }
        Ok(())
    }
}
