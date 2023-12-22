//! Bigquery Connector. I allows to connect locally to a [big-query-emulator](https://github.com/goccy/bigquery-emulator) server.
//! Utils to run the docker with the big-query-emulator if is not running, load tables and run sql queris. 
//! The bigquery client is created using gcp_bigquery_client rust library. Since it doesn't support the authentication using
//! dummy credentials, as a workaround, we create a mocked google authentication server 
//! Inspired by this https://github.com/lquerel/gcp-bigquery-client/blob/main/examples/local.rs
//! 

use chrono::ParseError;
use serde::Serialize;
use tempfile::NamedTempFile;
use std::{ops::Deref, str::ParseBoolError};
use wiremock::{
    matchers::{method, path},
    Mock, MockServer, ResponseTemplate, Times,
};

use std::path::Path;

use fake::{Fake, StringFaker};
use gcp_bigquery_client::{
    model::{
        dataset::Dataset, query_request::QueryRequest, table::Table as BQTable,
        table_data_insert_all_request::TableDataInsertAllRequest, table_field_schema::TableFieldSchema,
        table_schema::TableSchema, query_response::ResultSet, field_type,
    },
    Client,
    table::{TableApi, ListOptions},
};

use super::{
    Database as DatabaseTrait,
    Error,
    Result,
    DATA_GENERATION_SEED
};

use crate::{
    data_type::{
        generator::Generator,
        value::{self, Value, Variant},
        DataTyped, List, self,
    },
    namer,
    relation::{Schema, Table, TableBuilder, Variant as _},
    DataType, Ready as _,
};
use colored::Colorize;
use rand::{rngs::StdRng, SeedableRng};
use std::{
    env, fmt, process::Command, str::FromStr, sync::Arc, sync::Mutex, thread, time,
};

//use crate::dialect_translation::mssql::BigQueryTranslator;


const DB: &str = "qrlew-bigquery-test";
const PORT: u16 = 9050;
const PROJECT_ID: &str = "test";
const DATASET_ID: &str = "dataset1";

impl From<gcp_bigquery_client::error::BQError> for Error {
    fn from(err: gcp_bigquery_client::error::BQError) -> Self {
        Error::Other(err.to_string())
    }
}

impl From<std::num::ParseFloatError> for Error {
    fn from(err: std::num::ParseFloatError) -> Self {
        Error::Other(err.to_string())
    }
}

impl From<ParseBoolError> for Error {
    fn from(err: ParseBoolError) -> Self {
        Error::Other(err.to_string())
    }
}

impl From<ParseError> for Error {
    fn from(err: ParseError) -> Self {
        Error::Other(err.to_string())
    }
}

const NAME_COLUMN: &str = "name";
const TABLE_ID: &str = "table";
pub const AUTH_TOKEN_ENDPOINT: &str = "/:o/oauth2/token";


pub struct GoogleAuthMock {
    server: MockServer,
}

impl Deref for GoogleAuthMock {
    type Target = MockServer;
    fn deref(&self) -> &Self::Target {
        &self.server
    }
}
impl GoogleAuthMock {
    pub async fn start() -> Self {
        Self {
            server: MockServer::start().await,
        }
    }
}

#[derive(Eq, PartialEq, Serialize, Debug, Clone)]
pub struct Token {
    access_token: String,
    token_type: String,
    expires_in: u32,
}
impl Token {
    fn fake() -> Self {
        Self {
            access_token: "aaaa".to_string(),
            token_type: "bearer".to_string(),
            expires_in: 9999999,
        }
    }
}

impl GoogleAuthMock {
    /// Mock token, given how many times the endpoint will be called.
    pub async fn mock_token<T: Into<Times>>(&self, n_times: T) {
        let response = ResponseTemplate::new(200).set_body_json(Token::fake());
        Mock::given(method("POST"))
            .and(path(AUTH_TOKEN_ENDPOINT))
            .respond_with(response)
            .named("mock token")
            .expect(n_times)
            .mount(self)
            .await;
    }
}

pub fn dummy_configuration(oauth_server: &str) -> serde_json::Value {
    let oauth_endpoint = format!("{oauth_server}/:o/oauth2");
    serde_json::json!({
      "type": "service_account",
      "project_id": "dummy",
      "private_key_id": "dummy",
      "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvwIBADANBgkqhkiG9w0BAQEFAASCBKkwggSlAgEAAoIBAQDNk6cKkWP/4NMu\nWb3s24YHfM639IXzPtTev06PUVVQnyHmT1bZgQ/XB6BvIRaReqAqnQd61PAGtX3e\n8XocTw+u/ZfiPJOf+jrXMkRBpiBh9mbyEIqBy8BC20OmsUc+O/YYh/qRccvRfPI7\n3XMabQ8eFWhI6z/t35oRpvEVFJnSIgyV4JR/L/cjtoKnxaFwjBzEnxPiwtdy4olU\nKO/1maklXexvlO7onC7CNmPAjuEZKzdMLzFszikCDnoKJC8k6+2GZh0/JDMAcAF4\nwxlKNQ89MpHVRXZ566uKZg0MqZqkq5RXPn6u7yvNHwZ0oahHT+8ixPPrAEjuPEKM\nUPzVRz71AgMBAAECggEAfdbVWLW5Befkvam3hea2+5xdmeN3n3elrJhkiXxbAhf3\nE1kbq9bCEHmdrokNnI34vz0SWBFCwIiWfUNJ4UxQKGkZcSZto270V8hwWdNMXUsM\npz6S2nMTxJkdp0s7dhAUS93o9uE2x4x5Z0XecJ2ztFGcXY6Lupu2XvnW93V9109h\nkY3uICLdbovJq7wS/fO/AL97QStfEVRWW2agIXGvoQG5jOwfPh86GZZRYP9b8VNw\ntkAUJe4qpzNbWs9AItXOzL+50/wsFkD/iWMGWFuU8DY5ZwsL434N+uzFlaD13wtZ\n63D+tNAxCSRBfZGQbd7WxJVFfZe/2vgjykKWsdyNAQKBgQDnEBgSI836HGSRk0Ub\nDwiEtdfh2TosV+z6xtyU7j/NwjugTOJEGj1VO/TMlZCEfpkYPLZt3ek2LdNL66n8\nDyxwzTT5Q3D/D0n5yE3mmxy13Qyya6qBYvqqyeWNwyotGM7hNNOix1v9lEMtH5Rd\nUT0gkThvJhtrV663bcAWCALmtQKBgQDjw2rYlMUp2TUIa2/E7904WOnSEG85d+nc\norhzthX8EWmPgw1Bbfo6NzH4HhebTw03j3NjZdW2a8TG/uEmZFWhK4eDvkx+rxAa\n6EwamS6cmQ4+vdep2Ac4QCSaTZj02YjHb06Be3gptvpFaFrotH2jnpXxggdiv8ul\n6x+ooCffQQKBgQCR3ykzGoOI6K/c75prELyR+7MEk/0TzZaAY1cSdq61GXBHLQKT\nd/VMgAN1vN51pu7DzGBnT/dRCvEgNvEjffjSZdqRmrAVdfN/y6LSeQ5RCfJgGXSV\nJoWVmMxhCNrxiX3h01Xgp/c9SYJ3VD54AzeR/dwg32/j/oEAsDraLciXGQKBgQDF\nMNc8k/DvfmJv27R06Ma6liA6AoiJVMxgfXD8nVUDW3/tBCVh1HmkFU1p54PArvxe\nchAQqoYQ3dUMBHeh6ZRJaYp2ATfxJlfnM99P1/eHFOxEXdBt996oUMBf53bZ5cyJ\n/lAVwnQSiZy8otCyUDHGivJ+mXkTgcIq8BoEwERFAQKBgQDmImBaFqoMSVihqHIf\nDa4WZqwM7ODqOx0JnBKrKO8UOc51J5e1vpwP/qRpNhUipoILvIWJzu4efZY7GN5C\nImF9sN3PP6Sy044fkVPyw4SYEisxbvp9tfw8Xmpj/pbmugkB2ut6lz5frmEBoJSN\n3osZlZTgx+pM3sO6ITV6U4ID2Q==\n-----END PRIVATE KEY-----\n",
      "client_email": "dummy@developer.gserviceaccount.com",
      "client_id": "dummy",
      "auth_uri": format!("{oauth_endpoint}/auth"),
      "token_uri": format!("{}{}", oauth_server, AUTH_TOKEN_ENDPOINT),
      "auth_provider_x509_cert_url": format!("{oauth_endpoint}/v1/certs"),
      "client_x509_cert_url": format!("{oauth_server}/robot/v1/metadata/x509/457015483506-compute%40developer.gserviceaccount.com")
    })
}

pub struct BQ {
    client: Client,
    project_id: String,
    dataset_id: String,
    table_id: String,
}

#[derive(Serialize, Debug, Clone, PartialEq, Eq)]
pub struct Row {
    pub name: String,
}

impl BQ {
    pub async fn new(sa_config_path: &Path, big_query_auth_base_url: String) -> Self {
        let client = gcp_bigquery_client::client_builder::ClientBuilder::new()
            .with_auth_base_url(big_query_auth_base_url)
            // Url of the BigQuery emulator docker image.
            .with_v2_base_url("http://localhost:9050".to_string())
            .build_from_service_account_key_file(sa_config_path.to_str().unwrap())
            .await
            .unwrap();
        // Use a random dataset id, so that each run is isolated.
        let dataset_id: String = {
            const LETTERS: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
            let f = StringFaker::with(Vec::from(LETTERS), 8);
            f.fake()
        };
        // Create a new dataset
        let dataset = client
            .dataset()
            .create(Dataset::new(PROJECT_ID, &dataset_id))
            .await
            .unwrap();
        create_table(&client, &dataset).await;
        Self {
            client,
            project_id: PROJECT_ID.to_string(),
            dataset_id: dataset_id.to_string(),
            table_id: TABLE_ID.to_string(),
        }
    }
    pub fn dataset_id(&self) -> String {
        self.dataset_id.clone()
    }
    pub async fn delete_dataset(&self) {
        // Delete the table previously created
        self.client
            .table()
            .delete(&self.project_id, &self.dataset_id, &self.table_id)
            .await
            .unwrap();
        // Delete the dataset previously created
        self.client
            .dataset()
            .delete(&self.project_id, &self.dataset_id, true)
            .await
            .unwrap();
    }
    pub async fn insert_row(&self, name: String) {
        let mut insert_request = TableDataInsertAllRequest::new();
        insert_request.add_row(None, Row { name }).unwrap();
        self.client
            .tabledata()
            .insert_all(&self.project_id, &self.dataset_id, &self.table_id, insert_request)
            .await
            .unwrap();
    }
    pub async fn get_rows(&self) -> Vec<String> {
        let mut rs = self
            .client
            .job()
            .query(
                &self.project_id,
                QueryRequest::new(format!(
                    "SELECT * FROM `{}.{}.{}`",
                    &self.project_id, &self.dataset_id, &self.table_id
                )),
            )
            .await
            .unwrap();
        let mut rows: Vec<String> = vec![];
        while rs.next_row() {
            let name = rs.get_string_by_name(NAME_COLUMN).unwrap().unwrap();
            rows.push(name)
        }
        rows
    }
    pub async fn async_query(&self, query_str: &str) -> ResultSet {
        let mut rs = self
            .client
            .job()
            .query(
                &self.project_id,
                QueryRequest::new(query_str),
            )
            .await
            .unwrap();
        rs
    }
}

// I can create it with a query actually.
async fn create_table(client: &Client, dataset: &Dataset) {
    dataset
        .create_table(
            client,
            BQTable::from_dataset(
                dataset,
                TABLE_ID,
                TableSchema::new(vec![TableFieldSchema::string(NAME_COLUMN)]),
            ),
        )
        .await
        .unwrap();
}

pub struct Database {
    name: String,
    tables: Vec<Table>,
    client: Client,
    google_authenticator: GoogleAuthMock, // Do we really need to keep this alive?
}

pub static BQ_CLIENT: Mutex<Option<Client>> = Mutex::new(None);
pub static BIGQUERY_CONTAINER: Mutex<bool> = Mutex::new(false);

impl Database {
    fn db() -> String {
        env::var("BIGQUERY_DB").unwrap_or(DB.into())
    }

    fn port() -> u16 {
        match env::var("BIGQUERY_PORT") {
            Ok(port) => u16::from_str(&port).unwrap_or(PORT),
            Err(_) => PORT,
        }
    }

    fn project_id() -> String {
        env::var("BIGQUERY_PROJECT_ID").unwrap_or(PROJECT_ID.into())
    }

    fn build_pool_from_existing(auth: &GoogleAuthMock, credentials_file: &NamedTempFile) -> Result<Client> {
        let rt = tokio::runtime::Runtime::new()?;
        let client = rt.block_on(build_client(auth.uri(), credentials_file))?;
        Ok(client)
    }

    /// Get a Database from a container
    fn build_pool_from_container(name: String, auth: &GoogleAuthMock, credentials_file: &NamedTempFile) -> Result<Client> {
        let mut bq_container = BIGQUERY_CONTAINER.lock().unwrap();

        if *bq_container == false {
            // A new container will be started
            *bq_container = true;

            // Other threads will wait for this to be ready
            let name = namer::new_name(name);
            let port = PORT + namer::new_id("bigquery-port") as u16;

            // Test the connection and launch a test instance if necessary
            if !Command::new("docker")
                .arg("start")
                .arg(&name)
                .status()?
                .success()
            {
                log::debug!("Starting the DB");
                // If the container does not exist, start a new container
                // docker run --name bigquery_name -p 9050:9050 ghcr.io/goccy/bigquery-emulator:latest --project=PROJECT_ID --dataset=DATASET_ID
                // use a helthcheck that sleeps 10 seconds to make sure the service is ready
                // in principle we should execute a dummy query such as SELECT 1
                // from inside the docker
                // but is a bit difficult with bigquery
                let output = Command::new("docker")
                    .arg("run")
                    .arg("--name")
                    .arg(&name)
                    .arg("-d")
                    .arg("--rm")
                    .arg("-p")
                    .arg(format!("{}:9050", port))
                    .arg("--health-cmd=sleep 10")
                    .arg("--health-interval=5s")
                    .arg("--health-timeout=20s") // greater than sleep
                    .arg("--health-retries=3")
                    .arg("ghcr.io/goccy/bigquery-emulator:latest")
                    .arg(format!("--project={}", PROJECT_ID))
                    .arg(format!("--dataset={}", DATASET_ID))
                    .output()?;
                log::info!("{:?}", output);
                log::info!("Waiting for the DB to start");
                log::info!("{}", "DB ready");
            }
        }
        Database::build_pool_from_existing(auth, credentials_file)
    }
}


async fn build_auth() -> Result<(GoogleAuthMock, NamedTempFile)> {
    let google_auth = GoogleAuthMock::start().await;
    google_auth.mock_token(1).await;

    let google_config = dummy_configuration(&google_auth.uri());
    println!("Write google configuration to file.");
    let temp_file: tempfile::NamedTempFile = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(temp_file.path(), serde_json::to_string_pretty(&google_config).unwrap()).unwrap();
    Ok((google_auth, temp_file))
}

async fn build_client(auth_uri: String, tmp_file_credentials: &NamedTempFile) -> Result<Client> {
    let client = gcp_bigquery_client::client_builder::ClientBuilder::new()
        .with_auth_base_url(auth_uri)
        // Url of the BigQuery emulator docker image.
        .with_v2_base_url("http://localhost:9050".to_string())
        .build_from_service_account_key_file(tmp_file_credentials.path().to_str().unwrap())
        .await?;
    Ok(client)
}

pub async fn async_row_query(query_str: &str, client: &Client) -> ResultSet {
    let mut rs = 
        client
        .job()
        .query(
            PROJECT_ID,
            QueryRequest::new(query_str),
        )
        .await
        .unwrap();
    rs
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
        let rt = tokio::runtime::Runtime::new().unwrap();
        let (auth_server, tmp_file_credentials) = rt.block_on(build_auth()).unwrap();
        
        let mut bq_client = BQ_CLIENT.lock().unwrap();
        if let None = *bq_client {
            *bq_client = Some(
                Database::build_pool_from_existing(&auth_server, &tmp_file_credentials)
                    .or_else(|_| Database::build_pool_from_container(name.clone(), &auth_server, &tmp_file_credentials))?,
            );
        }

        let client = bq_client.as_ref().unwrap().clone();

        let list_tabs = rt.block_on(client.table().list(PROJECT_ID, DATASET_ID, ListOptions::default())).unwrap();
        let table_names_in_db: Vec<String> = list_tabs
            .tables
            .unwrap_or_default()
            .into_iter()
            .map(|t| t.table_reference.table_id)
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
                client,
                google_authenticator: auth_server,
            }
            .with_tables(tables_to_be_created)
        } else {
            Ok(Database {
                name,
                tables,
                client,
                google_authenticator: auth_server,
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
        todo!()
    }

    fn insert_data(&mut self, table: &Table) -> Result<()> {
        todo!()
    }

    fn query(&mut self, query: &str) -> Result<Vec<value::List>> {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async_query(query, &self.client))
    }
}

async fn async_query(query_str: &str, client: &Client) -> Result<Vec<value::List>> {
    let mut rs = client
        .job()
        .query(PROJECT_ID,QueryRequest::new(query_str),
        )
        .await
        .unwrap();
    let query_response = rs.query_response();
    let schema = &query_response.schema;
    if let Some(table_schema) = schema {
        let fields = table_schema.fields().as_ref().unwrap();
        Ok(query_response.rows.as_ref()
            .unwrap()
            .iter()
            .map(|row| {
                // iterate over columns. There will be as many columns as
                // there are fields in the schema
                let cells = row.columns.as_ref().unwrap();
                println!("row: {:?}", row);
                println!("cells: {:?}", cells.len());
                println!("fields: {:?}", fields.len());
                let values: Vec<SqlValue> = (0..fields.len())
                    .map(|i| {
                        let cell_value = cells
                            .get(i)
                            .unwrap()
                            .clone()
                            .value;
                        let field_type = fields.get(i)
                            .unwrap()
                            .r#type.clone();
                        let val = SqlValue::try_from((cell_value, field_type)).unwrap();
                        val
                    })
                    .collect();
                value::List::from_iter(values.into_iter().map(|v| v.try_into().expect("Convert")))
            })
            .collect()
        )
    } else {
        Ok(vec![])
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
    Null(value::Unit),
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
            Value::Unit(u) => Ok(SqlValue::Null(u)),
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
            SqlValue::Null(u) => Ok(Value::Unit(u)),
        }
    }
}

impl TryFrom<(Option<serde_json::Value>, field_type::FieldType)> for SqlValue {
    // Type convertion from what is provided from the database to SqlValue
    // (a wrapper to qrlew value)
    // Data type in the query output is probided by the query_response table schema
    // field_type::FieldType. However we don't know from the type if the result
    // will contain Null or not. 
    // Data Values comes as a serde_json::Value which I only see String values
    // This is optional. Here, If the Value is None we map it to value::Value::unit()
    // As an alternative if We can map all columns to an Optional Value.
    type Error = Error;

    fn try_from(value: (Option<serde_json::Value>, field_type::FieldType)) -> Result<Self> {
        let (val, dtype) = value;        
        if let Some(v) = val {
            let val_as_str = extract_value(v)?;
            match dtype {
                field_type::FieldType::String => value::Value::text(val_as_str).try_into(),
                field_type::FieldType::Bytes => todo!(),
                field_type::FieldType::Integer => value::Value::integer(val_as_str.parse()?).try_into(),
                field_type::FieldType::Int64 => value::Value::integer(val_as_str.parse()?).try_into(),
                field_type::FieldType::Float => value::Value::float(val_as_str.parse()?).try_into(),
                field_type::FieldType::Float64 => value::Value::float(val_as_str.parse()?).try_into(),
                field_type::FieldType::Numeric => value::Value::float(val_as_str.parse()?).try_into(),
                field_type::FieldType::Bignumeric => value::Value::float(val_as_str.parse()?).try_into(),
                field_type::FieldType::Boolean => value::Value::boolean(val_as_str.parse()?).try_into(),
                field_type::FieldType::Bool => value::Value::boolean(val_as_str.parse()?).try_into(),
                field_type::FieldType::Timestamp => {
                    let timestamp: f64 = val_as_str.parse()?;
                    let seconds = timestamp as i64; // Whole seconds part
                    let nanoseconds = ((timestamp - seconds as f64) * 1_000_000_000.0) as u32; // Fractional part in nanoseconds
                    let datetime = chrono::NaiveDateTime::from_timestamp_opt(seconds, nanoseconds).unwrap();
                    value::Value::date_time(datetime).try_into()
                },
                field_type::FieldType::Date => value::Value::date(chrono::NaiveDate::parse_from_str(&val_as_str[..], "%Y-%m-%d")?).try_into(),
                field_type::FieldType::Time => value::Value::time(chrono::NaiveTime::parse_from_str(&val_as_str[..], "%H:%M:%S%.f")?).try_into(),
                field_type::FieldType::Datetime => value::Value::date_time(chrono::NaiveDateTime::parse_from_str(&val_as_str[..], "%Y-%m-%dT%H:%M:%S%.f")?).try_into(),
                field_type::FieldType::Record => todo!(),
                field_type::FieldType::Struct => todo!(),
                field_type::FieldType::Geography => todo!(),
                field_type::FieldType::Json => todo!(),
            }
        } else {
            value::Value::unit().try_into()
        }
    }
}

fn extract_value(val: serde_json::Value) -> Result<String>{
    match val {
        serde_json::Value::Null => todo!(),
        serde_json::Value::Bool(_) => todo!(),
        serde_json::Value::Number(_) => todo!(),
        serde_json::Value::String(s) => Ok(s),
        serde_json::Value::Array(_) => todo!(),
        serde_json::Value::Object(_) => todo!(),
    }

}

#[cfg(test)]
mod tests {

    use gcp_bigquery_client::table::ListOptions;

    use super::*;

    #[tokio::test]
    async fn test_bq_connector() {
        println!("Connecting to a mocked server");
        let google_auth = GoogleAuthMock::start().await;
        google_auth.mock_token(1).await;

        let google_config = dummy_configuration(&google_auth.uri());
        println!("Write google configuration to file.");
        let temp_file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(temp_file.path(), serde_json::to_string_pretty(&google_config).unwrap()).unwrap();

        println!("Create a dataset.");
        let bq = BQ::new(temp_file.path(), google_auth.uri()).await;
        let name = "foo";

        println!("Insert row to dataset.");
        bq.insert_row(name.to_string()).await;

        println!("Get rows from dataset.");
        let rows = bq.get_rows().await;
        assert_eq!(rows, vec![name]);
        println!("That's all Folks!");

        // let dataset_id = bq.dataset_id();
        // let query = format!("SELECT * FROM `{}.INFORMATION_SCHEMA.TABLES`", dataset_id);
        // println!("{:?}", query);
        // let res = bq.async_query(&query[..]).await;
        // println!("{:?}", res);
        bq.delete_dataset().await;
    }

    #[tokio::test]
    async fn test_table_list() {
        println!("Connecting to a mocked server");

        let google_auth = GoogleAuthMock::start().await;
        google_auth.mock_token(1).await;

        let google_config = dummy_configuration(&google_auth.uri());
        println!("Write google configuration to file.");
        let temp_file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(temp_file.path(), serde_json::to_string_pretty(&google_config).unwrap()).unwrap();

        let client = gcp_bigquery_client::client_builder::ClientBuilder::new()
            .with_auth_base_url(google_auth.uri())
            // Url of the BigQuery emulator docker image.
            .with_v2_base_url("http://localhost:9050".to_string())
            .build_from_service_account_key_file(temp_file.path().to_str().unwrap())
            .await
            .unwrap();

        let table_api = client.table();
        let list_tabs = table_api.list(PROJECT_ID, DATASET_ID, ListOptions::default()).await.unwrap();
        let tables_as_str: Vec<String> = list_tabs
            .tables
            .unwrap_or_default()
            .into_iter()
            .map(|t| t.table_reference.table_id)
            .collect();

        println!("{:?}", tables_as_str);
    }

    #[test]
    fn test_client() {
        let mut rt = tokio::runtime::Runtime::new().unwrap();

        let (auth_server, tmp_file_credentials) = rt.block_on(build_auth()).unwrap();
        let client = rt.block_on(build_client(auth_server.uri(), &tmp_file_credentials)).unwrap();
        let list_tabs = rt.block_on(client.table().list(PROJECT_ID, DATASET_ID, ListOptions::default())).unwrap();
        let tables_as_str: Vec<String> = list_tabs
            .tables
            .unwrap_or_default()
            .into_iter()
            .map(|t| t.table_reference.table_id)
            .collect();
        println!("{:?}", tables_as_str);

        // let query = "SELECT * FROM mytable";
        // let query= "SELECT CURRENT_TIMESTAMP() AS now;";
        // let res: ResultSet = rt.block_on(async_query(query, &client));
        // let query_response: &gcp_bigquery_client::model::query_response::QueryResponse = res.query_response();
        // if let Some(tab_schema) = &query_response.schema {
        //     println!("{:?}", tab_schema);
        // }
        // println!("{:?}", query_response);
    }

    #[test]
    fn test_mapping() {
        let mut rt = tokio::runtime::Runtime::new().unwrap();

        let (auth_server, tmp_file_credentials) = rt.block_on(build_auth()).unwrap();
        let client = rt.block_on(build_client(auth_server.uri(), &tmp_file_credentials)).unwrap();
        let list_tabs = rt.block_on(client.table().list(PROJECT_ID, DATASET_ID, ListOptions::default())).unwrap();
        let tables_as_str: Vec<String> = list_tabs
            .tables
            .unwrap_or_default()
            .into_iter()
            .map(|t| t.table_reference.table_id)
            .collect();
        println!("{:?}", tables_as_str);

        let query = "
        SELECT 
            *,
            CURRENT_TIMESTAMP() AS now,
            CURRENT_DATETIME() as now_datetime,
            CURRENT_DATE() AS date_utc, 
            CURRENT_TIME() AS time_utc, 
            1.00 AS int_v, 
            'AHAhA' AS mysrt, 
            True AS mybool, 
            Null AS mynull 
        FROM dataset1.mytable2;";

        let res: ResultSet = rt.block_on(async_row_query(query, &client));
        //println!("{:?}", res);
        let query_response = res.query_response();
        if let Some(tab_schema) = &query_response.schema {
            println!("{:?}", tab_schema);
            let fields = tab_schema.fields().as_ref().unwrap();
            //let i = ..fields.len();//iterator over columns
            for (index, field) in fields.iter().enumerate() {
                println!("ID={}, Type={:?}", index, field.r#type)
            }
            
            for row in query_response.rows.as_ref().unwrap().iter() {
                println!("ROW ITERATOR");
                let cells = row.columns.as_ref().unwrap();
                for cell in cells {
                    if let Some(value) = cell.value.as_ref() {
                        match value {
                            serde_json::Value::Null => println!("NULL INNER"),
                            serde_json::Value::Bool(b) => println!("BOOL: {}", b),
                            serde_json::Value::Number(n) => println!("NUM: {}", n),
                            serde_json::Value::String(s) => println!("STR: {}", s),
                            serde_json::Value::Array(a) => todo!(),
                            serde_json::Value::Object(o) => todo!(),
                        }
                    } else {
                        println!("NULL")
                    }
                }
            }
        }
        
    }

    #[test]
    fn test_timestamp() {
        let timestamp = 1703273535.453880;
        let seconds = timestamp as i64; // Whole seconds part
        let nanoseconds = ((timestamp - seconds as f64) * 1_000_000_000.0) as u32; // Fractional part in nanoseconds
        let datetime = chrono::NaiveDateTime::from_timestamp_opt(seconds, nanoseconds);
        println!("Datetime: {:?}", datetime);
    }

    #[test]
    fn test_datetime() {
        let datetime = "2023-12-22T19:50:11.637687";
        let datetime = chrono::NaiveDateTime::parse_from_str(datetime, "%Y-%m-%dT%H:%M:%S%.f").unwrap();
        println!("Datetime: {:?}", datetime);
    }

    #[test]
    fn test_date() {
        let date = "2023-12-22";
        let date = chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d").unwrap();
        println!("Datetime: {:?}", date);
    }

    #[test]
    fn test_time() {
        let time = "19:50:11.637698";
        let time =  chrono::NaiveTime::parse_from_str(time, "%H:%M:%S%.f").unwrap();
        println!("Datetime: {:?}", time);
    }

    #[test]
    fn test_mapping_bis() {
        let mut rt = tokio::runtime::Runtime::new().unwrap();

        let (auth_server, tmp_file_credentials) = rt.block_on(build_auth()).unwrap();
        let client = rt.block_on(build_client(auth_server.uri(), &tmp_file_credentials)).unwrap();
        let list_tabs = rt.block_on(client.table().list(PROJECT_ID, DATASET_ID, ListOptions::default())).unwrap();
        let tables_as_str: Vec<String> = list_tabs
            .tables
            .unwrap_or_default()
            .into_iter()
            .map(|t| t.table_reference.table_id)
            .collect();
        println!("{:?}", tables_as_str);

        let query = "
        SELECT 
            *,
            CURRENT_TIMESTAMP() AS now,
            CURRENT_DATETIME() as now_datetime,
            CURRENT_DATE() AS date_utc, 
            CURRENT_TIME() AS time_utc, 
            1.00 AS int_v, 
            'AHAhA' AS mysrt, 
            True AS mybool, 
            Null AS mynull 
        FROM dataset1.mytable2;";

        let res = rt.block_on(async_query(query, &client)).unwrap();
        println!("{:?}", res);
    }
// Can I not create the dataset?
// 
}