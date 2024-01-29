//! Bigquery Connector. It allows to connect locally to a [big-query-emulator](https://github.com/goccy/bigquery-emulator) server.
//! Utils to run the docker with the big-query-emulator if is not running, load tables and run sql queries.
//! The bigquery client is created using gcp_bigquery_client rust library. Since it doesn't support the authentication using
//! dummy credentials, as a workaround, we create a mocked google authentication server
//! Inspired by this https://github.com/lquerel/gcp-bigquery-client/blob/main/examples/local.rs
//!

use chrono::ParseError;
use serde::{ser, Serialize};
use serde_json;
use std::{collections::HashMap, ops::Deref, str::ParseBoolError};
use tempfile::NamedTempFile;
use wiremock::{
    matchers::{method, path},
    Mock, MockServer, ResponseTemplate, Times,
};

use gcp_bigquery_client::{
    model::{
        dataset_reference::DatasetReference, field_type, query_parameter::QueryParameter,
        query_request::QueryRequest, query_response::ResultSet, table::Table as BQTable,
        table_data_insert_all_request::TableDataInsertAllRequest,
        table_data_insert_all_request_rows::TableDataInsertAllRequestRows,
        table_field_schema::TableFieldSchema, table_schema::TableSchema,
    },
    table::ListOptions,
    Client,
};

use super::{Database as DatabaseTrait, Error, Result, DATA_GENERATION_SEED};

use crate::{
    data_type::{
        self,
        generator::Generator,
        value::{self, Value, Variant},
        DataTyped, List,
    },
    namer,
    relation::{Constraint, Schema, Table, TableBuilder, Variant as _},
    DataType, Ready as _,
};
use rand::{rngs::StdRng, SeedableRng};
use std::{env, fmt, process::Command, result, str::FromStr, sync::Arc, sync::Mutex, thread, time};

const DB: &str = "qrlew-bigquery-test";
const PORT: u16 = 9050;
const PROJECT_ID: &str = "test";
const DATASET_ID: &str = "dataset1";
const AUTH_TOKEN_ENDPOINT: &str = "/:o/oauth2/token";

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

pub struct Database {
    name: String,
    tables: Vec<Table>,
    client: Client,
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

    fn check_client(client: &Client) -> Result<()> {
        let rt = tokio::runtime::Runtime::new()?;
        let res = rt.block_on(async_query("SELECT 1", &client, None))?;
        Ok(())
    }

    /// Get a Database from a container
    fn build_pool_from_container(name: String, client: &Client) -> Result<()> {
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
                // use a health check that sleeps 10 seconds to make sure the service gets ready
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

                let max_seconds = 10;
                let max_duration = time::Duration::from_secs(max_seconds); // Set maximum duration for the loop
                let start_time = time::Instant::now();

                loop {
                    match Database::check_client(&client) {
                        Ok(_) => {
                            log::info!("BQ emulator ready!");
                            break;
                        }
                        Err(_) => {
                            if start_time.elapsed() > max_duration {
                                return Err(Error::other(format!(
                                    "BQ emulator couldn't be ready in {} seconds!",
                                    max_seconds
                                )));
                            }
                            // Optional: sleep for a bit before retrying
                            thread::sleep(time::Duration::from_millis(500));
                        }
                    }
                }
            }
            Ok(())
        } else {
            Err(Error::other("Could find the container!"))
        }
    }

    // Overriding test_tables because we there is a maximum allowed table size
    // imposed by the bigquery emulator. more_users is too big.
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
                .path(["table_2"])
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
                .path(["user_table"])
                .name("users")
                .size(100)
                .schema(
                    Schema::empty()
                        .with(("id", DataType::integer_interval(0, 100)))
                        .with(("name", DataType::text(), Constraint::Unique))
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
                .size(1000)
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
            // TableBuilder::new()
            //     .path(["MY SPECIAL TABLE"])
            //     .name("my_table")
            //     .size(100)
            //     .schema(
            //         Schema::empty()
            //             .with(("Id", DataType::integer_interval(0, 1000)))
            //             .with(("Na.Me", DataType::text()))
            //             .with(("inc&ome", DataType::float_interval(100.0, 200000.0)))
            //             .with(("normal_col", DataType::text())),
            //     )
            //     .build(),
        ]
    }
}

async fn build_auth() -> Result<(GoogleAuthMock, NamedTempFile)> {
    let google_auth = GoogleAuthMock::start().await;
    google_auth.mock_token(1).await;

    let google_config = dummy_configuration(&google_auth.uri());
    let temp_file: tempfile::NamedTempFile = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(
        temp_file.path(),
        serde_json::to_string_pretty(&google_config).unwrap(),
    )
    .unwrap();
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
    let mut rs = client
        .job()
        .query(PROJECT_ID, QueryRequest::new(query_str))
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
        *bq_client = Some(rt.block_on(build_client(auth_server.uri(), &tmp_file_credentials))?);
        let client = bq_client.as_ref().unwrap().clone();

        // make sure you check there is a bigquery instance up and running
        // or try to start an existing one
        // or create a new one.
        Database::check_client(&client)
            .or_else(|_| Database::build_pool_from_container(name.clone(), &client))?;
        let list_tabs = rt
            .block_on(
                client
                    .table()
                    .list(PROJECT_ID, DATASET_ID, ListOptions::default()),
            )
            .unwrap();
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
            }
            .with_tables(tables_to_be_created)
        } else {
            Ok(Database {
                name,
                tables,
                client,
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
        let mut rt = tokio::runtime::Runtime::new()?;
        let bq_table: BQTable = table.clone().try_into()?;

        rt.block_on(self.client.table().create(bq_table))?;
        Ok(1)
    }

    fn insert_data(&mut self, table: &Table) -> Result<()> {
        let mut rt = tokio::runtime::Runtime::new()?;
        let mut rng = StdRng::seed_from_u64(DATA_GENERATION_SEED);
        let size = Database::MAX_SIZE.min(table.size().generate(&mut rng) as usize);

        let mut insert_query = TableDataInsertAllRequest::new();
        let mut rows_for_bq: Vec<TableDataInsertAllRequestRows> = vec![];
        for _ in 1..size {
            let structured: value::Struct =
                table.schema().data_type().generate(&mut rng).try_into()?;
            let keys: Vec<String> = table
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().into())
                .collect();
            let values: Result<Vec<SqlValue>> = structured
                .into_iter()
                .map(|(_, v)| (**v).clone().try_into())
                .collect();
            let values = values?;
            let map: HashMap<String, SqlValue> = keys.into_iter().zip(values.into_iter()).collect();
            let map_as_json = serde_json::json!(map);
            rows_for_bq.push(TableDataInsertAllRequestRows {
                insert_id: None,
                json: map_as_json,
            });
        }
        
        insert_query.add_rows(rows_for_bq.clone())?;

        rt.block_on(self.client.tabledata().insert_all(
            PROJECT_ID,
            DATASET_ID,
            table.path().head().unwrap().to_string().as_str(),
            insert_query.clone(),
        ))?;
        Ok(())
    }

    fn query(&mut self, query: &str) -> Result<Vec<value::List>> {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async_query(query, &self.client, None))
    }
}

async fn async_query(
    query_str: &str,
    client: &Client,
    query_parameters: Option<Vec<QueryParameter>>,
) -> Result<Vec<value::List>> {
    let parameter_mode: Option<String> = if let Some(_) = query_parameters {
        Some("NAMED".to_string())
    } else {
        None
    };

    let query_request = QueryRequest {
        connection_properties: None,
        default_dataset: Some(DatasetReference {
            dataset_id: DATASET_ID.to_string(),
            project_id: PROJECT_ID.to_string(),
        }),
        dry_run: None,
        kind: None,
        labels: None,
        location: None,
        max_results: None,
        maximum_bytes_billed: None,
        parameter_mode,
        preserve_nulls: None,
        query: query_str.into(),
        query_parameters,
        request_id: None,
        timeout_ms: None,
        use_legacy_sql: false, // force standard SQL by default
        use_query_cache: None,
        format_options: None,
    };
    let mut rs = client.job().query(PROJECT_ID, query_request).await?;
    let query_response = rs.query_response();
    let schema = &query_response.schema;
    if let Some(table_schema) = schema {
        let fields = table_schema.fields().as_ref().unwrap();
        Ok(query_response
            .rows
            .as_ref()
            .unwrap()
            .iter()
            .map(|row| {
                // iterate over columns. There will be as many columns as
                // there are fields in the schema
                let cells = row.columns.as_ref().unwrap();
                let values: Vec<SqlValue> = (0..fields.len())
                    .map(|i| {
                        let cell_value = cells.get(i).unwrap().clone().value;
                        let field_type = fields.get(i).unwrap().r#type.clone();
                        let val = SqlValue::try_from((cell_value, field_type)).unwrap();
                        val
                    })
                    .collect();
                value::List::from_iter(values.into_iter().map(|v| v.try_into().expect("Convert")))
            })
            .collect())
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

// Implementing Serialize for SqlValue
impl Serialize for SqlValue {
    fn serialize<S>(&self, serializer: S) -> result::Result<S::Ok, S::Error>
    where
        S: ser::Serializer,
    {
        // You can customize how each variant is serialized
        match self {
            SqlValue::Boolean(b) => serializer.serialize_bool(*b.deref()),
            SqlValue::Integer(i) => serializer.serialize_i64(*i.deref()),
            SqlValue::Float(f) => serializer.serialize_f64(*f.deref()),
            SqlValue::Text(t) => serializer.serialize_str(t.deref().as_str()),
            SqlValue::Optional(o) => match o {
                Some(value) => value.clone().serialize(serializer),
                None => serializer.serialize_none(),
            },
            SqlValue::Date(d) => {
                serializer.serialize_str(d.deref().format("%Y-%m-%d").to_string().as_str())
            }
            SqlValue::Time(t) => {
                serializer.serialize_str(t.deref().format("%H:%M:%S").to_string().as_str())
            }
            SqlValue::DateTime(dt) => serializer.serialize_str(
                dt.deref()
                    .format("%Y-%m-%dT%H:%M:%S%.f")
                    .to_string()
                    .as_str(),
            ),
            SqlValue::Id(id) => serializer.serialize_str(id.deref().as_str()),
            SqlValue::Null(_) => serializer.serialize_none(),
        }
    }
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
                field_type::FieldType::Bytes => value::Value::bytes(val_as_str).try_into(),
                field_type::FieldType::Integer => {
                    value::Value::integer(val_as_str.parse()?).try_into()
                }
                field_type::FieldType::Int64 => {
                    value::Value::integer(val_as_str.parse()?).try_into()
                }
                field_type::FieldType::Float => value::Value::float(val_as_str.parse()?).try_into(),
                field_type::FieldType::Float64 => {
                    value::Value::float(val_as_str.parse()?).try_into()
                }
                field_type::FieldType::Numeric => {
                    value::Value::float(val_as_str.parse()?).try_into()
                }
                field_type::FieldType::Bignumeric => {
                    value::Value::float(val_as_str.parse()?).try_into()
                }
                field_type::FieldType::Boolean => {
                    value::Value::boolean(val_as_str.parse()?).try_into()
                }
                field_type::FieldType::Bool => {
                    value::Value::boolean(val_as_str.parse()?).try_into()
                }
                field_type::FieldType::Timestamp => {
                    let timestamp: f64 = val_as_str.parse()?;
                    let seconds = timestamp as i64; // Whole seconds part
                    let nanoseconds = ((timestamp - seconds as f64) * 1_000_000_000.0) as u32; // Fractional part in nanoseconds
                    let datetime =
                        chrono::NaiveDateTime::from_timestamp_opt(seconds, nanoseconds).unwrap();
                    value::Value::date_time(datetime).try_into()
                }
                field_type::FieldType::Date => value::Value::date(
                    chrono::NaiveDate::parse_from_str(&val_as_str[..], "%Y-%m-%d")?,
                )
                .try_into(),
                field_type::FieldType::Time => value::Value::time(
                    chrono::NaiveTime::parse_from_str(&val_as_str[..], "%H:%M:%S%.f")?,
                )
                .try_into(),
                field_type::FieldType::Datetime => value::Value::date_time(
                    chrono::NaiveDateTime::parse_from_str(&val_as_str[..], "%Y-%m-%dT%H:%M:%S%.f")?,
                )
                .try_into(),
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

// impl From<SqlValue> for

fn extract_value(val: serde_json::Value) -> Result<String> {
    match val {
        serde_json::Value::Null => todo!(),
        serde_json::Value::Bool(_) => todo!(),
        serde_json::Value::Number(_) => todo!(),
        serde_json::Value::String(s) => Ok(s),
        serde_json::Value::Array(_) => todo!(),
        serde_json::Value::Object(_) => todo!(),
    }
}

impl TryFrom<DataType> for field_type::FieldType {
    type Error = Error;

    fn try_from(dtype: DataType) -> Result<Self> {
        match dtype {
            DataType::Null => todo!(),
            DataType::Unit(_) => todo!(),
            DataType::Boolean(_) => Ok(field_type::FieldType::Boolean),
            DataType::Integer(_) => Ok(field_type::FieldType::Integer),
            DataType::Enum(_) => todo!(),
            DataType::Float(_) => Ok(field_type::FieldType::Float),
            DataType::Text(_) => Ok(field_type::FieldType::String),
            DataType::Bytes(_) => Ok(field_type::FieldType::Bytes),
            DataType::Struct(_) => todo!(),
            DataType::Union(_) => todo!(),
            DataType::Optional(o) => field_type::FieldType::try_from(o.data_type().to_owned()),
            DataType::List(_) => todo!(),
            DataType::Set(_) => todo!(),
            DataType::Array(_) => todo!(),
            DataType::Date(_) => Ok(field_type::FieldType::Date),
            DataType::Time(_) => Ok(field_type::FieldType::Time),
            DataType::DateTime(_) => Ok(field_type::FieldType::Datetime),
            DataType::Duration(_) => todo!(),
            DataType::Id(i) => Ok(field_type::FieldType::String),
            DataType::Function(_) => todo!(),
            DataType::Any => todo!(),
        }
    }
}

impl TryFrom<Table> for BQTable {
    type Error = Error;

    fn try_from(table: Table) -> Result<Self> {
        let fields: Vec<TableFieldSchema> = table
            .schema()
            .fields()
            .iter()
            .map(|f| {
                let name = f.name();
                let mode = if f.all_values() == true {
                    String::from("REQUIRED")
                } else {
                    String::from("NULLABLE")
                };
                let bq_type = field_type::FieldType::try_from(f.data_type()).unwrap();
                TableFieldSchema {
                    categories: None,
                    description: None,
                    fields: None,
                    mode: Some(mode),
                    name: name.to_string(),
                    policy_tags: None,
                    r#type: bq_type,
                }
            })
            .collect();

        let table_schema = TableSchema::new(fields);
        Ok(BQTable::new(
            PROJECT_ID,
            DATASET_ID,
            table.path().head().unwrap().to_string().as_str(),
            table_schema,
        ))
    }
}

pub fn test_database() -> Database {
    // Database::test()
    Database::new(DB.into(), Database::test_tables()).expect("Database")
}

#[cfg(test)]
mod tests {

    use std::{collections::HashMap, fmt::format};

    use gcp_bigquery_client::{
        model::table_data_insert_all_request_rows::TableDataInsertAllRequestRows,
        table::ListOptions,
    };
    use serde_json::json;

    use crate::dialect_translation::bigquery::BigQueryTranslator;

    use super::*;

    #[tokio::test]
    async fn test_table_list() {
        println!("Connecting to a mocked server");

        let google_auth = GoogleAuthMock::start().await;
        google_auth.mock_token(1).await;

        let google_config = dummy_configuration(&google_auth.uri());
        println!("Write google configuration to file.");
        let temp_file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            temp_file.path(),
            serde_json::to_string_pretty(&google_config).unwrap(),
        )
        .unwrap();

        let client = gcp_bigquery_client::client_builder::ClientBuilder::new()
            .with_auth_base_url(google_auth.uri())
            // Url of the BigQuery emulator docker image.
            .with_v2_base_url("http://localhost:9050".to_string())
            .build_from_service_account_key_file(temp_file.path().to_str().unwrap())
            .await
            .unwrap();

        let table_api = client.table();
        let list_tabs = table_api
            .list(PROJECT_ID, DATASET_ID, ListOptions::default())
            .await
            .ok();
        if let Some(tabs) = list_tabs {
            let tables_as_str: Vec<String> = tabs
            .tables
            .unwrap_or_default()
            .into_iter()
            .map(|t| t.table_reference.table_id)
            .collect();
            println!("{:?}", tables_as_str);
        }
        
    }

    // #[tokio::test]
    // async fn test_delete_all_tables() {
    //     println!("Connecting to a mocked server");

    //     let google_auth = GoogleAuthMock::start().await;
    //     google_auth.mock_token(1).await;

    //     let google_config = dummy_configuration(&google_auth.uri());
    //     println!("Write google configuration to file.");
    //     let temp_file = tempfile::NamedTempFile::new().unwrap();
    //     std::fs::write(
    //         temp_file.path(),
    //         serde_json::to_string_pretty(&google_config).unwrap(),
    //     )
    //     .unwrap();

    //     let client = gcp_bigquery_client::client_builder::ClientBuilder::new()
    //         .with_auth_base_url(google_auth.uri())
    //         // Url of the BigQuery emulator docker image.
    //         .with_v2_base_url("http://localhost:9050".to_string())
    //         .build_from_service_account_key_file(temp_file.path().to_str().unwrap())
    //         .await
    //         .unwrap();

    //     let table_api = client.table();
    //     let list_tabs = table_api
    //         .list(PROJECT_ID, DATASET_ID, ListOptions::default())
    //         .await
    //         .unwrap();
    //     let tables_as_str: Vec<String> = list_tabs
    //         .tables
    //         .unwrap_or_default()
    //         .into_iter()
    //         .map(|t| t.table_reference.table_id)
    //         .collect();

    //     println!("Table to be deleted {:?}", tables_as_str);

    //     for table_name in tables_as_str {
    //         client
    //             .table()
    //             .delete(PROJECT_ID, DATASET_ID, table_name.as_str())
    //             .await
    //             .unwrap();
    //     }
    // }

    // #[test]
    // fn test_client() {
    //     let mut rt = tokio::runtime::Runtime::new().unwrap();

    //     let (auth_server, tmp_file_credentials) = rt.block_on(build_auth()).unwrap();
    //     let client = rt
    //         .block_on(build_client(auth_server.uri(), &tmp_file_credentials))
    //         .unwrap();
    //     let list_tabs = rt
    //         .block_on(
    //             client
    //                 .table()
    //                 .list(PROJECT_ID, DATASET_ID, ListOptions::default()),
    //         )
    //         .unwrap();
    //     let tables_as_str: Vec<String> = list_tabs
    //         .tables
    //         .unwrap_or_default()
    //         .into_iter()
    //         .map(|t| t.table_reference.table_id)
    //         .collect();
    //     println!("{:?}", tables_as_str);
    // }

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
        let datetime =
            chrono::NaiveDateTime::parse_from_str(datetime, "%Y-%m-%dT%H:%M:%S%.f").unwrap();
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
        let time = chrono::NaiveTime::parse_from_str(time, "%H:%M:%S%.f").unwrap();
        println!("Datetime: {:?}", time);
    }

    // #[tokio::test]
    // async fn test_create_table() {
    //     let (auth_server, tmp_file_credentials) = build_auth().await.unwrap();
    //     let client = build_client(auth_server.uri(), &tmp_file_credentials).await.unwrap();

    //     let table_name = "MY TABLE";
    //     let table: Table = TableBuilder::new()
    //         .path([table_name])
    //         .name("aaaa")
    //         .size(10)
    //         .schema(
    //             Schema::empty()
    //                 .with(("f", DataType::float_interval(0.0, 10.0)))
    //                 .with(("z", DataType::text_values(["Foo".into(), "Bar".into()]))) // .with(("x", DataType::integer_interval(0, 100)))
    //                 .with(("y", DataType::optional(DataType::text()))), // .with(("z", DataType::text_values(["Foo".into(), "Bar".into()])))
    //         )
    //         .build();

    //     let bq_table: BQTable = table.try_into().unwrap();
    //     println!("{:?}",bq_table);
    //     let res = (client.table().create(bq_table)).await.unwrap();
    //     println!("ROWS: {:?}", res.num_rows);
    //     // rt.block_on(client
    //     //     .table()
    //     //     .delete(PROJECT_ID, DATASET_ID, table_name)).unwrap();
    // }

    #[tokio::test]
    async fn test_insert_into_table() {
        let (auth_server, tmp_file_credentials) = build_auth().await.unwrap();
        let client = build_client(auth_server.uri(), &tmp_file_credentials)
            .await
            .unwrap();
        let table_api = client.tabledata();
        let table_name = "mytable5";
        let table: Table = TableBuilder::new()
            .path(["dataset1", table_name])
            .name(table_name)
            .size(10)
            .schema(
                Schema::empty()
                    .with(("f", DataType::float_interval(0.0, 10.0)))
                    .with(("z", DataType::text_values(["Foo".into(), "Bar".into()]))) // .with(("x", DataType::integer_interval(0, 100)))
                    .with(("y", DataType::optional(DataType::text()))), // .with(("z", DataType::text_values(["Foo".into(), "Bar".into()])))
            )
            .build();
        let size = 10;
        let mut rng = StdRng::seed_from_u64(1234);
        let mut insert_query = TableDataInsertAllRequest::new();
        let mut rows_for_bq: Vec<TableDataInsertAllRequestRows> = vec![];
        for _ in 1..size {
            let structured: value::Struct = table
                .schema()
                .data_type()
                .generate(&mut rng)
                .try_into()
                .unwrap();
            let keys: Vec<String> = table
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().into())
                .collect();
            let values: Result<Vec<SqlValue>> = structured
                .into_iter()
                .map(|(_, v)| (**v).clone().try_into())
                .collect();
            let values = values.unwrap();
            let map: HashMap<String, SqlValue> = keys.into_iter().zip(values.into_iter()).collect();
            let map_as_json = json!(map);
            println!("{}", map_as_json);
            rows_for_bq.push(TableDataInsertAllRequestRows {
                insert_id: None,
                json: map_as_json,
            });
        }
        insert_query.add_rows(rows_for_bq.clone()).unwrap();
        //
        let res = table_api
            .insert_all(PROJECT_ID, DATASET_ID, table_name, insert_query.clone())
            .await
            .ok();
        println!("{:?}", res)
    }

    #[tokio::test]
    async fn test_insert_structured_rows() {
        let table_name = "mytable5";

        #[derive(Serialize, Debug, Clone, PartialEq, Eq)]
        pub struct Row {
            pub f: String,
            pub z: String,
            pub y: Option<String>,
        }

        //let (auth_server, tmp_file_credentials) = rt.block_on(build_auth()).unwrap();
        let (auth_server, tmp_file_credentials) = build_auth().await.unwrap();
        //let client = rt.block_on(build_client(auth_server.uri(), &tmp_file_credentials)).unwrap();
        let client = build_client(auth_server.uri(), &tmp_file_credentials)
            .await
            .unwrap();

        let mut insert_request = TableDataInsertAllRequest::new();
        //let row_as_json = serde_json::to_string_pretty(&row).expect("json value");
        insert_request
            .add_row(
                None,
                Row {
                    f: "1.3".to_string(),
                    z: "val1".to_string(),
                    y: Some("ljsdncssd".to_string()),
                },
            )
            .unwrap();
        insert_request
            .add_row(
                None,
                Row {
                    f: "2.3".to_string(),
                    z: "val2".to_string(),
                    y: Some("ljc".to_string()),
                },
            )
            .unwrap();
        insert_request
            .add_row(
                None,
                Row {
                    f: "3.3".to_string(),
                    z: "val3".to_string(),
                    y: None,
                },
            )
            .unwrap();
        insert_request
            .add_row(
                None,
                Row {
                    f: "4.3".to_string(),
                    z: "val4".to_string(),
                    y: Some("de".to_string()),
                },
            )
            .unwrap();

        let my_value = Row {
            f: "4.3".to_string(),
            z: "val4".to_string(),
            y: None,
        };
        let json_as_value = serde_json::to_value(my_value).unwrap();
        println!("VALUE: {}", json_as_value);
        let res = client
            .tabledata()
            .insert_all(PROJECT_ID, DATASET_ID, table_name, insert_request)
            .await
            .ok();
        let res_as_json = serde_json::to_string_pretty(&res).expect("json value");
        println!("{}", res_as_json);
    }

    #[tokio::test]
    async fn test_insert_structured_rows_bis() {
        let table_name = "mytable5";

        //let (auth_server, tmp_file_credentials) = rt.block_on(build_auth()).unwrap();
        let (auth_server, tmp_file_credentials) = build_auth().await.unwrap();
        //let client = rt.block_on(build_client(auth_server.uri(), &tmp_file_credentials)).unwrap();
        let client = build_client(auth_server.uri(), &tmp_file_credentials)
            .await
            .unwrap();

        let mut insert_request = TableDataInsertAllRequest::new();
        let rows: Vec<TableDataInsertAllRequestRows> = vec![TableDataInsertAllRequestRows {
            insert_id: None,
            json: json!({"f":8.3, "z":"ahaha1", "y":"sjkd"}),
        }];
        insert_request.add_rows(rows).unwrap();

        let res = client
            .tabledata()
            .insert_all(PROJECT_ID, DATASET_ID, table_name, insert_request)
            .await
            .ok();
        let res_as_json = serde_json::to_string_pretty(&res).expect("json value");
        println!("{}", res_as_json);
    }

    #[test]
    fn database_display() -> Result<()> {
        let mut database = test_database();
        let query = "SELECT * FROM table_1 LIMIT 10";
        println!("\n{query}");
        for row in database.query(query)? {
            println!("{}", row);
        }
        let query = "SELECT * FROM table_2 LIMIT 10";
        println!("\n{query}");
        for row in database.query(query)? {
            println!("{}", row);
        }

        let query = "SELECT * FROM user_table LIMIT 10";
        println!("\n{query}");
        for row in database.query(query)? {
            println!("{}", row);
        }

        let query = "SELECT * FROM large_user_table LIMIT 10";
        println!("\n{query}");
        for row in database.query(query)? {
            println!("{}", row);
        }

        let query = "SELECT * FROM order_table LIMIT 10";
        println!("\n{query}");
        for row in database.query(query)? {
            println!("{}", row);
        }

        let query = "SELECT * FROM item_table LIMIT 10";
        println!("\n{query}");
        for row in database.query(query)? {
            println!("{}", row);
        }

        // TODO: uncomment ones we manage to push MY SPECIAL TABLE
        // let query = r"SELECT * FROM `MY SPECIAL TABLE` LIMIT 10";
        // println!("\n{query}");
        // for row in database.query(query)? {
        //     println!("{}", row);
        // }

        Ok(())
    }
}
