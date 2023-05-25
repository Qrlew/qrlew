#![recursion_limit = "1024"]
pub mod data_type;
pub mod setup;
#[macro_use]
pub mod expr;
pub mod builder;
pub mod debug;
pub mod encoder;
pub mod hierarchy;
pub mod io;
pub mod namer;
pub mod relation;
pub mod sql;
pub mod types;
pub mod visitor;

// TODO we should add and reexport all the object we want to expose to the user of the lib
pub use builder::{Ready, With, WithContext, WithIterator, WithoutContext};
pub use sqlparser;
pub use types::{And, Factor, Or, Term};
