use env_logger::Builder;
use log::LevelFilter;
pub use log::{debug, info, warn};

pub fn init() {
    // Init the logger
    let mut builder = Builder::from_default_env();
    builder.filter(None, LevelFilter::Info).init();
}
