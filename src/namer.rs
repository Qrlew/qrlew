//! # Naming utilities
//!
//! Module dedicated to naming of objects such as Qrlew expressions and relations
//!

use std::{
    collections::{hash_map::DefaultHasher, HashMap, HashSet},
    hash::{Hash, Hasher},
    sync::Mutex,
};

use crate::encoder::{Encoder, BASE_37};

pub const FIELD: &str = "field";
pub const TABLE: &str = "table";
pub const MAP: &str = "map";
pub const REDUCE: &str = "reduce";
pub const JOIN: &str = "join";
pub const SET: &str = "set";
pub const RELATION: &str = "relation";

/// A global shared counter
static COUNTER: Mutex<Option<HashMap<String, usize>>> = Mutex::new(None);

/// A function used to count named objects
fn count<S: Into<String>>(key: S) -> usize {
    *COUNTER
        .lock()
        .unwrap()
        .get_or_insert_with(|| HashMap::new())
        .entry(key.into())
        .and_modify(|count| *count += 1)
        .or_default()
}

/// A function used to hash named objects
fn hash<H: Hash>(content: &H) -> u64 {
    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    hasher.finish()
}

/// A function used to count named objects
pub fn reset() {
    *COUNTER.lock().unwrap() = None;
}

pub fn new_name<S: Into<String>>(prefix: S) -> String {
    let prefix = prefix.into();
    if prefix.len() == 0 {
        format!("{}", count(prefix))
    } else {
        format!("{}_{}", prefix.clone(), count(prefix))
    }
}

pub fn new_id<S: Into<String>>(prefix: S) -> usize {
    count(prefix)
}

pub fn new_name_outside<S: Into<String>, T: Into<String>, H: IntoIterator<Item = T>>(
    prefix: S,
    existing: H,
) -> String {
    let prefix = prefix.into();
    let existing: HashSet<String> = existing.into_iter().map(|name| name.into()).collect();
    (0u64..)
        .map(|i| {
            if prefix.len() == 0 {
                format!("{}", i)
            } else {
                format!("{}_{}", prefix.clone(), i)
            }
        })
        .find(|name| !existing.contains(name))
        .unwrap()
}

pub fn name_from_content<S: Into<String>, H: Hash>(prefix: S, content: &H) -> String {
    format!(
        "{}_{}",
        prefix.into(),
        Encoder::new(BASE_37, 4).encode(hash(content))
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter() {
        println!("# foo = {}", count("foo"));
        println!("# foo = {}", count("foo"));
        println!("# bar = {}", count("bar"));
        println!("# foo = {}", count("foo"));
        println!("# bar = {}", count("bar"));
        println!("# foo = {}", count("foo"));
        println!("# bar = {}", count("bar"));
        println!("# foo = {}", count("foo"));
        println!("# bar = {}", count("bar"));
        println!("# foo = {}", count("foo"));
        println!("COUNTER = {:?}", COUNTER);
    }

    #[test]
    fn test_simple_namer() {
        println!("foo name = {}", new_name("foo"));
        println!("foo name = {}", new_name("foo"));
        for _ in 0..100 {
            new_name("bar");
            new_name("");
        }
        println!("bar name = {}", new_name("bar"));
        println!("foo name = {}", new_name("foo"));
        println!("bar name = {}", new_name("bar"));
        println!("foo name = {}", new_name("foo"));
        println!("bar name = {}", new_name("bar"));
        println!("foo name = {}", new_name("foo"));
        println!("bar name = {}", new_name("bar"));
        println!("foo name = {}", new_name("foo"));
        println!("empty name = {}", new_name(""));
        println!("COUNTER = {:?}", COUNTER);
        reset();
        println!("COUNTER = {:?}", COUNTER);
        println!("bar name = {}", new_name("bar"));
        println!("foo name = {}", new_name("foo"));
        println!("bar name = {}", new_name("bar"));
        println!("foo name = {}", new_name("foo"));
        println!("bar name = {}", new_name("bar"));
        println!("foo name = {}", new_name("foo"));
        println!("bar name = {}", new_name("bar"));
        println!("foo name = {}", new_name("foo"));
        println!("empty name = {}", new_name(""));
        println!("COUNTER = {:?}", COUNTER);
    }

    #[test]
    fn test_content_based_namer() {
        println!("foo name = {}", name_from_content("foo", &"A"));
        println!("foo name = {}", name_from_content("foo", &"B"));
        println!("bar name = {}", name_from_content("bar", &"A"));
        println!("foo name = {}", name_from_content("foo", &"B"));
        println!("bar name = {}", name_from_content("bar", &"B"));
        println!("foo name = {}", name_from_content("foo", &"C"));
        println!("bar name = {}", name_from_content("bar", &"C"));
        println!("foo name = {}", name_from_content("foo", &"D"));
        println!("bar name = {}", name_from_content("bar", &"D"));
        println!("foo name = {}", name_from_content("foo", &"EFG"));
        println!("COUNTER = {:?}", COUNTER);
    }

    #[test]
    fn test_outside_namer() {
        println!("name = {}", new_name_outside("", ["0", "1", "other"]));
        println!(
            "name = {}",
            new_name_outside("foo", ["foo_0", "foo_1", "ok"])
        );
    }
}
