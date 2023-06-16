//! # Debugging utilities
//!
// For debugging purpose
thread_local! {
    pub static DEPTH: std::cell::RefCell<u32>  = std::cell::RefCell::new(0);
}

pub fn debug<T, F: Fn() -> T>(name: &str, code: F) -> T {
    DEPTH.with(|depth| depth.replace_with(|old| *old + 1));
    println!(
        "Enterring {}, depth {}",
        name,
        DEPTH.with(|depth| depth.borrow().clone())
    );
    let result = code();
    DEPTH.with(|depth| depth.replace_with(|old| *old - 1));
    return result;
}
