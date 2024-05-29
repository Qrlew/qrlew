//! A simple DSL to write expressions

// https://veykril.github.io/tlborm/introduction.html
// https://stackoverflow.com/questions/36721733/is-there-a-way-to-pattern-match-infix-operations-with-precedence-in-rust-macros
// Macro DSL for exprs
#![allow(unused)]
macro_rules! expr {
    // Process functions
    (@expf [$($f:tt)*][$([$($x:tt)*])*]) => {$($f)*($(expr!(@exp+ $($x)*)),*)};
    (@expf [$($f:tt)*][$([$($x:tt)*])*][$($y:tt)*]) => {expr!(@expf [$($f)*][$([$($x)*])*[$($y)*]])}; // Consume until the op
    (@expf [$($f:tt)*][$([$($x:tt)*])*][$($y:tt)*] , $($t:tt)*) => {expr!(@expf [$($f)*][$([$($x)*])*[$($y)*]][] $($t)*)}; // Consume until the op
    (@expf [$($f:tt)*][$([$($x:tt)*])*][$($y:tt)*] $h:tt $($t:tt)*) => {expr!(@expf [$($f)*][$([$($x)*])*][$($y)*$h] $($t)*)}; // Consume the tokens until we find the right op
    (@expf [$($f:tt)*] $($t:tt)*) => {expr!(@expf [$($f)*][][] $($t)*)}; // Start consuming tokens
    // Look for terminal nodes
    (@expt $fun:ident($($t:tt)*)) => {expr!(@expf [Expr::$fun] $($t)*)};
    // (@expt $fun:ident($($t:tt)*)) => {Expr::$fun(expr!($($t)*))};
    (@expt $col:ident) => {Expr::col(stringify!($col))};
    (@expt $val:literal) => {Expr::val($val)};
    (@expt ($($t:tt)*)) => {(expr!($($t)*))};
    // Look for /
    (@exp/ [$($x:tt)*]) => {expr!(@expt $($x)*)}; // We are done, look for lower priority ops
    (@exp/ [$($x:tt)*] / $($t:tt)*) => {Expr::divide(expr!(@expt $($x)*), expr!(@exp/ $($t)*))}; // Consume until the op
    (@exp/ [$($x:tt)*] $h:tt $($t:tt)*) => {expr!(@exp/ [$($x)* $h] $($t)*)}; // Consume the tokens until we find the right op
    (@exp/ $($t:tt)*) => {expr!(@exp/ [] $($t)*)}; // Start consuming tokens
    // Look for *
    (@exp* [$($x:tt)*]) => {expr!(@exp/ $($x)*)}; // We are done, look for lower priority ops
    (@exp* [$($x:tt)*] * $($t:tt)*) => {Expr::multiply(expr!(@exp/ $($x)*), expr!(@exp* $($t)*))}; // Consume until the op
    (@exp* [$($x:tt)*] $h:tt $($t:tt)*) => {expr!(@exp* [$($x)* $h] $($t)*)}; // Consume the tokens until we find the right op
    (@exp* $($t:tt)*) => {expr!(@exp* [] $($t)*)}; // Start consuming tokens
    // Look for -
    (@exp- [$($x:tt)*]) => {expr!(@exp* $($x)*)}; // We are done, look for lower priority ops
    (@exp- [$($x:tt)*] - $($t:tt)*) => {Expr::minus(expr!(@exp* $($x)*), expr!(@exp- $($t)*))}; // Consume until the op
    (@exp- [$($x:tt)*] $h:tt $($t:tt)*) => {expr!(@exp- [$($x)* $h] $($t)*)}; // Consume the tokens until we find the right op
    (@exp- $($t:tt)*) => {expr!(@exp- [] $($t)*)}; // Start consuming tokens
    // Look for +
    (@exp+ [$($x:tt)*]) => {expr!(@exp- $($x)*)}; // We are done, look for lower priority ops
    (@exp+ [$($x:tt)*] + $($t:tt)*) => {Expr::plus(expr!(@exp- $($x)*), expr!(@exp+ $($t)*))}; // Consume until the op
    (@exp+ [$($x:tt)*] $h:tt $($t:tt)*) => {expr!(@exp+ [$($x)* $h] $($t)*)}; // Consume the tokens until we find the right op
    (@exp+ $($t:tt)*) => {expr!(@exp+ [] $($t)*)}; // Start consuming tokens
    // Look for high priority ops first
    ($($t:tt)*) => {expr!(@exp+ $($t)*)};
}
