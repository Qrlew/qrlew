use super::{aggregate::Aggregate, function::Function};
use crate::data_type::{
    function::{self, Extended, Optional},
    DataType,
};
use paste::paste;
use rand::rngs::OsRng;
use std::sync::{Arc, Mutex};

macro_rules! function_implementations {
    ([$($unary:ident),*], [$($binary:ident),*], [$($ternary:ident),*], $function:ident, $default:block) => {
        paste! {
            // A (thread local) global map
            thread_local! {
                static FUNCTION_IMPLEMENTATIONS: FunctionImplementations = FunctionImplementations {
                    $([< $unary:snake >]: Arc::new(Optional::new(function::[< $unary:snake >]())),)*
                    $([< $binary:snake >]: Arc::new(Optional::new(function::[< $binary:snake >]())),)*
                    $([< $ternary:snake >]: Arc::new(Optional::new(function::[< $ternary:snake >]())),)*
                };
            }

            /// A struct containing all implementations
            struct FunctionImplementations {
                $(pub [< $unary:snake >]: Arc<dyn function::Function>,)*
                $(pub [< $binary:snake >]: Arc<dyn function::Function>,)*
                $(pub [< $ternary:snake >]: Arc<dyn function::Function>,)*
            }

            /// The object to access implementations
            pub fn function(function: Function) -> Arc<dyn function::Function> {
                match function {
                    $(Function::$unary => FUNCTION_IMPLEMENTATIONS.with(|impls| impls.[< $unary:snake >].clone()),)*
                    $(Function::$binary => FUNCTION_IMPLEMENTATIONS.with(|impls| impls.[< $binary:snake >].clone()),)*
                    $(Function::$ternary => FUNCTION_IMPLEMENTATIONS.with(|impls| impls.[< $ternary:snake >].clone()),)*
                    $function => $default
                }
            }
        }
    };
}

// All functions:
// Unary: Opposite, Not, Exp, Ln, Abs, Sin, Cos, CharLength, Lower, Upper, Md5
// Binary: Plus, Minus, Multiply, Divide, Modulo, StringConcat, Gt, Lt, GtEq, LtEq, Eq, NotEq, And, Or, Xor, BitwiseOr, BitwiseAnd, BitwiseXor, Position, Concat, Greatest, Least, Coalesce
// Ternary: Case, Position
// Nary: Concat
function_implementations!(
    [Opposite, Not, Exp, Ln, Log, Abs, Sin, Cos, Sqrt, Md5],
    [
        Plus,
        Minus,
        Multiply,
        Divide,
        Modulo,
        StringConcat,
        Gt,
        Lt,
        GtEq,
        LtEq,
        Eq,
        NotEq,
        And,
        Or,
        Xor,
        BitwiseOr,
        BitwiseAnd,
        BitwiseXor,
        Pow,
        CharLength,
        Lower,
        Upper,
        InList,
        Least,
        Greatest,
        Coalesce
    ],
    [Case, Position],
    x,
    {
        match x {
            Function::CastAsText => Arc::new(function::cast(DataType::text())),
            Function::CastAsInteger => Arc::new(function::cast(DataType::integer())),
            Function::CastAsFloat => Arc::new(function::cast(DataType::float())),
            Function::CastAsDateTime => Arc::new(function::cast(DataType::date_time())),
            Function::Concat(n) => Arc::new(function::concat(n)),
            Function::Random(n) => Arc::new(function::random(Mutex::new(OsRng))), //TODO change this initialization
            _ => unreachable!(),
        }
    }
);

macro_rules! aggregate_implementations {
    ([$($implementation:ident),*], $aggregate:ident, $default:block) => {
        paste! {
            // A (thread local) global map
            thread_local! {
                static AGGREGATE_IMPLEMENTATIONS: AggregateImplementations = AggregateImplementations {
                    $([< $implementation:snake >]: Arc::new(Optional::new(function::[< $implementation:snake >]())),)*
                };
            }

            /// A struct containing all implementations
            struct AggregateImplementations {
                $(pub [< $implementation:snake >]: Arc<dyn function::Function>,)*
            }

            /// The object to access implementations
            pub fn aggregate(aggregate: Aggregate) -> Arc<dyn function::Function> {
                match aggregate {
                    $(Aggregate::$implementation => AGGREGATE_IMPLEMENTATIONS.with(|impls| impls.[< $implementation:snake >].clone()),)*
                    $aggregate => $default
                }
            }
        }
    };
}

aggregate_implementations!(
    [Min, Max, Median, NUnique, First, Last, Mean, List, Count, Sum, AggGroups, Std, Var],
    x,
    {
        match x {
            Aggregate::Quantile(p) => Arc::new(function::quantile(p)),
            Aggregate::Quantiles(p) => Arc::new(function::quantiles(p.iter().cloned().collect())),
            _ => unreachable!(),
        }
    }
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_implementations() {
        println!("exp = {}", function(Function::Exp));
        println!("plus = {}", function(Function::Plus));
        println!(
            "plus.super_image({}) = {}",
            &(DataType::float() & DataType::float()),
            function(Function::Plus)
                .super_image(&(DataType::float() & DataType::float()))
                .unwrap()
        );
        println!("count = {}", aggregate(Aggregate::Count));
        println!("quantile = {}", aggregate(Aggregate::Quantile(5.0)));
    }
}
