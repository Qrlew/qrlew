use std::fmt;

use super::{implementation, Result};
use crate::data_type::{value::Value, DataType};

/// The list of operators
/// inspired by: https://docs.rs/sqlparser/latest/sqlparser/ast/enum.BinaryOperator.html
/// and mostly: https://docs.rs/polars/latest/polars/prelude/enum.Operator.html
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Function {
    // TODO use directly the data_type function
    // Unary operators, see: https://docs.rs/sqlparser/latest/sqlparser/ast/enum.UnaryOperator.html
    Opposite,
    Not,
    // Binary operator
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
    InList,
    Coalesce,
    // Functions
    Exp,
    Ln,
    Log,
    Abs,
    Sin,
    Cos,
    Sqrt,
    Pow,
    Case,
    Concat(usize),
    CharLength,
    Lower,
    Upper,
    Md5,
    Position,
    Random(usize),
    CastAsText,
    CastAsFloat,
    CastAsInteger,
    CastAsBoolean,
    CastAsDateTime,
    CastAsDate,
    CastAsTime,
    Least,
    Greatest,
    Rtrim,
    Ltrim,
    Substr,
    SubstrWithSize
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Style {
    UnaryOperator,
    BinaryOperator,
    Function,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Arity {
    Unary,
    Nary(usize),
    Varying,
}

impl Function {
    /// Return the style of display
    pub fn style(self) -> Style {
        match self {
            // Unary Operators
            Function::Opposite | Function::Not => Style::UnaryOperator,
            // Binary Operators
            Function::Plus
            | Function::Minus
            | Function::Multiply
            | Function::Divide
            | Function::Modulo
            | Function::StringConcat
            | Function::Gt
            | Function::Lt
            | Function::GtEq
            | Function::LtEq
            | Function::Eq
            | Function::NotEq
            | Function::And
            | Function::Or
            | Function::Xor
            | Function::BitwiseOr
            | Function::BitwiseAnd
            | Function::BitwiseXor
            | Function::InList => Style::BinaryOperator,
            // Zero arg Functions
            Function::Random(_)
            // Unary Functions
            | Function::Exp
            | Function::Ln
            | Function::Log
            | Function::Abs
            | Function::Sin
            | Function::Cos
            | Function::Sqrt
            | Function::CharLength
            | Function::Lower
            | Function::Upper
            | Function::Md5
            | Function::CastAsText
            | Function::CastAsFloat
            | Function::CastAsInteger
            | Function::CastAsBoolean
            | Function::CastAsDateTime
            | Function::CastAsDate
            | Function::CastAsTime
            // Binary Functions
            | Function::Pow
            | Function::Position
            | Function::Least
            | Function::Greatest
            | Function::Coalesce
            | Function::Rtrim
            | Function::Ltrim
            | Function::Substr
            // Ternary Function
            | Function::Case
            | Function::SubstrWithSize
            // Nary Function
            | Function::Concat(_) => Style::Function,
        }
    }

    /// Return the arity of the function
    pub fn arity(self) -> Arity {
        match self {
            // Unary Operators
            Function::Opposite | Function::Not => Arity::Unary,
            // Binary Operators
            Function::Plus
            | Function::Minus
            | Function::Multiply
            | Function::Divide
            | Function::Modulo
            | Function::StringConcat
            | Function::Gt
            | Function::Lt
            | Function::GtEq
            | Function::LtEq
            | Function::Eq
            | Function::NotEq
            | Function::And
            | Function::Or
            | Function::Xor
            | Function::BitwiseOr
            | Function::BitwiseAnd
            | Function::BitwiseXor
            | Function::InList => Arity::Nary(2),
            // Zero arg Functions
            Function::Random(_) => Arity::Nary(0),
            // Unary Functions
            Function::Exp
            | Function::Ln
            | Function::Log
            | Function::Abs
            | Function::Sin
            | Function::Cos
            | Function::Sqrt
            | Function::CharLength
            | Function::Lower
            | Function::Upper
            | Function::Md5
            | Function::CastAsText
            | Function::CastAsFloat
            | Function::CastAsInteger
            | Function::CastAsBoolean
            | Function::CastAsDateTime
            | Function::CastAsDate
            | Function::CastAsTime => Arity::Unary,
            // Binary Function
            Function::Pow
            | Function::Position
            | Function::Least
            | Function::Greatest
            | Function::Coalesce
            | Function::Rtrim
            | Function::Ltrim
            | Function::Substr => Arity::Nary(2),
            // Ternary Function
            Function::Case | Function::SubstrWithSize => Arity::Nary(3),
            // Nary Function
            Function::Concat(_) => Arity::Varying,
        }
    }

    /// Return the function object implementing the function
    pub fn super_image(self, sets: &[DataType]) -> Result<DataType> {
        let set = match self.arity() {
            Arity::Unary => sets.as_ref()[0].clone(),
            Arity::Nary(n) => DataType::structured_from_data_types(&sets[0..n]),
            Arity::Varying => DataType::structured_from_data_types(sets),
        };
        Ok(implementation::function(self).super_image(&set)?)
    }

    /// Return the function object implementing the function
    pub fn value(self, args: &[Value]) -> Result<Value> {
        let arg = match self.arity() {
            Arity::Unary => args.as_ref()[0].clone(),
            Arity::Nary(n) => Value::structured_from_values(&args[0..n]),
            Arity::Varying => Value::structured_from_values(&args),
        };
        Ok(implementation::function(self).value(&arg)?)
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            // Unary Operators
            Function::Opposite => "-",
            Function::Not => "not",
            // Binary Operators
            Function::Plus => "+",
            Function::Minus => "-",
            Function::Multiply => "*",
            Function::Divide => "/",
            Function::Modulo => "%",
            Function::StringConcat => "||",
            Function::Gt => ">",
            Function::Lt => "<",
            Function::GtEq => ">=",
            Function::LtEq => "<=",
            Function::Eq => "=",
            Function::NotEq => "<>",
            Function::And => "and",
            Function::Or => "or",
            Function::Xor => "xor",
            Function::BitwiseOr => "|",
            Function::BitwiseAnd => "&",
            Function::BitwiseXor => "^",
            Function::InList => "in",
            // Zero arg Functions
            Function::Random(_) => "random",
            // Unary Functions
            Function::Exp => "exp",
            Function::Ln => "ln",
            Function::Log => "log",
            Function::Abs => "abs",
            Function::Sin => "sin",
            Function::Cos => "cos",
            Function::Sqrt => "sqrt",
            Function::CharLength => "char_length",
            Function::Lower => "lower",
            Function::Upper => "upper",
            Function::Md5 => "md5",
            Function::CastAsText => "cast_as_text",
            Function::CastAsInteger => "cast_as_integer",
            Function::CastAsFloat => "cast_as_float",
            Function::CastAsBoolean => "cast_as_boolean",
            Function::CastAsDateTime => "cast_as_date_time",
            Function::CastAsDate => "cast_as_date",
            Function::CastAsTime => "cast_as_time",
            // Binary Functions
            Function::Pow => "pow",
            Function::Position => "position",
            Function::Least => "least",
            Function::Greatest => "greatest",
            Function::Coalesce => "coalesce",
            Function::Rtrim => "rtrim",
            Function::Ltrim => "ltrim",
            Function::Substr => "substr",
            // Ternary Functions
            Function::Case => "case",
            Function::SubstrWithSize => "substr",
            // Nary Functions
            Function::Concat(_) => "concat",
        })
    }
}
