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
    Md5,
    Concat(usize),
    CharLength,
    Lower,
    Position,
    Upper,
    Random(usize),
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Style {
    UnaryOperator,
    BinaryOperator,
    Function,
    Case,
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
            | Function::Md5
            | Function::Lower
            | Function::Upper
            // Binary Functions
            | Function::Pow
            | Function::CharLength
            | Function::Position
            // Nary Function
            | Function::Concat(_) => Style::Function,
            // Case Function
            Function::Case => Style::Case,
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
            | Function::Md5
            | Function::CharLength
            | Function::Lower
            | Function::Upper => Arity::Unary,
            // Binary Function
            Function::Pow | Function::Position => Arity::Nary(2),
            // Ternary Function
            Function::Case => Arity::Nary(3),
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
            // Binary Functions
            Function::Pow => "pow",
            Function::Concat(_) => "concat",
            Function::Position => "position",
            // Ternary Functions
            Function::Case => "case",
            // Nary Functions
            Function::Md5 => "md5",
        })
    }
}
