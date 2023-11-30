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
    Sign,
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
    Pi,
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
    SubstrWithSize,
    Ceil,
    Floor,
    Round,
    Trunc,
    RegexpContains,
    RegexpExtract,
    RegexpReplace,
    Newid,
    Encode,
    Decode,
    Unhex,
    CurrentDate,
    CurrentTime,
    CurrentTimestamp,
    ExtractYear,
    ExtractMonth,
    ExtractDay,
    ExtractHour,
    ExtractMinute,
    ExtractSecond,
    ExtractMicrosecond,
    ExtractMillisecond,
    ExtractDow,
    ExtractWeek,
    Dayname,
    FromUnixtime,
    UnixTimestamp,
    DateFormat,
    Quarter,
    DatetimeDiff,
    Date
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
            | Function::Pi
            | Function::Newid
            | Function::CurrentDate
            | Function::CurrentTime
            | Function::CurrentTimestamp
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
            | Function::Ceil
            | Function::Floor
            | Function::CastAsDate
            | Function::CastAsTime
            | Function::Sign
            | Function::Unhex
            | Function::ExtractYear
            | Function::ExtractMonth
            | Function::ExtractDay
            | Function::ExtractHour
            | Function::ExtractMinute
            | Function::ExtractSecond
            | Function::ExtractMicrosecond
            | Function::ExtractMillisecond
            | Function::ExtractDow
            | Function::ExtractWeek
            | Function::Dayname
            | Function::UnixTimestamp
            | Function::Quarter
            | Function::Date
            // Binary Functions
            | Function::Pow
            | Function::Position
            | Function::Least
            | Function::Greatest
            | Function::Coalesce
            | Function::Rtrim
            | Function::Ltrim
            | Function::Substr
            | Function::Round
            | Function::Trunc
            | Function::RegexpContains
            | Function::Encode
            | Function::Decode
            | Function::FromUnixtime
            | Function::DateFormat
            // Ternary Function
            | Function::Case
            | Function::SubstrWithSize
            | Function::RegexpExtract
            | Function::RegexpReplace
            | Function::DatetimeDiff
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
            | Function::InList
            | Function::Encode
            | Function::Decode => Arity::Nary(2),
            // Zero arg Functions
            Function::Random(_)
            | Function::Pi
            | Function::Newid
            | Function::CurrentDate
            | Function::CurrentTime
            | Function::CurrentTimestamp => Arity::Nary(0),
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
            | Function::CastAsTime
            | Function::Ceil
            | Function::Floor
            | Function::Sign
            | Function::Unhex
            | Function::ExtractYear
            | Function::ExtractMonth
            | Function::ExtractDay
            | Function::ExtractHour
            | Function::ExtractMinute
            | Function::ExtractSecond
            | Function::ExtractMicrosecond
            | Function::ExtractMillisecond
            | Function::ExtractDow
            | Function::ExtractWeek
            | Function::Dayname
            | Function::UnixTimestamp
            | Function::Quarter
            | Function::Date => Arity::Unary,
            // Binary Function
            Function::Pow
            | Function::Position
            | Function::Least
            | Function::Greatest
            | Function::Coalesce
            | Function::Rtrim
            | Function::Ltrim
            | Function::Substr
            | Function::Round
            | Function::Trunc
            | Function::RegexpContains
            | Function::FromUnixtime
            | Function::DateFormat => {
                Arity::Nary(2)
            }
            // Ternary Function
            Function::Case | Function::SubstrWithSize | Function::RegexpReplace | Function::DatetimeDiff => Arity::Nary(3),
            // Quaternary Function
            Function::RegexpExtract => Arity::Nary(4),
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
            Function::Pi => "pi",
            Function::Newid => "newid",
            Function::CurrentDate => "current_date",
            Function::CurrentTime => "current_time",
            Function::CurrentTimestamp => "current_timestamp",
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
            Function::Ceil => "ceil",
            Function::Floor => "floor",
            Function::CastAsDate => "cast_as_date",
            Function::CastAsTime => "cast_as_time",
            Function::Sign => "sign",
            Function::Unhex => "unhex",
            Function::ExtractYear => "extract_year",
            Function::ExtractMonth => "extract_month",
            Function::ExtractDay => "extract_day",
            Function::ExtractHour => "extract_hour",
            Function::ExtractMinute => "extract_minute",
            Function::ExtractSecond => "extract_second",
            Function::ExtractMicrosecond => "extract_microsecond",
            Function::ExtractMillisecond => "extract_millisecond",
            Function::ExtractDow => "extract_dow",
            Function::ExtractWeek => "extract_week",
            Function::Dayname => "dayname",
            Function::UnixTimestamp => "unix_timestamp",
            Function::Quarter => "quarter",
            Function::Date => "date",
            // Binary Functions
            Function::Pow => "pow",
            Function::Position => "position",
            Function::Least => "least",
            Function::Greatest => "greatest",
            Function::Coalesce => "coalesce",
            Function::Rtrim => "rtrim",
            Function::Ltrim => "ltrim",
            Function::Substr => "substr",
            Function::Round => "round",
            Function::Trunc => "trunc",
            Function::RegexpContains => "regexp_contains",
            Function::Encode => "encode",
            Function::Decode => "decode",
            Function::FromUnixtime => "from_unixtime",
            Function::DateFormat => "date_format",
            // Ternary Functions
            Function::Case => "case",
            Function::SubstrWithSize => "substr",
            Function::RegexpExtract => "regexp_extract",
            Function::RegexpReplace => "regexp_replace",
            Function::DatetimeDiff => "datetime_diff",
            // Nary Functions
            Function::Concat(_) => "concat",
        })
    }
}
