use super::DataType;
use sqlparser::ast;

/// Based on the FromRelationVisitor implement the From trait
impl From<DataType> for ast::DataType {
    fn from(value: DataType) -> Self {
        match value {
            DataType::Unit(_) => ast::DataType::Varchar(None),
            DataType::Boolean(_) => ast::DataType::Boolean,
            DataType::Integer(_) => ast::DataType::BigInt(None),
            DataType::Enum(e) => ast::DataType::Enum(e.iter().map(|(n, _)| n.clone()).collect()),
            DataType::Float(_) => ast::DataType::Float(None),
            DataType::Text(_) => ast::DataType::Varchar(None),
            DataType::Bytes(_) => ast::DataType::Blob(None),
            DataType::Date(_) => ast::DataType::Date,
            DataType::Time(_) => ast::DataType::Time(None, ast::TimezoneInfo::None),
            DataType::DateTime(_) => ast::DataType::Datetime(None),
            DataType::Optional(o) => ast::DataType::from(o.data_type().clone()),
            _ => todo!(),
        }
    }
}
