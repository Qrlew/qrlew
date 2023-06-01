//! A few transforms for relations
//! 

use super::{Relation, display, Field, Variant as _};
use crate::{
    expr::Expr,
    builder::{With, Ready, WithIterator},
};

impl Relation {
    pub fn with_computed_field(self, name: &str, expr: Expr) -> Relation {
        Relation::map()
            .with((name, expr))
            .with_iter(self.schema().iter().map(|f| (f.name(), Expr::col(f.name()))))
            .input(self)
            .build()
    }

    pub fn filter_fields<F: Fn(&Field) -> bool>(self, predicate:F) -> Relation {
        Relation::map()
            .with_iter(self.schema().iter().filter_map(|f| if predicate(f) {Some((f.name(), Expr::col(f.name())))} else {None}))
            .input(self)
            .build()
    }

    pub fn filter_field_names<F: Fn(&str) -> bool>(self, predicate:F) -> Relation {
        self.filter_fields(|f| predicate(f.name()))
    }
}

impl With<(&str, Expr)> for Relation {
    fn with(self, (name, expr): (&str, Expr)) -> Self {
        self.with_computed_field(name, expr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        data_type::DataType,
        sql::parse,
        io::{Database, postgresql},
    };

    #[test]
    fn test_with_computed_field() {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        let table = relations.get(&["table_1".into()]).unwrap().as_ref().clone();
        let relation = Relation::try_from(parse("SELECT * FROM table_1").unwrap().with(&relations)).unwrap();
        // Table
        assert!(table.schema()[0].name()!="peid");
        let table = table.with_computed_field("peid", expr!(a+b));
        assert!(table.schema()[0].name()=="peid");
        // Relation
        assert!(relation.schema()[0].name()!="peid");
        let relation = relation.with_computed_field("peid", expr!(cos(a)));
        assert!(relation.schema()[0].name()=="peid");
    }

    #[ignore]
    #[test]
    fn test_filter_field_names() {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        let relation = Relation::try_from(parse("SELECT * FROM table_1").unwrap().with(&relations)).unwrap();
        let relation = relation.with_computed_field("peid", expr!(cos(a)));
        display(&relation);
        let relation = relation.filter_field_names(|n| n!="peid");
        display(&relation);
    }
}