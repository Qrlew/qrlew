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

    /// Returns the L1 norm of
    pub fn make_l1_norm(self, vector: &str, bases: Vec<&str>, coordinates: Vec<&str>) -> Relation {
        /// Returns a Relation that compute the L1 norm of the columns in coordinates and
        ///
        /// # Arguments
        ///
        /// * `vector` - peid
        /// * `bases` - A vector of the names of the columns we want to to decompose `vector` on
        /// * `coordinates` -
        ///
        /// # Examples
        /// For the clipping,

        // Build the first Reduce that sum up the coordinates according to `vector` and `bases`
        let mut reduce_builder = Relation::reduce();
        reduce_builder = bases.iter()
            .fold(
                reduce_builder,
                |acc, b| acc.group_by(Expr::col(b.to_string()))
            );
        reduce_builder = reduce_builder.group_by(Expr::col(vector.to_string()));
        reduce_builder = reduce_builder.with_iter(coordinates.iter().map(|x| Expr::sum(Expr::col(x.to_string()))));
        let reduce1: Relation = reduce_builder.input(self).build();
        display(&reduce1);
        println!("SCHEMA: {:?}", reduce1.schema());

        let mut map_builder = Relation::map();
        map_builder = map_builder.with_iter(reduce1.input_fields().iter().map(|f| Expr::abs(Expr::col(f.name()))));
        let map: Relation = map_builder.input(reduce1).build();
        map
        // let mut named_group_by: Vec<(String, Expr)> = vec![];
        // bases.iter()
        //     .for_each(|x| named_group_by.push((x.to_string(), Expr::col(x.to_string()))));
        // named_group_by.push((vector.to_string(), Expr::col(vector)));
        // let mut named_exprs: Vec<(String, Expr)> = named_group_by.clone();
        // coordinates.iter()
        //     .for_each(|c| named_exprs.push((format!("sum_{}", c), Expr::sum(Expr::col(c.to_string())))));
        // let reduce = Relation::from(Reduce::new(
        //     "group_by_vector_and_base".to_string(),
        //     named_exprs.clone(),
        //     named_group_by.iter().map(|(_,x)| x.clone()).collect(),
        //     self.into()
        // ));



        // Build the Map that takes the absolute value of the summation of coordinates
        // let mut map_named_exprs: Vec<(String, Expr)> = named_exprs.iter()
        //     .enumerate()
        //     .map(|(i, (s, x))| if i < bases.len() + 1 {
        //         (s.to_string(), x.clone())
        //     } else {
        //         (format!("normed_{}", s), Expr::abs(x.clone()))
        //     })
        //     .collect();
        // let map = Relation::from(Map::new(
        //     "norm_coordinates".to_string(),
        //     named_exprs.clone(),
        //     None,
        //     vec![],
        //     None,
        //     reduce.into()
        // ));

        // // Build the Reduce that sum up along bases
        // let reduce_named_exprs = map_named_exprs.split_off(bases.len());
        // Relation::from(Reduce::new(
        //     "normalize_coordinates".to_string(),
        //     reduce_named_exprs,
        //     map_named_exprs.iter().map(|(_,x)| x.clone()).collect(),
        //     map.into()
        // ))
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
        relation::{Table, schema::Schema, builder::*}
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

    #[test]
    fn test_make_l1_norm() {
        let database = postgresql::test_database();
        let relations = database.relations();

        //  Table
        let table = relations.get(&["secondary_table".into()])
            .unwrap()
            .as_ref()
            .clone();
        display(&table);
        // let amount_norm = table.make_l1_norm(
        //     "primary_id",
        //     vec!["transaction_name"],
        //     vec!["amount"]
        // );
        // display(&amount_norm);
        let reduce: Relation = Relation::reduce()
            .group_by(Expr::col("primary_id"))
            .with(Expr::sum(Expr::col("primary_id")))
            .with(Expr::sum(Expr::col("amount")))
            .input(table)
            // .with(Expr::count(Expr::col("b")))
            .build();
        display(&reduce);

        //let relation = Relation::try_from(parse("SELECT d AS peid As c, c, a FROM table_1").unwrap().with(&relations)).unwrap();
        //let relation = relation.with_computed_field("peid", expr!(cos(a)));
        //
        // // peid: d, groupby: c, coordinates: sum(a)
        // // let relation = relation.make_l1_norm();
        // // display(&relation);
    }
}