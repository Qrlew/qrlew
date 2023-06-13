use std::collections::HashMap;
use std::{ops::Deref, rc::Rc};
use itertools::Itertools;
use crate::{
    builder::{Ready, With, WithIterator},
    expr::{aggregate, Aggregate, Expr, Value},
    hierarchy::Hierarchy,
    DataType,
    relation::{Table, Map, Reduce, Join, Set, Relation, Variant as _},
    display::Dot
};

/* Reduce
 */

 impl Reduce {

    pub fn dp_compilation<'a>(
        self,
        relations: &'a Hierarchy<Rc<Relation>>,
        protected_entity: &'a [(&'a str, &'a [(&'a str, &'a str, &'a str)], &'a str)],
        epsilon: f64,
        delta: f64
) -> Relation {
        // fn (Reduce, epsilon, delta) -> Relation
        // 0. protection
        // 1. Recupérer les intervals des aggs
        // 2. Pour chaque colonne, c = max(abs(min), abs(max)) * 1
        // 3. clipping avec un c par colonne
        // 4. ajout de bruit avec sigma(c, epsilon, delta) par col
        let protected_relation = Relation::Reduce(self).force_protect_from_field_paths(
            relations,
            protected_entity
        );


        todo!()
    }

 }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Relation,
        display::Dot,
        io::{postgresql, Database},
        relation::{Variant as _},
        sql::parse,
        builder::With,
    };
    use colored::Colorize;
    use itertools::Itertools;
    use sqlparser::ast;

    #[test]
    fn test_table_with_noise() {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        // // CReate a relation to add noise to
        // let relation = Relation::try_from(
        //     parse("SELECT sum(price) FROM item_table GROUP BY order_id")
        //         .unwrap()
        //         .with(&relations),
        // )
        // .unwrap();
        // println!("Schema = {}", relation.schema());
        // relation.display_dot().unwrap();

        // Add noise directly
        for row in database.query("SELECT random(), sum(price) FROM item_table GROUP BY order_id").unwrap() {
            println!("Row = {row}");
        }
    }
}