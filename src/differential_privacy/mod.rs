




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
    fn test_table_protection() {
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