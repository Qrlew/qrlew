use std::collections::HashMap;
use std::{ops::Deref, rc::Rc};
use itertools::Itertools;
use crate::{
    builder::{Ready, With, WithIterator},
    expr::{aggregate, Aggregate, Expr, Value},
    hierarchy::Hierarchy,
    DataType,
    relation::{Table, Map, Reduce, Join, Set, Relation, Variant as _, field::Field},
    display::Dot,
    protected::PE_ID,
};


impl Field {
    pub fn clipping_value(self, multiplicity: i64) -> f64 {
        println!("Field: {:?}", self);
        todo!()
    }
}

pub fn gaussian_noise(epsilon: f64, delta: f64, sensitivity: f64) -> f64 {
    (2. * (1.25_f64.ln() / delta)).sqrt() * sensitivity / epsilon
}

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
        let protected_relation = Relation::Reduce(self.clone()).force_protect_from_field_paths(
            relations,
            protected_entity
        );

        let multiplicity = 1; // TODO
        println!("{:?}", self.input.schema());
        let (clipping_values, name_sigmas): (Vec<(String, f64)>, Vec<(String, f64)>) = self
            .schema()
            .clone()
            .iter()
            .zip(self.aggregate.into_iter())
            .fold((vec![], vec![]), |(c, s), (f, x)| {
                if let (name, Expr::Aggregate(agg)) = (f.name(), x) {
                    match agg.aggregate() {
                        aggregate::Aggregate::Sum => {
                            let mut c = c;
                            let cvalue = self.input.schema()
                                .field(agg.argument_name().unwrap())
                                .unwrap()
                                .clone()
                                .clipping_value(multiplicity);
                            c.push((agg.argument_name().unwrap().to_string(), cvalue));
                            let mut s = s;
                            s.push(
                                (name.to_string(), gaussian_noise(epsilon, delta, cvalue))
                            );
                            (c, s)
                        }
                        _ => (c, s),
                    }
                } else {
                    (c, s)
                }
            });

        let clipping_values = clipping_values.iter().map(|(n, v)| (n.as_str(), *v)).collect();
        let clipped_relation = protected_relation.clip_aggregates(PE_ID, clipping_values);

        let name_sigmas = name_sigmas.iter().map(|(n, v)| (n.as_str(), *v)).collect();
        clipped_relation.add_gaussian_noise(name_sigmas)
    }

}


impl Relation {
    pub fn dp_compilation<'a>(
        self,
        relations: &'a Hierarchy<Rc<Relation>>,
        protected_entity: &'a [(&'a str, &'a [(&'a str, &'a str, &'a str)], &'a str)],
        epsilon: f64,
        delta: f64
    ) -> Relation {
        match self {
            Relation::Reduce(reduce) => reduce.dp_compilation(relations, protected_entity, epsilon, delta),
            _ => todo!(),
        }
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

    #[test]
    fn test_dp_compilation() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        let table = relations
            .get(&["item_table".into()])
            .unwrap()
            .as_ref()
            .clone();

        // with GROUP BY
        let relation: Relation = Relation::reduce()
            .input(table.clone())
            .with(("sum_price", Expr::sum(Expr::col("price"))))
            .with_group_by_column("order_id")
            .build();


        let epsilon = 1.;
        let delta = 1e-3;
        let relation = relation.dp_compilation(
            &relations,
            &[
                (
                    "item_table",
                    &[
                        ("order_id", "order_table", "id"),
                        ("user_id", "user_table", "id"),
                    ],
                    "name",
                ),
                ("order_table", &[("user_id", "user_table", "id")], "name"),
                ("user_table", &[], "name"),
            ],
            epsilon,
            delta,
        );
        relation.display_dot().unwrap();
    }
}