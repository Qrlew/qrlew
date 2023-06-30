//! # Given a Relation representing a computaion on a sampled table, a table (representing the schema of original dataset) and a weight representing 
//! 
//! This is experimental and little tested yet.
//!

use rand::distributions::weighted;

use crate::{
    builder::{self, Ready, With},
    display::Dot,
    expr::{identifier::Identifier, Expr, aggregate},
    hierarchy::{Hierarchy, Path},
    relation::{Join, Map, Reduce, Relation, Set, Table, Variant as _, Visitor},
    visitor::Acceptor,
};
use std::{error, fmt, rc::Rc, result};

#[derive(Debug, Clone)]
pub enum Error {
    Other(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Other(err) => writeln!(f, "{}", err),
        }
    }
}

impl error::Error for Error {}

pub type Result<T> = result::Result<T, Error>;

#[derive(Clone, Debug)]
pub struct RelationWithMultiplicity(Relation, f64);

/// A visitor to compute Relation protection
#[derive(Clone, Debug)]

// It can be a closer as in protect. So that you can accept a wight and apply it to all table without specifing it.
//struct MultiplicityVisitor<'a>(Vec<(&'a Table, f64)>);
struct MultiplicityVisitor<F: Fn(&Table) -> RelationWithMultiplicity> {
    weight_table: F
}

impl<F: Fn(&Table) -> RelationWithMultiplicity> MultiplicityVisitor<F> {
    pub fn new(weight_table: F) -> Self {
        MultiplicityVisitor {weight_table}
    }
}


impl<'a, F: Fn(&Table) -> RelationWithMultiplicity> Visitor<'a, RelationWithMultiplicity> for MultiplicityVisitor<F> {
    fn table(&self, table: &'a Table) -> RelationWithMultiplicity {
        (self.weight_table)(table)
    }

    /// take the Map and the weight of the input and generate the RelationWithMultiplicity
    fn map(&self, map: &'a Map, input: RelationWithMultiplicity) -> RelationWithMultiplicity {
        RelationWithMultiplicity(Relation::from(map.clone()), input.1)
    }

    /// take the weight of the input, create a new reduce with modified aggregate expressions -> RelationWithMultiplicity
    fn reduce(&self, reduce: &'a Reduce, input: RelationWithMultiplicity) -> RelationWithMultiplicity {
        let (schema, exprs) = reduce.clone().schema_exprs();
        // let relation = Relation::from(reduce.clone());
        // print!("{}", &relation);
        // let new_map = relation.map_fields(|_, expr| {
        //     match &expr {
        //         Expr::Aggregate(agg) => {
        //             match agg.aggregate() {
        //                 aggregate::Aggregate::Count => Expr::multiply(Expr::val(1.0/input.1), expr.clone()),
        //                 aggregate::Aggregate::Sum => Expr::multiply(Expr::val(1.0/input.1), expr.clone()),
        //                 _ => expr
        //             }
        //         },
        //         _ => {print!("{expr}"); expr}
        //     }
        // });
        //RelationWithMultiplicity(new_map, 1.0)
        // let expr = &reduce.aggregate[0];


        // let new_map = Relation::map()
        //     .with(new_expr)
        //     .input(reduce.clone().into())
        //     .build();
        // If I constuct a new reduce, the expression 1/w * agg will not be allowed, I would need to add a map on top of the reduce.
        // let new_reduce: Relation = Relation::reduce()
        //     .with((field, new_expr))
        //     .group_by_iter(reduce.clone().group_by.into_iter())
        //     .input(input.0)
        //     .build();

        // RelationWithMultiplicity(new_map, 1.0)
        // let input_weight = input.1;
        // let weighted_aggs = reduce.aggregate.iter()
        // .map(|expr| match expr {
        //     Expr::Aggregate(agg) => {
        //         let match agg.aggregate() {
        //             aggregate::Aggregate::Count => todo!(),
        //             _ => _
        //         }
        //     }
        //     _ => expr
        // });

        // let weighted_reduce: Relation = Relation::reduce()
        //     .input(input.0)
        //     .with(("sum_price", Expr::sum(Expr::col("price"))))
        //     .with_group_by_column("order_id")
        //     .build();

    }

    /// take the weight the left (wl), the weight of the right (wr) and create a RelationWithMultiplicity with the Join with weight = wl * wr
    fn join(&self, join: &'a Join, left: RelationWithMultiplicity, right: RelationWithMultiplicity) -> RelationWithMultiplicity {
        RelationWithMultiplicity(Relation::from(join.clone()), left.1*right.1)
    }

    fn set(&self, set: &'a Set, left: RelationWithMultiplicity, right: RelationWithMultiplicity) -> RelationWithMultiplicity {
        todo!()
    }
}

/// Build a visitor for uniform multiplicity
/// Apply the same weight to all tables
fn uniform_multiplicity_visitor(
    weight: f64
) -> MultiplicityVisitor<impl Fn(&Table) -> RelationWithMultiplicity> {
    MultiplicityVisitor::new(
        move |table: &Table| RelationWithMultiplicity(Relation::from(table.clone()), weight) 
    )
}


impl Relation {
    /// it sets the same weight for all tables in the relation
    pub fn uniform_multiplicity_visitor<'a>(&'a self, weight: f64) -> RelationWithMultiplicity {
        self.accept(uniform_multiplicity_visitor(weight))
    }
}


#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::{
        display::Dot,
        data_type::{value::List, self},
        io::{postgresql, Database},
        sql::parse,
    };

    #[test]
    fn test_table() {
        let mut database = postgresql::test_database();
        let weight: f64 = 2.0;
        let relations = database.relations();
        let table = relations
            .get(&["item_table".into()])
            .unwrap()
            .as_ref()
            .clone();
        let wtable = table.uniform_multiplicity_visitor(weight);
        wtable.0.display_dot().unwrap();
        assert!(wtable.1 == 2.0)
    }

    #[test]
    fn test_map() {
        let mut database = postgresql::test_database();
        let weight: f64 = 2.0;
        let relations = database.relations();
        let query = "SELECT * FROM item_table";
        let relation = Relation::try_from(parse(query).unwrap().with(&relations)).unwrap();
        let wrelation = relation.uniform_multiplicity_visitor(weight);
        wrelation.0.display_dot().unwrap();
        assert!(wrelation.1 == 2.0)
    }
    #[test]
    fn test_reduce() {
        let mut database = postgresql::test_database();
        let weight: f64 = 2.0;
        let relations = database.relations();
        let query = "SELECT COUNT(id) FROM order_table";
        let relation = Relation::try_from(parse(query).unwrap().with(&relations)).unwrap();
        let wrelation = relation.uniform_multiplicity_visitor(weight);
        //relation.display_dot().unwrap();
        wrelation.0.display_dot().unwrap();
        assert!(wrelation.1 == 1.0)
    }

    #[test]
    fn test_joins() {
        let mut database = postgresql::test_database();
        let weight: f64 = 2.0;
        let relations = database.relations();
        let relation = Relation::try_from(
            parse("SELECT * FROM order_table JOIN item_table ON id=order_id")
                .unwrap()
                .with(&relations),
        )
        .unwrap();
        let wrelation: RelationWithMultiplicity = relation.uniform_multiplicity_visitor(weight);
        wrelation.0.display_dot().unwrap();
        assert!(wrelation.1 == 4.0)
    }

    // halper function to create relations with sampling queries.
    // The resulting relations will be used to test RelationWithMultiplicity
    // provide tables which have an `id` column.
    // TODO we would like to implement a transform sample. see transforms.rs
    fn sample_relations(
        sample_number: usize,
        fraction: f64,
        tables: &Vec<Relation>
    ) -> Hierarchy<Rc<Relation>> {

        // let sampled_relations: Vec<Relation> = tables.iter().map(|table| {
        //     let size = table.size().max().unwrap();
        //     let limit = (*size as f64 * fraction) as i32;
        //     let sampled_query = format!("SELECT MD5(CONCAT({}, id)) AS rand_col, * FROM {} ORDER BY rand_col LIMIT {}", sample_number, table.name(), limit);
        //     let input: Hierarchy<Rc<Relation>> = Hierarchy::from([([table.name().clone()], Rc::new(table.clone()))]);
        //     let relation = Relation::try_from(parse(&sampled_query[..]).unwrap().with(&input)).unwrap();
        //     relation
        // }).collect();
        // // let name_map: HashMap<&str, &str> = tables
        // // .iter()
        // // .map(|tab: &Relation| tab.name().clone())
        // // .zip(sampled_relations.iter().map(|rel| rel.name().clone()))
        // // .collect();
        // Hierarchy::from_iter(sampled_relations.iter().map(|rel| ([rel.name()], Rc::new(rel.clone()))))

        // to remove rand_col column use relation.filter_fields(|n| n != "rand_col");

        tables.iter().map(|table| {
                let size = table.size().max().unwrap();
                let limit = (*size as f64 * fraction) as i32;
                let sampled_query = format!("SELECT MD5(CONCAT({}, id)) AS rand_col, * FROM {} ORDER BY rand_col LIMIT {}", sample_number, table.name(), limit);
                let input: Hierarchy<Rc<Relation>> = Hierarchy::from([([table.name().clone()], Rc::new(table.clone()))]);
                let relation = Relation::try_from(parse(&sampled_query[..]).unwrap().with(&input)).unwrap();
                let path = relation.name().path();
                // I can't let a &str created in this scope to get out from it
                (path, Rc::new(relation))
            }).collect()
    }

    #[test]
    fn test_sampled_ds() {
        let mut database = postgresql::test_database();
        let weight: f64 = 2.0;
        let fraction: f64 = 1.0 / weight;
        //let sampled_query = format!("SELECT * FROM user_table WHERE random() > {}", fraction);
        // print!("{}", sampled_query);
        // print!("\n");
        // let mut sampled_results: Vec<f64> = vec![];
        // let sampled_query = "WITH cte1 AS (SELECT MD5(CONCAT(2, id)) AS rand_col, * FROM order_table ORDER BY rand_col LIMIT 10), cte2 AS (SELECT MD5(CONCAT(2, order_id)) AS rand_col, * FROM item_table ORDER BY rand_col LIMIT 10) SELECT * FROM cte1";
        // let my_res = database.query(&sampled_query[..]).unwrap();
        // print!("{:?}", my_res[0]);
        let relations: Hierarchy<Rc<Relation>> = database.relations();
        let table_name = "order_table";
        let tables = vec![relations
            .get(&[table_name.into()])
            .unwrap()
            .as_ref()
            .clone()];
        let sampled_relations = sample_relations(1, fraction, &tables);
        
        let name_map: HashMap<&str, &str> = tables
            .iter()
            .map(|tab: &Relation| tab.name().clone())
            .zip(sampled_relations.iter().map(|(_, rel)| rel.name().clone()))
            .collect();

        println!("{}", sampled_relations);
        println!("{:?}", name_map);

        let rel_name: &str =  match name_map.get(table_name) {
            Some(&name) => name,
            _ => panic!(),
        };

        let query = format!("SELECT * FROM (SELECT COUNT(id) AS count_id FROM {}) AS subtable", rel_name);
        println!("{:?}", query);

        let final_relation_from_sampled = Relation::try_from(parse(&query[..]).unwrap().with(&sampled_relations)).unwrap();
        // apply multiplicity
        // execute
        // TODO: implement LIMIT on the dot
        let weighted_relation = final_relation_from_sampled.uniform_multiplicity_visitor(weight);
        print!("{:?}", weighted_relation);
        final_relation_from_sampled.display_dot().unwrap();

        // let query = format!("SELECT * FROM (SELECT COUNT(id) AS count_id FROM {}) AS subtable", table_name);
        // println!("{:?}", query);
        // let final_relation_from_source = Relation::try_from(parse(&query[..]).unwrap().with(&relations)).unwrap();


        // final_relation_from_source.display_dot().unwrap();
        //let query = format!("SELECT COUNT(*) FROM {}", 
        // the idea: 
        // query with at least 1 redue -> complex relation
        // results <- execute the query on the database
        // use sample_relations() -> sampled_relations: Hierarchy<Rc<Relation>> 
        // relation from query (with input sampled_relations) -> we need to change the source name; 
        // apply multiplicity (it will change the reduce expressions)
        // results <- execute the relaion
        // repeat and average the results and compare them with results on original source.


        // print!("{:?}", relations);
        // let relation = Relation::try_from(parse(sampled_query).unwrap().with(&relations)).unwrap();
        // relation.display_dot().unwrap();
        // for _ in 0..100 { sampled_results.append(database.query(sampled_query).unwrap()[0][0])}
        // for row in database.query(sampled_query).unwrap() {
        //     println!("{}", row[0]);
        // }

        // let table = relations
        //     .get(&["item_table".into()])
        //     .unwrap()
        //     .as_ref()
        //     .clone();

        // // let sampled_relation = sample_relations(1, fraction, &vec![table]);

        // let n = 1;
        // let sampled_query = format!("SELECT id, user_id, description FROM (SELECT MD5(CONCAT({}, id)) AS rand_col, * FROM order_table ORDER BY rand_col LIMIT 10) )", n);
        // let relation = Relation::try_from(parse(&sampled_query[..]).unwrap().with(&relations)).unwrap();
        
        // let name = relation.name();
        // let new_rc = Rc::new(relation.clone());
        // let hier = Hierarchy::from([([name], new_rc)]);

        // let new_relations: Hierarchy<Rc<Relation>> = Hierarchy::from([
        //     (vec![relation.name().clone()], Rc::new(relation.clone())),
        // ]);
        
        // let sampled_relation = Relation::try_from(
        //     parse("SELECT * FROM map_lddf").unwrap().with(&new_relations)
        // ).unwrap();
        // sampled_relation.display_dot().unwrap();
        //print!("{:?}", hier);
    }

}
