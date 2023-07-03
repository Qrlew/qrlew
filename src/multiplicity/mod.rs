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
    visitor::Acceptor, WithIterator,
};
use std::{error, fmt, rc::Rc, result, collections::HashMap};

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
        let mew_map: Relation = Relation::map()
            .with(map.clone())
            .input(input.0)
            .build();
        RelationWithMultiplicity(mew_map, input.1)
    }

    /// take the weight of the input, create a new reduce with modified aggregate expressions -> RelationWithMultiplicity
    fn reduce(&self, reduce: &'a Reduce, input: RelationWithMultiplicity) -> RelationWithMultiplicity {
        // get (str, Expr) from reduce
        let field_aggexpr_map: Vec<(&str, &Expr)> = reduce
        .schema()
        .fields()
        .iter()
        .map(|field| field.name())
        .zip((&reduce.aggregate).iter())
        .collect();

        // map (str, Expr) to -> Expr (1/weight * Expr::col(str))
        let new_exprs: Vec<(&str, Expr)> = field_aggexpr_map.into_iter().map(|(name, expr)| {
            match &expr {
                Expr::Aggregate(agg) => {
                    match agg.aggregate() {
                        aggregate::Aggregate::Count => (name, Expr::multiply(Expr::val(input.1), Expr::col(name))),
                        aggregate::Aggregate::Sum => (name, Expr::multiply(Expr::val(input.1), Expr::col(name))),
                        _ => (name, Expr::col(name))
                    }
                },
                _ => (name, Expr::col(name))
            }
        }).collect();

        // create a Map wich weights Reduce's expressions based on the epxr type.
        // the Map takes as an imput the reduce.
        let new_map: Relation = Relation::map()
            .with_iter(new_exprs.into_iter())
            .input(Relation::from(reduce.clone()))
            .build();
        RelationWithMultiplicity(new_map, 1.0)
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
        ast,
    };


    use colored::Colorize;
    use itertools::Itertools;

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

        // When it does not work. ok now it works
        //let query = "SELECT * FROM (SELECT COUNT(id) FROM order_table) AS temp";
        
        // When it works
        let query = "SELECT COUNT(id) FROM order_table";
        let relation = Relation::try_from(parse(query).unwrap().with(&relations)).unwrap();
        let wrelation = relation.uniform_multiplicity_visitor(weight);
        print!("{:?}", wrelation);
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

        tables.iter().map(|table| {
                let size = table.size().max().unwrap();
                let limit = (*size as f64 * fraction) as i32;
                let sampled_query = format!("SELECT MD5(CONCAT({}, id)) AS rand_col, * FROM {} ORDER BY rand_col LIMIT {}", sample_number, table.name(), limit);
                let input: Hierarchy<Rc<Relation>> = Hierarchy::from([([table.name().clone()], Rc::new(table.clone()))]);
                let relation = Relation::try_from(parse(&sampled_query[..]).unwrap().with(&input)).unwrap();
                let path = relation.name().path();
                (path, Rc::new(relation))
            }).collect()
    }

    #[test]
    fn test_sampled_ds() {
        let mut database = postgresql::test_database();
        let weight: f64 = 2.0;
        let fraction: f64 = 1.0 / weight;

        let relations: Hierarchy<Rc<Relation>> = database.relations();

        // Query on the sampled relataion. Propagate weights
        // TODO: sampling should be a Relation transform. 
        let table_name = "order_table";
        let tables = vec![relations
            .get(&[table_name.into()])
            .unwrap()
            .as_ref()
            .clone()];
        // sampled_relations
        let sampled_relations = sample_relations(1, fraction, &tables);
        // table name mapping.
        let name_map: HashMap<&str, &str> = tables
            .iter()
            .map(|tab: &Relation| tab.name().clone())
            .zip(sampled_relations.iter().map(|(_, rel)| rel.name().clone()))
            .collect();
        // table name for the sampled relation 
        let rel_name: &str =  match name_map.get(table_name) {
            Some(&name) => name,
            _ => panic!(),
        };
        let sampled_query = format!("SELECT * FROM (SELECT COUNT(id) AS count_id FROM {}) AS subtable", rel_name);
        let relation_from_sampled_query = Relation::try_from(parse(&sampled_query[..]).unwrap().with(&sampled_relations)).unwrap();
        let weighted_relation_from_sampled_query = relation_from_sampled_query.uniform_multiplicity_visitor(weight);
        weighted_relation_from_sampled_query.0.display_dot().unwrap();
        let query_from_weighted_relation: &str = &ast::Query::from(&(weighted_relation_from_sampled_query.0)).to_string();
        println!(
            "On Sampled: \n{}\n{}",
            format!("{query_from_weighted_relation}").yellow(),
            database
                .query(query_from_weighted_relation)
                .unwrap()
                .iter()
                .map(ToString::to_string)
                .join("\n")
        );

        // Query witout sampling.
        let source_query = format!("SELECT * FROM (SELECT COUNT(id) AS count_id FROM {}) AS subtable", table_name); 
        let source_relation =  Relation::try_from(parse(&source_query[..]).unwrap().with(&relations)).unwrap();
        source_relation.display_dot().unwrap();
        let query_source_relation: &str = &ast::Query::from(&(source_relation)).to_string();
        println!(
            "On Source: \n{}\n{}",
            format!("{query_source_relation}").yellow(),
            database
                .query(query_source_relation)
                .unwrap()
                .iter()
                .map(ToString::to_string)
                .join("\n")
        );
    }
}
