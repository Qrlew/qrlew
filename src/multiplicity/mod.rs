//! # Given a Relation representing a computaion on a sampled table, a table (representing the schema of original dataset) and a weight representing
//!
//! This is experimental and little tested yet.
//!

use itertools::Itertools;
use rand::distributions::weighted;

use crate::{
    builder::{self, Ready, With},
    display::Dot,
    expr::{aggregate, identifier::Identifier, Expr},
    hierarchy::{Hierarchy, Path},
    relation::{Join, Map, Reduce, Relation, Set, Table, Variant as _, Visitor},
    visitor::Acceptor,
    WithIterator,
};
use std::{collections::HashMap, error, fmt, rc::Rc, result};

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
    weight_table: F,
}

impl<F: Fn(&Table) -> RelationWithMultiplicity> MultiplicityVisitor<F> {
    pub fn new(weight_table: F) -> Self {
        MultiplicityVisitor { weight_table }
    }
}

impl<'a, F: Fn(&Table) -> RelationWithMultiplicity> Visitor<'a, RelationWithMultiplicity>
    for MultiplicityVisitor<F>
{
    fn table(&self, table: &'a Table) -> RelationWithMultiplicity {
        (self.weight_table)(table)
    }

    /// take the Map and the weight of the input and generate the RelationWithMultiplicity
    fn map(&self, map: &'a Map, input: RelationWithMultiplicity) -> RelationWithMultiplicity {
        let mew_relation: Relation = Relation::map().with(map.clone()).input(input.0).build();
        // mew_relation.display_dot().unwrap();
        RelationWithMultiplicity(mew_relation, input.1)
    }

    /// take the weight of the input, create a new reduce with modified aggregate expressions -> RelationWithMultiplicity
    fn reduce(
        &self,
        reduce: &'a Reduce,
        input: RelationWithMultiplicity,
    ) -> RelationWithMultiplicity {
        // Create a new reduce using the input

        // the reduce weight would depend on the number of expressions in the group by
        // 0 group by:
        //    The reduce would generate 1 line of results on a sampled and non-sampled dataset
        //    So the RelationWithMultiplicity would have weight of 1
        // 1 group by
        //    the number of rows will depend on unique values of group by col.
        //    the expected count of sampled values would be table size n *

        // let reduce_output_weight = if reduce.group_by.len()!=0 {input.1} else {1.0};

        let reduce: Reduce = Relation::reduce()
            .with(reduce.clone())
            .input(input.0.clone())
            .build();

        // Extract schema field names and associated expressions from the reduce
        let field_aggexpr_map: Vec<(&str, &Expr)> = reduce
            .schema()
            .fields()
            .iter()
            .map(|field| field.name())
            .zip((&reduce.aggregate).iter())
            .collect();

        // Create expressions for a Map on top of the reduce where we scale reduce expressions
        // according to the weight from the imput
        // (str, Expr) to -> Expr (weight * Expr::col(str))
        let new_exprs: Vec<(&str, Expr)> = field_aggexpr_map
            .into_iter()
            .map(|(name, expr)| match &expr {
                Expr::Aggregate(agg) => match agg.aggregate() {
                    aggregate::Aggregate::Count => {
                        (name, Expr::multiply(Expr::val(input.1), Expr::col(name)))
                    }
                    aggregate::Aggregate::Sum => {
                        (name, Expr::multiply(Expr::val(input.1), Expr::col(name)))
                    }
                    _ => (name, Expr::col(name)),
                },
                _ => (name, Expr::col(name)),
            })
            .collect();

        // create a Map wich weights Reduce's expressions based on the epxr type.
        // the Map takes as an imput the reduce.
        let new_map: Relation = Relation::map()
            .with_iter(new_exprs.into_iter())
            .input(Relation::from(reduce))
            .build();

        RelationWithMultiplicity(new_map, 1.0)
    }

    /// take the weight the left (wl), the weight of the right (wr) and create a RelationWithMultiplicity with the Join with weight = wl * wr
    fn join(
        &self,
        join: &'a Join,
        left: RelationWithMultiplicity,
        right: RelationWithMultiplicity,
    ) -> RelationWithMultiplicity {
        let RelationWithMultiplicity(left, left_weight) = left;
        let RelationWithMultiplicity(right, right_weight) = right;

        let left_new_name = left.name().to_string();
        let right_new_name = right.name().to_string();

        // Preserve the schema names of the existing JOIN
        let schema_names: Vec<String> =
            join.schema().iter().map(|f| f.name().to_string()).collect();
        let left_names: Vec<String> = schema_names
            .iter()
            .take(join.left.schema().len())
            .cloned()
            .collect();
        let right_names: Vec<String> = schema_names
            .iter()
            .skip(join.left.schema().len())
            .cloned()
            .collect();

        // map old columns names (from the join) into new column names from the left and right
        let columns_mapping: Hierarchy<Identifier> = join
            .left
            .schema()
            .iter()
            .zip(left.schema().iter())
            .map(|(o, n)| {
                (
                    vec![join.left.name().to_string(), o.name().to_string()],
                    Identifier::from(vec![left_new_name.clone(), n.name().to_string()]),
                )
            })
            .chain(
                join.right
                    .schema()
                    .iter()
                    .zip(right.schema().iter())
                    .map(|(o, n)| {
                        (
                            vec![join.right.name().to_string(), o.name().to_string()],
                            Identifier::from(vec![right_new_name.clone(), n.name().to_string()]),
                        )
                    }),
            )
            .collect();

        // build the output relation
        RelationWithMultiplicity(
            Relation::join()
                .left_names(left_names)
                .right_names(right_names)
                .operator(join.operator.clone().rename(&columns_mapping))
                .left(left)
                .right(right)
                .build(),
            left_weight * right_weight,
        )
    }

    fn set(
        &self,
        set: &'a Set,
        left: RelationWithMultiplicity,
        right: RelationWithMultiplicity,
    ) -> RelationWithMultiplicity {
        todo!()
    }
}

/// Build a visitor for uniform multiplicity
/// Apply the same weight to all tables
fn uniform_multiplicity_visitor(
    weight: f64,
) -> MultiplicityVisitor<impl Fn(&Table) -> RelationWithMultiplicity> {
    MultiplicityVisitor::new(move |table: &Table| {
        RelationWithMultiplicity(Relation::from(table.clone()), weight)
    })
}

struct TableSamplerVisitor<F: Fn(&Table) -> Relation> {
    table_sampler: F,
}

impl<F: Fn(&Table) -> Relation> TableSamplerVisitor<F> {
    pub fn new(table_sampler: F) -> Self {
        TableSamplerVisitor { table_sampler }
    }
}

impl<'a, F: Fn(&Table) -> Relation> Visitor<'a, Relation> for TableSamplerVisitor<F> {
    fn table(&self, table: &'a Table) -> Relation {
        (self.table_sampler)(table)
    }

    /// take the Map and the weight of the input and generate the RelationWithMultiplicity
    fn map(&self, map: &'a Map, input: Relation) -> Relation {
        Relation::map().with(map.clone()).input(input).build()
    }

    /// take the weight of the input, create a new reduce with modified aggregate expressions -> RelationWithMultiplicity
    fn reduce(&self, reduce: &'a Reduce, input: Relation) -> Relation {
        Relation::reduce().with(reduce.clone()).input(input).build()
    }

    /// take the weight the left (wl), the weight of the right (wr) and create a RelationWithMultiplicity with the Join with weight = wl * wr
    fn join(&self, join: &'a Join, left: Relation, right: Relation) -> Relation {
        let left_new_name = left.name().to_string();
        let right_new_name = right.name().to_string();

        // Preserve the schema names of the existing JOIN
        let schema_names: Vec<String> =
            join.schema().iter().map(|f| f.name().to_string()).collect();
        let left_names: Vec<String> = schema_names
            .iter()
            .take(join.left.schema().len())
            .cloned()
            .collect();
        let right_names: Vec<String> = schema_names
            .iter()
            .skip(join.left.schema().len())
            .cloned()
            .collect();

        // map old columns names (from the join) into new column names from the left and right
        let columns_mapping: Hierarchy<Identifier> = join
            .left
            .schema()
            .iter()
            .zip(left.schema().iter())
            .map(|(o, n)| {
                (
                    vec![join.left.name().to_string(), o.name().to_string()],
                    Identifier::from(vec![left_new_name.clone(), n.name().to_string()]),
                )
            })
            .chain(
                join.right
                    .schema()
                    .iter()
                    .zip(right.schema().iter())
                    .map(|(o, n)| {
                        (
                            vec![join.right.name().to_string(), o.name().to_string()],
                            Identifier::from(vec![right_new_name.clone(), n.name().to_string()]),
                        )
                    }),
            )
            .collect();

        // build the output relation
        Relation::join()
            .left_names(left_names)
            .right_names(right_names)
            .operator(join.operator.clone().rename(&columns_mapping))
            .left(left)
            .right(right)
            .build()
    }

    fn set(&self, set: &'a Set, left: Relation, right: Relation) -> Relation {
        todo!()
    }
}

// apply poisson sampling to each table using the same probability
fn poisson_sampling_table_visitor(proba: f64) -> TableSamplerVisitor<impl Fn(&Table) -> Relation> {
    TableSamplerVisitor::new(move |table: &Table| {
        Relation::from(table.clone()).poisson_sampling(proba)
    })
}

fn sampling_without_replacements_table_visitor(
    rate: f64,
    rate_multiplier: f64,
) -> TableSamplerVisitor<impl Fn(&Table) -> Relation> {
    TableSamplerVisitor::new(move |table: &Table| {
        Relation::from(table.clone()).sampling_without_replacements(rate, rate_multiplier)
    })
}

impl Relation {
    /// it sets the same weight for all tables in the relation
    pub fn uniform_multiplicity_visitor<'a>(&'a self, weight: f64) -> RelationWithMultiplicity {
        self.accept(uniform_multiplicity_visitor(weight))
    }

    pub fn poisson_sampling_table_visitor<'a>(&'a self, proba: f64) -> Relation {
        self.accept(poisson_sampling_table_visitor(proba))
    }

    pub fn sampling_without_replacements_table_visitor<'a>(
        &'a self,
        rate: f64,
        rate_multiplier: f64,
    ) -> Relation {
        self.accept(sampling_without_replacements_table_visitor(
            rate,
            rate_multiplier,
        ))
    }
}

macro_rules! assert_relative_eq {
    ($x:expr, $y:expr, $d:expr) => {
        if !((($x - $y) / $x).abs() <= $d) {
            panic!();
        }
    };
}

// To test the multiplicity propagation we would like to compare results of
// a complex query on dataset with those of the same query
// but issued from a RelationWithMultiplicity executed on a sampled dataset.

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::{
        ast,
        data_type::{self, value::List},
        display::Dot,
        io::{postgresql, Database},
        sql::parse,
    };

    use colored::Colorize;
    use itertools::Itertools;

    const TOLLERANCE: f64 = 0.03;
    const EXPERIMENTS: u32 = 100;

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

    // compute errors from results and from estimate
    fn errors(real: &Vec<f64>, estimate: &Vec<f64>) -> Vec<f64> {
        real.iter()
            .zip(estimate.iter())
            .map(|(source, sampled)| (source - sampled) / source)
            .collect()
    }

    // halper function that repeats the sampling multiple
    // times and it returns an average of the results.
    // it assumes that results are numeric single raw.
    fn collect_results_from_many_samples(
        relation_to_sample: &Relation,
        proba: f64,
        n_experiments: u32,
        use_possin_sampling: bool,
        tolerance: f64,
    ) {
        let mut database = postgresql::test_database();

        print!("\nFROM WEIGHTED RELATION partial reslts:\n");
        let mut res_holder: Vec<Vec<f64>> = vec![];
        for _ in 0..n_experiments {
            let sampled_relation = if use_possin_sampling {
                relation_to_sample.poisson_sampling_table_visitor(proba)
            } else {
                relation_to_sample.sampling_without_replacements_table_visitor(proba, 2.0)
            };
            let weighted_sampled_relation =
                sampled_relation.uniform_multiplicity_visitor(1.0 / proba);
            let query_weighted_relation: &str =
                &ast::Query::from(&(weighted_sampled_relation.0)).to_string();
            let res = database.query(query_weighted_relation).unwrap();
            assert!(res.len() == 1);
            let float_res: Vec<f64> = res[0]
                .iter()
                .filter_map(|f| f.to_string().parse::<f64>().ok())
                .collect();
            // print!("{:?}\n", float_res);
            res_holder.extend([float_res].to_vec().into_iter());
        }
        let num_rows = res_holder.len();
        let num_cols = res_holder[0].len();

        let avg_res: Vec<f64> = (0..num_cols)
            .map(|col| {
                let sum: f64 = res_holder.iter().map(|row| row[col]).sum();
                sum / num_rows as f64
            })
            .collect();

        let query_unsampled_relation: &str = &ast::Query::from(relation_to_sample).to_string();
        let results_unsampled = database.query(query_unsampled_relation).unwrap();
        let source_res: Vec<f64> = results_unsampled[0]
            .iter()
            .filter_map(|f| f.to_string().parse::<f64>().ok())
            .collect();

        let errs: Vec<f64> = errors(&source_res, &avg_res);

        println!("\n{}", format!("{query_unsampled_relation}").yellow());
        print!("\nFROM WEIGHTED RELATION:\n{:?}", avg_res);
        print!("\nFROM SOURCE RELATION:\n{:?}", source_res);
        print!("\nErrors:\n{:?}", errs);

        for err in errs {
            assert!(err.abs() < tolerance)
        }

        //for displaying purposes
        // relation_to_sample.display_dot().unwrap();
        // let sampled_relation = if use_possin_sampling {
        //     relation_to_sample.poisson_sampling_table_visitor(proba)
        // } else {
        //     // use a relatively safe rate_multiplier (2.0)
        //     relation_to_sample.sampling_without_replacements_table_visitor(proba, 2.0)
        // };

        // let weighted_sampled_relation= sampled_relation.uniform_multiplicity_visitor(1.0 / proba);
        // weighted_sampled_relation.0.display_dot().unwrap();
        // print!("Final weight: {}\n", weighted_sampled_relation.1);
    }

    #[test]
    fn test_multiplicity_simple_reduce() {
        // MIN() and MAX() not well supported.
        let mut database = postgresql::test_database();
        let relations: Hierarchy<Rc<Relation>> = database.relations();

        let query = "SELECT COUNT(order_id), SUM(price), AVG(price) FROM item_table";

        for weight in [2.0, 4.0, 8.0] {
            let fraction: f64 = 1.0 / weight;
            let relation = Relation::try_from(parse(query).unwrap().with(&relations)).unwrap();
            // with poisson sampling
            collect_results_from_many_samples(&relation, fraction, EXPERIMENTS, true, TOLLERANCE);
            // with sampling without replacements
            collect_results_from_many_samples(&relation, fraction, EXPERIMENTS, false, TOLLERANCE);
        }
    }

    #[test]
    fn test_multiplicity_join_reduce() {
        let mut database = postgresql::test_database();
        let relations: Hierarchy<Rc<Relation>> = database.relations();

        let query = "SELECT COUNT(id), SUM(price), AVG(price) FROM order_table JOIN item_table ON id=order_id";

        for weight in [2.0, 4.0, 8.0] {
            let fraction: f64 = 1.0 / weight;
            let relation = Relation::try_from(parse(query).unwrap().with(&relations)).unwrap();
            // with poisson sampling
            collect_results_from_many_samples(&relation, fraction, EXPERIMENTS, true, TOLLERANCE);
            // with sampling without replacements
            collect_results_from_many_samples(&relation, fraction, EXPERIMENTS, false, TOLLERANCE);
        }
    }

    #[test]
    fn test_multiplicity_reduce_reduce() {
        let mut database = postgresql::test_database();
        let relations: Hierarchy<Rc<Relation>> = database.relations();

        let query = "
        SELECT COUNT(city), SUM(sum_age), AVG(avg_age) 
        FROM (
            SELECT city, COUNT(age) AS count_age, SUM(age) AS sum_age, AVG(age) AS avg_age 
            FROM user_table GROUP BY city
        ) AS subq";

        for weight in [2.0, 4.0, 8.0] {
            let fraction: f64 = 1.0 / weight;
            let relation = Relation::try_from(parse(query).unwrap().with(&relations)).unwrap();
            // with poisson sampling
            collect_results_from_many_samples(&relation, fraction, EXPERIMENTS, true, TOLLERANCE);
            // with sampling without replacements
            collect_results_from_many_samples(&relation, fraction, EXPERIMENTS, false, TOLLERANCE);
        }
    }

    // not sure it makes sense this test.
    #[test]
    fn test_multiplicity_reduce_join_reduce() {
        let mut database = postgresql::test_database();
        let relations: Hierarchy<Rc<Relation>> = database.relations();

        // bug with USING (col) when there are 2 cols named at the same way in the 2 tables I can't use USING to join them
        // There is a bug with this king of query. the bug is related to addressing columns after a JOINs
        // probably a problem with aliasing as well?
        let query = "
        WITH tmp1 AS (select city, name, age from user_table where char_length(name) < 33),
        tmp2 AS (select city, AVG(age) AS avg_age from user_table GROUP BY city)
        SELECT COUNT(name), SUM(age), AVG(age-avg_age) FROM tmp1 JOIN tmp2 ON(tmp1.city=tmp2.city)
        ";

        for weight in [2.0, 4.0, 8.0] {
            let fraction: f64 = 1.0 / weight;
            let relation = Relation::try_from(parse(query).unwrap().with(&relations)).unwrap();
            // with poisson sampling
            collect_results_from_many_samples(&relation, fraction, EXPERIMENTS, true, TOLLERANCE);
            // with sampling without replacements
            collect_results_from_many_samples(&relation, fraction, EXPERIMENTS, false, TOLLERANCE);
        }
    }

    #[test]
    fn test_multiplicity_join_reduce_reduce() {
        let mut database = postgresql::test_database();
        let relations: Hierarchy<Rc<Relation>> = database.relations();

        // 2 reduce after the join
        let query = "
         WITH 
         tmp1 AS (SELECT user_id FROM order_table),
         tmp2 AS (SELECT id, age, city FROM user_table),
         tmp3 AS (SELECT age, city FROM tmp1 JOIN tmp2 ON tmp1.user_id=tmp2.id),
         tmp4 AS (SELECT COUNT(age) AS count_age, SUM(age) AS sum_age, AVG(age) AS avg_age FROM tmp3 GROUP BY city)
         SELECT COUNT(count_age), SUM(sum_age), AVG(avg_age) FROM tmp4
         ";
        for weight in [2.0, 4.0, 8.0] {
            let fraction: f64 = 1.0 / weight;
            let relation = Relation::try_from(parse(query).unwrap().with(&relations)).unwrap();
            // with poisson sampling
            collect_results_from_many_samples(&relation, fraction, EXPERIMENTS, true, TOLLERANCE);
            // with sampling without replacements
            collect_results_from_many_samples(&relation, fraction, EXPERIMENTS, false, TOLLERANCE);
        }
    }

    // TODO
    #[test]
    fn test_multiplicity_reduce_reduce_reduce() {
        let mut database = postgresql::test_database();
        let relations: Hierarchy<Rc<Relation>> = database.relations();

        // TODO try different weights
        let weight: f64 = 2.0;
        let fraction: f64 = 1.0 / weight;
        let use_possin_sampling: bool = true;

        // 3 reduce
        let query = "
        WITH 
        tmp1 AS (
            SELECT id, city, COUNT(age) AS count_age, AVG(age) AS avg_age
            FROM user_table
            GROUP BY id, city
        ),
        tmp2 AS (
            SELECT id, COUNT(city) as count_city, SUM(count_age) as sum_count_age, AVG(avg_age) AS avg_avg_age
            FROM tmp1
            GROUP BY id
        )
        SELECT COUNT(id), SUM(count_city), AVG(sum_count_age), SUM(avg_avg_age), AVG(avg_avg_age) FROM tmp2
        ";

        let relation = Relation::try_from(parse(query).unwrap().with(&relations)).unwrap();
        // collect_results_from_many_samples(&relation, fraction, EXPERIMENTS, use_possin_sampling, TOLLERANCE);

        // let sampled = relation.poisson_sampling_table_visitor(fraction);
        // sampled.display_dot().unwrap();
        // let weighted_sampled_relation= sampled.uniform_multiplicity_visitor(1.0 / fraction);
        // print!("\n{}\n", weighted_sampled_relation.0);
        // // With this query, it seems that weighted_sampled_relation.0.display_dot().unwrap() enters in an infinite
        // // loop. It is very slow also the query execution. probably it generates a huge graph?
        // weighted_sampled_relation.0.display_dot().unwrap();
        // print!("Final weight: {}\n", weighted_sampled_relation.1);
        // let query_from_rel: &str = &ast::Query::from(&(weighted_sampled_relation.0)).to_string();
        // println!("\n{}", format!("{query_from_rel}").yellow());
        // println!("\n{}\n", database
        //         .query(query_from_rel)
        //         .unwrap()
        //         .iter()
        //         .map(ToString::to_string)
        //         .join("\n")
        // );

        // let query_from_rel: &str = &ast::Query::from(&relation).to_string();
        // relation.display_dot().unwrap();
        // println!("\n{}", format!("{query_from_rel}").yellow());
        // println!("\n{}\n", database
        //         .query(query_from_rel)
        //         .unwrap()
        //         .iter()
        //         .map(ToString::to_string)
        //         .join("\n")
        // );
    }

    // TODO
    #[test]
    fn test_multiplicity_reduce_reduce_join_reduce() {
        let mut database = postgresql::test_database();
        let relations: Hierarchy<Rc<Relation>> = database.relations();

        // TODO try different weights
        let weight: f64 = 2.0;
        let fraction: f64 = 1.0 / weight;
        let use_possin_sampling: bool = true;

        // 2 reduce before the join and 1 after
        let query = "
        WITH tmp1 AS (SELECT id, COUNT(user_id) AS count_user_id, AVG(user_id) AS avg_user_id, SUM(user_id) AS sum_user_id FROM order_table GROUP BY id),
        tmp2 AS (SELECT id, COUNT(age) AS count_age, AVG(age) AS avg_age, SUM(age) AS sum_age FROM user_table GROUP BY id),
        tmp3 AS (SELECT count_user_id, count_age, avg_user_id, avg_age, sum_user_id, sum_age FROM tmp1 JOIN tmp2 ON (tmp1.id = tmp2.id))
        SELECT COUNT(count_user_id), COUNT(count_age), AVG(avg_user_id), AVG(avg_age), SUM(sum_user_id), SUM(sum_age) FROM tmp3
        ";

        let relation = Relation::try_from(parse(query).unwrap().with(&relations)).unwrap();
        // collect_results_from_many_samples(&relation, fraction, EXPERIMENTS, use_possin_sampling, TOLLERANCE);

        // relation.display_dot().unwrap();
        // let sampled = relation.poisson_sampling_table_visitor(fraction);

        // let weighted_sampled_relation= sampled.uniform_multiplicity_visitor(1.0 / fraction);
        // weighted_sampled_relation.0.display_dot().unwrap();
        // print!("Final weight: {}\n", weighted_sampled_relation.1);
        // let query_from_rel: &str = &ast::Query::from(&relation).to_string();
        // relation.display_dot().unwrap();
        // println!("\n{}", format!("{query_from_rel}").yellow());
        // println!("\n{}\n", database
        //         .query(query_from_rel)
        //         .unwrap()
        //         .iter()
        //         .map(ToString::to_string)
        //         .join("\n")
        // );
    }
}
