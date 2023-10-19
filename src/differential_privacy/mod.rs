//! # Methods to transform `Relation`s into differentially private ones
//!
//! This is experimental and little tested yet.
//!

pub mod aggregates;
pub mod budget;
pub mod group_by;
pub mod private_query;

use crate::{
    builder::With,
    differential_privacy::private_query::PrivateQuery,
    expr,
    protection,
    relation::{rewriting, Reduce, Relation},
    Ready,
};
use std::{error, fmt, ops::Deref, result};

#[derive(Debug, PartialEq, Clone)]
pub enum Error {
    InvalidRelation(String),
    DPCompilationError(String),
    GroupingKeysError(String),
    Other(String),
}

impl Error {
    pub fn invalid_relation(relation: impl fmt::Display) -> Error {
        Error::InvalidRelation(format!("{relation} is invalid"))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidRelation(relation) => writeln!(f, "{relation} invalid."),
            Error::DPCompilationError(desc) => writeln!(f, "DPCompilationError: {}", desc),
            Error::GroupingKeysError(desc) => writeln!(f, "GroupingKeysError: {}", desc),
            Error::Other(err) => writeln!(f, "{}", err),
        }
    }
}

impl From<expr::Error> for Error {
    fn from(err: expr::Error) -> Self {
        Error::Other(err.to_string())
    }
}
impl From<rewriting::Error> for Error {
    fn from(err: rewriting::Error) -> Self {
        Error::Other(err.to_string())
    }
}
impl From<protection::Error> for Error {
    fn from(err: protection::Error) -> Self {
        Error::Other(err.to_string())
    }
}

impl error::Error for Error {}
pub type Result<T> = result::Result<T, Error>;

/// A DP Relation
#[derive(Clone, Debug)]
pub struct DPRelation {
    relation: Relation,
    private_query: PrivateQuery,
}

impl From<DPRelation> for Relation {
    fn from(value: DPRelation) -> Self {
        value.relation
    }
}

impl DPRelation {
    pub fn new(relation: Relation, private_query: PrivateQuery) -> Self {
        DPRelation {
            relation,
            private_query,
        }
    }

    pub fn relation(&self) -> &Relation {
        &self.relation
    }

    pub fn private_query(&self) -> &PrivateQuery {
        &self.private_query
    }
}

impl Deref for DPRelation {
    type Target = Relation;

    fn deref(&self) -> &Self::Target {
        &self.relation
    }
}

impl From<DPRelation> for (Relation, PrivateQuery) {
    fn from(value: DPRelation) -> Self {
        (value.relation, value.private_query)
    }
}

impl From<(Relation, PrivateQuery)> for DPRelation {
    fn from(value: (Relation, PrivateQuery)) -> Self {
        DPRelation::new(value.0, value.1)
    }
}

impl Reduce {
    /// Compiles a `Reduce` into DP:
    ///     - Protect the grouping keys
    ///     - Add noise on the aggregations
    pub fn differential_privacy(
        self,
        epsilon: f64,
        delta: f64,
        epsilon_tau_thresholding: f64,
        delta_tau_thresholding: f64,
    ) -> Result<DPRelation> {
        let mut private_query = PrivateQuery::null();

        // DP compile group by
        let reduce_with_dp_group_by = if self.group_by().is_empty() {
            self
        } else {
            let (dp_grouping_values, private_query_group_by) = self
                .dp_compile_group_by(epsilon_tau_thresholding, delta_tau_thresholding)?
                .into();
            let input_relation_with_protected_group_by = self
                .input()
                .clone()
                .join_with_grouping_values(dp_grouping_values)?;
            let reduce: Reduce = Reduce::builder()
                .with(self)
                .input(input_relation_with_protected_group_by)
                .build();
            private_query = private_query.compose(private_query_group_by);
            reduce
        };

        // DP compile aggregates
        let (dp_relation, private_query_agg) = reduce_with_dp_group_by
            .differential_privacy_aggregates(epsilon, delta)?
            .into();
        private_query = private_query.compose(private_query_agg);
        Ok((dp_relation, private_query).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ast,
        builder::With,
        data_type::{DataType, DataTyped, Variant as _},
        display::Dot,
        expr::{AggregateColumn, Expr},
        hierarchy::Hierarchy,
        io::{postgresql, Database},
        relation::{Schema, Variant},
        sql::parse,
        Relation,
        protection::{Protection, Strategy}
    };
    use std::sync::Arc;

    #[test]
    fn test_dp_compile_reduce_without_group_by() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        let table = relations
            .get(&["item_table".to_string()])
            .unwrap()
            .deref()
            .clone();
        let (epsilon, delta) = (1., 1e-3);
        let (epsilon_tau_thresholding, delta_tau_thresholding) = (0.5, 2e-3);

        // protect the inputs
        let protection = Protection::from((
            &relations,
            vec![
                (
                    "item_table",
                    vec![("order_id", "order_table", "id")],
                    "date",
                ),
                ("order_table", vec![], "date"),
            ],
            Strategy::Hard,
        ));
        let pep_table = protection.table(&table.try_into().unwrap()).unwrap();
        let reduce = Reduce::new(
            "my_reduce".to_string(),
            vec![("sum_price".to_string(), AggregateColumn::sum("price"))],
            vec![],
            pep_table.deref().clone().into()
        );
        let relation = Relation::from(reduce.clone());
        relation.display_dot().unwrap();

        let (dp_relation, private_query) = reduce
            .differential_privacy(
                epsilon,
                delta,
                epsilon_tau_thresholding,
                delta_tau_thresholding,
            )
            .unwrap()
            .into();
        dp_relation.display_dot().unwrap();
        assert_eq!(
            private_query,
            PrivateQuery::gaussian_privacy_pars(epsilon, delta, 50.)
        );
        assert!(dp_relation
            .data_type()
            .is_subset_of(&DataType::structured([("sum_price", DataType::float())])));

        let query: &str = &ast::Query::from(&dp_relation).to_string();
        println!("{query}");
        _ = database
            .query(query)
            .unwrap()
            .iter()
            .map(ToString::to_string);

        // // With GROUPBY. Only one column with possible values
        // let relation: Relation = Relation::reduce()
        //     .name("reduce_relation")
        //     .with(("sum_a".to_string(), AggregateColumn::sum("a")))
        //     .with(("b".to_string(), AggregateColumn::first("b")))
        //     .group_by(expr!(b))
        //     .input(table.clone())
        //     .build();
        // let pep_relation =
        //     relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]);
        // let (dp_relation, private_query) = pep_relation
        //     .dp_compile(
        //         epsilon,
        //         delta,
        //         epsilon_tau_thresholding,
        //         delta_tau_thresholding,
        //     )
        //     .unwrap()
        //     .into();
        // dp_relation.display_dot().unwrap();
        // assert_eq!(
        //     private_query,
        //     PrivateQuery::gaussian_privacy_pars(epsilon, delta, 10.)
        // );
        // assert!(dp_relation.data_type().is_subset_of(&DataType::structured([
        //     ("sum_a", DataType::float()),
        //     ("b", DataType::integer_values([1, 2, 5, 6, 7, 8]))
        // ])));

        // // With GROUPBY. Only one column with tau-thresholding values
        // let relation: Relation = Relation::reduce()
        //     .name("reduce_relation")
        //     .with(("sum_a".to_string(), AggregateColumn::sum("a")))
        //     .group_by(expr!(c))
        //     .with(("c".to_string(), AggregateColumn::first("c")))
        //     .input(table.clone())
        //     .build();
        // let pep_relation =
        //     relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]);
        // //pep_relation.display_dot().unwrap();
        // let (dp_relation, private_query) = pep_relation
        //     .dp_compile(
        //         epsilon,
        //         delta,
        //         epsilon_tau_thresholding,
        //         delta_tau_thresholding,
        //     )
        //     .unwrap()
        //     .into();
        // assert_eq!(
        //     private_query,
        //     vec![
        //         PrivateQuery::EpsilonDelta(epsilon_tau_thresholding, delta_tau_thresholding),
        //         PrivateQuery::gaussian_privacy_pars(epsilon, delta, 10.)
        //     ]
        //     .into()
        // );
        // assert!(dp_relation.data_type().is_subset_of(&DataType::structured([
        //     ("sum_a", DataType::float()),
        //     ("c", DataType::integer_range(5..=20))
        // ])));

        // // With GROUPBY. Both tau-thresholding and possible values
        // let relation: Relation = Relation::reduce()
        //     .name("reduce_relation")
        //     .with(("sum_a".to_string(), AggregateColumn::sum("a")))
        //     .group_by(expr!(c))
        //     .group_by(expr!(b))
        //     .with(("b".to_string(), AggregateColumn::first("b")))
        //     .input(table.clone())
        //     .build();
        // let pep_relation =
        //     relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]);
        // //pep_relation.display_dot().unwrap();
        // let (dp_relation, private_query) = pep_relation
        //     .dp_compile(
        //         epsilon,
        //         delta,
        //         epsilon_tau_thresholding,
        //         delta_tau_thresholding,
        //     )
        //     .unwrap()
        //     .into();
        // assert_eq!(
        //     private_query,
        //     vec![
        //         PrivateQuery::EpsilonDelta(epsilon_tau_thresholding, delta_tau_thresholding),
        //         PrivateQuery::gaussian_privacy_pars(epsilon, delta, 10.)
        //     ]
        //     .into()
        // );
        // assert!(dp_relation.data_type().is_subset_of(&DataType::structured([
        //     ("sum_a", DataType::float()),
        //     ("b", DataType::integer_values([1, 2, 5, 6, 7, 8]))
        // ])));
    }

    // #[test]
    // fn test_dp_compile_map() {
    //     let table: Relation = Relation::table()
    //         .name("table")
    //         .schema(
    //             Schema::builder()
    //                 .with(("a", DataType::integer_range(1..=10)))
    //                 .with(("b", DataType::integer_values([1, 2, 5, 6, 7, 8])))
    //                 .with(("c", DataType::integer_range(5..=20)))
    //                 .with(("id", DataType::integer_range(1..=100)))
    //                 .build(),
    //         )
    //         .build();
    //     let relations: Hierarchy<Arc<Relation>> = vec![("table", Arc::new(table.clone()))]
    //         .into_iter()
    //         .collect();
    //     let (epsilon, delta) = (1., 1e-3);
    //     let (epsilon_tau_thresholding, delta_tau_thresholding) = (0.5, 2e-3);

    //     let reduce: Relation = Relation::reduce()
    //         .name("reduce_relation")
    //         .with(("sum_a".to_string(), AggregateColumn::sum("a")))
    //         .group_by(expr!(b))
    //         .group_by(expr!(c))
    //         .with(("group_b".to_string(), AggregateColumn::first("b")))
    //         .with(("c".to_string(), AggregateColumn::first("c")))
    //         .input(table)
    //         .build();
    //     let relation: Relation = Relation::map()
    //         .name("map_relation")
    //         .with(("my_sum_a", expr!(0 * sum_a)))
    //         .with(("my_group_b".to_string(), expr!(group_b)))
    //         .with(("c".to_string(), expr!(4 * c)))
    //         .filter(Expr::gt(Expr::col("group_b"), Expr::val(6)))
    //         .input(reduce)
    //         .build();

    //     let pep_relation =
    //         relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]);
    //     let (dp_relation, private_query) = pep_relation
    //         .dp_compile(
    //             epsilon,
    //             delta,
    //             epsilon_tau_thresholding,
    //             delta_tau_thresholding,
    //         )
    //         .unwrap()
    //         .into();
    //     dp_relation.display_dot().unwrap();
    //     assert_eq!(
    //         private_query,
    //         vec![
    //             PrivateQuery::EpsilonDelta(epsilon_tau_thresholding, delta_tau_thresholding),
    //             PrivateQuery::gaussian_privacy_pars(epsilon, delta, 10.)
    //         ]
    //         .into()
    //     );
    //     assert_eq!(
    //         dp_relation.data_type()["my_sum_a"],
    //         DataType::float_value(0.)
    //     );
    //     assert_eq!(
    //         dp_relation.data_type()["my_group_b"],
    //         DataType::integer_values([6, 7, 8])
    //     );
    //     assert_eq!(
    //         dp_relation.data_type()["c"],
    //         DataType::integer_range(20..=80)
    //     );
    // }

    // #[test]
    // #[ignore]
    // fn test_dp_compile_join() {
    //     let table: Relation = Relation::table()
    //         .name("table")
    //         .schema(
    //             Schema::builder()
    //                 .with(("a", DataType::integer_range(1..=10)))
    //                 .with(("b", DataType::integer_values([1, 2, 5, 6, 7, 8])))
    //                 .with(("c", DataType::integer_range(5..=20)))
    //                 .with(("id", DataType::integer_range(1..=100)))
    //                 .build(),
    //         )
    //         .build();
    //     let relations: Hierarchy<Arc<Relation>> = vec![("table", Arc::new(table.clone()))]
    //         .into_iter()
    //         .collect();
    //     let (epsilon, delta) = (1., 1e-3);
    //     let (epsilon_tau_thresholding, delta_tau_thresholding) = (0.5, 2e-3);

    //     let reduce: Relation = Relation::reduce()
    //         .name("reduce_relation")
    //         .with(("sum_a".to_string(), AggregateColumn::sum("a")))
    //         .group_by(expr!(b))
    //         .with(("my_b".to_string(), AggregateColumn::first("b")))
    //         .input(table.clone())
    //         .build();
    //     let right: Relation = Relation::map()
    //         .with(("c".to_string(), expr!(my_b - 1)))
    //         .with(("sum_a".to_string(), expr!(2 * sum_a)))
    //         .input(reduce.clone())
    //         .build();
    //     let relation: Relation = Relation::join()
    //         .left(reduce)
    //         .right(right)
    //         .left_names(vec!["left_sum", "b"])
    //         .right_names(vec!["c", "right_sum"])
    //         .inner()
    //         .on(Expr::eq(Expr::col("my_b"), Expr::col("c")))
    //         .build();
    //     relation.display_dot().unwrap();

    //     let pep_relation =
    //         relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]);
    //     pep_relation.display_dot().unwrap();
    //     let (dp_relation, private_query) = pep_relation
    //         .dp_compile(
    //             epsilon,
    //             delta,
    //             epsilon_tau_thresholding,
    //             delta_tau_thresholding,
    //         )
    //         .unwrap()
    //         .into();
    //     dp_relation.display_dot().unwrap();
    //     assert_eq!(
    //         private_query,
    //         vec![
    //             PrivateQuery::EpsilonDelta(epsilon_tau_thresholding, delta_tau_thresholding),
    //             PrivateQuery::gaussian_privacy_pars(epsilon, delta, 10.)
    //         ]
    //         .into()
    //     );
    //     matches!(dp_relation.data_type()["left_sum"], DataType::Float(_));
    //     assert_eq!(
    //         dp_relation.data_type()["b"],
    //         DataType::integer_values([6, 7, 8])
    //     );
    //     assert_eq!(
    //         dp_relation.data_type()["c"],
    //         DataType::integer_range(20..=80)
    //     );
    //     matches!(dp_relation.data_type()["left_sum"], DataType::Float(_));

    //     // AutoJoin
    // }

    // #[test]
    // fn test_dp_compile_simple() {
    //     let mut database = postgresql::test_database();
    //     let relations = database.relations();

    //     // No GROUPING cols
    //     let str_query = "SELECT sum(x) AS sum_x FROM table_2";
    //     let query = parse(str_query).unwrap();
    //     let relation = Relation::try_from(query.with(&relations)).unwrap();

    //     let pep_relation =
    //         relation.force_protect_from_field_paths(&relations, vec![("table_2", vec![], "y")]);

    //     let (dp_relation, private_query) =
    //         pep_relation.dp_compile(1., 1e-3, 1., 1e-3).unwrap().into();
    //     dp_relation.display_dot().unwrap();
    //     assert!(matches!(
    //         dp_relation.data_type()["sum_x"],
    //         DataType::Float(_)
    //     ));
    //     assert_eq!(
    //         private_query,
    //         PrivateQuery::gaussian_privacy_pars(1., 1e-3, 100.0)
    //     );
    //     assert_eq!(dp_relation.schema().len(), 1);
    //     let dp_query = ast::Query::from(&dp_relation);
    //     database.query(&dp_query.to_string()).unwrap();

    //     // GROUPING col in the SELECT clause
    //     let str_query = "SELECT z, sum(x) AS sum_x FROM table_2 GROUP BY z";
    //     let query = parse(str_query).unwrap();
    //     let relation = Relation::try_from(query.with(&relations)).unwrap();

    //     let pep_relation =
    //         relation.force_protect_from_field_paths(&relations, vec![("table_2", vec![], "y")]);

    //     let (dp_relation, private_query) =
    //         pep_relation.dp_compile(1., 1e-3, 1., 1e-3).unwrap().into();
    //     dp_relation.display_dot().unwrap();

    //     assert_eq!(
    //         dp_relation.data_type()["z"],
    //         DataType::text_values(["Foo".into(), "Bar".into()])
    //     );
    //     assert!(matches!(
    //         dp_relation.data_type()["sum_x"],
    //         DataType::Float(_)
    //     ));
    //     assert_eq!(
    //         private_query,
    //         PrivateQuery::gaussian_privacy_pars(1., 1e-3, 100.0)
    //     );
    //     assert_eq!(dp_relation.schema().len(), 2);
    //     let dp_query = ast::Query::from(&dp_relation);
    //     database.query(&dp_query.to_string()).unwrap();

    //     // GROUPING col NOT in the SELECT clause
    //     let str_query = "SELECT sum(x) AS sum_x FROM table_2 GROUP BY z";
    //     let query = parse(str_query).unwrap();
    //     let relation = Relation::try_from(query.with(&relations)).unwrap();

    //     let pep_relation =
    //         relation.force_protect_from_field_paths(&relations, vec![("table_2", vec![], "y")]);

    //     let (dp_relation, private_query) =
    //         pep_relation.dp_compile(1., 1e-3, 1., 1e-3).unwrap().into();
    //     //dp_relation.display_dot().unwrap();
    //     assert_eq!(
    //         private_query,
    //         PrivateQuery::gaussian_privacy_pars(1., 1e-3, 100.0)
    //     );
    //     assert_eq!(dp_relation.schema().len(), 1);
    //     assert!(matches!(
    //         dp_relation.data_type()["sum_x"],
    //         DataType::Float(_)
    //     ));
    //     let dp_query = ast::Query::from(&dp_relation);
    //     database.query(&dp_query.to_string()).unwrap();

    //     // GROUPING col has no possible values
    //     let str_query = "SELECT y, sum(x) AS sum_x FROM table_2 GROUP BY y";
    //     let query = parse(str_query).unwrap();
    //     let relation = Relation::try_from(query.with(&relations)).unwrap();

    //     let pep_relation =
    //         relation.force_protect_from_field_paths(&relations, vec![("table_2", vec![], "z")]);

    //     let (dp_relation, private_query) =
    //         pep_relation.dp_compile(1., 1e-3, 1., 1e-3).unwrap().into();
    //     dp_relation.display_dot().unwrap();
    //     assert_eq!(
    //         private_query,
    //         vec![
    //             PrivateQuery::EpsilonDelta(1., 1e-3),
    //             PrivateQuery::gaussian_privacy_pars(1., 1e-3, 100.0)
    //         ]
    //         .into()
    //     );
    //     assert_eq!(dp_relation.schema().len(), 2);
    //     assert!(matches!(
    //         dp_relation.data_type()["sum_x"],
    //         DataType::Float(_)
    //     ));
    //     assert_eq!(
    //         dp_relation.data_type()["y"],
    //         DataType::optional(DataType::text())
    //     );
    //     let dp_query = ast::Query::from(&dp_relation);
    //     database.query(&dp_query.to_string()).unwrap();
    // }

    // #[test]
    // fn test_dp_compile() {
    //     let mut database = postgresql::test_database();
    //     let relations = database.relations();

    //     let query = parse(
    //         "SELECT order_id, sum(price) AS sum_price,
    //     count(price) AS count_price,
    //     avg(price) AS mean_price
    //     FROM item_table WHERE order_id IN (1,2,3,4,5,6,7,8,9,10) GROUP BY order_id",
    //     )
    //     .unwrap();
    //     let relation = Relation::try_from(query.with(&relations)).unwrap();
    //     relation.display_dot().unwrap();

    //     let pep_relation = relation.force_protect_from_field_paths(
    //         &relations,
    //         vec![
    //             (
    //                 "item_table",
    //                 vec![
    //                     ("order_id", "order_table", "id"),
    //                     ("user_id", "user_table", "id"),
    //                 ],
    //                 "name",
    //             ),
    //             ("order_table", vec![("user_id", "user_table", "id")], "name"),
    //             ("user_table", vec![], "name"),
    //         ],
    //     );
    //     pep_relation.display_dot().unwrap();

    //     let epsilon = 1.;
    //     let delta = 1e-3;
    //     let epsilon_tau_thresholding = 1.;
    //     let delta_tau_thresholding = 1e-3;
    //     let (dp_relation, private_query) = pep_relation
    //         .dp_compile(
    //             epsilon,
    //             delta,
    //             epsilon_tau_thresholding,
    //             delta_tau_thresholding,
    //         )
    //         .unwrap()
    //         .into();
    //     dp_relation.display_dot().unwrap();
    //     assert_eq!(
    //         private_query,
    //         vec![
    //             PrivateQuery::gaussian_privacy_pars(epsilon, delta, 50.0),
    //             PrivateQuery::gaussian_privacy_pars(epsilon, delta, 1.0),
    //         ]
    //         .into()
    //     );
    //     let dp_query = ast::Query::from(&dp_relation);
    //     for row in database.query(&dp_query.to_string()).unwrap() {
    //         println!("{row}");
    //     }
    // }

}
