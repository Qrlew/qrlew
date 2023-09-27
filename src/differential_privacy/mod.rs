//! # Methods to transform `Relation`s into differentially private ones
//!
//! This is experimental and little tested yet.
//!

pub mod private_query;
pub mod group_by;
pub mod aggregates;

use crate::{
    builder::With,
    differential_privacy::private_query::PrivateQuery,
    expr,
    protection::{self, PEPRelation, PEPReduce},
    relation::{Visitor, transforms, Join, Map, Reduce, Relation, Table, Variant},
    Ready,
    visitor::Acceptor
};
use std::{error, fmt, result, ops::Deref};

#[derive(Debug, PartialEq, Clone)]
pub enum Error {
    InvalidRelation(String),
    UnsafeGroups(String),
    GroupingKeysError(String),
    Other(String),
}

impl Error {
    pub fn invalid_relation(relation: impl fmt::Display) -> Error {
        Error::InvalidRelation(format!("{relation} is invalid"))
    }
    pub fn unsafe_groups(groups: impl fmt::Display) -> Error {
        Error::UnsafeGroups(format!("{groups} should be public"))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidRelation(relation) => writeln!(f, "{relation} invalid."),
            Error::UnsafeGroups(groups) => writeln!(f, "{groups} should be public."),
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
impl From<transforms::Error> for Error {
    fn from(err: transforms::Error) -> Self {
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

impl PEPReduce {
    /// Compiles a protected Relation into DP:
    ///     - Replace the grouping keys by their DP values
    /// protected the grouping keys and the results of the aggregations
    pub fn dp_compile(
        self,
        epsilon: f64,
        delta: f64,
        epsilon_tau_thresholding: f64,
        delta_tau_thresholding: f64,
    ) -> Result<DPRelation> {
        let protected_entity_id = self.protected_entity_id().to_string();
        let mut private_query = PrivateQuery::null();

        // DP compile group by
        let pep_reduce_with_dp_group_by = if self.group_by_names() == vec![protected_entity_id] {
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
                .with(self.deref().clone())
                .input(input_relation_with_protected_group_by)
                .build();
            private_query = private_query.compose(private_query_group_by);
            PEPReduce::try_from(reduce)?
        };

        // DP compile aggregates
        let (dp_relation, private_query_agg) = pep_reduce_with_dp_group_by.dp_compile_aggregates(epsilon, delta)?.into();
        private_query = private_query.compose(private_query_agg);
        Ok((dp_relation, private_query).into())
    }
}


// struct DPCompilator{
//     protected_entity_id: String,
//     protected_entity_weight: String,
//     epsilon: f64,
//     delta: f64,
//     epsilon_tau_thresholding: f64,
//     delta_tau_thresholding: f64,
// }

// impl<'a> Visitor<'a, Result<DPRelation>> for DPCompilator {
//     fn table(&self, table: &'a Table) -> Result<DPRelation> {
//         PEPRelation::try_from(Relation::from(table.clone()))?
//             .dp_values(self.epsilon_tau_thresholding, self.delta_tau_thresholding)
//     }

//     fn map(&self, map: &'a Map, input: Result<DPRelation>) -> Result<DPRelation> {
//         let (dp_input, private_query) = input?.into();
//         let relation = Map::builder()
//             .filter_fields_with(map.clone(), |f| {
//                 f != self.protected_entity_id.as_str() && f != self.protected_entity_weight.as_str()
//             })
//             .input(dp_input)
//             .build();
//         Ok(DPRelation::new(relation, private_query))
//     }

//     fn reduce(&self, reduce: &'a Reduce, _input: Result<DPRelation>) -> Result<DPRelation> {
//         PEPReduce::try_from(reduce.clone())?.dp_compile(
//             self.epsilon,
//             self.delta,
//             self.epsilon_tau_thresholding,
//             self.delta_tau_thresholding
//         )
//     }

//     fn join(&self, join: &'a Join, left: Result<DPRelation>, right: Result<DPRelation>) -> Result<DPRelation> {
//         let (left_dp_relation, left_private_query) = left?.into();
//         let (right_dp_relation, right_private_query) = right?.into();
//         let relation: Relation = Join::builder()
//             .left(left_dp_relation)
//             .right(right_dp_relation)
//             .with(join.clone())
//             .build();
//         Ok(DPRelation::new(
//             relation,
//             vec![left_private_query, right_private_query].into(),
//         ))
//     }

//     fn set(&self, set: &'a crate::relation::Set, left: Result<DPRelation>, right: Result<DPRelation>) -> Result<DPRelation> {
//         todo!()
//     }

//     fn values(&self, values: &'a crate::relation::Values) -> Result<DPRelation> {
//         Ok(DPRelation::new(
//             Relation::from(values.clone()),
//             PrivateQuery::null(),
//         ))
//     }
// }

impl PEPRelation {
    /// Compile a protected Relation into DP: protected the grouping keys and the results of the aggregations
    // pub fn dp_compile(
    //     self,
    //     epsilon: f64,
    //     delta: f64,
    //     epsilon_tau_thresholding: f64,
    //     delta_tau_thresholding: f64,
    // ) -> Result<DPRelation> {
    //     let protected_entity_id = self.protected_entity_id().to_string();
    //     let protected_entity_weight = self.protected_entity_weight().to_string();
    //     Relation::from(self).accept(
    //         DPCompilator{
    //             protected_entity_id,
    //             protected_entity_weight,
    //             epsilon,
    //             delta,
    //             epsilon_tau_thresholding,
    //             delta_tau_thresholding
    //         }
    //     )
    // }

    /// Compile a protected Relation into DP: protected the grouping keys and the results of the aggregations
    pub fn dp_compile(
        self,
        epsilon: f64,
        delta: f64,
        epsilon_tau_thresholding: f64,
        delta_tau_thresholding: f64,
    ) -> Result<DPRelation> {
        let protected_entity_id = self.protected_entity_id().to_string();
        let protected_entity_weight = self.protected_entity_weight().to_string();

        match Relation::from(self) {
            Relation::Table(_) => todo!(),
            Relation::Map(m) => m.dp_compile(
                protected_entity_id.as_str(),
                protected_entity_weight.as_str(),
                epsilon,
                delta,
                epsilon_tau_thresholding,
                delta_tau_thresholding
            ),
            Relation::Reduce(r) => PEPReduce::try_from(r)?.dp_compile(epsilon, delta, epsilon_tau_thresholding, delta_tau_thresholding),
            Relation::Join(j) => todo!(),
            Relation::Set(_) => todo!(),
            Relation::Values(v) => Ok(DPRelation::new(Relation::from(v), PrivateQuery::null())),
        }
    }
}

impl Map {
    pub fn dp_compile(
        self,
        protected_entity_id: &str,
        protected_entity_weight: &str,
        epsilon: f64,
        delta: f64,
        epsilon_tau_thresholding: f64,
        delta_tau_thresholding: f64,
    ) -> Result<DPRelation> {
        let dp_input = PEPRelation::try_from(self.input().clone())?.dp_compile(epsilon, delta, epsilon_tau_thresholding, delta_tau_thresholding)?;
        let relation = Map::builder()
            .filter_fields_with(self, |f| {
                f != protected_entity_id && f != protected_entity_weight
            })
            .input(dp_input.relation().clone())
            .build();
        Ok(DPRelation::new(relation, dp_input.private_query().clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ast,
        builder::With,
        display::Dot,
        io::{postgresql, Database},
        sql::parse,
        Relation,
        data_type::{DataType, DataTyped, Variant as _},
        relation::{Variant, Schema},
        hierarchy::Hierarchy,
        expr::{Expr, AggregateColumn},
    };
    use std::rc::Rc;

    #[test]
    fn test_dp_compile_reduce() {
        let table: Relation = Relation::table()
            .name("table")
            .schema(
                Schema::builder()
                    .with(("a", DataType::integer_range(1..=10)))
                    .with(("b", DataType::integer_values([1, 2, 5, 6, 7, 8])))
                    .with(("c", DataType::integer_range(5..=20)))
                    .with(("id", DataType::integer_range(1..=100)))
                    .build(),
            )
            .build();
        let relations: Hierarchy<Rc<Relation>> = vec![("table", Rc::new(table.clone()))]
            .into_iter()
            .collect();
        let (epsilon, delta) = (1., 1e-3);
        let (epsilon_tau_thresholding, delta_tau_thresholding) = (0.5, 2e-3);

        // Without GROUPBY
        let relation: Relation = Relation::reduce()
            .name("reduce_relation")
            .with(("sum_a".to_string(), AggregateColumn::sum("a")))
            .input(table.clone())
            .build();
        //relation.display_dot().unwrap();
        let pep_relation = Relation::from(relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]));
        //pep_relation.display_dot().unwrap();
        if let Relation::Reduce(reduce) = pep_relation {
            let pep_reduce = PEPReduce::try_from(reduce).unwrap();
            let (dp_relation, private_query) = pep_reduce.dp_compile(epsilon, delta, epsilon_tau_thresholding, delta_tau_thresholding).unwrap().into();
            assert_eq!(private_query, PrivateQuery::gaussian_privacy_pars(epsilon, delta, 10.));
            assert!(dp_relation.data_type().is_subset_of(&DataType::structured([("sum_a", DataType::float())])));
        } else {
            panic!()
        }

        // With GROUPBY. Only one column with possible values
        let relation: Relation = Relation::reduce()
            .name("reduce_relation")
            .with(("sum_a".to_string(), AggregateColumn::sum("a")))
            .with(("b".to_string(), AggregateColumn::first("b")))
            .group_by(expr!(b))
            .input(table.clone())
            .build();
        let pep_relation = Relation::from(relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]));
        //pep_relation.display_dot().unwrap();
        if let Relation::Reduce(reduce) = pep_relation {
            let pep_reduce = PEPReduce::try_from(reduce).unwrap();
            let (dp_relation, private_query) = pep_reduce.dp_compile(epsilon, delta, epsilon_tau_thresholding, delta_tau_thresholding).unwrap().into();
            assert_eq!(private_query, PrivateQuery::gaussian_privacy_pars(epsilon, delta, 10.));
            assert!(dp_relation.data_type().is_subset_of(&DataType::structured([("sum_a", DataType::float()), ("b", DataType::integer_values([1, 2, 5, 6, 7, 8]))])));
        } else {
            panic!()
        }

        // With GROUPBY. Only one column with tau-thresholding values
        let relation: Relation = Relation::reduce()
            .name("reduce_relation")
            .with(("sum_a".to_string(), AggregateColumn::sum("a")))
            .group_by(expr!(c))
            .with(("c".to_string(), AggregateColumn::first("c")))
            .input(table.clone())
            .build();
        let pep_relation = Relation::from(relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]));
        //pep_relation.display_dot().unwrap();
        if let Relation::Reduce(reduce) = pep_relation {
            let pep_reduce = PEPReduce::try_from(reduce).unwrap();
            let (dp_relation, private_query) = pep_reduce.dp_compile(epsilon, delta, epsilon_tau_thresholding, delta_tau_thresholding).unwrap().into();
            assert_eq!(
                private_query,
                vec![
                    PrivateQuery::EpsilonDelta(epsilon_tau_thresholding, delta_tau_thresholding),
                    PrivateQuery::gaussian_privacy_pars(epsilon, delta, 10.)
                ].into()
            );
            assert!(dp_relation.data_type().is_subset_of(&DataType::structured([("sum_a", DataType::float()), ("c", DataType::integer_range(5..=20))])));
        } else {
            panic!()
        }

        // With GROUPBY. Both tau-thresholding and possible values
        let relation: Relation = Relation::reduce()
            .name("reduce_relation")
            .with(("sum_a".to_string(), AggregateColumn::sum("a")))
            .group_by(expr!(c))
            .group_by(expr!(b))
            .with(("b".to_string(), AggregateColumn::first("b")))
            .input(table.clone())
            .build();
        let pep_relation = Relation::from(relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]));
        //pep_relation.display_dot().unwrap();
        if let Relation::Reduce(reduce) = pep_relation {
            let pep_reduce = PEPReduce::try_from(reduce).unwrap();
            let (dp_relation, private_query) = pep_reduce.dp_compile(epsilon, delta, epsilon_tau_thresholding, delta_tau_thresholding).unwrap().into();
            assert_eq!(
                private_query,
                vec![
                    PrivateQuery::EpsilonDelta(epsilon_tau_thresholding, delta_tau_thresholding),
                    PrivateQuery::gaussian_privacy_pars(epsilon, delta, 10.)
                ].into()
            );
            assert!(
                dp_relation.data_type().is_subset_of(
                    &DataType::structured([
                        ("sum_a", DataType::float()),
                        ("b", DataType::integer_values([1, 2, 5, 6, 7, 8]))
                    ])
                )
            );
        } else {
            panic!()
        }
    }

    #[test]
    fn test_dp_compile_map() {
        let table: Relation = Relation::table()
            .name("table")
            .schema(
                Schema::builder()
                    .with(("a", DataType::integer_range(1..=10)))
                    .with(("b", DataType::integer_values([1, 2, 5, 6, 7, 8])))
                    .with(("c", DataType::integer_range(5..=20)))
                    .with(("id", DataType::integer_range(1..=100)))
                    .build(),
            )
            .build();
        let relations: Hierarchy<Rc<Relation>> = vec![("table", Rc::new(table.clone()))]
            .into_iter()
            .collect();
        let (epsilon, delta) = (1., 1e-3);
        let (epsilon_tau_thresholding, delta_tau_thresholding) = (0.5, 2e-3);

        let reduce: Relation = Relation::reduce()
            .name("reduce_relation")
            .with(("sum_a".to_string(), AggregateColumn::sum("a")))
            .group_by(expr!(b))
            .group_by(expr!(c))
            .with(("group_b".to_string(), AggregateColumn::first("b")))
            .with(("c".to_string(), AggregateColumn::first("c")))
            .input(table)
            .build();
        let relation: Relation = Relation::map()
            .name("map_relation")
            .with(("my_sum_a", expr!(0*sum_a)))
            .with(("my_group_b".to_string(), expr!(group_b)))
            .with(("c".to_string(), expr!(4*c)))
            .filter(Expr::gt(Expr::col("group_b"), Expr::val(6)))
            .input(reduce)
            .build();

        let pep_relation = relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]);
        let (dp_relation, private_query) = pep_relation.dp_compile(epsilon, delta, epsilon_tau_thresholding, delta_tau_thresholding).unwrap().into();
        dp_relation.display_dot().unwrap();
        assert_eq!(
            private_query,
            vec![
                PrivateQuery::EpsilonDelta(epsilon_tau_thresholding, delta_tau_thresholding),
                PrivateQuery::gaussian_privacy_pars(epsilon, delta, 10.)
            ].into()
        );
        assert_eq!(dp_relation.data_type()["my_sum_a"], DataType::float_value(0.));
        assert_eq!(dp_relation.data_type()["my_group_b"], DataType::integer_values([6, 7, 8]));
        assert_eq!(dp_relation.data_type()["c"], DataType::integer_range(20..=80));
    }

    #[test]
    fn test_dp_compile_simple() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        // No GROUPING cols
        let str_query = "SELECT sum(x) AS sum_x FROM table_2";
        let query = parse(str_query).unwrap();
        let relation = Relation::try_from(query.with(&relations)).unwrap();

        let pep_relation =
            relation.force_protect_from_field_paths(&relations, vec![("table_2", vec![], "y")]);

        let (dp_relation, private_query) =
            pep_relation.dp_compile(1., 1e-3, 1., 1e-3).unwrap().into();
        dp_relation.display_dot().unwrap();
        assert!(matches!(
            dp_relation.data_type()["sum_x"],
            DataType::Float(_)
        ));
        assert_eq!(
            private_query,
            PrivateQuery::gaussian_privacy_pars(1., 1e-3, 100.0)
        );
        assert_eq!(dp_relation.schema().len(), 1);
        let dp_query = ast::Query::from(&dp_relation);
        database.query(&dp_query.to_string()).unwrap();

        // GROUPING col in the SELECT clause
        let str_query = "SELECT z, sum(x) AS sum_x FROM table_2 GROUP BY z";
        let query = parse(str_query).unwrap();
        let relation = Relation::try_from(query.with(&relations)).unwrap();

        let pep_relation =
            relation.force_protect_from_field_paths(&relations, vec![("table_2", vec![], "y")]);

        let (dp_relation, private_query) =
            pep_relation.dp_compile(1., 1e-3, 1., 1e-3).unwrap().into();
        dp_relation.display_dot().unwrap();

        assert_eq!(
            dp_relation.data_type()["z"],
            DataType::text_values(["Foo".into(), "Bar".into()])
        );
        assert!(matches!(
            dp_relation.data_type()["sum_x"],
            DataType::Float(_)
        ));
        assert_eq!(
            private_query,
            PrivateQuery::gaussian_privacy_pars(1., 1e-3, 100.0)
        );
        assert_eq!(dp_relation.schema().len(), 2);
        let dp_query = ast::Query::from(&dp_relation);
        database.query(&dp_query.to_string()).unwrap();

        // GROUPING col NOT in the SELECT clause
        let str_query = "SELECT sum(x) AS sum_x FROM table_2 GROUP BY z";
        let query = parse(str_query).unwrap();
        let relation = Relation::try_from(query.with(&relations)).unwrap();

        let pep_relation =
            relation.force_protect_from_field_paths(&relations, vec![("table_2", vec![], "y")]);

        let (dp_relation, private_query) =
            pep_relation.dp_compile(1., 1e-3, 1., 1e-3).unwrap().into();
        //dp_relation.display_dot().unwrap();
        assert_eq!(
            private_query,
            PrivateQuery::gaussian_privacy_pars(1., 1e-3, 100.0)
        );
        assert_eq!(dp_relation.schema().len(), 1);
        assert!(matches!(
            dp_relation.data_type()["sum_x"],
            DataType::Float(_)
        ));
        let dp_query = ast::Query::from(&dp_relation);
        database.query(&dp_query.to_string()).unwrap();

        // GROUPING col has no possible values
        let str_query = "SELECT y, sum(x) AS sum_x FROM table_2 GROUP BY y";
        let query = parse(str_query).unwrap();
        let relation = Relation::try_from(query.with(&relations)).unwrap();

        let pep_relation =
            relation.force_protect_from_field_paths(&relations, vec![("table_2", vec![], "z")]);

        let (dp_relation, private_query) =
            pep_relation.dp_compile(1., 1e-3, 1., 1e-3).unwrap().into();
        dp_relation.display_dot().unwrap();
        assert_eq!(
            private_query,
            vec![
                PrivateQuery::EpsilonDelta(1., 1e-3),
                PrivateQuery::gaussian_privacy_pars(1., 1e-3, 100.0)
            ]
            .into()
        );
        assert_eq!(dp_relation.schema().len(), 2);
        assert!(matches!(
            dp_relation.data_type()["sum_x"],
            DataType::Float(_)
        ));
        assert_eq!(
            dp_relation.data_type()["y"],
            DataType::optional(DataType::text())
        );
        let dp_query = ast::Query::from(&dp_relation);
        database.query(&dp_query.to_string()).unwrap();
    }

    #[test]
    fn test_dp_compile() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        let query = parse(
            "SELECT order_id, sum(price) AS sum_price,
        count(price) AS count_price,
        avg(price) AS mean_price
        FROM item_table WHERE order_id IN (1,2,3,4,5,6,7,8,9,10) GROUP BY order_id",
        )
        .unwrap();
        let relation = Relation::try_from(query.with(&relations)).unwrap();
        relation.display_dot().unwrap();

        let pep_relation = relation.force_protect_from_field_paths(
            &relations,
            vec![
                (
                    "item_table",
                    vec![
                        ("order_id", "order_table", "id"),
                        ("user_id", "user_table", "id"),
                    ],
                    "name",
                ),
                ("order_table", vec![("user_id", "user_table", "id")], "name"),
                ("user_table", vec![], "name"),
            ],
        );
        pep_relation.display_dot().unwrap();

        let epsilon = 1.;
        let delta = 1e-3;
        let epsilon_tau_thresholding = 1.;
        let delta_tau_thresholding = 1e-3;
        let (dp_relation, private_query) = pep_relation
            .dp_compile(
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
            vec![
                PrivateQuery::gaussian_privacy_pars(epsilon, delta, 50.0),
                PrivateQuery::gaussian_privacy_pars(epsilon, delta, 1.0),
            ]
            .into()
        );
        let dp_query = ast::Query::from(&dp_relation);
        for row in database.query(&dp_query.to_string()).unwrap() {
            println!("{row}");
        }
    }

}
