//! # Methods to transform `Relation`s into differentially private ones
//!
//! This is experimental and little tested yet.
//!

pub mod aggregates;
pub mod group_by;
pub mod private_query;

use crate::{
    builder::With,
    differential_privacy::private_query::PrivateQuery,
    expr,
    protection::{self, PEPReduce, PEPRelation},
    relation::{transforms, Join, Map, Reduce, Relation, Table, Visitor},
    visitor::Acceptor,
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
        let (dp_relation, private_query_agg) = pep_reduce_with_dp_group_by
            .dp_compile_aggregates(epsilon, delta)?
            .into();
        private_query = private_query.compose(private_query_agg);
        Ok((dp_relation, private_query).into())
    }
}

struct DPCompilator {
    protected_entity_id: String,
    protected_entity_weight: String,
    epsilon: f64,
    delta: f64,
    epsilon_tau_thresholding: f64,
    delta_tau_thresholding: f64,
}

impl<'a> Visitor<'a, Result<DPRelation>> for DPCompilator {
    fn table(&self, table: &'a Table) -> Result<DPRelation> {
        Err(Error::DPCompilationError(
            "A Relation::Table cannot be compiled into DP".to_string(),
        ))
    }

    fn map(&self, map: &'a Map, input: Result<DPRelation>) -> Result<DPRelation> {
        if let Ok(dp_input) = input {
            let (dp_relation, private_query) = dp_input.into();
            let relation = Map::builder()
                .filter_fields_with(map.clone(), |f| {
                    f != self.protected_entity_id.as_str()
                        && f != self.protected_entity_weight.as_str()
                })
                .input(dp_relation)
                .build();
            Ok(DPRelation::new(relation, private_query))
        } else {
            Err(Error::DPCompilationError(
                "Cannot compile into DP a Relation::Map that does not input a DPRelation"
                    .to_string(),
            ))
        }
    }

    fn reduce(&self, reduce: &'a Reduce, _input: Result<DPRelation>) -> Result<DPRelation> {
        PEPReduce::try_from(reduce.clone())?.dp_compile(
            self.epsilon,
            self.delta,
            self.epsilon_tau_thresholding,
            self.delta_tau_thresholding,
        )
    }

    fn join(
        &self,
        join: &'a Join,
        left: Result<DPRelation>,
        right: Result<DPRelation>,
    ) -> Result<DPRelation> {
        if let (Ok(left_input), Ok(right_input)) = (left, right) {
            todo!()
            // let (left_dp_relation, left_private_query) = left_input.into();
            // left_dp_relation.display_dot().unwrap();
            // let (right_dp_relation, right_private_query) = right_input.into();
            // right_dp_relation.display_dot().unwrap();

            // let left_names = join.schema()
            //     .iter()
            //     .take(join.left().schema().len())
            //     .map(|f| f.name())
            //     .collect::<Vec<_>>();

            // let right_names = join.schema()
            //     .iter()
            //     .skip(join.left().schema().len())
            //     .map(|f| f.name())
            //     .collect::<Vec<_>>();

            // let relation: Relation = Join::builder()
            //     .operator(join.operator().clone()) // This does work because it should not be protected
            //     .left_names(left_names)
            //     .right_names(right_names)
            //     .left(left_dp_relation)
            //     .right(right_dp_relation)
            //     .build();
            // Ok(DPRelation::new(
            //     relation,
            //     vec![left_private_query, right_private_query].into(),
            // ))
        } else {
            Err(Error::DPCompilationError(
                "Cannot compile into DP a Relation::Join that does not input two DPRelations"
                    .to_string(),
            ))
        }
    }

    fn set(
        &self,
        set: &'a crate::relation::Set,
        left: Result<DPRelation>,
        right: Result<DPRelation>,
    ) -> Result<DPRelation> {
        todo!()
    }

    fn values(&self, values: &'a crate::relation::Values) -> Result<DPRelation> {
        todo!()
    }
}

impl PEPRelation {
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
        Relation::from(self).accept(DPCompilator {
            protected_entity_id,
            protected_entity_weight,
            epsilon,
            delta,
            epsilon_tau_thresholding,
            delta_tau_thresholding,
        })
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
    };
    use std::sync::Arc;

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
        let relations: Hierarchy<Arc<Relation>> = vec![("table", Arc::new(table.clone()))]
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
        let pep_relation =
            relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]);
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
            PrivateQuery::gaussian_privacy_pars(epsilon, delta, 10.)
        );
        assert!(dp_relation
            .data_type()
            .is_subset_of(&DataType::structured([("sum_a", DataType::float())])));

        // With GROUPBY. Only one column with possible values
        let relation: Relation = Relation::reduce()
            .name("reduce_relation")
            .with(("sum_a".to_string(), AggregateColumn::sum("a")))
            .with(("b".to_string(), AggregateColumn::first("b")))
            .group_by(expr!(b))
            .input(table.clone())
            .build();
        let pep_relation =
            relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]);
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
            PrivateQuery::gaussian_privacy_pars(epsilon, delta, 10.)
        );
        assert!(dp_relation.data_type().is_subset_of(&DataType::structured([
            ("sum_a", DataType::float()),
            ("b", DataType::integer_values([1, 2, 5, 6, 7, 8]))
        ])));

        // With GROUPBY. Only one column with tau-thresholding values
        let relation: Relation = Relation::reduce()
            .name("reduce_relation")
            .with(("sum_a".to_string(), AggregateColumn::sum("a")))
            .group_by(expr!(c))
            .with(("c".to_string(), AggregateColumn::first("c")))
            .input(table.clone())
            .build();
        let pep_relation =
            relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]);
        //pep_relation.display_dot().unwrap();
        let (dp_relation, private_query) = pep_relation
            .dp_compile(
                epsilon,
                delta,
                epsilon_tau_thresholding,
                delta_tau_thresholding,
            )
            .unwrap()
            .into();
        assert_eq!(
            private_query,
            vec![
                PrivateQuery::EpsilonDelta(epsilon_tau_thresholding, delta_tau_thresholding),
                PrivateQuery::gaussian_privacy_pars(epsilon, delta, 10.)
            ]
            .into()
        );
        assert!(dp_relation.data_type().is_subset_of(&DataType::structured([
            ("sum_a", DataType::float()),
            ("c", DataType::integer_range(5..=20))
        ])));

        // With GROUPBY. Both tau-thresholding and possible values
        let relation: Relation = Relation::reduce()
            .name("reduce_relation")
            .with(("sum_a".to_string(), AggregateColumn::sum("a")))
            .group_by(expr!(c))
            .group_by(expr!(b))
            .with(("b".to_string(), AggregateColumn::first("b")))
            .input(table.clone())
            .build();
        let pep_relation =
            relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]);
        //pep_relation.display_dot().unwrap();
        let (dp_relation, private_query) = pep_relation
            .dp_compile(
                epsilon,
                delta,
                epsilon_tau_thresholding,
                delta_tau_thresholding,
            )
            .unwrap()
            .into();
        assert_eq!(
            private_query,
            vec![
                PrivateQuery::EpsilonDelta(epsilon_tau_thresholding, delta_tau_thresholding),
                PrivateQuery::gaussian_privacy_pars(epsilon, delta, 10.)
            ]
            .into()
        );
        assert!(dp_relation.data_type().is_subset_of(&DataType::structured([
            ("sum_a", DataType::float()),
            ("b", DataType::integer_values([1, 2, 5, 6, 7, 8]))
        ])));
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
        let relations: Hierarchy<Arc<Relation>> = vec![("table", Arc::new(table.clone()))]
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
            .with(("my_sum_a", expr!(0 * sum_a)))
            .with(("my_group_b".to_string(), expr!(group_b)))
            .with(("c".to_string(), expr!(4 * c)))
            .filter(Expr::gt(Expr::col("group_b"), Expr::val(6)))
            .input(reduce)
            .build();

        let pep_relation =
            relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]);
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
                PrivateQuery::EpsilonDelta(epsilon_tau_thresholding, delta_tau_thresholding),
                PrivateQuery::gaussian_privacy_pars(epsilon, delta, 10.)
            ]
            .into()
        );
        assert_eq!(
            dp_relation.data_type()["my_sum_a"],
            DataType::float_value(0.)
        );
        assert_eq!(
            dp_relation.data_type()["my_group_b"],
            DataType::integer_values([6, 7, 8])
        );
        assert_eq!(
            dp_relation.data_type()["c"],
            DataType::integer_range(20..=80)
        );
    }

    #[test]
    #[ignore]
    fn test_dp_compile_join() {
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
        let relations: Hierarchy<Arc<Relation>> = vec![("table", Arc::new(table.clone()))]
            .into_iter()
            .collect();
        let (epsilon, delta) = (1., 1e-3);
        let (epsilon_tau_thresholding, delta_tau_thresholding) = (0.5, 2e-3);

        let reduce: Relation = Relation::reduce()
            .name("reduce_relation")
            .with(("sum_a".to_string(), AggregateColumn::sum("a")))
            .group_by(expr!(b))
            .with(("my_b".to_string(), AggregateColumn::first("b")))
            .input(table.clone())
            .build();
        let right: Relation = Relation::map()
            .with(("c".to_string(), expr!(my_b - 1)))
            .with(("sum_a".to_string(), expr!(2 * sum_a)))
            .input(reduce.clone())
            .build();
        let relation: Relation = Relation::join()
            .left(reduce)
            .right(right)
            .left_names(vec!["left_sum", "b"])
            .right_names(vec!["c", "right_sum"])
            .inner()
            .on(Expr::eq(Expr::col("my_b"), Expr::col("c")))
            .build();
        relation.display_dot().unwrap();

        let pep_relation =
            relation.force_protect_from_field_paths(&relations, vec![("table", vec![], "id")]);
        pep_relation.display_dot().unwrap();
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
                PrivateQuery::EpsilonDelta(epsilon_tau_thresholding, delta_tau_thresholding),
                PrivateQuery::gaussian_privacy_pars(epsilon, delta, 10.)
            ]
            .into()
        );
        matches!(dp_relation.data_type()["left_sum"], DataType::Float(_));
        assert_eq!(
            dp_relation.data_type()["b"],
            DataType::integer_values([6, 7, 8])
        );
        assert_eq!(
            dp_relation.data_type()["c"],
            DataType::integer_range(20..=80)
        );
        matches!(dp_relation.data_type()["left_sum"], DataType::Float(_));

        // AutoJoin
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
