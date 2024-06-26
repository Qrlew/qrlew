//! # Given a Relation representing a computaion on a sampled table, a table (representing the schema of original dataset) and a weight representing
//!
//! WARNING This is experimental and little tested yet.
// TODO Test and document this properly

use crate::{
    builder::{Ready, With},
    expr::{aggregate, identifier::Identifier, AggregateColumn, Expr},
    hierarchy::Hierarchy,
    relation::{Join, Map, Reduce, Relation, Set, Table, Values, Variant as _, Visitor},
    visitor::Acceptor,
    WithIterator,
};
use std::{error, fmt, result, sync::Arc, vec};

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
pub struct RelationWithWeight(Relation, f64);

/// Getters
impl RelationWithWeight {
    pub fn relation(&self) -> Relation {
        self.0.clone().filter_fields(|n| n != ROW_WEIGHT)
    }

    pub fn weight(&self) -> &f64 {
        &self.1
    }
}

// Visitors

/// A visitor to compute RelationWithWeight propagationing table weights
/// and ROW_WEIGHT.
#[derive(Clone, Debug)]
struct WeightRelationVisitor<F: Fn(&Table) -> RelationWithWeight> {
    weight_table: F,
}

pub const ROW_WEIGHT: &str = "_ROW_WEIGHT_";
pub const ONE_COUNT_ROW_WEIGHT: &str = "_ONE_COUNT_ROW_WEIGHT_";
pub const GREATER_THAN_ONE_COUNT_ROW_WEIGHT: &str = "_GREATER_THAN_ONE_COUNT_ROW_WEIGHT_";
pub const ACTUAL_GROUPS_COUNT: &str = "_ACTUAL_GROUPS_COUNT_";
pub const ESTIMATED_GROUPS_COUNT: &str = "_ESTIMATED_GROUPS_COUNT_";
pub const CORRECTION_FACTOR: &str = "_CORRECTION_FACTOR_";
pub const PROPAGATED_COLUMNS: usize = 1;

impl<F: Fn(&Table) -> RelationWithWeight> WeightRelationVisitor<F> {
    pub fn new(weight_table: F) -> Self {
        WeightRelationVisitor { weight_table }
    }
}

impl<'a, F: Fn(&Table) -> RelationWithWeight> Visitor<'a, RelationWithWeight>
    for WeightRelationVisitor<F>
{
    /// Apply the weight from weight_table and add ROW_WEIGHT field initialized to 1
    fn table(&self, table: &'a Table) -> RelationWithWeight {
        let weighted_table = (self.weight_table)(table);
        let new_rel = weighted_table.0.insert_field(0, ROW_WEIGHT, Expr::val(1));
        RelationWithWeight(new_rel, weighted_table.1)
    }

    /// Propagate the relation weight and ROW_WEIGHT.
    fn map(&self, map: &'a Map, input: RelationWithWeight) -> RelationWithWeight {
        let mew_relation: Relation = Relation::map()
            .with((ROW_WEIGHT, Expr::col(ROW_WEIGHT)))
            .with(map.clone())
            .input(input.0)
            .build();
        RelationWithWeight(mew_relation, input.1)
    }

    /// It constructs the needed relations to compute the CORRECTION_FACTOR from the ROW_WEIGHT
    /// to compensate for the lost of groups during sampling
    /// CORRECTION_FACTOR = ESTIMATED_GROUPS_COUNT / ACTUAL_GROUPS_COUNT
    /// ESTIMATED_GROUPS_COUNT = sqrt(w)*sum(ONE_COUNT_ROW_WEIGHT) + sum(GREATER_THAN_ONE_COUNT_ROW_WEIGHT)
    /// for more details see: https://dl.acm.org/doi/pdf/10.1145/276305.276343
    /// In the paper ONE_COUNT_ROW_WEIGHT is indicated as f1
    /// and GREATER_THAN_ONE_COUNT_ROW_WEIGHT is indicated as fj
    /// It applies the corrections to the aggregation functions
    /// The table weight of the output RelationWithWeight is 1.0.
    fn reduce(&self, reduce: &'a Reduce, input: RelationWithWeight) -> RelationWithWeight {
        // construct the new reduce from the visited reduce and the input relation.
        // with sum of ROW_WEIGHT
        let new_reduce: Relation = Relation::reduce()
            .with((ROW_WEIGHT, Expr::sum(Expr::col(ROW_WEIGHT))))
            .with(reduce.clone())
            .input(input.0.clone())
            .build();

        // compute the counts of line weight
        let counts_row_weight = &format!("_COUNTS{ROW_WEIGHT}")[..];
        let group_by_line_weight: Relation = Relation::reduce()
            .with((ROW_WEIGHT, Expr::first(Expr::col(ROW_WEIGHT))))
            .with((counts_row_weight, Expr::count(Expr::col(ROW_WEIGHT))))
            .group_by(Expr::col(ROW_WEIGHT))
            .input(new_reduce.clone())
            .build();

        let one_where_f1 = Expr::case(
            Expr::eq(Expr::col(ROW_WEIGHT), Expr::val(1)),
            Expr::val(1),
            Expr::val(0),
        );
        let one_where_fj = Expr::case(
            Expr::gt(Expr::col(ROW_WEIGHT), Expr::val(1)),
            Expr::col(counts_row_weight),
            Expr::val(0),
        );

        let tmp_map: Relation = Relation::map()
            .with((ROW_WEIGHT, Expr::col(ROW_WEIGHT)))
            .with((ONE_COUNT_ROW_WEIGHT, one_where_f1))
            .with((GREATER_THAN_ONE_COUNT_ROW_WEIGHT, one_where_fj))
            .input(group_by_line_weight)
            .build();

        // compute sum_f1: sum over number of distinct values which occur exactly 1 times
        // sum_fj: sum over number of distinct values which occur exactly j times
        let tmp_reduce: Relation = Relation::reduce()
            .with((
                format!("_SUM{ONE_COUNT_ROW_WEIGHT}"),
                Expr::sum(Expr::col(ONE_COUNT_ROW_WEIGHT)),
            ))
            .with((
                format!("_SUM{GREATER_THAN_ONE_COUNT_ROW_WEIGHT}"),
                Expr::sum(Expr::col(GREATER_THAN_ONE_COUNT_ROW_WEIGHT)),
            ))
            .input(tmp_map)
            .build();

        let estimated_groups = Expr::multiply(
            Expr::sqrt(Expr::val(input.1)),
            Expr::col(format!("_SUM{ONE_COUNT_ROW_WEIGHT}")),
        ) + Expr::col(format!("_SUM{GREATER_THAN_ONE_COUNT_ROW_WEIGHT}"));

        let estimated_and_actual_groups: Relation = Relation::map()
            .with((ESTIMATED_GROUPS_COUNT, estimated_groups))
            .with((
                ACTUAL_GROUPS_COUNT,
                Expr::col(format!("_SUM{ONE_COUNT_ROW_WEIGHT}"))
                    + Expr::col(format!("_SUM{GREATER_THAN_ONE_COUNT_ROW_WEIGHT}")),
            ))
            .input(tmp_reduce)
            .build();

        let correction_factor = Expr::divide(
            Expr::col(ESTIMATED_GROUPS_COUNT),
            Expr::col(ACTUAL_GROUPS_COUNT),
        );
        let correction_factor_map: Relation = Relation::map()
            .with((CORRECTION_FACTOR, correction_factor))
            .filter(Expr::and(
                Expr::gt_eq(Expr::col(ESTIMATED_GROUPS_COUNT), Expr::val(1.0)),
                Expr::gt_eq(Expr::col(ACTUAL_GROUPS_COUNT), Expr::val(1.0)),
            ))
            .limit(1)
            .input(estimated_and_actual_groups)
            .build();

        // join correction_factor_map with the new reduce.
        // correction_factor_map has 1 line result so the join type would be cross
        let reduce_with_correction: Relation = Relation::join()
            .left_names(
                new_reduce
                    .schema()
                    .iter()
                    .map(|f| f.name().to_string())
                    .collect(),
            )
            .right_names(vec![CORRECTION_FACTOR.to_string()])
            .cross()
            .left(new_reduce)
            .right(correction_factor_map)
            .build();

        // Extract field from the visited reduce
        let field_aggexpr_map: Vec<(&str, &AggregateColumn)> = reduce
            .schema()
            .fields()
            .iter()
            .map(|field| field.name())
            .zip(reduce.aggregate())
            .collect();

        // Apply corrections to aggregate function
        let new_exprs: Vec<(&str, Expr)> = field_aggexpr_map
            .into_iter()
            .map(|(name, agg)| match agg.aggregate() {
                aggregate::Aggregate::Count => (
                    name,
                    Expr::multiply(
                        Expr::col(CORRECTION_FACTOR),
                        Expr::multiply(Expr::val(input.1), Expr::col(name)),
                    ),
                ),
                aggregate::Aggregate::Sum => {
                    (name, Expr::multiply(Expr::val(input.1), Expr::col(name)))
                }
                aggregate::Aggregate::Mean => (
                    name,
                    Expr::divide(Expr::col(name), Expr::col(CORRECTION_FACTOR)),
                ),
                aggregate::Aggregate::First | aggregate::Aggregate::Last => (name, Expr::col(name)),
                // todo for aggregation function that we don't know how to correct yet such as MIN and MAX.
                _ => todo!(),
            })
            .collect();

        let new_map: Relation = Relation::map()
            .with((ROW_WEIGHT, Expr::col(ROW_WEIGHT)))
            .with_iter(new_exprs.into_iter())
            .input(reduce_with_correction)
            .build();

        RelationWithWeight(new_map, 1.0)
    }

    /// propagate table weight as table weight left * table weight right
    /// propagate ROW_WEIGHT as _LEFT_ROW_WEIGHT_ * _RIGHT_ROW_WEIGHT_
    fn join(
        &self,
        join: &'a Join,
        left: RelationWithWeight,
        right: RelationWithWeight,
    ) -> RelationWithWeight {
        let RelationWithWeight(left, left_weight) = &left;
        let RelationWithWeight(right, right_weight) = &right;

        let left_new_name = left.name().to_string();
        let right_new_name = right.name().to_string();
        let schema_names: Vec<String> =
            join.schema().iter().map(|f| f.name().to_string()).collect();

        let mut left_names = vec![format!("_LEFT{ROW_WEIGHT}")];
        left_names.extend(
            schema_names
                .iter()
                .take(join.left().schema().len())
                .cloned(),
        );

        let mut right_names = vec![format!("_RIGHT{ROW_WEIGHT}")];
        right_names.extend(
            schema_names
                .iter()
                .skip(join.left().schema().len())
                .cloned(),
        );

        // map old columns names (from the join) into new column names from the left and right
        let columns_mapping: Hierarchy<Identifier> = join
            .left()
            .schema()
            .iter()
            // skip 1 because the left (coming from the RelationWithWeight)
            // has the ROW_WEIGHT colum
            .zip(left.schema().iter().skip(PROPAGATED_COLUMNS))
            .map(|(o, n)| {
                (
                    vec![join.left().name().to_string(), o.name().to_string()],
                    Identifier::from(vec![left_new_name.clone(), n.name().to_string()]),
                )
            })
            .chain(
                join.right()
                    .schema()
                    .iter()
                    .zip(right.schema().iter().skip(PROPAGATED_COLUMNS))
                    .map(|(o, n)| {
                        (
                            vec![join.right().name().to_string(), o.name().to_string()],
                            Identifier::from(vec![right_new_name.clone(), n.name().to_string()]),
                        )
                    }),
            )
            .collect();

        let builder = Relation::join()
            .left_names(left_names)
            .right_names(right_names)
            .operator(join.operator().clone().rename(&columns_mapping))
            .left(left.clone())
            .right(right.clone());

        let join: Join = builder.build();

        let mut builder = Relation::map();
        builder = builder.with((
            ROW_WEIGHT,
            Expr::multiply(
                Expr::col(format!("_LEFT{ROW_WEIGHT}")),
                Expr::col(format!("_RIGHT{ROW_WEIGHT}")),
            ),
        ));
        builder = join.names().iter().fold(builder, |b, (p, n)| {
            if [ROW_WEIGHT].contains(&p[1].as_str()) {
                b
            } else {
                b.with((n, Expr::col(n)))
            }
        });
        let builder = builder.input(Arc::new(join.into()));

        let final_map: Relation = builder.build();

        RelationWithWeight(final_map, left_weight * right_weight)
    }

    fn set(
        &self,
        _set: &'a Set,
        _left: RelationWithWeight,
        _right: RelationWithWeight,
    ) -> RelationWithWeight {
        todo!()
    }

    fn values(&self, _values: &'a Values) -> RelationWithWeight {
        todo!()
    }
}

/// A visitor to samaple tables in a relation
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

    fn map(&self, map: &'a Map, input: Relation) -> Relation {
        Relation::map().with(map.clone()).input(input).build()
    }

    fn reduce(&self, reduce: &'a Reduce, input: Relation) -> Relation {
        Relation::reduce().with(reduce.clone()).input(input).build()
    }

    fn join(&self, join: &'a Join, left: Relation, right: Relation) -> Relation {
        let left_new_name = left.name().to_string();
        let right_new_name = right.name().to_string();

        // Preserve the schema names of the existing JOIN
        let schema_names: Vec<String> =
            join.schema().iter().map(|f| f.name().to_string()).collect();
        let left_names: Vec<String> = schema_names
            .iter()
            .take(join.left().schema().len())
            .cloned()
            .collect();
        let right_names: Vec<String> = schema_names
            .iter()
            .skip(join.left().schema().len())
            .cloned()
            .collect();

        let columns_mapping: Hierarchy<Identifier> = join
            .left()
            .schema()
            .iter()
            .zip(left.schema().iter())
            .map(|(o, n)| {
                (
                    vec![join.left().name().to_string(), o.name().to_string()],
                    Identifier::from(vec![left_new_name.clone(), n.name().to_string()]),
                )
            })
            .chain(
                join.right()
                    .schema()
                    .iter()
                    .zip(right.schema().iter())
                    .map(|(o, n)| {
                        (
                            vec![join.right().name().to_string(), o.name().to_string()],
                            Identifier::from(vec![right_new_name.clone(), n.name().to_string()]),
                        )
                    }),
            )
            .collect();

        // build the output relation
        Relation::join()
            .left_names(left_names)
            .right_names(right_names)
            .operator(join.operator().clone().rename(&columns_mapping))
            .left(left)
            .right(right)
            .build()
    }

    fn set(&self, _set: &'a Set, _left: Relation, _right: Relation) -> Relation {
        todo!()
    }

    fn values(&self, values: &'a Values) -> Relation {
        Relation::Values(values.clone())
    }
}

// Visitor builders

/// Creates a WeightRelationVisitor that applies the same weight to all tables
fn uniform_adjustment_table_visitor(
    weight: f64,
) -> WeightRelationVisitor<impl Fn(&Table) -> RelationWithWeight> {
    WeightRelationVisitor::new(move |table: &Table| {
        RelationWithWeight(Relation::from(table.clone()), weight)
    })
}

/// Creates a WeightRelationVisitor that applies the different weights to tables according to the input
fn differenciated_adjustment_table_visitor<'a>(
    relations: &'a Hierarchy<Arc<Relation>>,
    tables_and_weights: Vec<(Vec<String>, f64)>,
) -> WeightRelationVisitor<impl Fn(&Table) -> RelationWithWeight + 'a> {
    WeightRelationVisitor::new(move |table: &Table| {
        match tables_and_weights.iter().find(|(tab, _)| {
            relations
                .get(tab)
                .map_or(false, |rel| rel.name() == table.name())
        }) {
            Some((_, weight)) => RelationWithWeight(Relation::from(table.clone()), *weight),
            None => RelationWithWeight(Relation::from(table.clone()), 1.0),
        }
    })
}

/// Creates a TableSamplerVisitor that applies poisson sampling with the same probability to all tables.
fn poisson_sampling_table_visitor(proba: f64) -> TableSamplerVisitor<impl Fn(&Table) -> Relation> {
    TableSamplerVisitor::new(move |table: &Table| {
        Relation::from(table.clone()).poisson_sampling(proba)
    })
}

/// Creates a TableSamplerVisitor that applies poisson sampling with different probability to tables as specified in the input
fn differenciated_poisson_sampling_table_visitor<'a>(
    relations: &'a Hierarchy<Arc<Relation>>,
    tables_and_sampling_probabilities: Vec<(Vec<String>, f64)>,
) -> TableSamplerVisitor<impl Fn(&Table) -> Relation + 'a> {
    TableSamplerVisitor::new(move |table: &Table| {
        match tables_and_sampling_probabilities.iter().find(|(tab, _)| {
            relations
                .get(tab)
                .map_or(false, |rel| rel.name() == table.name())
        }) {
            Some((_, proba)) => Relation::from(table.clone()).poisson_sampling(*proba),
            None => table.clone().into(),
        }
    })
}

/// Creates a TableSamplerVisitor that applies random samping with without replacement using the same rate for all tables.
fn sampling_without_replacements_table_visitor(
    rate: f64,
    rate_multiplier: f64,
) -> TableSamplerVisitor<impl Fn(&Table) -> Relation> {
    TableSamplerVisitor::new(move |table: &Table| {
        Relation::from(table.clone()).sampling_without_replacements(rate, rate_multiplier)
    })
}

impl Relation {
    pub fn uniform_adjustment<'a>(&'a self, weight: f64) -> RelationWithWeight {
        self.accept(uniform_adjustment_table_visitor(weight))
    }

    pub fn differenciated_adjustment<'a>(
        &'a self,
        relations: &'a Hierarchy<Arc<Relation>>,
        tables_and_weights: Vec<(Vec<String>, f64)>,
    ) -> RelationWithWeight {
        self.accept(differenciated_adjustment_table_visitor(
            relations,
            tables_and_weights,
        ))
    }

    pub fn uniform_poisson_sampling<'a>(&'a self, proba: f64) -> Relation {
        self.accept(poisson_sampling_table_visitor(proba))
    }

    pub fn differenciated_poisson_sampling<'a>(
        &'a self,
        relations: &'a Hierarchy<Arc<Relation>>,
        tables_and_sampling_probabilities: Vec<(Vec<String>, f64)>,
    ) -> Relation {
        self.accept(differenciated_poisson_sampling_table_visitor(
            relations,
            tables_and_sampling_probabilities,
        ))
    }

    pub fn uniform_sampling_without_replacements<'a>(
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

#[cfg(feature = "tested_sampling_adjustment")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ast,
        display::Dot,
        io::{postgresql, Database},
        namer,
        sql::parse,
    };

    use colored::Colorize;
    use itertools::Itertools as _;

    #[test]
    fn test_uniform_poisson_sampling() {
        let database = postgresql::test_database();
        let proba = 0.1;
        let relations = database.relations();
        let relation = Relation::try_from(
            parse("SELECT * FROM order_table JOIN item_table ON id=order_id")
                .unwrap()
                .with(&relations),
        )
        .unwrap();
        namer::reset();
        let sampled = relation.uniform_poisson_sampling(proba);
        namer::reset();
        // build the resulting relation
        let item_table = relations
            .get(&["item_table".into()])
            .unwrap()
            .as_ref()
            .clone();

        let sampled_item_table = item_table.poisson_sampling(proba);

        let order_table = relations
            .get(&["order_table".into()])
            .unwrap()
            .as_ref()
            .clone();

        let sampled_order_table = order_table.poisson_sampling(proba);

        let relation_schema_names: Vec<&str> = relation.schema().iter().map(|f| f.name()).collect();
        let relation_name = relation.name();
        let left_names: Vec<String> = relation_schema_names
            .iter()
            .take(sampled_order_table.schema().len())
            .map(|s| s.to_string())
            .collect();
        let right_exprs: Vec<String> = relation_schema_names
            .iter()
            .skip(sampled_order_table.schema().len())
            .map(|s| s.to_string())
            .collect();

        let join: Relation = Relation::join()
            .left_names(left_names)
            .right_names(right_exprs)
            .left(sampled_order_table)
            .right(sampled_item_table)
            .on_eq("order_id", "id")
            .build();
        let exprs: Vec<(&str, Expr)> = join
            .schema()
            .iter()
            .map(|f| (f.name(), Expr::col(f.name())))
            .collect();

        let final_map: Relation = Relation::map()
            .name(relation_name)
            .with_iter(exprs.into_iter())
            .input(join)
            .build();

        assert_eq!(final_map, sampled);

        relation.display_dot().unwrap();
        sampled.display_dot().unwrap();
        final_map.display_dot().unwrap();
    }

    #[test]
    fn test_differenciated_poisson_sampling() {
        let database = postgresql::test_database();
        let tables_and_proba: Vec<(Vec<String>, f64)> = vec![
            (vec!["order_table".to_string()], 0.1),
            (vec!["item_table".to_string()], 0.5),
        ];
        let relations = database.relations();
        let relation = Relation::try_from(
            parse("SELECT * FROM order_table JOIN item_table ON id=order_id")
                .unwrap()
                .with(&relations),
        )
        .unwrap();
        namer::reset();
        let sampled = relation.differenciated_poisson_sampling(&relations, tables_and_proba);
        // build the resulting relation
        namer::reset();
        let item_table = relations
            .get(&["item_table".into()])
            .unwrap()
            .as_ref()
            .clone();

        let sampled_item_table = item_table.poisson_sampling(0.5);

        let order_table = relations
            .get(&["order_table".into()])
            .unwrap()
            .as_ref()
            .clone();

        let sampled_order_table = order_table.poisson_sampling(0.1);

        let relation_schema_names: Vec<&str> = relation.schema().iter().map(|f| f.name()).collect();
        let relation_name = relation.name();
        let left_names: Vec<String> = relation_schema_names
            .iter()
            .take(sampled_order_table.schema().len())
            .map(|s| s.to_string())
            .collect();
        let right_exprs: Vec<String> = relation_schema_names
            .iter()
            .skip(sampled_order_table.schema().len())
            .map(|s| s.to_string())
            .collect();
        let join: Relation = Relation::join()
            .left_names(left_names)
            .right_names(right_exprs)
            .left(sampled_order_table)
            .right(sampled_item_table)
            .on_eq("order_id", "id")
            .build();
        let exprs: Vec<(&str, Expr)> = join
            .schema()
            .iter()
            .map(|f| (f.name(), Expr::col(f.name())))
            .collect();

        let final_map: Relation = Relation::map()
            .name(relation_name)
            .with_iter(exprs.into_iter())
            .input(join)
            .build();

        assert_eq!(final_map, sampled);

        relation.display_dot().unwrap();
        sampled.display_dot().unwrap();
        final_map.display_dot().unwrap();
    }

    #[test]
    fn test_sampling_without_replacements() {
        let database = postgresql::test_database();
        let proba = 0.1;
        let relations = database.relations();
        namer::reset();
        let relation = Relation::try_from(
            parse("SELECT * FROM order_table JOIN item_table ON id=order_id")
                .unwrap()
                .with(&relations),
        )
        .unwrap();
        namer::reset();
        let sampled = relation.uniform_sampling_without_replacements(proba, 1.0);
        // build the resulting relation
        namer::reset();
        let item_table = relations
            .get(&["item_table".into()])
            .unwrap()
            .as_ref()
            .clone();

        let sampled_item_table = item_table.sampling_without_replacements(proba, 1.0);

        let order_table = relations
            .get(&["order_table".into()])
            .unwrap()
            .as_ref()
            .clone();

        let sampled_order_table = order_table.sampling_without_replacements(proba, 1.0);

        let relation_schema_names: Vec<&str> = relation.schema().iter().map(|f| f.name()).collect();
        let relation_name = relation.name();
        let left_names: Vec<String> = relation_schema_names
            .iter()
            .take(sampled_order_table.schema().len())
            .map(|s| s.to_string())
            .collect();
        let right_exprs: Vec<String> = relation_schema_names
            .iter()
            .skip(sampled_order_table.schema().len())
            .map(|s| s.to_string())
            .collect();

        let join: Relation = Relation::join()
            .left_names(left_names)
            .right_names(right_exprs)
            .left(sampled_order_table)
            .right(sampled_item_table)
            .on_eq("order_id", "id")
            .build();
        let exprs: Vec<(&str, Expr)> = join
            .schema()
            .iter()
            .map(|f| (f.name(), Expr::col(f.name())))
            .collect();

        let final_map: Relation = Relation::map()
            .name(relation_name)
            .with_iter(exprs.into_iter())
            .input(join)
            .build();

        assert_eq!(final_map, sampled);

        relation.display_dot().unwrap();
        sampled.display_dot().unwrap();
        final_map.display_dot().unwrap();
    }

    #[test]
    fn test_table_with_weight() {
        let mut database = postgresql::test_database();
        let weight: f64 = 2.0;
        let relations = database.relations();
        let table = relations
            .get(&["item_table".into()])
            .unwrap()
            .as_ref()
            .clone();
        let uniform_weighted_relation = table.uniform_adjustment(weight);
        uniform_weighted_relation.relation().display_dot().unwrap();
        assert!(*uniform_weighted_relation.weight() == 2.0);

        let differenciated_weighted_relation = table
            .differenciated_adjustment(&relations, vec![(vec!["item_table".to_string()], weight)]);
        differenciated_weighted_relation
            .relation()
            .display_dot()
            .unwrap();
        assert!(*differenciated_weighted_relation.weight() == 2.0);

        assert_eq!(
            uniform_weighted_relation.relation(),
            differenciated_weighted_relation.relation()
        );

        let query_from_rel: &str =
            &ast::Query::from(&(uniform_weighted_relation.relation())).to_string();
        println!("\n{}", format!("{query_from_rel}").yellow());
        println!(
            "\n{}\n",
            database
                .query(query_from_rel)
                .unwrap()
                .iter()
                .map(ToString::to_string)
                .join("\n")
        );
    }

    #[test]
    fn test_map_with_weight() {
        let mut database = postgresql::test_database();
        let weight: f64 = 2.0;
        let relations = database.relations();
        let query = "SELECT * FROM item_table";
        let relation = Relation::try_from(parse(query).unwrap().with(&relations)).unwrap();

        let uniform_weighted_relation = relation.uniform_adjustment(weight);
        uniform_weighted_relation.relation().display_dot().unwrap();
        assert!(*uniform_weighted_relation.weight() == 2.0);

        let differenciated_weighted_relation = relation
            .differenciated_adjustment(&relations, vec![(vec!["item_table".to_string()], weight)]);
        differenciated_weighted_relation
            .relation()
            .display_dot()
            .unwrap();
        assert!(*differenciated_weighted_relation.weight() == 2.0);

        assert_eq!(
            uniform_weighted_relation.relation(),
            differenciated_weighted_relation.relation()
        );

        let query_from_rel: &str =
            &ast::Query::from(&(uniform_weighted_relation.relation())).to_string();
        println!("\n{}", format!("{query_from_rel}").yellow());
        println!(
            "\n{}\n",
            database
                .query(query_from_rel)
                .unwrap()
                .iter()
                .map(ToString::to_string)
                .join("\n")
        );
    }

    #[test]
    fn test_reduce_with_weight() {
        let mut database = postgresql::test_database();
        let weight: f64 = 2.0;
        let relations = database.relations();

        let query = "
        WITH tmp AS (
            SELECT
                id,
                AVG(income) AS avg_income
            FROM large_user_table
            GROUP BY id
        ) SELECT AVG(avg_income) FROM tmp";

        let relation = Relation::try_from(parse(query).unwrap().with(&relations)).unwrap();
        relation.display_dot().unwrap();
        let uniform_weighted_relation = relation.uniform_adjustment(weight);
        uniform_weighted_relation.relation().display_dot().unwrap();
        assert!(*uniform_weighted_relation.weight() == 1.0);
        let differenciated_weighted_relation = relation.differenciated_adjustment(
            &relations,
            vec![(vec!["large_user_table".to_string()], weight)],
        );
        differenciated_weighted_relation
            .relation()
            .display_dot()
            .unwrap();
        assert!(*differenciated_weighted_relation.weight() == 1.0);

        assert_eq!(
            uniform_weighted_relation.relation(),
            differenciated_weighted_relation.relation()
        );

        let query_from_rel: &str =
            &ast::Query::from(&(uniform_weighted_relation.relation())).to_string();
        println!("\n{}", format!("{query_from_rel}").yellow());
        println!(
            "\n{}\n",
            database
                .query(query_from_rel)
                .unwrap()
                .iter()
                .map(ToString::to_string)
                .join("\n")
        );
    }

    #[test]
    fn test_join_with_weight() {
        let mut database = postgresql::test_database();
        let weight: f64 = 2.0;
        let relations = database.relations();
        let relation = Relation::try_from(
            parse("SELECT * FROM order_table JOIN item_table ON id=order_id LIMIT 10")
                .unwrap()
                .with(&relations),
        )
        .unwrap();
        let uniform_weighted_relation = relation.uniform_adjustment(weight);
        uniform_weighted_relation.relation().display_dot().unwrap();
        assert!(*uniform_weighted_relation.weight() == 4.0);

        let differenciated_weighted_relation = relation.differenciated_adjustment(
            &relations,
            vec![
                (vec!["item_table".to_string()], weight),
                (vec!["order_table".to_string()], weight),
            ],
        );
        differenciated_weighted_relation
            .relation()
            .display_dot()
            .unwrap();
        assert!(*differenciated_weighted_relation.weight() == 4.0);

        assert_eq!(
            uniform_weighted_relation.relation(),
            differenciated_weighted_relation.relation()
        );

        let query_from_rel: &str =
            &ast::Query::from(&(uniform_weighted_relation.relation())).to_string();
        println!("\n{}", format!("{query_from_rel}").yellow());
        println!(
            "\n{}\n",
            database
                .query(query_from_rel)
                .unwrap()
                .iter()
                .map(ToString::to_string)
                .join("\n")
        );
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
        use_poisson_sampling: bool,
    ) {
        let mut database = postgresql::test_database();

        print!("\nFROM WEIGHTED RELATION partial reslts:\n");
        let mut res_holder: Vec<Vec<f64>> = vec![];
        for _ in 0..n_experiments {
            let sampled_relation = if use_poisson_sampling {
                relation_to_sample.uniform_poisson_sampling(proba)
            } else {
                relation_to_sample.uniform_sampling_without_replacements(proba, 2.0)
            };
            let weighted_sampled_relation: RelationWithWeight =
                sampled_relation.uniform_adjustment(1.0 / proba);

            let weighted_filtered_sampled_relation =
                (weighted_sampled_relation.0).filter_fields(|n| n != ROW_WEIGHT);

            let query_weighted_relation: &str =
                &ast::Query::from(&weighted_filtered_sampled_relation).to_string();
            let res = database.query(query_weighted_relation).unwrap();
            assert!(res.len() == 1);
            let float_res: Vec<f64> = res[0]
                .iter()
                .filter_map(|f| f.to_string().parse::<f64>().ok())
                .collect();
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

        print!("\nFROM WEIGHTED RELATION:\n{:?}", avg_res);
        print!("\nFROM SOURCE RELATION:\n{:?}", source_res);
        print!("\nErrors:\n{:?}", errs);

        //for displaying purposes
        relation_to_sample.display_dot().unwrap();
        let sampled_relation = if use_poisson_sampling {
            relation_to_sample.uniform_poisson_sampling(proba)
        } else {
            // use a relatively safe rate_multiplier (2.0) for optimization purposes
            relation_to_sample.uniform_sampling_without_replacements(proba, 2.0)
        };

        let weighted_sampled_relation = sampled_relation.uniform_adjustment(1.0 / proba);
        weighted_sampled_relation.0.display_dot().unwrap();
        let query_weighted_relation: &str =
            &ast::Query::from(&(weighted_sampled_relation.0)).to_string();
        print!("Final weight: {}\n", weighted_sampled_relation.1);
        println!(
            "\nOriginal query \n{}",
            format!("{query_unsampled_relation}").yellow()
        );
        println!(
            "\nFinal readdressed query \n{}",
            format!("{query_weighted_relation}").yellow()
        );
    }

    #[cfg(feature = "tested_sampling_adjustment")]
    #[test]
    fn test_adjustment_simple_reduce() {
        let database = postgresql::test_database();
        let relations: Hierarchy<Arc<Relation>> = database.relations();

        let query = "SELECT COUNT(order_id), SUM(price), AVG(price) FROM item_table";

        let weight: f64 = 5.0;
        let fraction: f64 = 1.0 / weight;
        let relation = Relation::try_from(parse(query).unwrap().with(&relations)).unwrap();

        collect_results_from_many_samples(&relation, fraction, 100, true)
    }

    #[cfg(feature = "tested_sampling_adjustment")]
    #[test]
    fn test_adjustment_join_reduce() {
        let database = postgresql::test_database();
        let relations: Hierarchy<Arc<Relation>> = database.relations();

        let query = "SELECT COUNT(id), SUM(price), AVG(price) FROM order_table JOIN item_table ON id=order_id";

        let weight: f64 = 2.0;
        let fraction: f64 = 1.0 / weight;
        let relation = Relation::try_from(parse(query).unwrap().with(&relations)).unwrap();

        collect_results_from_many_samples(&relation, fraction, 100, true)
    }

    #[cfg(feature = "tested_sampling_adjustment")]
    #[test]
    fn test_adjustment_reduce_reduce() {
        let database = postgresql::test_database();
        let relations: Hierarchy<Arc<Relation>> = database.relations();

        let query = "
        WITH tmp1 AS (
          SELECT
            id,
            AVG(income) AS avg_inc,
            SUM(income) AS sum_inc,
            COUNT(city) AS count_city
        FROM large_user_table GROUP BY id
        )
        SELECT
            COUNT(id),
            AVG(avg_inc) AS avg_avg_inc,
            SUM(avg_inc) AS sum_avg_inc,
            AVG(sum_inc) AS avg_sum_inc,
            SUM(sum_inc) AS sum_sum_inc,
            AVG(count_city) AS avg_count_city,
            SUM(count_city) AS sum_count_city
        FROM tmp1;";

        let weight: f64 = 6.0;
        let fraction: f64 = 1.0 / weight;
        let relation = Relation::try_from(parse(query).unwrap().with(&relations)).unwrap();

        collect_results_from_many_samples(&relation, fraction, 100, true)
    }

    #[cfg(feature = "tested_sampling_adjustment")]
    #[test]
    fn test_adjustment_reduce_join_reduce() {
        let database = postgresql::test_database();
        let relations: Hierarchy<Arc<Relation>> = database.relations();

        // bug with USING (col)
        let query = "
        WITH
        tmp1 AS (select city, name, income from large_user_table),
        tmp2 AS (select city, SUM(age) AS sum_age from user_table GROUP BY city),
        tmp3 AS (SELECT name, income, sum_age FROM tmp1 JOIN tmp2 ON(tmp1.city=tmp2.city))
        SELECT COUNT(name), SUM(sum_age), AVG(income) FROM tmp3
        ";

        let weight = 4.0;
        let fraction: f64 = 1.0 / weight;
        let relation = Relation::try_from(parse(query).unwrap().with(&relations)).unwrap();
        collect_results_from_many_samples(&relation, fraction, 100, true)
    }

    #[cfg(feature = "tested_sampling_adjustment")]
    #[test]
    fn test_adjustment_join_reduce_reduce() {
        let database = postgresql::test_database();
        let relations: Hierarchy<Arc<Relation>> = database.relations();

        // 2 reduce after the join
        let query = "
         WITH
         tmp1 AS (SELECT user_id FROM order_table),
         tmp2 AS (SELECT id, income, city FROM large_user_table),
         tmp3 AS (SELECT income, city FROM tmp1 JOIN tmp2 ON tmp1.user_id=tmp2.id),
         tmp4 AS (
            SELECT
                city,
                COUNT(income) AS count_income,
                SUM(income) AS sum_income,
                AVG(income) AS avg_income
            FROM tmp3
            GROUP BY city
        )
         SELECT COUNT(count_income), SUM(sum_income), AVG(avg_income), AVG(count_income), AVG(sum_income) FROM tmp4
         ";
        let weight = 4.0;
        let fraction: f64 = 1.0 / weight;
        let relation = Relation::try_from(parse(query).unwrap().with(&relations)).unwrap();
        collect_results_from_many_samples(&relation, fraction, 100, true)
    }

    #[cfg(feature = "tested_sampling_adjustment")]
    #[test]
    fn test_adjustment_reduce_reduce_reduce() {
        let database = postgresql::test_database();
        let relations: Hierarchy<Arc<Relation>> = database.relations();

        let weight: f64 = 2.0;
        let fraction: f64 = 1.0 / weight;

        let query = "
        WITH tmp1 AS (
          SELECT id, city, AVG(income) AS avg_inc, SUM(income) AS sum_inc FROM large_user_table GROUP BY id, city
        ),
        tmp2 AS (
            SELECT
                id,
                COUNT(city) AS count_city,
                AVG(avg_inc) AS avg_avg_inc,
                SUM(avg_inc) AS sum_avg_inc,
                AVG(sum_inc) AS avg_sum_inc,
                SUM(sum_inc) AS sum_sum_inc
            FROM tmp1 GROUP BY id
          )
        SELECT
            COUNT(id) AS count_id,
            SUM(count_city) AS sum_count_city,
            AVG(count_city) AS avg_count_city,
            AVG(avg_avg_inc) AS avg_avg_avg_inc,
            AVG(sum_avg_inc) AS avg_sum_avg_inc,
            SUM(avg_sum_inc) AS sum_avg_sum_inc,
            SUM(sum_sum_inc) AS sum_sum_sum_inc
        FROM tmp2;";

        let relation = Relation::try_from(parse(query).unwrap().with(&relations)).unwrap();

        collect_results_from_many_samples(&relation, fraction, 100, true)
    }

    #[cfg(feature = "tested_sampling_adjustment")]
    #[test]
    fn test_adjustment_reduce_reduce_join_reduce() {
        let database = postgresql::test_database();
        let relations: Hierarchy<Arc<Relation>> = database.relations();

        let weight: f64 = 2.0;
        let fraction: f64 = 1.0 / weight;

        let query = "
        WITH
        tmp1 AS (
            SELECT
                id,
                COUNT(user_id) AS count_user_id,
                AVG(user_id) AS avg_user_id,
                SUM(user_id) AS sum_user_id
            FROM order_table
            GROUP BY id
        ),
        tmp2 AS (
            SELECT
                id,
                AVG(income) AS avg_income,
                SUM(income) AS sum_income
            FROM large_user_table
            GROUP BY id
        ),
        tmp3 AS (
            SELECT
                count_user_id,
                avg_user_id,
                avg_income,
                sum_user_id,
                sum_income
            FROM tmp1
            JOIN tmp2 ON (tmp1.id = tmp2.id)
        )
        SELECT
            COUNT(count_user_id),
            AVG(avg_user_id),
            AVG(avg_income),
            SUM(sum_user_id),
            AVG(sum_income),
            SUM(avg_income)
        FROM tmp3
        ";

        let relation = Relation::try_from(parse(query).unwrap().with(&relations)).unwrap();

        collect_results_from_many_samples(&relation, fraction, 100, true)
    }
}
