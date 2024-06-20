use crate::{
    builder::{Ready, With},
    expr::identifier::Identifier,
    hierarchy::Hierarchy,
    relation::{Join, Map, Reduce, Relation, Set, Table, Values, Variant as _, Visitor},
    visitor::Acceptor,
};
use std::{ops::Deref, sync::Arc, vec};

use super::Result;

struct ComposerVisitor<F: Fn(&Table) -> Relation> {
    composer: F,
}

impl<F: Fn(&Table) -> Relation> ComposerVisitor<F> {
    pub fn new(composer: F) -> Self {
        ComposerVisitor { composer }
    }
}

impl<'a, F: Fn(&Table) -> Relation> Visitor<'a, Relation> for ComposerVisitor<F> {
    fn table(&self, table: &'a Table) -> Relation {
        (self.composer)(table)
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

    fn set(&self, set: &'a Set, left: Relation, right: Relation) -> Relation {
        Relation::set()
            .with(set.clone())
            .left(left)
            .right(right)
            .build()
    }

    fn values(&self, values: &'a Values) -> Relation {
        Relation::Values(values.clone())
    }
}

fn composer_visitor<'a>(
    relations: &'a Hierarchy<Arc<Relation>>,
) -> ComposerVisitor<impl Fn(&Table) -> Relation + 'a> {
    ComposerVisitor::new(move |table: &Table| match relations.get(table.path()) {
        Some(r) => r.deref().clone(),
        None => table.clone().into(),
    })
}

impl Relation {
    pub fn compose<'a>(&'a self, relations: &'a Hierarchy<Arc<Relation>>) -> Relation {
        self.accept(composer_visitor(relations))
    }
}

impl Hierarchy<Arc<Relation>> {
    /// It composes itself with another Hierarchy of relations.
    /// It substitute its Tables with the corresponding relation in relations
    /// with the same path.
    /// The output Hierarchy of relations will have the same paths as self.
    /// Schemas in the relations to be composed should be compatible with
    /// the schema of the corresponding table otherwise an error is raised.
    pub fn compose<'a>(
        &'a self,
        relations: &'a Hierarchy<Arc<Relation>>,
    ) -> Result<Hierarchy<Arc<Relation>>> {
        Ok(self
            .iter()
            .map(|(outer_rel_path, rel)| (outer_rel_path.clone(), Arc::new(rel.compose(relations))))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use crate::{display::Dot as _, hierarchy::Path, relation::Schema, DataType, Expr};

    use super::*;

    fn build_complex_relation_1() -> Arc<Relation> {
        let schema: Schema = vec![
            ("a", DataType::float()),
            ("b", DataType::float_interval(-2., 2.)),
            ("c", DataType::float()),
            ("d", DataType::float_interval(0., 1.)),
        ]
        .into_iter()
        .collect();
        let table: Arc<Relation> = Arc::new(
            Relation::table()
                .name("table")
                .schema(schema.clone())
                .size(1000)
                .build(),
        );
        let map: Arc<Relation> = Arc::new(
            Relation::map()
                .name("map_1")
                .with(Expr::exp(Expr::col("a")))
                .input(table.clone())
                .with(Expr::col("b") + Expr::col("d"))
                .build(),
        );

        let other_map: Arc<Relation> = Arc::new(
            Relation::map()
                .with(Expr::col("a"))
                .with(Expr::col("b"))
                .with(Expr::col("c"))
                .with(Expr::col("d"))
                .input(table.clone())
                .build(),
        );
        let join: Arc<Relation> = Arc::new(
            Relation::join()
                .name("join")
                .cross()
                .left(other_map.clone())
                .right(map.clone())
                .build(),
        );
        let map_2: Arc<Relation> = Arc::new(
            Relation::map()
                .name("map_2")
                .with(Expr::exp(Expr::col(join[4].name())))
                .input(join.clone())
                .with(Expr::col(join[0].name()) + Expr::col(join[1].name()))
                .build(),
        );
        let join_2: Arc<Relation> = Arc::new(
            Relation::join()
                .name("join_2")
                .cross()
                .left(join.clone())
                .right(map_2.clone())
                .build(),
        );
        join_2
    }

    fn build_complex_relation_2() -> Arc<Relation> {
        let table1: Arc<Relation> = Arc::new(
            Relation::table()
                .name("table1")
                .path(["private", "table1"].path())
                .schema(
                    vec![
                        ("a", DataType::float()),
                        ("b", DataType::float_interval(-2., 2.)),
                        ("c", DataType::float()),
                        ("d", DataType::float_interval(0., 1.)),
                    ]
                    .into_iter()
                    .collect::<Schema>(),
                )
                .size(1000)
                .build(),
        );

        let table2: Arc<Relation> = Arc::new(
            Relation::table()
                .name("table2")
                .path(["private", "table2"].path())
                .schema(
                    vec![
                        ("e", DataType::integer_interval(-10, 10)),
                        ("f", DataType::text()),
                    ]
                    .into_iter()
                    .collect::<Schema>(),
                )
                .size(300)
                .build(),
        );

        let join: Arc<Relation> = Arc::new(
            Relation::join()
                .cross()
                .left(table1.clone())
                .right(table2.clone())
                .build(),
        );
        let map_2: Arc<Relation> = Arc::new(
            Relation::map()
                .with(Expr::exp(Expr::col(join[4].name())))
                .input(join.clone())
                .with(Expr::col(join[0].name()) + Expr::col(join[1].name()))
                .build(),
        );
        map_2
    }

    #[test]
    fn test_simple_renamer() {
        let binding = build_complex_relation_1();
        let rel = binding.deref();
        rel.display_dot().unwrap();

        let schema: Schema = vec![
            ("a", DataType::float()),
            ("b", DataType::float_interval(-2., 2.)),
            ("c", DataType::float()),
            ("d", DataType::float_interval(0., 1.)),
            ("e", DataType::float()),
            ("f", DataType::float_interval(-2., 2.)),
            ("g", DataType::float()),
            ("h", DataType::float_interval(0., 1.)),
        ]
        .into_iter()
        .collect();

        let table_1: Relation = Relation::table()
            .name("real_table")
            .schema(schema)
            .size(10000)
            .build();

        let relations = Hierarchy::from([(vec!["table"], Arc::new(table_1))]);

        let composed = rel.compose(&relations);
        composed.display_dot().unwrap();
    }

    #[test]
    fn test_simple_composition() {
        let binding = build_complex_relation_1();
        let rel = binding.deref();
        rel.display_dot().unwrap();

        let schema: Schema = vec![
            ("a", DataType::float()),
            ("b", DataType::float_interval(-2., 2.)),
            ("c", DataType::float()),
            ("d", DataType::float_interval(0., 1.)),
        ]
        .into_iter()
        .collect();

        let table_1: Relation = Relation::table()
            .name("real_table")
            .schema(schema)
            .size(10000)
            .build();
        let map: Relation = Relation::map()
            .with(("a", Expr::col("a")))
            .with(("b", Expr::col("b")))
            .with(("c", Expr::col("c")))
            .with(("d", Expr::col("d")))
            .filter(Expr::gt(Expr::col("a"), Expr::val(0.5)))
            .input(table_1.clone())
            .build();

        map.display_dot().unwrap();
        let relations = Hierarchy::from([(vec!["table"], Arc::new(map))]);

        let composed = rel.compose(&relations);
        composed.display_dot().unwrap();
    }

    #[test]
    fn test_compose_with_different_schema() {
        let binding = build_complex_relation_1();
        let rel = binding.deref();
        rel.display_dot().unwrap();

        // Composing with a relation that has one more field
        let table_1: Relation = Relation::table()
            .name("real_table")
            .schema(
                vec![
                    ("a", DataType::float()),
                    ("b", DataType::float_interval(-2., 2.)),
                    ("c", DataType::float()),
                    ("d", DataType::float_interval(0., 1.)),
                    ("e", DataType::text()),
                ]
                .into_iter()
                .collect::<Schema>(),
            )
            .size(10000)
            .build();
        let map: Relation = Relation::map()
            .with(("a", Expr::col("a")))
            .with(("b", Expr::col("b")))
            .with(("c", Expr::col("c")))
            .with(("d", Expr::col("d")))
            .with(("e", Expr::col("e")))
            .filter(Expr::gt(Expr::col("a"), Expr::val(0.5)))
            .input(table_1.clone())
            .build();

        map.display_dot().unwrap();
        let relations = Hierarchy::from([(vec!["table"], Arc::new(map))]);

        let composed = rel.compose(&relations);
        composed.display_dot().unwrap();
    }

    #[test]
    fn test_compose_relations() {
        let r = build_complex_relation_1();
        r.deref().display_dot().unwrap();

        let b = build_complex_relation_2();
        b.deref().display_dot().unwrap();

        // building outer relations
        let outer_relations = Hierarchy::from([
            (vec!["my", "first", "relation"], build_complex_relation_1()),
            (vec!["my", "second", "relation"], build_complex_relation_2()),
        ]);

        // building inner relations to be composed with the outer
        let table_x: Relation = Relation::table()
            .name("real_table")
            .schema(
                vec![
                    ("x", DataType::integer_interval(-2, 2)),
                    ("y", DataType::integer_interval(0, 10)),
                ]
                .into_iter()
                .collect::<Schema>(),
            )
            .size(1000)
            .build();

        // map substitutes the table in build_complex_relation_1
        let map: Relation = Relation::map()
            .with(("a", Expr::col("x")))
            .with(("b", Expr::col("x") + Expr::col("y")))
            .with(("c", Expr::col("x") * Expr::col("y")))
            .with(("d", Expr::col("y")))
            .filter(Expr::gt(Expr::col("x"), Expr::val(0.5)))
            .input(table_x.clone())
            .build();

        // table_1 substitutes the table_1 in build_complex_relation_2
        let table_1: Relation = Relation::table()
            .name("real_table1")
            .schema(
                vec![
                    ("a", DataType::float()),
                    ("b", DataType::float_interval(-2., 2.)),
                    ("c", DataType::float()),
                    ("d", DataType::float_interval(0., 1.)),
                ]
                .into_iter()
                .collect::<Schema>(),
            )
            .size(1000)
            .build();

        // table_2 substitutes the table_1 in build_complex_relation_2
        let table_2: Relation = Relation::table()
            .name("real_table2")
            .schema(
                vec![("e", DataType::text()), ("x", DataType::text())]
                    .into_iter()
                    .collect::<Schema>(),
            )
            .size(1000)
            .build();

        let inner_relations = Hierarchy::from([
            (vec!["table"], Arc::new(map)),
            (vec!["private", "table1"], Arc::new(table_1)),
            (vec!["private", "table2"], Arc::new(table_2)),
        ]);

        let composed = outer_relations.compose(&inner_relations).unwrap();
        assert!(composed.contains_key(&["my", "first", "relation"].path()));
        assert!(composed.contains_key(&["my", "second", "relation"].path()));
        let first = composed
            .get(&["my", "first", "relation"].path())
            .unwrap()
            .deref();
        let second = composed
            .get(&["my", "second", "relation"].path())
            .unwrap()
            .deref();
        first.display_dot().unwrap();
        second.display_dot().unwrap();
    }
}
