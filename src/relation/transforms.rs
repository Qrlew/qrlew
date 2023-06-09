//! A few transforms for relations
//!

use std::{ops::Deref, rc::Rc};

use itertools::Itertools;

use super::{Table, Map, Reduce, Join, Set, Relation, Variant as _};
use crate::{
    builder::{Ready, With, WithIterator},
    expr::Expr,
    hierarchy::Hierarchy,
    DataType,
};

/* Reduce
 */

impl Table {
    /// Rename a Table
    pub fn with_name(mut self, name: String) -> Table {
        self.name = name;
        self
    }
}

/* Map
 */

impl Map {
    /// Rename a Map
    pub fn with_name(mut self, name: String) -> Map {
        self.name = name;
        self
    }
    /// Prepend a field to a Map
    pub fn with_field(self, name: &str, expr: Expr) -> Map {
        Relation::map().with((name, expr)).with(self).build()
    }
    /// Insert a field in a Map at position index
    pub fn insert_field(self, index: usize, inserted_name: &str, inserted_expr: Expr) -> Map {
        let Map {
            name,
            projection,
            filter,
            order_by,
            limit,
            schema,
            input,
            ..
        } = self;
        let mut builder = Map::builder().name(name);
        let field_exprs: Vec<_> = schema.into_iter().zip(projection).collect();
        for (f, e) in &field_exprs[0..index] {
            builder = builder.with((f.name().to_string(), e.clone()));
        }
        builder = builder.with((inserted_name, inserted_expr));
        for (f, e) in &field_exprs[index..field_exprs.len()] {
            builder = builder.with((f.name().to_string(), e.clone()));
        }
        // Filter
        builder = filter.into_iter().fold(builder, |b, f| b.filter(f));
        // Order by
        builder = order_by
            .into_iter()
            .fold(builder, |b, o| b.order_by(o.expr, o.asc));
        // Limit
        builder = limit.into_iter().fold(builder, |b, l| b.limit(l));
        builder
            .input(input)
            .build()
    }
    /// Filter fields
    pub fn filter_fields<P: Fn(&str) -> bool>(self, predicate: P) -> Map {
        Relation::map().filter_with(self, predicate).build()
    }
    /// Map fields
    pub fn map_fields<F: Fn(&str, Expr) -> Expr>(self, f: F) -> Map {
        Relation::map().map_with(self, f).build()
    }
}

// A few utility objects
#[derive(Clone, Debug)]
pub struct Step<'a> {
    pub referring_id: &'a str,
    pub referred_relation: &'a str,
    pub referred_id: &'a str,
}

impl<'a> From<(&'a str, &'a str, &'a str)> for Step<'a> {
    fn from((referring_id, referred_relation, referred_id): (&'a str, &'a str, &'a str)) -> Self {
        Step {
            referring_id,
            referred_relation,
            referred_id,
        }
    }
}

/* Reduce
 */

 impl Reduce {
    /// Rename a Reduce
    pub fn with_name(mut self, name: String) -> Reduce {
        self.name = name;
        self
    }
}

/* Join
 */

 impl Join {
    /// Rename a Join
    pub fn with_name(mut self, name: String) -> Join {
        self.name = name;
        self
    }
}

/* Set
 */

 impl Set {
    /// Rename a Join
    pub fn with_name(mut self, name: String) -> Set {
        self.name = name;
        self
    }
}

#[derive(Clone, Debug)]
pub struct Path<'a>(pub Vec<Step<'a>>);

impl<'a> Deref for Path<'a> {
    type Target = Vec<Step<'a>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> FromIterator<&'a (&'a str, &'a str, &'a str)> for Path<'a> {
    fn from_iter<T: IntoIterator<Item = &'a (&'a str, &'a str, &'a str)>>(iter: T) -> Self {
        Path(
            iter.into_iter()
                .map(|(referring_id, referred_relation, referred_id)| Step {
                    referring_id,
                    referred_relation,
                    referred_id,
                })
                .collect(),
        )
    }
}

impl<'a> IntoIterator for Path<'a> {
    type Item = Step<'a>;
    type IntoIter = <Vec<Step<'a>> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

/// A link to a relation and a field to keep with a new name
#[derive(Clone, Debug)]
pub struct ReferredField<'a> {
    pub referring_id: &'a str,
    pub referred_relation: &'a str,
    pub referred_id: &'a str,
    pub referred_field: &'a str,
    pub referred_field_name: &'a str,
}

#[derive(Clone, Debug)]
pub struct FieldPath<'a>(pub Vec<ReferredField<'a>>);

impl<'a> FieldPath<'a> {
    pub fn from_path(
        path: Path<'a>,
        referred_field: &'a str,
        referred_field_name: &'a str,
    ) -> Self {
        let mut field_path = FieldPath(Vec::new());
        let mut last_step: Option<Step> = None;
        // Fill the vec
        for step in path {
            if let Some(last_step) = &mut last_step {
                field_path.0.push(ReferredField {
                    referring_id: last_step.referring_id,
                    referred_relation: last_step.referred_relation,
                    referred_id: last_step.referred_id,
                    referred_field: step.referring_id,
                    referred_field_name,
                });
                *last_step = Step {
                    referring_id: referred_field_name,
                    referred_relation: step.referred_relation,
                    referred_id: step.referred_id,
                };
            } else {
                last_step = Some(step);
            }
        }
        if let Some(last_step) = last_step {
            field_path.0.push(ReferredField {
                referring_id: last_step.referring_id,
                referred_relation: last_step.referred_relation,
                referred_id: last_step.referred_id,
                referred_field,
                referred_field_name,
            });
        }
        field_path
    }
}

impl<'a> Deref for FieldPath<'a> {
    type Target = Vec<ReferredField<'a>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> IntoIterator for FieldPath<'a> {
    type Item = ReferredField<'a>;
    type IntoIter = <Vec<ReferredField<'a>> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl Relation {
    /// Rename a Relation
    pub fn with_name(self, name: String) -> Relation {
        match self {
            Relation::Table(t) => t.with_name(name).into(),
            Relation::Map(m) => m.with_name(name).into(),
            Relation::Reduce(r) => r.with_name(name).into(),
            Relation::Join(j) => j.with_name(name).into(),
            Relation::Set(s) => s.with_name(name).into(),
        }
    }
    /// Add a field that derives from existing fields
    pub fn identity_with_field(self, name: &str, expr: Expr) -> Relation {
        Relation::map()
            .with((name, expr))
            .with_iter(
                self.schema()
                    .iter()
                    .map(|f| (f.name(), Expr::col(f.name()))),
            )
            .input(self)
            .build()
    }
    /// Insert a field that derives from existing fields
    pub fn identity_insert_field(self, index: usize, inserted_name: &str, inserted_expr: Expr) -> Relation {
        let mut builder = Relation::map();
        let named_exprs: Vec<_> = self.schema().iter().map(|f| (f.name(), Expr::col(f.name()))).collect();
        for (n, e) in &named_exprs[0..index] {
            builder = builder.with((n.to_string(), e.clone()));
        }
        builder = builder.with((inserted_name, inserted_expr));
        for (n, e) in &named_exprs[index..named_exprs.len()] {
            builder = builder.with((n.to_string(), e.clone()));
        }
        builder.input(self).build()
    }
    /// Add a field that derives from input fields
    pub fn with_field(self, name: &str, expr: Expr) -> Relation {
        match self {
            // Simply add a column on Maps
            Relation::Map(map) => map.with_field(name, expr).into(),
            relation => relation.identity_with_field(name, expr),
        }
    }
    /// Insert a field that derives from input fields
    pub fn insert_field(self, index: usize, inserted_name: &str, inserted_expr: Expr) -> Relation {
        match self {
            // Simply add a column on Maps
            Relation::Map(map) => map.insert_field(index, inserted_name, inserted_expr).into(),
            relation => relation.identity_insert_field(index, inserted_name, inserted_expr),
        }
    }
    /// Add a field designated with a foreign relation and a field
    pub fn with_referred_field<'a>(
        self,
        referring_id: &'a str,
        referred_relation: Rc<Relation>,
        referred_id: &'a str,
        referred_field: &'a str,
        referred_field_name: &'a str,
    ) -> Relation {
        let left_size = referred_relation.schema().len();
        let names: Vec<String> = self
            .schema()
            .iter()
            .map(|f| f.name().to_string())
            .filter(|name| name != referred_field_name)
            .collect();
        let join: Relation = Relation::join()
            .inner()
            .on(Expr::eq(
                Expr::qcol(self.name(), referring_id),
                Expr::qcol(referred_relation.name(), referred_id),
            ))
            .left(referred_relation)
            .right(self)
            .build();
        let left: Vec<_> = join
            .schema()
            .iter()
            .zip(join.input_fields())
            .take(left_size)
            .collect();
        let right: Vec<_> = join
            .schema()
            .iter()
            .zip(join.input_fields())
            .skip(left_size)
            .collect();
        Relation::map()
            .with_iter(left.into_iter().find_map(|(o, i)| {
                (referred_field == i.name()).then_some((referred_field_name, Expr::col(o.name())))
            }))
            .with_iter(right.into_iter().filter_map(|(o, i)| {
                names
                    .contains(&i.name().to_string())
                    .then_some((i.name(), Expr::col(o.name())))
            }))
            .input(join)
            .build()
    }

    /// Add a field designated with a "fiald path"
    pub fn with_field_path<'a>(
        self,
        relations: &'a Hierarchy<Rc<Relation>>,
        path: &'a [(&'a str, &'a str, &'a str)],
        referred_field: &'a str,
        referred_field_name: &'a str,
    ) -> Relation {
        if path.is_empty() {
            self.identity_with_field(referred_field_name, Expr::col(referred_field))
        } else {
            let path = Path::from_iter(path);
            let field_path = FieldPath::from_path(path, referred_field, referred_field_name);
            // Build the relation following the path to compute the new field
            field_path.into_iter().fold(
                self,
                |relation,
                 ReferredField {
                     referring_id,
                     referred_relation,
                     referred_id,
                     referred_field,
                     referred_field_name,
                 }| {
                    relation.with_referred_field(
                        referring_id,
                        relations
                            .get(&[referred_relation.to_string()])
                            .unwrap()
                            .clone(),
                        referred_id,
                        referred_field,
                        referred_field_name,
                    )
                },
            )
        }
    }

    pub fn filter_fields<P: Fn(&str) -> bool>(self, predicate: P) -> Relation {
        match self {
            Relation::Map(map) => map.filter_fields(predicate).into(),
            relation => {
                Relation::map()
                    .with_iter(relation.schema().iter().filter_map(|f| {
                        predicate(f.name()).then_some((f.name(), Expr::col(f.name())))
                    }))
                    .input(relation)
                    .build()
            }
        }
    }

    pub fn map_fields<F: Fn(&str, Expr) -> Expr>(self, f: F) -> Relation {
        match self {
            Relation::Map(map) => map.map_fields(f).into(),
            relation => Relation::map()
                .with_iter(
                    relation
                        .schema()
                        .iter()
                        .map(|field| (field.name(), f(field.name(), Expr::col(field.name())))),
                )
                .input(relation)
                .build(),
        }
    }

    pub fn l1_norm(self, vector: &str, base: Vec<&str>, coordinates: Vec<&str>) -> Self {
        // group by base, coordinates
        let mut reduce = Relation::reduce().input(self.clone());
        reduce = reduce.with_group_by_column(vector);
        reduce = base
            .iter()
            .fold(reduce, |acc, s| acc.with_group_by_column(s.to_string()));
        reduce = reduce.with_iter(
            coordinates
                .iter()
                .map(|c| Expr::sum(Expr::col(c.to_string()))),
        );
        let reduce_rel: Relation = reduce.build();

        // group by base
        let mut reduce2 = Relation::reduce().input(reduce_rel.clone());
        for i in 1..(1 + base.len()) {
            reduce2 = reduce2.with_group_by_column(reduce_rel.field_from_index(i).unwrap().name())
        }
        for i in (1 + base.len())..(1 + base.len() + coordinates.len()) {
            let agg = Expr::abs(Expr::col(reduce_rel.field_from_index(i).unwrap().name()));
            reduce2 = reduce2.with(Expr::sum(agg));
        }
        reduce2.build()
    }

    pub fn l2_norm(self, vector: &str, base: Vec<&str>, coordinates: Vec<&str>) -> Self {
        // group by base, coordinates
        let mut reduce = Relation::reduce().input(self.clone());
        reduce = reduce.with_group_by_column(vector);
        reduce = base
            .iter()
            .fold(reduce, |acc, s| acc.with_group_by_column(s.to_string()));
        reduce = reduce.with_iter(
            coordinates
                .iter()
                .map(|c| Expr::sum(Expr::col(c.to_string()))),
        );
        let reduce_rel: Relation = reduce.build();

        // group by base
        let mut reduce = Relation::reduce().input(reduce_rel.clone());
        for i in 1..(1 + base.len()) {
            reduce = reduce.with_group_by_column(reduce_rel.field_from_index(i).unwrap().name())
        }
        for i in (1 + base.len())..(1 + base.len() + coordinates.len()) {
            let agg = Expr::col(reduce_rel.field_from_index(i).unwrap().name());
            let sqr = Expr::multiply(agg.clone(), agg);
            reduce = reduce.with(Expr::sum(sqr));
        }
        let reduce_rel2: Relation = reduce.build();
        // sqrt
        let mut map = Relation::map().input(reduce_rel2.clone());
        for i in 0..(base.len()) {
            map = map.with(Expr::col(reduce_rel2.field_from_index(i).unwrap().name()));
        }
        for i in base.len()..(base.len() + coordinates.len()) {
            map = map.with(Expr::sqrt(Expr::col(
                reduce_rel2.field_from_index(i).unwrap().name(),
            )));
        }
        let map_rel: Relation = map.build();
        map_rel
    }
}

impl With<(&str, Expr)> for Relation {
    fn with(self, (name, expr): (&str, Expr)) -> Self {
        self.identity_with_field(name, expr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        display::Dot,
        io::{postgresql, Database},
        relation::{builder::*, schema::Schema, Table},
        sql::parse,
    };
    use colored::Colorize;
    use itertools::Itertools;
    use sqlparser::ast;

    #[test]
    fn test_with_computed_field() {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        let table = relations.get(&["table_1".into()]).unwrap().as_ref().clone();
        let relation =
            Relation::try_from(parse("SELECT * FROM table_1").unwrap().with(&relations)).unwrap();
        // Table
        assert!(table.schema()[0].name() != "peid");
        let table = table.identity_with_field("peid", expr!(a + b));
        assert!(table.schema()[0].name() == "peid");
        // Relation
        assert!(relation.schema()[0].name() != "peid");
        let relation = relation.identity_with_field("peid", expr!(cos(a)));
        assert!(relation.schema()[0].name() == "peid");
    }

    #[test]
    fn test_filter_fields() {
        let database = postgresql::test_database();
        let relations = database.relations();
        let relation =
            Relation::try_from(parse("SELECT * FROM table_1").unwrap().with(&relations)).unwrap();
        let relation = relation.with_field("peid", expr!(cos(a)));
        assert!(relation.schema()[0].name() == "peid");
        let relation = relation.filter_fields(|n| n != "peid");
        assert!(relation.schema()[0].name() != "peid");
    }

    #[test]
    fn test_referred_field() {
        let database = postgresql::test_database();
        let relations = database.relations();
        let orders =
            Relation::try_from(parse("SELECT * FROM order_table").unwrap().with(&relations))
                .unwrap();
        let user = relations.get(&["user_table".to_string()]).unwrap().as_ref();
        let relation =
            orders.with_referred_field("user_id", Rc::new(user.clone()), "id", "id", "peid");
        assert!(relation.schema()[0].name() == "peid");
        let relation = relation.filter_fields(|n| n != "peid");
        assert!(relation.schema()[0].name() != "peid");
    }

    #[test]
    fn test_field_path() {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        // Link orders to users
        let orders = relations
            .get(&["order_table".to_string()])
            .unwrap()
            .as_ref();
        let relation = orders.clone().with_field_path(
            &relations,
            &[("user_id", "user_table", "id")],
            "id",
            "peid",
        );
        assert!(relation.schema()[0].name() == "peid");
        // Link items to orders
        let items = relations.get(&["item_table".to_string()]).unwrap().as_ref();
        let relation = items.clone().with_field_path(
            &relations,
            &[
                ("order_id", "order_table", "id"),
                ("user_id", "user_table", "id"),
            ],
            "name",
            "peid",
        );
        assert!(relation.schema()[0].name() == "peid");
        // Produce the query
        relation.display_dot();
        let query: &str = &ast::Query::from(&relation).to_string();
        println!("{query}");
        println!(
            "{}\n{}",
            format!("{query}").yellow(),
            database
                .query(query)
                .unwrap()
                .iter()
                .map(ToString::to_string)
                .join("\n")
        );
        let relation = relation.filter_fields(|n| n != "peid");
        assert!(relation.schema()[0].name() != "peid");
    }

    #[test]
    fn test_compute_norm_for_table() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        let table = relations
            .get(&["item_table".into()])
            .unwrap()
            .as_ref()
            .clone();
        // L1 Norm
        let amount_norm = table
            .clone()
            .l1_norm("order_id", vec!["item"], vec!["price"]);
        // amount_norm.display_dot().unwrap();
        let query: &str = &ast::Query::from(&amount_norm).to_string();
        //println!("Query = {}", query);
        let valid_query = "SELECT item, SUM(sum_by_peid) FROM (SELECT order_id, item, SUM(ABS(price)) AS sum_by_peid FROM item_table GROUP BY order_id, item) AS subquery GROUP BY item";
        assert_eq!(
            database.query(query).unwrap(),
            database.query(valid_query).unwrap()
        );
        // L2 Norm
        let amount_norm = table.l2_norm("order_id", vec!["item"], vec!["price"]);
        amount_norm.display_dot().unwrap();
        let query: &str = &ast::Query::from(&amount_norm).to_string();
        let valid_query = "SELECT item, SQRT(SUM(sum_by_peid)) FROM (SELECT order_id, item, POWER(SUM(price), 2) AS sum_by_peid FROM item_table GROUP BY order_id, item) AS subquery GROUP BY item";
        assert_eq!(
            database.query(query).unwrap(),
            database.query(valid_query).unwrap()
        );
    }

    #[test]
    fn test_compute_norm_for_map() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        let relation = Relation::try_from(
            parse("SELECT price - 25 AS std_price, * FROM item_table")
                .unwrap()
                .with(&relations),
        )
        .unwrap();
        relation.display_dot().unwrap();
        // L1 Norm
        let relation_norm =
            relation
                .clone()
                .l1_norm("order_id", vec!["item"], vec!["price", "std_price"]);
        relation_norm.display_dot().unwrap();
        let query: &str = &ast::Query::from(&relation_norm).to_string();
        //println!("Query = {}", query);
        let valid_query = "SELECT item, SUM(sum_1), SUM(sum_2) FROM (SELECT order_id, item, ABS(SUM(price)) AS sum_1, ABS(SUM(std_price)) AS sum_2 FROM ( SELECT price - 25 AS std_price, * FROM item_table ) AS intermediate_table GROUP BY order_id, item) AS subquery GROUP BY item";
        assert_eq!(
            database.query(query).unwrap(),
            database.query(valid_query).unwrap()
        );
        // L2 Norm
        let relation_norm = relation.l2_norm("order_id", vec!["item"], vec!["price", "std_price"]);
        relation_norm.display_dot().unwrap();
        let query: &str = &ast::Query::from(&relation_norm).to_string();
        let valid_query = "SELECT item, SQRT(SUM(sum_1)), SQRT(SUM(sum_2)) FROM (SELECT order_id, item, POWER(SUM(price), 2) AS sum_1, POWER(SUM(std_price), 2) AS sum_2 FROM ( SELECT price - 25 AS std_price, * FROM item_table ) AS intermediate_table GROUP BY order_id, item) AS subquery GROUP BY item";
        assert_eq!(
            database.query(query).unwrap(),
            database.query(valid_query).unwrap()
        );
    }

    #[test]
    fn test_compute_norm_for_join() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        let left: Relation = relations
            .get(&["item_table".into()])
            .unwrap()
            .as_ref()
            .clone();
        let right: Relation = relations
            .get(&["order_table".into()])
            .unwrap()
            .as_ref()
            .clone();
        let relation: Relation = Relation::join()
            .left(left)
            .right(right)
            .on(Expr::eq(
                Expr::qcol("item_table", "order_id"),
                Expr::qcol("order_table", "id"),
            ))
            .build();
        relation.display_dot().unwrap();
        let schema = relation.schema().clone();
        let item = schema.field_from_index(1).unwrap().name();
        let price = schema.field_from_index(2).unwrap().name();
        let user_id = schema.field_from_index(4).unwrap().name();
        let date = schema.field_from_index(6).unwrap().name();

        // L1 Norm
        let relation_norm = relation
            .clone()
            .l1_norm(user_id, vec![item, date], vec![price]);
        relation_norm.display_dot().unwrap();
        let query: &str = &ast::Query::from(&relation_norm).to_string();
        println!("Query = {}", query);

        let valid_query = "SELECT item, date, SUM(sum_1) FROM (SELECT user_id, item, date, ABS(SUM(price)) AS sum_1 FROM item_table JOIN order_table ON item_table.order_id = order_table.id GROUP BY user_id, item, date) AS subquery GROUP BY item, date";
        assert_eq!(
            database.query(query).unwrap(),
            database.query(valid_query).unwrap()
        );
        // L2 Norm
        let relation_norm = relation.l2_norm(user_id, vec![item, date], vec![price]);
        relation_norm.display_dot().unwrap();
        let query: &str = &ast::Query::from(&relation_norm).to_string();
        let valid_query = "SELECT item, date, SQRT(SUM(sum_1)) FROM (SELECT user_id, item, date, POWER(SUM(price), 2) AS sum_1 FROM item_table JOIN order_table ON item_table.order_id = order_table.id GROUP BY user_id, item, date) AS subquery GROUP BY item, date";
        assert_eq!(
            database.query(query).unwrap(),
            database.query(valid_query).unwrap()
        );
        // DEBUG
        for row in database.query(query).unwrap() {
            println!("{row}")
        }
    }
}
