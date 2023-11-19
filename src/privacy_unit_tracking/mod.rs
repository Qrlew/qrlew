//! # Methods to define `Relation`s' Privacy Unit and propagate it
//!
//! The definitions and names are inspired by:
//! - https://pipelinedp.io/key-definitions/
//! - https://programming-dp.com/ch3.html#the-unit-of-privacy
//! - https://arxiv.org/pdf/2212.04133.pdf
//!
pub mod privacy_unit;

use crate::{
    builder::{Ready, With, WithIterator},
    expr::{AggregateColumn, Expr},
    hierarchy::Hierarchy,
    relation::{Join, Map, Reduce, Relation, Table, Values, Variant as _},
};
pub use privacy_unit::{PrivacyUnit, PrivacyUnitPath};
use std::{error, fmt, ops::Deref, result, sync::Arc};

#[derive(Debug, Clone)]
pub enum Error {
    NotPrivacyUnitPreserving(String),
    NoPrivateTable(String),
    Other(String),
}

impl Error {
    pub fn not_privacy_unit_preserving(relation: impl fmt::Display) -> Error {
        Error::NotPrivacyUnitPreserving(format!("{} is not PUP", relation))
    }
    pub fn no_private_table(table: impl fmt::Display) -> Error {
        Error::NoPrivateTable(format!("{} is not private", table))
    }
    pub fn other(value: impl fmt::Display) -> Error {
        Error::Other(format!("{} is not PUP", value))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NotPrivacyUnitPreserving(desc) => {
                writeln!(f, "NotPrivacyUnitPreserving: {}", desc)
            }
            Error::NoPrivateTable(desc) => {
                writeln!(f, "NoPrivateTable: {}", desc)
            }
            Error::Other(err) => writeln!(f, "{}", err),
        }
    }
}

impl error::Error for Error {}

pub type Result<T> = result::Result<T, Error>;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Default)]
pub enum Strategy {
    /// Protect only when it does not affect the meaning of the original query.
    /// Fail otherwise.
    #[default]
    Soft,
    /// Protect at all cost.
    /// Will succeede most of the time.
    Hard,
}

#[derive(Clone, Debug)]
pub struct PUPRelation(pub Relation);

impl PUPRelation {
    pub fn privacy_unit(&self) -> &str {
        PrivacyUnit::privacy_unit()
    }

    pub fn privacy_unit_default(&self) -> &str {
        PrivacyUnit::privacy_unit_default()
    }

    pub fn privacy_unit_weight(&self) -> &str {
        PrivacyUnit::privacy_unit_weight()
    }

    pub fn with_name(self, name: String) -> Result<Self> {
        PUPRelation::try_from(Relation::from(self).with_name(name))
    }

    pub fn filter_fields<P: Fn(&str) -> bool>(self, predicate: P) -> Result<Self> {
        PUPRelation::try_from(Relation::from(self).filter_fields(|f| predicate(f)))
    }
}

impl From<PUPRelation> for Relation {
    fn from(value: PUPRelation) -> Self {
        value.0
    }
}

impl TryFrom<Relation> for PUPRelation {
    type Error = Error;

    fn try_from(value: Relation) -> Result<Self> {
        if value.schema().field(PrivacyUnit::privacy_unit()).is_ok()
            && value
                .schema()
                .field(PrivacyUnit::privacy_unit_weight())
                .is_ok()
        {
            Ok(PUPRelation(value))
        } else {
            Err(Error::NotPrivacyUnitPreserving(
                format!(
                    "Cannot convert to PUPRelation a relation that does not contains both {} and {} columns. \nGot: {}",
                    PrivacyUnit::privacy_unit(), PrivacyUnit::privacy_unit_weight(), value.schema().iter().map(|f| f.name()).collect::<Vec<_>>().join(",")
                )
            ))
        }
    }
}

impl Deref for PUPRelation {
    type Target = Relation;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Relation {
    /// Add a field designated with a foreign relation and a field
    pub fn with_referred_field(
        self,
        referring_id: String,
        referred_relation: Arc<Relation>,
        referred_id: String,
        referred_field: String,
        referred_field_name: String,
    ) -> Relation {
        let left_size = referred_relation.schema().len();
        let names: Vec<String> = self
            .schema()
            .iter()
            .map(|f| f.name().to_string())
            .filter(|name| name != &referred_field_name)
            .collect();
        let join: Relation = Relation::join()
            .inner()
            .on(Expr::eq(
                Expr::qcol(Join::right_name(), &referring_id),
                Expr::qcol(Join::left_name(), &referred_id),
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
                (referred_field == i.name())
                    .then_some((referred_field_name.clone(), Expr::col(o.name())))
            }))
            .with_iter(right.into_iter().filter_map(|(o, i)| {
                names
                    .contains(&i.name().to_string())
                    .then_some((i.name(), Expr::col(o.name())))
            }))
            .input(join)
            .build()
    }
    /// Add a field designated with a "field path"
    pub fn with_field_path(
        self,
        relations: &Hierarchy<Arc<Relation>>,
        field_path: PrivacyUnitPath,
    ) -> Relation {
        if field_path.path().is_empty() {
            // TODO Remove this?
            self.identity_with_field(
                PrivacyUnitPath::privacy_unit(),
                Expr::col(field_path.referred_field()),
            )
        } else {
            field_path
                .into_iter()
                .fold(self, |relation, referred_field| {
                    relation.with_referred_field(
                        referred_field.referring_id,
                        relations
                            .get(&[referred_field.referred_relation.to_string()])
                            .unwrap()
                            .clone(),
                        referred_field.referred_id,
                        referred_field.referred_field,
                        referred_field.referred_field_name,
                    )
                })
        }
    }
}

/// Implements the privacy tracking of various relations
pub struct PrivacyUnitTracking<'a> {
    relations: &'a Hierarchy<Arc<Relation>>,
    privacy_unit: PrivacyUnit,
    strategy: Strategy,
}

impl<'a> PrivacyUnitTracking<'a> {
    pub fn new(
        relations: &'a Hierarchy<Arc<Relation>>,
        privacy_unit: PrivacyUnit,
        strategy: Strategy,
    ) -> PrivacyUnitTracking {
        PrivacyUnitTracking {
            relations,
            privacy_unit,
            strategy,
        }
    }

    /// Table privacy tracking
    pub fn table(&self, table: &'a Table) -> Result<PUPRelation> {
        let (_, field_path) = self
            .privacy_unit
            .iter()
            .find(|(name, _field_path)| table.name() == self.relations[name.as_str()].name())
            .ok_or(Error::no_private_table(table.path()))?;
        PUPRelation::try_from(
            Relation::from(table.clone())
                .with_field_path(self.relations, field_path.clone())
                .map_fields(|name, expr| {
                    if name == PrivacyUnit::privacy_unit() {
                        Expr::md5(Expr::cast_as_text(expr))
                    } else {
                        expr
                    }
                })
                .insert_field(1, PrivacyUnit::privacy_unit_weight(), Expr::val(1)),
        )
    }

    /// Map privacy tracking from another PUP relation
    pub fn map(&self, map: &'a Map, input: PUPRelation) -> Result<PUPRelation> {
        let relation: Relation = Relation::map()
            .with((
                PrivacyUnit::privacy_unit(),
                Expr::col(PrivacyUnit::privacy_unit()),
            ))
            .with((
                PrivacyUnit::privacy_unit_weight(),
                Expr::col(PrivacyUnit::privacy_unit_weight()),
            ))
            .with(map.clone())
            .input(Relation::from(input))
            .build();
        PUPRelation::try_from(relation)
    }

    /// Reduce privacy tracking from another PUP relation
    pub fn reduce(&self, reduce: &'a Reduce, input: PUPRelation) -> Result<PUPRelation> {
        match self.strategy {
            Strategy::Soft => Err(Error::not_privacy_unit_preserving(reduce.name())),
            Strategy::Hard => {
                let relation: Relation = Relation::reduce()
                    .with_group_by_column(PrivacyUnit::privacy_unit())
                    .with((
                        PrivacyUnit::privacy_unit_weight(),
                        AggregateColumn::sum(PrivacyUnit::privacy_unit_weight()),
                    ))
                    .with(reduce.clone())
                    .input(Relation::from(input))
                    .build();
                PUPRelation::try_from(relation)
            }
        }
    }

    /// Join privacy tracking from 2 PUP relations
    pub fn join(
        &self,
        join: &'a crate::relation::Join,
        left: PUPRelation,
        right: PUPRelation,
    ) -> Result<PUPRelation> {
        // Create the privacy tracked join
        match self.strategy {
            Strategy::Soft => Err(Error::not_privacy_unit_preserving(join)),
            Strategy::Hard => {
                let name = join.name();
                let operator = join.operator().clone();
                let names = join.names();
                let names = names.with(vec![
                    (
                        vec![Join::left_name(), PrivacyUnit::privacy_unit()],
                        format!("_LEFT{}", PrivacyUnit::privacy_unit()),
                    ),
                    (
                        vec![Join::left_name(), PrivacyUnit::privacy_unit_weight()],
                        format!("_LEFT{}", PrivacyUnit::privacy_unit_weight()),
                    ),
                    (
                        vec![Join::right_name(), PrivacyUnit::privacy_unit()],
                        format!("_RIGHT{}", PrivacyUnit::privacy_unit()),
                    ),
                    (
                        vec![Join::right_name(), PrivacyUnit::privacy_unit_weight()],
                        format!("_RIGHT{}", PrivacyUnit::privacy_unit_weight()),
                    ),
                ]);
                let join: Join = Relation::join()
                    .names(names)
                    .operator(operator)
                    .and(Expr::eq(
                        Expr::qcol(Join::left_name(), PrivacyUnit::privacy_unit()),
                        Expr::qcol(Join::right_name(), PrivacyUnit::privacy_unit()),
                    ))
                    .left(Relation::from(left))
                    .right(Relation::from(right))
                    .build();
                let mut builder = Relation::map().name(name);
                builder = builder.with((
                    PrivacyUnit::privacy_unit(),
                    Expr::col(format!("_LEFT{}", PrivacyUnit::privacy_unit())),
                ));
                builder = builder.with((
                    PrivacyUnit::privacy_unit_weight(),
                    Expr::multiply(
                        Expr::col(format!("_LEFT{}", PrivacyUnit::privacy_unit_weight())),
                        Expr::col(format!("_RIGHT{}", PrivacyUnit::privacy_unit_weight())),
                    ),
                ));
                builder = join.names().iter().fold(builder, |b, (p, n)| {
                    if [
                        PrivacyUnit::privacy_unit(),
                        PrivacyUnit::privacy_unit_weight(),
                    ]
                    .contains(&p[1].as_str())
                    {
                        b
                    } else {
                        b.with((n, Expr::col(n)))
                    }
                });
                let relation: Relation = builder.input(Arc::new(join.into())).build();
                PUPRelation::try_from(relation)
            }
        }
    }

    /// Join privacy tracking from 2 PUP relations
    pub fn join_left_published(
        //TODO this need to be cleaned (really)
        &self,
        join: &'a crate::relation::Join,
        left: Relation,
        right: PUPRelation,
    ) -> Result<PUPRelation> {
        let name = join.name();
        let operator = join.operator().clone();
        let names = join.names();
        let names = names.with(vec![
            (
                vec![Join::right_name(), PrivacyUnit::privacy_unit()],
                format!("_RIGHT{}", PrivacyUnit::privacy_unit()),
            ),
            (
                vec![Join::right_name(), PrivacyUnit::privacy_unit_weight()],
                format!("_RIGHT{}", PrivacyUnit::privacy_unit_weight()),
            ),
        ]);
        let join: Join = Relation::join()
            .names(names)
            .operator(operator)
            .left(Relation::from(left))
            .right(Relation::from(right))
            .build();
        let mut builder = Relation::map()
            .name(name)
            .with((
                PrivacyUnit::privacy_unit(),
                Expr::col(format!("_RIGHT{}", PrivacyUnit::privacy_unit())),
            ))
            .with((
                PrivacyUnit::privacy_unit_weight(),
                Expr::col(format!("_RIGHT{}", PrivacyUnit::privacy_unit_weight())),
            ));
        builder = join.names().iter().fold(builder, |b, (p, n)| {
            if [
                PrivacyUnit::privacy_unit(),
                PrivacyUnit::privacy_unit_weight(),
            ]
            .contains(&p[1].as_str())
            {
                b
            } else {
                b.with((n, Expr::col(n)))
            }
        });
        let relation: Relation = builder.input(Arc::new(join.into())).build();
        PUPRelation::try_from(relation)
    }

    /// Join privacy tracking from 2 PUP relations
    pub fn join_right_published(
        //TODO this need to be cleaned (really)
        &self,
        join: &'a crate::relation::Join,
        left: PUPRelation,
        right: Relation,
    ) -> Result<PUPRelation> {
        let name = join.name();
        let operator = join.operator().clone();
        let names = join.names();
        let names = names.with(vec![
            (
                vec![Join::left_name(), PrivacyUnit::privacy_unit()],
                format!("_LEFT{}", PrivacyUnit::privacy_unit()),
            ),
            (
                vec![Join::left_name(), PrivacyUnit::privacy_unit_weight()],
                format!("_LEFT{}", PrivacyUnit::privacy_unit_weight()),
            ),
        ]);
        let join: Join = Relation::join()
            .names(names)
            .operator(operator)
            .left(Relation::from(left))
            .right(Relation::from(right))
            .build();
        let mut builder = Relation::map()
            .name(name)
            .with((
                PrivacyUnit::privacy_unit(),
                Expr::col(format!("_LEFT{}", PrivacyUnit::privacy_unit())),
            ))
            .with((
                PrivacyUnit::privacy_unit_weight(),
                Expr::col(format!("_LEFT{}", PrivacyUnit::privacy_unit_weight())),
            ));
        builder = join.names().iter().fold(builder, |b, (p, n)| {
            if [
                PrivacyUnit::privacy_unit(),
                PrivacyUnit::privacy_unit_weight(),
            ]
            .contains(&p[1].as_str())
            {
                b
            } else {
                b.with((n, Expr::col(n)))
            }
        });
        let relation: Relation = builder.input(Arc::new(join.into())).build();
        PUPRelation::try_from(relation)
    }

    /// Set privacy tracking from 2 PUP relations
    pub fn set(
        &self,
        set: &'a crate::relation::Set,
        left: Result<PUPRelation>,
        right: Result<PUPRelation>,
    ) -> Result<PUPRelation> {
        let relation: Relation = Relation::set()
            .name(set.name())
            .operator(set.operator().clone())
            .quantifier(set.quantifier().clone())
            .left(Relation::from(left?))
            .right(Relation::from(right?))
            .build();
        PUPRelation::try_from(relation)
    }

    /// Values privacy tracking
    pub fn values(&self, values: &'a Values) -> Result<PUPRelation> {
        PUPRelation::try_from(Relation::Values(values.clone()))
    }
}

impl<'a>
    From<(
        &'a Hierarchy<Arc<Relation>>,
        Vec<(&str, Vec<(&str, &str, &str)>, &str)>,
        Strategy,
    )> for PrivacyUnitTracking<'a>
{
    fn from(
        value: (
            &'a Hierarchy<Arc<Relation>>,
            Vec<(&str, Vec<(&str, &str, &str)>, &str)>,
            Strategy,
        ),
    ) -> Self {
        let (relations, privacy_unit, strategy) = value;
        let privacy_unit: Vec<_> = privacy_unit
            .into_iter()
            .map(|(table, privacy_tracking, referred_field)| {
                (table, privacy_tracking, referred_field)
            })
            .collect();
        PrivacyUnitTracking::new(relations, PrivacyUnit::from(privacy_unit), strategy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ast,
        data_type::{DataType, DataTyped},
        display::Dot,
        expr::Identifier,
        io::{postgresql, Database},
        relation::{Constraint, Schema, Variant},
    };
    use colored::Colorize;
    use itertools::Itertools;

    #[test]
    fn test_field_path() {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        // Link orders to users
        let orders = relations.get(&["orders".to_string()]).unwrap().as_ref();
        let relation = orders.clone().with_field_path(
            &relations,
            PrivacyUnitPath::from((vec![("user_id", "users", "id")], "id")),
        );
        assert!(relation.schema()[0].name() == PrivacyUnit::privacy_unit());
        // // Link items to orders
        let items = relations.get(&["items".to_string()]).unwrap().as_ref();
        let relation = items.clone().with_field_path(
            &relations,
            PrivacyUnitPath::from((
                vec![("order_id", "orders", "id"), ("user_id", "users", "id")],
                "name",
            )),
        );
        assert!(relation.schema()[0].name() == PrivacyUnit::privacy_unit());
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
        // let relation = relation.filter_fields(|n| n != "peid");
        // assert!(relation.schema()[0].name() != "peid");
    }

    #[test]
    fn test_table_privacy_tracking_from_field_paths() {
        let database = postgresql::test_database();
        let relations = database.relations();
        let table = relations
            .get(&["item_table".into()])
            .unwrap()
            .as_ref()
            .clone();
        let privacy_unit_tracking = PrivacyUnitTracking::from((
            &relations,
            vec![(
                "item_table",
                vec![("order_id", "order_table", "id")],
                "date",
            )],
            Strategy::Soft,
        ));
        // Table
        let table = privacy_unit_tracking
            .table(&table.try_into().unwrap())
            .unwrap();
        table.display_dot().unwrap();
        println!("Schema privacy_tracked = {}", table.schema());
        println!("Query privacy tracked = {}", ast::Query::from(&*table));
        assert_eq!(table.schema()[0].name(), PrivacyUnit::privacy_unit())
    }

    #[test]
    fn test_join_privacy_tracking() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        let left = relations
            .get(&["item_table".to_string()])
            .unwrap()
            .deref()
            .clone();
        let right = relations
            .get(&["order_table".to_string()])
            .unwrap()
            .deref()
            .clone();
        let join: Join = Join::builder()
            .inner()
            .on_eq("order_id", "id")
            .left(left.clone())
            .right(right.clone())
            .build();
        Relation::from(join.clone()).display_dot().unwrap();
        let privacy_unit_tracking = PrivacyUnitTracking::from((
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
        let pup_left = privacy_unit_tracking
            .table(&left.try_into().unwrap())
            .unwrap();
        let pup_right = privacy_unit_tracking
            .table(&right.try_into().unwrap())
            .unwrap();
        let pup_join = privacy_unit_tracking
            .join(&join, pup_left, pup_right)
            .unwrap();
        pup_join.display_dot().unwrap();

        let fields: Vec<(&str, DataType)> = join
            .schema()
            .iter()
            .map(|f| (f.name(), f.data_type()))
            .collect::<Vec<_>>();

        let mut true_fields = vec![
            (PrivacyUnit::privacy_unit(), DataType::text()),
            (
                PrivacyUnit::privacy_unit_weight(),
                DataType::integer_value(1),
            ),
        ];
        true_fields.extend(fields.into_iter());
        assert_eq!(
            pup_join.deref().data_type(),
            DataType::structured(true_fields)
        );

        let query: &str = &ast::Query::from(pup_join.deref()).to_string();
        println!("{query}");
        _ = database
            .query(query)
            .unwrap()
            .iter()
            .map(ToString::to_string)
            .join("\n");
    }

    #[test]
    fn test_auto_join_privacy_tracking() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        let table = relations
            .get(&["item_table".to_string()])
            .unwrap()
            .deref()
            .clone();
        let join: Join = Join::builder()
            .inner()
            .on_eq("item", "item")
            .left(table.clone())
            .right(table.clone())
            .build();
        Relation::from(join.clone()).display_dot().unwrap();
        let privacy_unit_tracking = PrivacyUnitTracking::from((
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
        let pup_table = privacy_unit_tracking
            .table(&table.try_into().unwrap())
            .unwrap();
        let pup_join = privacy_unit_tracking
            .join(&join, pup_table.clone(), pup_table.clone())
            .unwrap();
        pup_join.display_dot().unwrap();

        let fields: Vec<(&str, DataType)> = join
            .schema()
            .iter()
            .map(|f| (f.name(), f.data_type()))
            .collect::<Vec<_>>();

        let mut true_fields = vec![
            (PrivacyUnit::privacy_unit(), DataType::text()),
            (
                PrivacyUnit::privacy_unit_weight(),
                DataType::integer_value(1),
            ),
        ];
        true_fields.extend(fields.into_iter());
        assert_eq!(
            pup_join.deref().data_type(),
            DataType::structured(true_fields)
        );

        let query: &str = &ast::Query::from(pup_join.deref()).to_string();
        println!("{query}");
        _ = database
            .query(query)
            .unwrap()
            .iter()
            .map(ToString::to_string)
            .join("\n");
    }

    #[test]
    fn test_privacy_tracking_unique() {
        let table1: Table = Relation::table()
            .schema(
                Schema::empty()
                    .with(("id".to_string(), DataType::text()))
                    .with(("a", DataType::float(), Constraint::Unique)),
            )
            .name("table1")
            .size(10)
            .build();
        let table2: Table = Relation::table()
            .schema(
                Schema::empty()
                    .with(("a", DataType::float(), Constraint::Unique))
                    .with(("b", DataType::integer())),
            )
            .name("table2")
            .size(20)
            .build();
        let table3: Table = Relation::table()
            .schema(Schema::empty().with(("b", DataType::integer())).with((
                "c",
                DataType::float(),
                Constraint::Unique,
            )))
            .name("table3")
            .size(70)
            .build();
        let tables = vec![table1, table2, table3];
        let relations: Hierarchy<Arc<Relation>> = tables
            .iter()
            .map(|t| (Identifier::from(t.name()), Arc::new(t.clone().into()))) // Tables can be accessed from their name or path
            .chain(
                tables
                    .iter()
                    .map(|t| (t.path().clone(), Arc::new(t.clone().into()))),
            )
            .collect();

        let privacy_unit_tracking = PrivacyUnitTracking::from((
            &relations,
            vec![
                ("table1", vec![], "id"),
                ("table2", vec![("a", "table1", "a")], "id"),
                (
                    "table3",
                    vec![("b", "table2", "b"), ("a", "table1", "a")],
                    "id",
                ),
            ],
            Strategy::Hard,
        ));
        for table in tables {
            let pup_table = privacy_unit_tracking
                .table(&table.clone().try_into().unwrap())
                .unwrap();
            pup_table.deref().display_dot().unwrap();
        }
    }
}
