//! # Methods to define `Relation`s' protected entity and propagate it
//!
//! This is experimental and little tested yet.
//!
pub mod protected_entity;

use crate::{
    builder::{Ready, With, WithIterator},
    expr::{AggregateColumn, Expr},
    hierarchy::Hierarchy,
    relation::{Join, Map, Reduce, Relation, Table, Values, Variant as _},
};
use protected_entity::{FieldPath, ProtectedEntity};
use std::{error, fmt, ops::Deref, result, sync::Arc};

#[derive(Debug, Clone)]
pub enum Error {
    NotProtectedEntityPreserving(String),
    UnprotectedTable(String),
    Other(String),
}

impl Error {
    pub fn not_protected_entity_preserving(relation: impl fmt::Display) -> Error {
        Error::NotProtectedEntityPreserving(format!("{} is not PEP", relation))
    }
    pub fn unprotected_table(table: impl fmt::Display) -> Error {
        Error::NotProtectedEntityPreserving(format!("{} is not protected", table))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NotProtectedEntityPreserving(desc) => {
                writeln!(f, "NotProtectedEntityPreserving: {}", desc)
            }
            Error::UnprotectedTable(desc) => {
                writeln!(f, "UnprotectedTable: {}", desc)
            }
            Error::Other(err) => writeln!(f, "{}", err),
        }
    }
}

impl error::Error for Error {}

pub type Result<T> = result::Result<T, Error>;

pub const PROTECTION_PREFIX: &str = "_PROTECTED_";
pub const PROTECTION_COLUMNS: usize = 2;
pub const PE_ID: &str = "_PROTECTED_ENTITY_ID_";
pub const PE_WEIGHT: &str = "_PROTECTED_ENTITY_WEIGHT_";

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
pub struct PEPRelation(pub Relation);

impl PEPRelation {
    pub fn protected_entity_id(&self) -> &str {
        PE_ID
    }

    pub fn protected_entity_weight(&self) -> &str {
        PE_WEIGHT
    }

    pub fn with_name(self, name: String) -> Result<Self> {
        PEPRelation::try_from(Relation::from(self).with_name(name))
    }

    pub fn filter_fields<P: Fn(&str) -> bool>(self, predicate: P) -> Result<Self> {
        PEPRelation::try_from(Relation::from(self).filter_fields(|f| predicate(f)))
    }
}

impl From<PEPRelation> for Relation {
    fn from(value: PEPRelation) -> Self {
        value.0
    }
}

impl TryFrom<Relation> for PEPRelation {
    type Error = Error;

    fn try_from(value: Relation) -> Result<Self> {
        if value.schema().field(PE_ID).is_ok() && value.schema().field(PE_WEIGHT).is_ok() {
            Ok(PEPRelation(value))
        } else {
            Err(Error::NotProtectedEntityPreserving(
                format!(
                    "Cannot convert to PEPRelation a relation that does not contains both {} and {} columns. \nGot: {}",
                    PE_ID, PE_WEIGHT, value.schema().iter().map(|f| f.name()).collect::<Vec<_>>().join(",")
                )
            ))
        }
    }
}

impl From<PEPReduce> for PEPRelation {
    fn from(value: PEPReduce) -> Self {
        PEPRelation::try_from(Relation::from(value.0)).unwrap()
    }
}

impl Deref for PEPRelation {
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
        field_path: FieldPath,
    ) -> Relation {
        if field_path.path().is_empty() {
            // TODO Remove this?
            self.identity_with_field(
                field_path.referred_field_name(),
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

#[derive(Clone, Debug)]
pub struct PEPReduce(pub Reduce);

impl PEPReduce {
    pub fn protected_entity_id(&self) -> &str {
        PE_ID
    }

    pub fn protected_entity_weight(&self) -> &str {
        PE_WEIGHT
    }

    pub fn has_non_protected_entity_id_group_by(&self) -> bool {
        self.0.group_by().len() > 1
    }
}

impl From<PEPReduce> for Reduce {
    fn from(value: PEPReduce) -> Self {
        value.0
    }
}

impl TryFrom<Reduce> for PEPReduce {
    type Error = Error;

    fn try_from(value: Reduce) -> Result<Self> {
        if value.schema().field(PE_ID).is_ok() && value.schema().field(PE_WEIGHT).is_ok() {
            Ok(PEPReduce(value))
        } else {
            Err(Error::NotProtectedEntityPreserving(
                format!(
                    "Cannot convert to PEPReduce a reduce that does not contains both {} and {} columns. \nGot: {}",
                    PE_ID, PE_WEIGHT, value.schema().iter().map(|f| f.name()).collect::<Vec<_>>().join(",")
                )
            ))
        }
    }
}

impl Deref for PEPReduce {
    type Target = Reduce;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Implements the protection of various relations
pub struct Protection<'a> {
    relations: &'a Hierarchy<Arc<Relation>>,
    protected_entity: ProtectedEntity,
    strategy: Strategy,
}

impl<'a> Protection<'a> {
    pub fn new(
        relations: &'a Hierarchy<Arc<Relation>>,
        protected_entity: ProtectedEntity,
        strategy: Strategy,
    ) -> Protection {
        Protection {
            relations,
            protected_entity,
            strategy,
        }
    }

    /// Table protection
    pub fn table(&self, table: Table) -> Result<PEPRelation> {
        let (_, field_path) = self
            .protected_entity
            .iter()
            .find(|(name, _field_path)| table.name() == self.relations[name.as_str()].name())
            .ok_or(Error::unprotected_table(table.path()))?;
        PEPRelation::try_from(
            Relation::from(table)
                .with_field_path(self.relations, field_path.clone())
                .map_fields(|name, expr| {
                    if name == PE_ID {
                        Expr::md5(Expr::cast_as_text(expr))
                    } else {
                        expr
                    }
                })
                .insert_field(1, PE_WEIGHT, Expr::val(1)),
        )
    }

    /// Map protection from another PEP relation
    pub fn map(&self, map: &'a Map, input: PEPRelation) -> Result<PEPRelation> {
        let relation: Relation = Relation::map()
            .with(map.clone())
            .with((PE_ID, Expr::col(PE_ID)))
            .with((PE_WEIGHT, Expr::col(PE_WEIGHT)))
            .input(Relation::from(input))
            .build();
        PEPRelation::try_from(relation)
    }

    /// Reduce protection from another PEP relation
    pub fn reduce(&self, reduce: &'a Reduce, input: PEPRelation) -> Result<PEPRelation> {
        match self.strategy {
            Strategy::Soft => Err(Error::not_protected_entity_preserving(reduce.name())),
            Strategy::Hard => {
                let relation: Relation = Relation::reduce()
                    .with_group_by_column(PE_ID)
                    .with((PE_WEIGHT, AggregateColumn::sum(PE_WEIGHT)))
                    .with(reduce.clone())
                    .input(Relation::from(input))
                    .build();
                PEPRelation::try_from(relation)
            }
        }
    }

    /// Join protection from 2 PEP relations
    pub fn join(
        &self,
        join: &'a crate::relation::Join,
        left: PEPRelation,
        right: PEPRelation,
    ) -> Result<PEPRelation> {
        // Create the protected join
        match self.strategy {
            Strategy::Soft => Err(Error::not_protected_entity_preserving(join)),
            Strategy::Hard => {
                let name = join.name();
                let operator = join.operator().clone();
                let names = join.names();
                let names = names.with(vec![
                    (vec![Join::left_name(), PE_ID], format!("_LEFT{PE_ID}")),
                    (
                        vec![Join::left_name(), PE_WEIGHT],
                        format!("_LEFT{PE_WEIGHT}"),
                    ),
                    (vec![Join::right_name(), PE_ID], format!("_RIGHT{PE_ID}")),
                    (
                        vec![Join::right_name(), PE_WEIGHT],
                        format!("_RIGHT{PE_WEIGHT}"),
                    ),
                ]);
                let join: Join = Relation::join()
                    .names(names)
                    .operator(operator)
                    .and(Expr::eq(
                        Expr::qcol(Join::left_name(), PE_ID),
                        Expr::qcol(Join::right_name(), PE_ID),
                    ))
                    .left(Relation::from(left))
                    .right(Relation::from(right))
                    .build();
                let mut builder = Relation::map().name(name);
                builder = builder.with((PE_ID, Expr::col(format!("_LEFT{PE_ID}"))));
                builder = builder.with((
                    PE_WEIGHT,
                    Expr::multiply(
                        Expr::col(format!("_LEFT{PE_WEIGHT}")),
                        Expr::col(format!("_RIGHT{PE_WEIGHT}")),
                    ),
                ));
                builder = join.names().iter().fold(builder, |b, (p, n)| {
                    if [PE_ID, PE_WEIGHT].contains(&p[1].as_str()) {
                        b
                    } else {
                        b.with((n, Expr::col(n)))
                    }
                });
                let relation: Relation = builder.input(Arc::new(join.into())).build();

                PEPRelation::try_from(relation)
            }
        }
    }

    /// Join protection from 2 PEP relations
    pub fn join_left_published(
        //TODO this need to be cleaned (really)
        &self,
        join: &'a crate::relation::Join,
        left: Relation,
        right: PEPRelation,
    ) -> Result<PEPRelation> {
        todo!()
    }

    /// Join protection from 2 PEP relations
    pub fn join_right_published(
        //TODO this need to be cleaned (really)
        &self,
        join: &'a crate::relation::Join,
        left: PEPRelation,
        right: Relation,
    ) -> Result<PEPRelation> {
        todo!()
    }

    /// Set protection from 2 PEP relations
    pub fn set(
        &self,
        set: &'a crate::relation::Set,
        left: Result<PEPRelation>,
        right: Result<PEPRelation>,
    ) -> Result<PEPRelation> {
        let relation: Relation = Relation::set()
            .name(set.name())
            .operator(set.operator().clone())
            .quantifier(set.quantifier().clone())
            .left(Relation::from(left?))
            .right(Relation::from(right?))
            .build();
        PEPRelation::try_from(relation)
    }

    /// Values protection
    pub fn values(&self, values: &'a Values) -> Result<PEPRelation> {
        PEPRelation::try_from(Relation::Values(values.clone()))
    }
}

impl<'a>
    From<(
        &'a Hierarchy<Arc<Relation>>,
        Vec<(&str, Vec<(&str, &str, &str)>, &str)>,
        Strategy,
    )> for Protection<'a>
{
    fn from(
        value: (
            &'a Hierarchy<Arc<Relation>>,
            Vec<(&str, Vec<(&str, &str, &str)>, &str)>,
            Strategy,
        ),
    ) -> Self {
        let (relations, protected_entity, strategy) = value;
        let protected_entity: Vec<_> = protected_entity
            .into_iter()
            .map(|(table, protection, referred_field)| (table, protection, referred_field, PE_ID))
            .collect();
        Protection::new(relations, ProtectedEntity::from(protected_entity), strategy)
    }
}
/// A visitor to compute Relation protection
#[derive(Clone, Debug)]
pub struct ProtectVisitor<F: Fn(&Table) -> Result<PEPRelation>> {
    /// The protected entity definition
    protect_tables: F,
    /// Strategy used
    strategy: Strategy,
}

impl<F: Fn(&Table) -> Result<PEPRelation>> ProtectVisitor<F> {
    pub fn new(protect_tables: F, strategy: Strategy) -> Self {
        ProtectVisitor {
            protect_tables,
            strategy,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ast,
        data_type::{DataType, DataTyped},
        display::Dot,
        io::{postgresql, Database},
        relation::Variant,
        sql::parse,
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
            FieldPath::from((vec![("user_id", "users", "id")], "id", "peid")),
        );
        assert!(relation.schema()[0].name() == "peid");
        // // Link items to orders
        let items = relations.get(&["items".to_string()]).unwrap().as_ref();
        let relation = items.clone().with_field_path(
            &relations,
            FieldPath::from((
                vec![("order_id", "orders", "id"), ("user_id", "users", "id")],
                "name",
                "peid",
            )),
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
        // let relation = relation.filter_fields(|n| n != "peid");
        // assert!(relation.schema()[0].name() != "peid");
    }

    // #[test]
    // fn test_table_protection_from_field_paths() {
    //     let database = postgresql::test_database();
    //     let relations = database.relations();
    //     let table = relations
    //         .get(&["item_table".into()])
    //         .unwrap()
    //         .as_ref()
    //         .clone();
    //     let protection = Protection::from((
    //         &relations,
    //         vec![(
    //             "item_table",
    //             vec![("order_id", "order_table", "id")],
    //             "date",
    //         )],
    //         Strategy::Soft,
    //     ));
    //     // Table
    //     let table = protection.table(table.try_into().unwrap()).unwrap();
    //     table.display_dot().unwrap();
    //     println!("Schema protected = {}", table.schema());
    //     println!("Query protected = {}", ast::Query::from(&*table));
    //     assert_eq!(table.schema()[0].name(), PE_ID)
    // }

    #[test]
    fn test_join_protection() {
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
        let protected_left = protection.table(left.try_into().unwrap()).unwrap();
        let protected_right = protection.table(right.try_into().unwrap()).unwrap();
        let protected_join = protection
            .join(&join, protected_left, protected_right)
            .unwrap();
        protected_join.display_dot().unwrap();

        let fields: Vec<(&str, DataType)> = join
            .schema()
            .iter()
            .map(|f| (f.name(), f.data_type()))
            .collect::<Vec<_>>();

        let mut true_fields = vec![
            (PE_ID, DataType::text()),
            (PE_WEIGHT, DataType::integer_value(1)),
        ];
        true_fields.extend(fields.into_iter());
        assert_eq!(
            protected_join.deref().data_type(),
            DataType::structured(true_fields)
        );

        let query: &str = &ast::Query::from(protected_join.deref()).to_string();
        println!("{query}");
        _ = database
            .query(query)
            .unwrap()
            .iter()
            .map(ToString::to_string)
            .join("\n");
    }

    #[test]
    fn test_auto_join_protection() {
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
        let protected_table = protection.table(table.try_into().unwrap()).unwrap();
        let protected_join = protection
            .join(&join, protected_table.clone(), protected_table.clone())
            .unwrap();
        protected_join.display_dot().unwrap();

        let fields: Vec<(&str, DataType)> = join
            .schema()
            .iter()
            .map(|f| (f.name(), f.data_type()))
            .collect::<Vec<_>>();

        let mut true_fields = vec![
            (PE_ID, DataType::text()),
            (PE_WEIGHT, DataType::integer_value(1)),
        ];
        true_fields.extend(fields.into_iter());
        assert_eq!(
            protected_join.deref().data_type(),
            DataType::structured(true_fields)
        );

        let query: &str = &ast::Query::from(protected_join.deref()).to_string();
        println!("{query}");
        _ = database
            .query(query)
            .unwrap()
            .iter()
            .map(ToString::to_string)
            .join("\n");
    }
}
