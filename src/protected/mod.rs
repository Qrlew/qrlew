//! # Methods to define `Relation`s' protected entity and propagate it
//! 
//! This is experimental and little tested yet.
//! 

use crate::{
    builder::{self, Ready, With},
    display::Dot,
    expr::{identifier::Identifier, Expr},
    hierarchy::{Hierarchy, Path},
    relation::{Join, Map, Reduce, Relation, Set, Table, Variant as _, Visitor},
    visitor::Acceptor,
};
use std::{error, fmt, rc::Rc, result};

#[derive(Debug, Clone)]
pub enum Error {
    NotProtectedEntityPreserving(String),
    Other(String),
}

impl Error {
    pub fn not_protected_entity_preserving(relation: impl fmt::Display) -> Error {
        Error::NotProtectedEntityPreserving(format!("{} is not PEP", relation))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NotProtectedEntityPreserving(desc) => {
                writeln!(f, "NotProtectedEntityPreserving: {}", desc)
            }
            Error::Other(err) => writeln!(f, "{}", err),
        }
    }
}

impl error::Error for Error {}

pub type Result<T> = result::Result<T, Error>;

pub const PROTECTION_PREFIX: &str = "_protected_";
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

/// A visitor to compute Relation protection
#[derive(Clone, Debug)]
pub struct ProtectVisitor<F: Fn(&Table) -> Relation> {
    /// The protected entity definition
    protect_tables: F,
    /// Strategy used
    strategy: Strategy,
}

impl<F: Fn(&Table) -> Relation> ProtectVisitor<F> {
    pub fn new(protect_tables: F, strategy: Strategy) -> Self {
        ProtectVisitor {
            protect_tables,
            strategy,
        }
    }
}

/// Build a visitor from exprs
pub fn protect_visitor_from_exprs<'a>(
    protected_entity: &'a [(&'a Table, Expr)],
    strategy: Strategy,
) -> ProtectVisitor<impl Fn(&Table) -> Relation + 'a> {
    ProtectVisitor::new(
        move |table: &Table| match protected_entity
            .iter()
            .find_map(|(t, e)| (table == *t).then(|| e.clone()))
        {
            Some(expr) => Relation::from(table.clone()).identity_with_field(PE_ID, expr.clone()),
            None => table.clone().into(),
        },
        strategy,
    )
}

/// Build a visitor from exprs
pub fn protect_visitor_from_field_paths<'a>(
    relations: &'a Hierarchy<Rc<Relation>>,
    protected_entity: &'a [(&'a str, &'a [(&'a str, &'a str, &'a str)], &'a str)],
    strategy: Strategy,
) -> ProtectVisitor<impl Fn(&Table) -> Relation + 'a> {
    ProtectVisitor::new(
        move |table: &Table| match protected_entity
            .iter()
            .find(|(tab, path, field)| table.name() == *tab)
        {
            Some((tab, path, field)) => Relation::from(table.clone())
                .with_field_path(relations, path, field, PE_ID)
                .map_fields(|n, e| if n == PE_ID { Expr::md5(e) } else { e }),
            None => table.clone().into(),
        },
        strategy,
    )
}

impl<'a, F: Fn(&Table) -> Relation> Visitor<'a, Result<Relation>> for ProtectVisitor<F> {
    fn table(&self, table: &'a Table) -> Result<Relation> {
        Ok((self.protect_tables)(table)
            .insert_field(1, PE_WEIGHT, Expr::val(1))
            // We preserve the name
            .with_name(format!("{}{}", PROTECTION_PREFIX, table.name())))
    }

    fn map(&self, map: &'a Map, input: Result<Relation>) -> Result<Relation> {
        let builder = Relation::map()
            .with((PE_ID, Expr::col(PE_ID)))
            .with((PE_WEIGHT, Expr::col(PE_WEIGHT)))
            .with(map.clone())
            .input(input?);
        Ok(builder.build())
    }

    fn reduce(&self, reduce: &'a Reduce, input: Result<Relation>) -> Result<Relation> {
        match self.strategy {
            Strategy::Soft => Err(Error::not_protected_entity_preserving(reduce)),
            Strategy::Hard => {
                let builder = Relation::reduce()
                    .with_group_by_column(PE_ID)
                    .with((PE_WEIGHT, Expr::sum(Expr::col(PE_WEIGHT))))
                    .with(reduce.clone())
                    .input(input?);
                Ok(builder.build())
            }
        }
    }

    fn join(
        //TODO this need to be cleaned (really)
        &self,
        join: &'a crate::relation::Join,
        left: Result<Relation>,
        right: Result<Relation>,
    ) -> Result<Relation> {
        let left_name = left.as_ref().unwrap().name().to_string();
        let right_name: String = right.as_ref().unwrap().name().to_string();
        // Preserve names
        let names: Vec<String> = join.schema().iter().map(|f| f.name().to_string()).collect();
        let mut left_names = vec![format!("_LEFT{PE_ID}"), format!("_LEFT{PE_WEIGHT}")];
        left_names.extend(names.iter().take(join.left.schema().len()).cloned());
        let mut right_names = vec![format!("_RIGHT{PE_ID}"), format!("_RIGHT{PE_WEIGHT}")];
        right_names.extend(names.iter().skip(join.left.schema().len()).cloned());
        // Create the protected join
        match self.strategy {
            Strategy::Soft => Err(Error::not_protected_entity_preserving(join)),
            Strategy::Hard => {
                let Join { name, operator, .. } = join;
                let left = left?;
                let right = right?;
                // Compute the mapping between current and new columns //TODO clean this code a bit
                let columns: Hierarchy<Identifier> = join
                    .left
                    .schema()
                    .iter()
                    .zip(left.schema().iter().skip(PROTECTION_COLUMNS))
                    .map(|(o, n)| {
                        (
                            vec![join.left.name().to_string(), o.name().to_string()],
                            Identifier::from(vec![left_name.clone(), n.name().to_string()]),
                        )
                    })
                    .chain(
                        join.right
                            .schema()
                            .iter()
                            .zip(right.schema().iter().skip(PROTECTION_COLUMNS))
                            .map(|(o, n)| {
                                (
                                    vec![join.right.name().to_string(), o.name().to_string()],
                                    Identifier::from(vec![
                                        right_name.clone(),
                                        n.name().to_string(),
                                    ]),
                                )
                            }),
                    )
                    .collect();
                // Rename expressions in the operator// TODO
                let builder = Relation::join()
                    .left_names(left_names)
                    .right_names(right_names)
                    .operator(operator.rename(&columns))
                    .and(Expr::eq(
                        Expr::qcol(left_name.as_str(), PE_ID),
                        Expr::qcol(right_name.as_str(), PE_ID),
                    ))
                    .left(left)
                    .right(right);
                let join: Join = builder.build();
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
                let builder = builder.input(Rc::new(join.into()));

                Ok(builder.build())
            }
        }
    }

    fn set(
        &self,
        set: &'a crate::relation::Set,
        left: Result<Relation>,
        right: Result<Relation>,
    ) -> Result<Relation> {
        let Set {
            name,
            operator,
            quantifier,
            ..
        } = set;
        let builder = Relation::set()
            .name(name)
            .operator(operator.clone())
            .quantifier(quantifier.clone())
            .left(left?)
            .right(right?);
        Ok(builder.build())
    }
}

impl Relation {
    /// Add protection
    pub fn protect_from_visitor<F: Fn(&Table) -> Relation>(
        self,
        protect_visitor: ProtectVisitor<F>,
    ) -> Result<Relation> {
        self.accept(protect_visitor)
    }

    /// Add protection
    pub fn protect<F: Fn(&Table) -> Relation>(self, protect_tables: F) -> Result<Relation> {
        self.accept(ProtectVisitor::new(protect_tables, Strategy::Soft))
    }

    /// Add protection
    pub fn protect_from_exprs<'a>(
        self,
        protected_entity: &'a [(&'a Table, Expr)],
    ) -> Result<Relation> {
        self.accept(protect_visitor_from_exprs(protected_entity, Strategy::Soft))
    }

    /// Add protection
    pub fn protect_from_field_paths<'a>(
        self,
        relations: &'a Hierarchy<Rc<Relation>>,
        protected_entity: &'a [(&'a str, &'a [(&'a str, &'a str, &'a str)], &'a str)],
    ) -> Result<Relation> {
        self.accept(protect_visitor_from_field_paths(
            relations,
            protected_entity,
            Strategy::Soft,
        ))
    }

    /// Force protection
    pub fn force_protect<F: Fn(&Table) -> Relation>(self, protect_tables: F) -> Relation {
        self.accept(ProtectVisitor::new(protect_tables, Strategy::Hard))
            .unwrap()
    }

    /// Force protection
    pub fn force_protect_from_exprs<'a>(
        self,
        protected_entity: &'a [(&'a Table, Expr)],
    ) -> Relation {
        self.accept(protect_visitor_from_exprs(protected_entity, Strategy::Hard))
            .unwrap()
    }

    /// Force protection
    pub fn force_protect_from_field_paths<'a>(
        self,
        relations: &'a Hierarchy<Rc<Relation>>,
        protected_entity: &'a [(&'a str, &'a [(&'a str, &'a str, &'a str)], &'a str)],
    ) -> Relation {
        self.accept(protect_visitor_from_field_paths(
            relations,
            protected_entity,
            Strategy::Hard,
        ))
        .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        display::Dot,
        io::{postgresql, Database},
        relation::Variant,
        sql::parse,
    };
    use colored::Colorize;
    use itertools::Itertools;
    use sqlparser::ast;

    #[test]
    fn test_table_protection() {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        let table = relations.get(&["table_1".into()]).unwrap().as_ref().clone();
        // Table
        let table = table
            .protect_from_exprs(&[(&database.tables()[0], expr!(md5(a)))])
            .unwrap();
        println!("Schema protected = {}", table.schema());
        assert_eq!(table.schema()[0].name(), PE_ID)
    }

    #[test]
    fn test_table_protection_from_field_paths() {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        let table = relations
            .get(&["item_table".into()])
            .unwrap()
            .as_ref()
            .clone();
        // Table
        let table = table
            .protect_from_field_paths(
                &relations,
                &[("item_table", &[("order_id", "order_table", "id")], "id")],
            )
            .unwrap();
        println!("Schema protected = {}", table.schema());
        assert_eq!(table.schema()[0].name(), PE_ID)
    }

    #[test]
    fn test_relation_protection() {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        let relation = Relation::try_from(
            parse("SELECT sum(price) FROM item_table GROUP BY order_id")
                .unwrap()
                .with(&relations),
        )
        .unwrap();
        // let relation = Relation::try_from(parse("SELECT * FROM primary_table").unwrap().with(&relations)).unwrap();
        // Table
        let relation = relation.force_protect_from_field_paths(
            &relations,
            &[
                (
                    "item_table",
                    &[
                        ("order_id", "order_table", "id"),
                        ("user_id", "user_table", "id"),
                    ],
                    "name",
                ),
                ("order_table", &[("user_id", "user_table", "id")], "name"),
                ("user_table", &[], "name"),
            ],
        );
        relation.display_dot().unwrap();
        println!("Schema protected = {}", relation.schema());
        assert_eq!(relation.schema()[0].name(), PE_ID);
        // Print query
        let query: &str = &ast::Query::from(&relation).to_string();
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
    }

    #[test]
    fn test_compute_norm_on_protected_relation() {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        let relation =
            Relation::try_from(parse("SELECT * FROM item_table").unwrap().with(&relations))
                .unwrap();
        let relation = relation.force_protect_from_field_paths(
            &relations,
            &[
                (
                    "item_table",
                    &[
                        ("order_id", "order_table", "id"),
                        ("user_id", "user_table", "id"),
                    ],
                    "name",
                ),
                ("order_table", &[("user_id", "user_table", "id")], "name"),
                ("user_table", &[], "name"),
            ],
        );
        //display(&relation);
        println!("Schema protected = {}", relation.schema());
        assert_eq!(relation.schema()[0].name(), PE_ID);

        let vector = PE_ID.clone();
        let base = vec!["item"];
        let coordinates = vec!["price"];
        let norm = relation.l2_norm(vector, base, coordinates);
        norm.display_dot().unwrap();
        // Print query
        let query: &str = &ast::Query::from(&norm).to_string();
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
    }

    #[test]
    fn test_relation_protection_weights() {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        let relation = Relation::try_from(
            parse("SELECT * FROM order_table JOIN item_table ON id=order_id")
                .unwrap()
                .with(&relations),
        )
        .unwrap();
        // Table
        let relation = relation.force_protect_from_field_paths(
            &relations,
            &[
                (
                    "item_table",
                    &[
                        ("order_id", "order_table", "id"),
                        ("user_id", "user_table", "id"),
                    ],
                    "name",
                ),
                ("order_table", &[("user_id", "user_table", "id")], "name"),
                ("user_table", &[], "name"),
            ],
        );
        relation.display_dot().unwrap();
        println!("Schema protected = {}", relation.schema());
        assert_eq!(relation.schema()[0].name(), PE_ID);
        // Print query
        let query: &str = &ast::Query::from(&relation).to_string();
        println!("{}", format!("{query}").yellow());
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
    }
}
