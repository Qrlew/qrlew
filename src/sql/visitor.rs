//! An Acceptor and Visitor implementation for ast::Query

use crate::visitor::{self, Acceptor, Dependencies, Visited};
use sqlparser::ast;
use std::iter::Iterator;

/// A type to hold queries and relations with their aliases
/// A Table with its dependencies (a table can be a simple name refenrencing another Table)
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum TableWithAlias<'a> {
    // Subqueries in the FROM or queries in CTEs
    Query(&'a ast::Query, Option<&'a ast::TableAlias>),
    // References to named objects, possibly aliased
    Name(&'a ast::ObjectName, Option<&'a ast::TableAlias>),
}

/// A Wrapper, newtype pattern, to add conversion methods to an existing object
pub struct TableWithJoins<'a>(&'a ast::TableWithJoins);

impl<'a> TableWithJoins<'a> {
    pub fn new(table_with_joins: &'a ast::TableWithJoins) -> Self {
        TableWithJoins(table_with_joins)
    }

    fn tables_with_aliases(self) -> impl Iterator<Item = TableWithAlias<'a>> {
        match &self.0.relation {
            ast::TableFactor::Derived {
                subquery, alias, ..
            } => Some(TableWithAlias::Query(subquery.as_ref(), alias.as_ref())),
            ast::TableFactor::Table { name, alias, .. } => {
                Some(TableWithAlias::Name(name, alias.as_ref()))
            }
            _ => None,
        }
        .into_iter()
        .chain(
            // Then get the joins
            self.0.joins.iter().filter_map(|join| match &join.relation {
                ast::TableFactor::Derived {
                    subquery, alias, ..
                } => Some(TableWithAlias::Query(subquery.as_ref(), alias.as_ref())),
                ast::TableFactor::Table { name, alias, .. } => {
                    Some(TableWithAlias::Name(name, alias.as_ref()))
                }
                _ => None,
            }),
        )
    }

    /// Iterate over queries
    pub fn queries(self) -> impl Iterator<Item = &'a ast::Query> {
        self.tables_with_aliases().filter_map(|t| match t {
            TableWithAlias::Query(q, _) => Some(q),
            _ => None,
        })
    }

    /// Iterate over names
    pub fn names(self) -> impl Iterator<Item = &'a ast::ObjectName> {
        self.tables_with_aliases().filter_map(|t| match t {
            TableWithAlias::Name(n, _) => Some(n),
            _ => None,
        })
    }
}

/// Implement the Acceptor trait
impl<'a> Acceptor<'a> for ast::Query {
    fn dependencies(&'a self) -> Dependencies<'a, Self> {
        let mut dependencies = Dependencies::empty();
        // Add CTEs subqueries
        dependencies.extend(
            self.with
                .iter()
                .flat_map(|with| with.cte_tables.iter().map(|cte| cte.query.as_ref())),
        );
        // Add subqueries from the body
        dependencies.extend(match self.body.as_ref() {
            ast::SetExpr::Select(select) => select
                .from
                .iter()
                .flat_map(|table_with_joins| TableWithJoins(table_with_joins).queries()),
            _ => todo!(), // Not implemented
        });
        dependencies
    }
}

/// A Visitor for Queries
/// Dependencies are overrdden because references have to be unfold after a first pass to build the links (see IntoQueryNamesVisitor)
pub trait Visitor<'a, T: Clone> {
    /// The query visitor can affect the set of sub-queries it will visit by overriding this method
    fn dependencies(&self, acceptor: &'a ast::Query) -> Dependencies<'a, ast::Query> {
        acceptor.dependencies()
    }
    fn query(&self, query: &'a ast::Query, dependencies: Visited<'a, ast::Query, T>) -> T;
}

/// Unpack the visited queries of visitor::Visitor to ease the writing of Visitor
impl<'a, T: Clone, V: Visitor<'a, T>> visitor::Visitor<'a, ast::Query, T> for V {
    fn visit(&self, acceptor: &'a ast::Query, dependencies: Visited<'a, ast::Query, T>) -> T {
        self.query(acceptor, dependencies)
    }
    /// We override the dependencies to allow visitor control on the acceptors' dependencies
    fn dependencies(&self, acceptor: &'a ast::Query) -> Dependencies<'a, ast::Query> {
        self.dependencies(acceptor)
    }
}
