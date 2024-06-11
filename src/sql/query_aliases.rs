//! Data structures and visitor to collect object names and the queries they refer to.

use crate::{
    ast,
    sql::{query_names, visitor::Visitor},
    visitor::Visited,
};
use colored::Colorize;
use itertools::Itertools;
use std::{
    collections::BTreeMap,
    fmt,
    iter::Iterator,
    ops::{Deref, DerefMut},
};

/// A mapping between an ObjectName in a Query and the column aliases for that query referred by the Name (when available).
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct QueryAliases<'a>(
    BTreeMap<(&'a ast::Query, ast::ObjectName), Option<&'a Vec<ast::Ident>>>,
);

impl<'a> QueryAliases<'a> {
    /// Build a new QueryAliases object
    pub fn new() -> Self {
        QueryAliases(BTreeMap::new())
    }

    /// Set all unresolved names
    pub fn set(&mut self, name: ast::ObjectName, referred: &'a Vec<ast::Ident>) -> &mut Self {
        for ((_, n), r) in self.iter_mut() {
            if (*n == name) && r.is_none() {
                *r = Some(referred);
            }
        }
        self
    }

    /// Return all the names and referred queries in a query
    pub fn name_referred(
        &self,
        query: &'a ast::Query,
    ) -> impl Iterator<Item = (&ast::ObjectName, &'a Vec<ast::Ident>)> {
        self.iter()
            .filter_map(move |((q, n), r)| if *q == query { Some((n, (*r)?)) } else { None })
    }
}

impl<'a> Deref for QueryAliases<'a> {
    type Target = BTreeMap<(&'a ast::Query, ast::ObjectName), Option<&'a Vec<ast::Ident>>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> DerefMut for QueryAliases<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a> IntoIterator for QueryAliases<'a> {
    type Item =
        <BTreeMap<(&'a ast::Query, ast::ObjectName), Option<&'a Vec<ast::Ident>>> as IntoIterator>::Item;
    type IntoIter = <BTreeMap<(&'a ast::Query, ast::ObjectName), Option<&'a Vec<ast::Ident>>> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> fmt::Display for QueryAliases<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Query Aliases\n{}",
            self.0
                .iter()
                .map(|((q, n), r)| match r {
                    Some(r) => format!(
                        "{} | {} -> {}",
                        format!("{q}").blue(),
                        format!("{n}").red(),
                        format!("({})", r.iter().join(", ")).green()
                    ),
                    None => format!(
                        "{} | {} -> {}",
                        format!("{q}").blue(),
                        format!("{n}").red(),
                        format!("?").bold().green()
                    ),
                })
                .join(",\n")
        )
    }
}

/// A visitor to build a reference free query tree
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct IntoQueryAliasesVisitor;

impl<'a> Visitor<'a, QueryAliases<'a>> for IntoQueryAliasesVisitor {
    fn query(
        &self,
        query: &'a ast::Query,
        dependencies: Visited<'a, ast::Query, QueryAliases<'a>>,
    ) -> QueryAliases<'a> {
        let mut query_aliases = QueryAliases::new();
        for (_, dependency) in dependencies {
            query_aliases.extend(dependency);
        }
        // Add reference elements from current query
        for name in query_names::names_from_set_expr(query.body.as_ref()) {
            query_aliases.insert((query, name.clone()), None);
        }
        // Set names
        if let Some(with) = &query.with {
            for cte in &with.cte_tables {
                if !cte.alias.columns.is_empty() {
                    query_aliases.set(
                        ast::ObjectName(vec![cte.alias.name.clone()]),
                        cte.alias.columns.as_ref(),
                    );
                }
            }
        }
        query_aliases
    }
}

#[cfg(test)]
mod tests {
    use colored::Colorize;

    use super::*;
    use crate::{sql::relation, visitor::Acceptor as _};

    #[test]
    fn test_query_aliases() {
        let query_1 = relation::parse("select * from table_1").unwrap();
        let query_2 = relation::parse("select * from table_2").unwrap();
        let aliases_query_2: Vec<ast::Ident> = vec!["a".into(), "b".into(), "c".into()];
        let query_3 = relation::parse("select * from table_3").unwrap();
        let aliases_query_3: Vec<ast::Ident> = vec!["aa".into(), "bb".into(), "cc".into()];
        let aliases_4: Vec<ast::Ident> = vec!["aa".into(), "bb".into(), "cc".into()];
        let name_1 = ast::ObjectName(vec!["name_1".into()]);
        let name_2 = ast::ObjectName(vec!["name_2".into()]);
        let name_3 = ast::ObjectName(vec!["name_3".into()]);
        println!("query_1 = {}", query_1.to_string().blue());
        let mut query_aliases_1 = QueryAliases::new();
        let mut query_aliases_2 = QueryAliases::new();
        let mut query_aliases_3 = QueryAliases::new();
        query_aliases_2.insert((&query_1, name_1), None);
        query_aliases_2.insert((&query_1, name_2), Some(&aliases_query_2));
        query_aliases_3.insert((&query_1, name_3.clone()), Some(&aliases_query_3));
        query_aliases_3.insert((&query_2, name_3.clone()), None);
        query_aliases_3.insert((&query_3, name_3.clone()), None);
        query_aliases_1.extend(query_aliases_2);
        query_aliases_1.extend(query_aliases_3);
        println!("BEFORE: {query_aliases_1}");
        query_aliases_1.set(name_3.clone(), &aliases_4);
        println!("AFTER: {query_aliases_1}");
    }

    const COMPLEX_QUERY: &str = "
        WITH
        view_1 (a, b, c) as (select * from schema.table_1),
        view_2 (aa, bb, cc) as (select * from view_1)
        SELECT 2*new_a, b+1
        FROM (
            SELECT view_2.cc as new_a, right.d as b FROM (SELECT * FROM view_2) LEFT OUTER JOIN schema.table_2 as right ON view_1.a = table_2.id
            );
        select * from table_1;
    ";

    #[test]
    fn test_query_aliases_visitor() {
        let query = relation::parse(COMPLEX_QUERY).unwrap();
        println!("Query = {}", query.to_string().blue());
        let visitor = IntoQueryAliasesVisitor;
        let query_aliases = query.accept(visitor);
        println!("{}", query_aliases);
    }
}
