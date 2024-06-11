//! Data structures and visitor to collect object names and the queries they refer to.

use crate::{
    ast,
    expr::Identifier,
    hierarchy::Path,
    sql::visitor::{TableWithJoins, Visitor},
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

/// A mapping between an ObjectName in a Query and the Query referred by the Name (when available)
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct QueryNames<'a>(BTreeMap<(&'a ast::Query, ast::ObjectName), Option<&'a ast::Query>>);

impl<'a> QueryNames<'a> {
    /// Build a new QueryNames object
    pub fn new() -> Self {
        QueryNames(BTreeMap::new())
    }

    /// Set all unresolved names
    pub fn set(&mut self, name: ast::ObjectName, referred: &'a ast::Query) -> &mut Self {
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
    ) -> impl Iterator<Item = (&ast::ObjectName, &'a ast::Query)> {
        self.iter()
            .filter_map(move |((q, n), r)| if *q == query { Some((n, (*r)?)) } else { None })
    }
}

impl<'a> Deref for QueryNames<'a> {
    type Target = BTreeMap<(&'a ast::Query, ast::ObjectName), Option<&'a ast::Query>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> DerefMut for QueryNames<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a> IntoIterator for QueryNames<'a> {
    type Item =
        <BTreeMap<(&'a ast::Query, ast::ObjectName), Option<&'a ast::Query>> as IntoIterator>::Item;
    type IntoIter = <BTreeMap<(&'a ast::Query, ast::ObjectName), Option<&'a ast::Query>> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> fmt::Display for QueryNames<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Query Names\n{}",
            self.0
                .iter()
                .map(|((q, n), r)| match r {
                    Some(r) => format!(
                        "{} | {} -> {}",
                        format!("{q}").blue(),
                        format!("{n}").red(),
                        format!("{r}").green()
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
pub struct IntoQueryNamesVisitor;

pub(super) fn names_from_set_expr<'a>(set_expr: &'a ast::SetExpr) -> Vec<&'a ast::ObjectName> {
    match set_expr {
        ast::SetExpr::Select(select) => select
            .from
            .iter()
            .flat_map(|table_with_joins| TableWithJoins::new(table_with_joins).names())
            .collect(),
        ast::SetExpr::SetOperation { left, right, .. } => names_from_set_expr(left.as_ref())
            .into_iter()
            .chain(names_from_set_expr(right.as_ref()))
            .collect(),
        _ => todo!(),
    }
}

impl<'a> Visitor<'a, QueryNames<'a>> for IntoQueryNamesVisitor {
    fn query(
        &self,
        query: &'a ast::Query,
        dependencies: Visited<'a, ast::Query, QueryNames<'a>>,
    ) -> QueryNames<'a> {
        let mut query_names = QueryNames::new();
        // Add all elemdnts in dependencies
        for (_, dependency) in dependencies {
            query_names.extend(dependency);
        }
        // Add reference elements from current query
        for name in names_from_set_expr(query.body.as_ref()) {
            query_names.insert((query, name.clone()), None);
        }
        // Set names
        if let Some(with) = &query.with {
            for cte in &with.cte_tables {
                query_names.set(
                    ast::ObjectName(vec![cte.alias.name.clone()]),
                    cte.query.as_ref(),
                );
            }
        }
        query_names
    }
}

/// Implement conversion from ObjectName to Path
impl Path for ast::ObjectName {
    fn path(self) -> Vec<String> {
        self.0.path()
    }
}

/// Implement conversion to Identifier
impl From<&ast::ObjectName> for Identifier {
    fn from(value: &ast::ObjectName) -> Self {
        value.0.iter().map(|i| i.value.clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use colored::Colorize;

    use super::*;
    use crate::{sql::relation, visitor::Acceptor as _};

    #[test]
    fn test_query_names() {
        let query_1 = relation::parse("select * from table_1").unwrap();
        let query_2 = relation::parse("select * from table_2").unwrap();
        let query_3 = relation::parse("select * from table_3").unwrap();
        let query_4 = relation::parse("select * from table_4").unwrap();
        let name_1 = ast::ObjectName(vec!["name_1".into()]);
        let name_2 = ast::ObjectName(vec!["name_2".into()]);
        let name_3 = ast::ObjectName(vec!["name_3".into()]);
        println!("query_1 = {}", query_1.to_string().blue());
        let mut query_names_1 = QueryNames::new();
        let mut query_names_2 = QueryNames::new();
        let mut query_names_3 = QueryNames::new();
        query_names_2.insert((&query_1, name_1), None);
        query_names_2.insert((&query_1, name_2), Some(&query_2));
        query_names_3.insert((&query_1, name_3.clone()), Some(&query_3));
        query_names_3.insert((&query_2, name_3.clone()), None);
        query_names_3.insert((&query_3, name_3.clone()), None);
        query_names_1.extend(query_names_2);
        query_names_1.extend(query_names_3);
        println!("BEFORE: {query_names_1}");
        query_names_1.set(name_3.clone(), &query_4);
        println!("AFTER: {query_names_1}");
    }

    const COMPLEX_QUERY: &str = "
        with view_1 as (select * from schema.table_1),
        view_2 as (select * from view_1)
        select 2*a, b+1 from (select view_2.c as a, right.d as b from (select * from view_2) LEFT OUTER JOIN schema.table_2 as right ON view_1.id = table_2.id);
        select * from table_1;
    ";

    #[test]
    fn test_query_names_visitor() {
        let query = relation::parse(COMPLEX_QUERY).unwrap();
        println!("Query = {}", query.to_string().blue());
        let visitor = IntoQueryNamesVisitor;
        let query_names = query.accept(visitor);
        println!("{}", query_names);
    }
}
