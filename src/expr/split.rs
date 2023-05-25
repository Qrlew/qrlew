//! The splits with some improvements
//! Each split has named Expr and anonymous exprs
use super::{
    aggregate, function, visitor::Acceptor, Aggregate, Column, Expr, Function, Identifier, Value,
    Visitor,
};
use crate::{
    namer::{self, FIELD},
    And, Factor,
};
use colored::Colorize;
use itertools::Itertools;
use std::{fmt, rc::Rc};

/* Basic rules
Insertion of a split along another should happen at the bottom of an existing split if it has columns and at the top else.
 */

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Split {
    Map(Map),
    Reduce(Reduce),
}

impl Split {
    pub fn filter(expr: Expr) -> Map {
        Map::new(vec![], Some(expr), vec![], None)
    }

    pub fn order_by(expr: Expr, asc: bool) -> Map {
        Map::new(vec![], None, vec![(expr, asc)], None)
    }

    pub fn group_by(expr: Expr) -> Reduce {
        Reduce::new(vec![], vec![expr], None)
    }

    pub fn into_map(self) -> Map {
        match self {
            Split::Map(map) => map,
            Split::Reduce(reduce) => reduce.into_map(),
        }
    }

    pub fn into_reduce(self, aggregate: aggregate::Aggregate) -> Reduce {
        match self {
            Split::Map(map) => map.into_reduce(aggregate),
            Split::Reduce(reduce) => reduce,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Split::Map(m) => m.len(),
            Split::Reduce(r) => r.len(),
        }
    }

    pub fn map_last<F: FnOnce(Split) -> Split>(self, f: F) -> Self {
        match self {
            Split::Map(m) => m.map_last(f).into(),
            Split::Reduce(r) => r.map_last(f).into(),
        }
    }

    pub fn map_last_map<F: FnOnce(Map) -> Map>(self, f: F) -> Self {
        match self {
            Split::Map(m) => m.map_last_map(f).into(),
            Split::Reduce(r) => r.map_last_map(f).into(),
        }
    }

    pub fn map_last_reduce<F: FnOnce(Reduce) -> Reduce>(self, f: F) -> Self {
        match self {
            Split::Map(m) => m.map_last_reduce(f).into(),
            Split::Reduce(r) => r.map_last_reduce(f).into(),
        }
    }
}

impl Default for Split {
    fn default() -> Self {
        Split::Reduce(Reduce::default())
    }
}

impl fmt::Display for Split {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Split::Map(map) => map.fmt(f),
            Split::Reduce(reduce) => reduce.fmt(f),
        }
    }
}

impl From<Map> for Split {
    fn from(map: Map) -> Self {
        Split::Map(map)
    }
}

impl From<Reduce> for Split {
    fn from(reduce: Reduce) -> Self {
        Split::Reduce(reduce)
    }
}

impl And<Split> for Split {
    type Product = Split;

    fn and(self, other: Split) -> Self::Product {
        match (self, other) {
            (Split::Map(s), Split::Map(o)) => s.and(o).into(),
            (Split::Map(s), Split::Reduce(o)) => s.and(o).into(),
            (Split::Reduce(s), Split::Map(o)) => s.and(o).into(),
            (Split::Reduce(s), Split::Reduce(o)) => s.and(o).into(),
        }
    }
}

/// A split with stable number of subsplits
#[derive(Clone, Default, Debug, Hash, PartialEq, Eq)]
pub struct Map {
    pub named_exprs: Vec<(String, Expr)>,
    pub filter: Option<Expr>,
    pub order_by: Vec<(Expr, bool)>,
    pub reduce: Option<Box<Reduce>>,
}

impl Map {
    /// Maps should be built using this builder
    pub fn new(
        named_exprs: Vec<(String, Expr)>,
        filter: Option<Expr>,
        order_by: Vec<(Expr, bool)>,
        reduce: Option<Reduce>,
    ) -> Self {
        Map {
            named_exprs: named_exprs.into_iter().unique().collect(),
            filter,
            order_by: order_by.into_iter().unique().collect(),
            reduce: reduce.map(Box::new),
        }
    }

    pub fn named_exprs(&self) -> &[(String, Expr)] {
        &self.named_exprs
    }

    pub fn filter(&self) -> &Option<Expr> {
        &self.filter
    }

    pub fn order_by(&self) -> &Vec<(Expr, bool)> {
        &self.order_by
    }

    pub fn reduce(&self) -> Option<&Reduce> {
        self.reduce.as_deref()
    }

    pub fn into_reduce(self, aggregate: aggregate::Aggregate) -> Reduce {
        let Map {
            named_exprs,
            filter,
            order_by,
            reduce,
        } = self;
        let (named_aliases, aliased_expr): (Vec<(String, Expr)>, Vec<(String, Expr)>) = named_exprs
            .into_iter()
            .map(|(name, expr)| {
                let alias = namer::name_from_content(FIELD, &expr);
                (
                    (
                        name,
                        Expr::Aggregate(Aggregate {
                            aggregate,
                            argument: Rc::new(Expr::col(alias.clone())),
                        }),
                    ),
                    (alias.clone(), expr),
                )
            })
            .unzip();
        Reduce::new(
            named_aliases,
            Vec::new(),
            Some(Map::new(aliased_expr, filter, order_by, reduce.map(|r| *r))),
        )
    }

    pub fn len(&self) -> usize {
        1 + self.reduce().map_or(0, |r| r.len())
    }

    pub fn map_last<F: FnOnce(Split) -> Split>(self, f: F) -> Self {
        match self.reduce {
            Some(reduce) => Map::new(
                self.named_exprs,
                self.filter,
                self.order_by,
                Some(reduce.map_last(f)),
            ),
            None => {
                let split = f(self.clone().into());
                if let Split::Map(map) = split {
                    map
                } else {
                    self
                }
            }
        }
    }

    pub fn map_last_map<F: FnOnce(Map) -> Map>(self, f: F) -> Self {
        match self.reduce {
            Some(reduce) => match reduce.map {
                Some(_) => Map::new(
                    self.named_exprs,
                    self.filter,
                    self.order_by,
                    Some(reduce.map_last_map(f)),
                ),
                None => f(Map::new(
                    self.named_exprs,
                    self.filter,
                    self.order_by,
                    Some(*reduce),
                )),
            },
            None => f(self),
        }
    }

    pub fn map_last_reduce<F: FnOnce(Reduce) -> Reduce>(self, f: F) -> Self {
        match self.reduce {
            Some(reduce) => Map::new(
                self.named_exprs,
                self.filter,
                self.order_by,
                Some(reduce.map_last_reduce(f)),
            ),
            None => self,
        }
    }
}

impl fmt::Display for Map {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Map {
            named_exprs,
            filter,
            order_by,
            reduce,
        } = self;
        write!(
            f,
            "{}\n{}",
            named_exprs
                .iter()
                .map(|(n, e)| format!("{} -> {}", n, e.to_string().yellow()))
                .chain(
                    filter
                        .into_iter()
                        .map(|e| format!("WHERE -> {}", e.to_string().yellow()))
                )
                .chain(
                    order_by
                        .iter()
                        .map(|(e, _)| format!("ORDER BY -> {}", e.to_string().yellow()))
                )
                .join("\n"),
            reduce.as_deref().map_or(String::new(), ToString::to_string),
        )
    }
}

// TODO so when concatenating a single split, its sub-expressions should be substituted
impl And<Self> for Map {
    type Product = Self;

    fn and(self, other: Self) -> Self::Product {
        match (self.reduce, other.reduce) {
            (None, None) => Map::new(
                self.named_exprs
                    .into_iter()
                    .chain(other.named_exprs)
                    .collect(),
                self.filter.into_iter().chain(other.filter).next(),
                self.order_by.into_iter().chain(other.order_by).collect(),
                None,
            ),
            (Some(s), Some(o)) => Map::new(
                self.named_exprs
                    .into_iter()
                    .chain(other.named_exprs)
                    .collect(),
                self.filter.into_iter().chain(other.filter).next(),
                self.order_by.into_iter().chain(other.order_by).collect(),
                Some(s.and(*o)),
            ),
            (None, Some(o)) => {
                let (reduce, named_exprs) = self.named_exprs.into_iter().fold(
                    (*o, Vec::new()),
                    |(reduce, mut named_exprs), (name, expr)| {
                        let (reduce, expr) = reduce.and(expr);
                        named_exprs.push((name, expr));
                        (reduce, named_exprs)
                    },
                );
                let (reduce, filter) =
                    self.filter
                        .into_iter()
                        .fold((reduce, None), |(reduce, _), expr| {
                            let (reduce, expr) = reduce.and(expr);
                            (reduce, Some(expr))
                        });
                let (reduce, order_by) = self.order_by.into_iter().fold(
                    (reduce, Vec::new()),
                    |(reduce, mut order_by), (expr, asc)| {
                        let (reduce, expr) = reduce.and(expr);
                        order_by.push((expr, asc));
                        (reduce, order_by)
                    },
                );
                Map::new(
                    named_exprs.into_iter().chain(other.named_exprs).collect(),
                    filter.into_iter().chain(other.filter).next(),
                    order_by.into_iter().chain(other.order_by).collect(),
                    Some(reduce),
                )
            }
            (Some(s), None) => {
                let (reduce, named_exprs) = other.named_exprs.into_iter().fold(
                    (*s, Vec::new()),
                    |(reduce, mut named_exprs), (name, expr)| {
                        let (reduce, expr) = reduce.and(expr);
                        named_exprs.push((name, expr));
                        (reduce, named_exprs)
                    },
                );
                let (reduce, filter) =
                    other
                        .filter
                        .into_iter()
                        .fold((reduce, None), |(reduce, _), expr| {
                            let (reduce, expr) = reduce.and(expr);
                            (reduce, Some(expr))
                        });
                let (reduce, order_by) = other.order_by.into_iter().fold(
                    (reduce, Vec::new()),
                    |(reduce, mut order_by), (expr, asc)| {
                        let (reduce, expr) = reduce.and(expr);
                        order_by.push((expr, asc));
                        (reduce, order_by)
                    },
                );
                Map::new(
                    self.named_exprs.into_iter().chain(named_exprs).collect(),
                    self.filter.into_iter().chain(filter).next(),
                    self.order_by.into_iter().chain(order_by).collect(),
                    Some(reduce),
                )
            }
        }
    }
}

impl And<Reduce> for Map {
    type Product = Self;

    fn and(self, other: Reduce) -> Self::Product {
        self.and(other.into_map())
    }
}

impl And<Expr> for Map {
    type Product = (Map, Expr);

    fn and(self, expr: Expr) -> Self::Product {
        let Map {
            named_exprs,
            filter,
            order_by,
            reduce,
        } = self;
        // Add the expr to the next split if needed
        let (reduce, expr) = if let Some(r) = reduce {
            let (r, expr) = r.and(expr);
            (Some(r), expr)
        } else {
            (None, expr)
        };
        // Collect sub-expressions
        let patterns: Vec<(String, Expr)> = expr
            .columns()
            .into_iter()
            .map(|c| {
                let column = Expr::Column(c.clone());
                (namer::name_from_content(FIELD, &column), column)
            })
            .chain(named_exprs.clone())
            .unique()
            .collect();
        // Replace the sub-expressions
        let (expr, matched) = expr.alias(patterns);
        // Add matched sub-expressions
        (
            Map::new(
                named_exprs.into_iter().chain(matched).collect(),
                filter,
                order_by,
                reduce,
            ),
            expr,
        )
    }
}

#[derive(Clone, Default, Debug, Hash, PartialEq, Eq)]
pub struct Reduce {
    pub named_exprs: Vec<(String, Expr)>,
    pub group_by: Vec<Expr>,
    pub map: Option<Box<Map>>,
}

impl Reduce {
    pub fn new(named_exprs: Vec<(String, Expr)>, group_by: Vec<Expr>, map: Option<Map>) -> Self {
        Reduce {
            named_exprs: named_exprs.into_iter().unique().collect(),
            group_by: group_by.into_iter().unique().collect(),
            map: map.map(Box::new),
        }
    }

    pub fn named_exprs(&self) -> &[(String, Expr)] {
        &self.named_exprs
    }

    pub fn group_by(&self) -> &[Expr] {
        &self.group_by
    }

    pub fn map(&self) -> Option<&Map> {
        self.map.as_deref()
    }

    pub fn into_map(self) -> Map {
        let Reduce {
            named_exprs,
            group_by,
            map,
        } = self;
        let (named_aliases, aliased_expr): (Vec<(String, Expr)>, Vec<(String, Expr)>) = named_exprs
            .into_iter()
            .map(|(name, expr)| {
                let alias = namer::name_from_content(FIELD, &expr);
                ((name, Expr::col(alias.clone())), (alias.clone(), expr))
            })
            .unzip();
        // If the reduce is empty, remove it
        if aliased_expr.is_empty() && group_by.is_empty() {
            Map::new(named_aliases, None, Vec::new(), None)
        } else {
            Map::new(
                named_aliases,
                None,
                Vec::new(),
                Some(Reduce::new(aliased_expr, group_by, map.map(|m| *m))),
            )
        }
    }

    pub fn len(&self) -> usize {
        1 + self.map().map_or(0, |m| m.len())
    }

    pub fn map_last<F: FnOnce(Split) -> Split>(self, f: F) -> Self {
        match self.map {
            Some(map) => Reduce::new(self.named_exprs, self.group_by, Some(map.map_last(f))),
            None => {
                let split = f(self.clone().into());
                if let Split::Reduce(reduce) = split {
                    reduce
                } else {
                    self
                }
            }
        }
    }

    pub fn map_last_map<F: FnOnce(Map) -> Map>(self, f: F) -> Self {
        match self.map {
            Some(map) => Reduce::new(self.named_exprs, self.group_by, Some(map.map_last_map(f))),
            None => self,
        }
    }

    pub fn map_last_reduce<F: FnOnce(Reduce) -> Reduce>(self, f: F) -> Self {
        match self.map {
            Some(map) => match map.reduce {
                Some(_) => Reduce::new(
                    self.named_exprs,
                    self.group_by,
                    Some(map.map_last_reduce(f)),
                ),
                None => f(Reduce::new(self.named_exprs, self.group_by, Some(*map))),
            },
            None => f(self),
        }
    }
}

impl fmt::Display for Reduce {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Reduce {
            named_exprs,
            group_by,
            map,
        } = self;
        write!(
            f,
            "{}\n{}",
            named_exprs
                .iter()
                .map(|(n, e)| format!("{} -> {}", n, e.to_string().red()))
                .chain(
                    group_by
                        .iter()
                        .map(|e| format!("GROUP BY -> {}", e.to_string().red()))
                )
                .join("\n"),
            map.as_deref().map_or(String::new(), ToString::to_string),
        )
    }
}

impl And<Self> for Reduce {
    type Product = Self;

    fn and(self, other: Self) -> Self::Product {
        match (self.map, other.map) {
            (None, None) => Reduce::new(
                self.named_exprs
                    .into_iter()
                    .chain(other.named_exprs)
                    .collect(),
                self.group_by.into_iter().chain(other.group_by).collect(),
                None,
            ),
            (Some(s), Some(o)) => Reduce::new(
                self.named_exprs
                    .into_iter()
                    .chain(other.named_exprs)
                    .collect(),
                self.group_by.into_iter().chain(other.group_by).collect(),
                Some(s.and(*o)),
            ),
            (None, Some(o)) => {
                let (map, named_exprs) = self.named_exprs.into_iter().fold(
                    (*o, Vec::new()),
                    |(map, mut named_exprs), (name, expr)| {
                        let (map, expr) = map.and(expr);
                        named_exprs.push((name, expr));
                        (map, named_exprs)
                    },
                );
                let (map, group_by) = self.group_by.into_iter().fold(
                    (map, Vec::new()),
                    |(map, mut group_by), expr| {
                        let (map, expr) = map.and(expr);
                        group_by.push(expr);
                        (map, group_by)
                    },
                );
                Reduce::new(
                    named_exprs.into_iter().chain(other.named_exprs).collect(),
                    group_by.into_iter().chain(other.group_by).collect(),
                    Some(map),
                )
            }
            (Some(s), None) => {
                let (map, named_exprs) = other.named_exprs.into_iter().fold(
                    (*s, Vec::new()),
                    |(map, mut named_exprs), (name, expr)| {
                        let (map, expr) = map.and(expr);
                        named_exprs.push((name, expr));
                        (map, named_exprs)
                    },
                );
                let (map, group_by) = other.group_by.into_iter().fold(
                    (map, Vec::new()),
                    |(map, mut group_by), expr| {
                        let (map, expr) = map.and(expr);
                        group_by.push(expr);
                        (map, group_by)
                    },
                );
                Reduce::new(
                    self.named_exprs.into_iter().chain(named_exprs).collect(),
                    self.group_by.into_iter().chain(group_by).collect(),
                    Some(map),
                )
            }
        }
    }
}

impl And<Map> for Reduce {
    type Product = Map;

    fn and(self, other: Map) -> Self::Product {
        self.into_map().and(other)
    }
}

impl And<Expr> for Reduce {
    type Product = (Reduce, Expr);

    fn and(self, expr: Expr) -> Self::Product {
        let Reduce {
            named_exprs,
            group_by,
            map,
        } = self;
        // Add the expr to the next split if needed
        let (map, expr) = if let Some(m) = map {
            let (m, expr) = m.and(expr);
            (Some(m), expr)
        } else {
            (None, expr)
        };
        // Collect sub-expressions
        let patterns: Vec<(String, Expr)> = expr
            .columns()
            .into_iter()
            .map(|c| {
                let column = Expr::Column(c.clone());
                (namer::name_from_content(FIELD, &column), column)
            })
            .chain(
                group_by
                    .clone()
                    .into_iter()
                    .map(|e| (namer::name_from_content(FIELD, &e), e)),
            )
            .unique()
            .collect();
        // Replace the sub-expressions
        let (expr, matched) = expr.alias(patterns);
        // Add matched sub-expressions
        (
            Reduce::new(
                named_exprs.into_iter().chain(matched).collect(),
                group_by,
                map,
            ),
            expr,
        )
    }
}

#[derive(Clone, Debug)]
pub struct SplitVisitor(String);

impl<'a> Visitor<'a, Split> for SplitVisitor {
    fn column(&self, column: &'a Column) -> Split {
        Map::new(
            vec![(self.0.clone(), Expr::Column(column.clone()))],
            None,
            Vec::new(),
            None,
        )
        .into()
    }

    fn value(&self, value: &'a Value) -> Split {
        Map::new(
            vec![(self.0.clone(), Expr::Value(value.clone()))],
            None,
            Vec::new(),
            None,
        )
        .into()
    }

    fn function(&self, function: &'a function::Function, arguments: Vec<Split>) -> Split {
        let Map {
            named_exprs,
            filter,
            order_by,
            reduce,
        } = Split::all(arguments).into_map();
        Map::new(
            vec![(
                self.0.clone(),
                Expr::Function(Function::new(
                    function.clone(),
                    named_exprs
                        .into_iter()
                        .filter_map(|(n, e)| (n == self.0).then(|| Rc::new(e)))
                        .collect(),
                )),
            )],
            filter,
            order_by,
            reduce.map(|r| *r),
        )
        .into()
    }

    fn aggregate(&self, aggregate: &'a aggregate::Aggregate, argument: Split) -> Split {
        argument.into_reduce(aggregate.clone()).into()
    }

    fn structured(&self, fields: Vec<(Identifier, Split)>) -> Split {
        let (identifiers, fields): (Vec<Identifier>, Vec<Split>) = fields.into_iter().unzip();
        let Map {
            named_exprs,
            filter,
            order_by,
            reduce,
        } = Split::all(fields).into_map();
        Map::new(
            vec![(
                self.0.clone(),
                Expr::Struct(
                    identifiers
                        .into_iter()
                        .zip(
                            named_exprs
                                .into_iter()
                                .filter_map(|(n, e)| (n == self.0).then(|| Rc::new(e))),
                        )
                        .collect(),
                ),
            )],
            filter,
            order_by,
            reduce.map(|r| *r),
        )
        .into()
    }
}

// Builds a Split out of a named Expr
impl<S: Into<String>> From<(S, Expr)> for Split {
    fn from((name, expr): (S, Expr)) -> Self {
        expr.accept(SplitVisitor(name.into()))
    }
}

impl<S: Into<String>> FromIterator<(S, Expr)> for Split {
    fn from_iter<T: IntoIterator<Item = (S, Expr)>>(iter: T) -> Self {
        Split::all(iter.into_iter().map_into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map() {
        let map = Map::new(
            vec![
                ("a".into(), expr!(cos(x + y))),
                ("b".into(), expr!(sin(x - y))),
            ],
            None,
            vec![],
            None,
        );
        println!("map = {map}");
    }

    #[test]
    fn test_reduce() {
        let reduce = Reduce::new(
            vec![("a".into(), expr!(count(x))), ("b".into(), expr!(sum(y)))],
            vec![],
            None,
        );
        println!("reduce = {reduce}");
        let reduce = reduce.into_map();
        println!("reduce into map = {}", reduce);
        let reduce = reduce.and(Reduce::new(Vec::new(), vec![Expr::col("z")], None));
        println!("reduce and group by = {}", reduce);
    }

    #[test]
    fn test_and_split() {
        let a = Split::default();
        println!("a = {a}");
        println!("a = {a:?}");
        let b = Split::from(("b", expr!(exp(a))));
        println!("b = {b}");
        println!("b = {b:?}");
        println!("a & b = {}", a.and(b));
    }

    #[test]
    fn test_split_merge() {
        let u = Split::from(("u", expr!(sum(cos(x)))));
        println!("u = {u}");
        let v = Split::from(("v", expr!(sin(y))));
        println!("v = {v}");
        println!("u & v = {}", u.and(v));
    }

    #[test]
    fn test_split_merge_all() {
        let u = Split::from(("u", expr!(1)));
        println!("u = {u}");
        let v = Split::from(("v", expr!(y)));
        println!("v = {v}");
        let u_v: Split = u.and(v);
        println!("u & v = {u_v}");
        if let Split::Map(m) = u_v {
            assert!(m.reduce == None)
        }
    }

    #[test]
    fn test_plus() {
        println!(
            "sum = {}",
            Split::from(("a", expr!(1 + sum(x)))).and(Split::from(("b", expr!(count(y)))))
        );
    }

    #[test]
    fn test_and_expr() {
        let s = Split::from(("a", expr!(1 + sum(x)))).and(Split::from(("b", expr!(count(1 + y)))));
        let e = expr!(x);
        println!("expr = {}", e);
        if let Split::Map(m) = s {
            let (m, e) = m.and(e);
            println!("replaced split = {}", m);
            println!("replaced expr = {}", e);
        }
    }

    #[test]
    fn test_split_map_reduce_map_expr() {
        let split = Split::from_iter([("a", expr!(1 + sum(x))), ("b", expr!(count(1 + y)))]);
        println!("split = {split}");
    }
}
