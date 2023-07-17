//! Methods to convert Relations to ast::Query
use super::{
    Error, Join, JoinConstraint, JoinOperator, Map, OrderBy, Reduce, Relation, Result, Set,
    SetOperator, SetQuantifier, Table, Variant as _, Visitor,
};
use crate::{
    ast,
    data_type::{DataType, DataTyped},
    expr::identifier::Identifier,
    visitor::Acceptor,
};
use std::{collections::HashSet, convert::TryFrom, iter::Iterator};

/// A simple Relation -> ast::Query conversion Visitor using CTE
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct FromRelationVisitor;

impl TryFrom<Identifier> for ast::Ident {
    type Error = Error;

    fn try_from(value: Identifier) -> Result<Self> {
        if value.len() == 1 {
            Ok(ast::Ident::new(value.head()?))
        } else {
            Err(Error::invalid_conversion(value, "ast::Ident"))
        }
    }
}

impl From<JoinConstraint> for ast::JoinConstraint {
    fn from(value: JoinConstraint) -> Self {
        match value {
            JoinConstraint::On(expr) => ast::JoinConstraint::On(ast::Expr::from(&expr)),
            JoinConstraint::Using(idents) => ast::JoinConstraint::Using(
                idents
                    .into_iter()
                    .map(|ident| ident.try_into().unwrap())
                    .collect(),
            ),
            JoinConstraint::Natural => ast::JoinConstraint::Natural,
            JoinConstraint::None => ast::JoinConstraint::None,
        }
    }
}

impl From<JoinOperator> for ast::JoinOperator {
    fn from(value: JoinOperator) -> Self {
        match value {
            JoinOperator::Inner(join_constraint) => {
                ast::JoinOperator::Inner(join_constraint.into())
            }
            JoinOperator::LeftOuter(join_constraint) => {
                ast::JoinOperator::LeftOuter(join_constraint.into())
            }
            JoinOperator::RightOuter(join_constraint) => {
                ast::JoinOperator::RightOuter(join_constraint.into())
            }
            JoinOperator::FullOuter(join_constraint) => {
                ast::JoinOperator::FullOuter(join_constraint.into())
            }
            JoinOperator::Cross => ast::JoinOperator::CrossJoin,
        }
    }
}

impl From<SetOperator> for ast::SetOperator {
    fn from(value: SetOperator) -> Self {
        match value {
            SetOperator::Union => ast::SetOperator::Union,
            SetOperator::Except => ast::SetOperator::Except,
            SetOperator::Intersect => ast::SetOperator::Intersect,
        }
    }
}

impl From<SetQuantifier> for ast::SetQuantifier {
    fn from(value: SetQuantifier) -> Self {
        match value {
            SetQuantifier::All => ast::SetQuantifier::All,
            SetQuantifier::Distinct => ast::SetQuantifier::Distinct,
            SetQuantifier::None => ast::SetQuantifier::None,
        }
    }
}

/// Build a Query from simple elements
/// Have a look at: https://docs.rs/sqlparser/latest/sqlparser/ast/struct.Query.html
/// Also this can help: https://www.postgresql.org/docs/current/sql-select.html
fn query(
    with: Vec<ast::Cte>,
    projection: Vec<ast::SelectItem>,
    from: ast::TableWithJoins,
    selection: Option<ast::Expr>,
    group_by: Vec<ast::Expr>,
    order_by: Vec<ast::OrderByExpr>,
    limit: Option<ast::Expr>,
) -> ast::Query {
    ast::Query {
        with: (!with.is_empty()).then_some(ast::With {
            recursive: false,
            cte_tables: with,
        }),
        body: Box::new(ast::SetExpr::Select(Box::new(ast::Select {
            distinct: None,
            top: None,
            projection,
            into: None,
            from: vec![from],
            lateral_views: Vec::new(),
            selection,
            group_by,
            cluster_by: Vec::new(),
            distribute_by: Vec::new(),
            sort_by: Vec::new(),
            having: None,
            qualify: None,
            named_window: Vec::new(),
        }))),
        order_by,
        limit,
        offset: None,
        fetch: None,
        locks: Vec::new(),
    }
}

fn table_factor(name: ast::Ident) -> ast::TableFactor {
    ast::TableFactor::Table {
        name: ast::ObjectName(vec![name]),
        alias: None,
        args: None,
        with_hints: Vec::new(),
    }
}

fn table_with_joins(name: ast::Ident, joins: Vec<ast::Join>) -> ast::TableWithJoins {
    ast::TableWithJoins {
        relation: table_factor(name),
        joins,
    }
}

fn ctes_from_query(query: ast::Query) -> Vec<ast::Cte> {
    query.with.map(|with| with.cte_tables).unwrap_or_default()
}

fn cte(name: ast::Ident, columns: Vec<ast::Ident>, query: ast::Query) -> ast::Cte {
    ast::Cte {
        alias: ast::TableAlias { name, columns },
        query: Box::new(query),
        from: None,
    }
}

fn all() -> Vec<ast::SelectItem> {
    vec![ast::SelectItem::Wildcard(
        ast::WildcardAdditionalOptions::default(),
    )]
}

fn select_from_query(query: ast::Query) -> ast::Select {
    match query.body.as_ref() {
        ast::SetExpr::Select(select) => select.as_ref().clone(),
        _ => panic!("Non select query"), // It is okay to panic as this should not happen in our context and is a private function
    }
}

/// Build a set operation
fn set_operation(
    with: Vec<ast::Cte>,
    operator: ast::SetOperator,
    quantifier: ast::SetQuantifier,
    left: ast::Select,
    right: ast::Select,
) -> ast::Query {
    ast::Query {
        with: (!with.is_empty()).then_some(ast::With {
            recursive: false,
            cte_tables: with,
        }),
        body: Box::new(ast::SetExpr::SetOperation {
            op: operator,
            set_quantifier: quantifier,
            left: Box::new(ast::SetExpr::Select(Box::new(left))),
            right: Box::new(ast::SetExpr::Select(Box::new(right))),
        }),
        order_by: Vec::new(),
        limit: None,
        offset: None,
        fetch: None,
        locks: Vec::new(),
    }
}

impl<'a> Visitor<'a, ast::Query> for FromRelationVisitor {
    fn table(&self, table: &'a Table) -> ast::Query {
        query(
            Vec::new(),
            vec![ast::SelectItem::Wildcard(
                ast::WildcardAdditionalOptions::default(),
            )],
            table_with_joins(table.name().into(), Vec::new()),
            None,
            Vec::new(),
            Vec::new(),
            None,
        )
    }

    fn map(&self, map: &'a Map, input: ast::Query) -> ast::Query {
        // Pull the existing CTEs
        let mut input_ctes = ctes_from_query(input);
        // Add input query to CTEs
        input_ctes.push(cte(
            map.name().into(),
            map.schema()
                .iter()
                .map(|field| ast::Ident::from(field.name()))
                .collect(),
            query(
                Vec::new(),
                map.projection
                    .clone()
                    .into_iter()
                    .zip(map.schema.clone())
                    .map(|(expr, field)| ast::SelectItem::ExprWithAlias {
                        expr: ast::Expr::from(&expr),
                        alias: field.name().into(),
                    })
                    .collect(),
                table_with_joins(map.input.name().into(), Vec::new()),
                map.filter.as_ref().map(ast::Expr::from),
                Vec::new(),
                map.order_by
                    .iter()
                    .map(|OrderBy { expr, asc }| ast::OrderByExpr {
                        expr: expr.into(),
                        asc: Some(*asc),
                        nulls_first: None,
                    })
                    .collect(),
                map.limit
                    .map(|limit| ast::Expr::Value(ast::Value::Number(limit.to_string(), false))),
            ),
        ));
        query(
            input_ctes,
            all(),
            table_with_joins(map.name().into(), Vec::new()),
            None,
            Vec::new(),
            Vec::new(),
            map.limit
                .map(|limit| ast::Expr::Value(ast::Value::Number(limit.to_string(), false))),
        )
    }

    fn reduce(&self, reduce: &'a Reduce, input: ast::Query) -> ast::Query {
        // Pull the existing CTEs
        let mut input_ctes = ctes_from_query(input);
        // Add input query to CTEs
        input_ctes.push(cte(
            reduce.name().into(),
            reduce
                .schema()
                .iter()
                .map(|field| ast::Ident::from(field.name()))
                .collect(),
            query(
                Vec::new(),
                reduce
                    .aggregate
                    .clone()
                    .into_iter()
                    .zip(reduce.schema.clone())
                    .map(|(expr, field)| ast::SelectItem::ExprWithAlias {
                        expr: ast::Expr::from(&expr),
                        alias: field.name().into(),
                    })
                    .collect(),
                table_with_joins(reduce.input.name().into(), Vec::new()),
                None,
                reduce.group_by.iter().map(ast::Expr::from).collect(),
                Vec::new(),
                None,
            ),
        ));
        query(
            input_ctes,
            all(),
            table_with_joins(reduce.name().into(), Vec::new()),
            None,
            Vec::new(),
            Vec::new(),
            None,
        )
    }

    fn join(&self, join: &'a Join, left: ast::Query, right: ast::Query) -> ast::Query {
        // Pull the existing CTEs
        let mut exist: HashSet<ast::Cte> = HashSet::new();
        let mut input_ctes: Vec<ast::Cte> = Vec::new();
        ctes_from_query(left).into_iter().for_each(|cte| {
            if exist.insert(cte.clone()) {
                input_ctes.push(cte)
            }
        });
        ctes_from_query(right).into_iter().for_each(|cte| {
            if exist.insert(cte.clone()) {
                input_ctes.push(cte)
            }
        });
        // Add input query to CTEs
        input_ctes.push(cte(
            join.name().into(),
            join.schema()
                .iter()
                .map(|field| ast::Ident::from(field.name()))
                .collect(),
            query(
                Vec::new(),
                all(),
                table_with_joins(
                    join.left.name().into(),
                    vec![ast::Join {
                        relation: table_factor(join.right.name().into()),
                        join_operator: join.operator.clone().into(),
                    }],
                ),
                None,
                Vec::new(),
                Vec::new(),
                None,
            ),
        ));
        query(
            input_ctes,
            all(),
            table_with_joins(join.name().into(), Vec::new()),
            None,
            Vec::new(),
            Vec::new(),
            None,
        )
    }

    fn set(&self, set: &'a Set, left: ast::Query, right: ast::Query) -> ast::Query {
        // Pull the existing CTEs
        let mut exist: HashSet<ast::Cte> = HashSet::new();
        let mut input_ctes: Vec<ast::Cte> = Vec::new();
        ctes_from_query(left.clone()).into_iter().for_each(|cte| {
            if exist.insert(cte.clone()) {
                input_ctes.push(cte)
            }
        });
        ctes_from_query(right.clone()).into_iter().for_each(|cte| {
            if exist.insert(cte.clone()) {
                input_ctes.push(cte)
            }
        });
        // Add input query to CTEs
        input_ctes.push(cte(
            set.name().into(),
            set.schema()
                .iter()
                .map(|field| ast::Ident::from(field.name()))
                .collect(),
            set_operation(
                Vec::new(),
                set.operator.clone().into(),
                set.quantifier.clone().into(),
                select_from_query(left),
                select_from_query(right),
            ),
        ));
        query(
            input_ctes,
            all(),
            table_with_joins(set.name().into(), Vec::new()),
            None,
            Vec::new(),
            Vec::new(),
            None,
        )
    }
}

/// Based on the FromRelationVisitor implement the From trait
impl From<&Relation> for ast::Query {
    fn from(value: &Relation) -> Self {
        value.accept(FromRelationVisitor)
    }
}

impl Table {
    /// Build the CREATE TABLE statement
    pub fn create(&self) -> ast::Statement {
        ast::Statement::CreateTable {
            or_replace: false,
            temporary: false,
            external: false,
            global: None,
            if_not_exists: true,
            transient: false,
            name: ast::ObjectName(vec![self.name().into()]),
            columns: self
                .schema()
                .iter()
                .map(|f| ast::ColumnDef {
                    name: f.name().into(),
                    data_type: f.data_type().into(),
                    collation: None,
                    options: if let DataType::Optional(_) = f.data_type() {
                        vec![]
                    } else {
                        vec![ast::ColumnOptionDef {
                            name: None,
                            option: ast::ColumnOption::NotNull,
                        }]
                    },
                })
                .collect(),
            constraints: Vec::new(),
            hive_distribution: ast::HiveDistributionStyle::NONE,
            hive_formats: None,
            table_properties: Vec::new(),
            with_options: Vec::new(),
            file_format: None,
            location: None,
            query: None,
            without_rowid: false,
            like: None,
            clone: None,
            engine: None,
            default_charset: None,
            collation: None,
            on_commit: None,
            on_cluster: None,
            order_by: None,
            strict: false,
        }
    }

    pub fn insert(&self, prefix: char) -> ast::Statement {
        ast::Statement::Insert {
            or: None,
            into: true,
            table_name: ast::ObjectName(vec![self.name().into()]),
            columns: self.schema().iter().map(|f| f.name().into()).collect(),
            overwrite: false,
            source: Box::new(ast::Query {
                with: None,
                body: Box::new(ast::SetExpr::Values(ast::Values {
                    explicit_row: false,
                    rows: vec![(1..=self.schema().len())
                        .map(|i| ast::Expr::Value(ast::Value::Placeholder(format!("{prefix}{i}"))))
                        .collect()],
                })),
                order_by: Vec::new(),
                limit: None,
                offset: None,
                fetch: None,
                locks: Vec::new(),
            }),
            partitioned: None,
            after_columns: Vec::new(),
            table: false,
            on: None,
            returning: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        builder::{Ready, With},
        data_type::DataType,
        display::Dot,
        expr::Expr,
        namer,
        relation::schema::Schema,
    };
    use std::rc::Rc;

    fn build_complex_relation() -> Rc<Relation> {
        namer::reset();
        let schema: Schema = vec![
            ("a", DataType::float()),
            ("b", DataType::float_interval(-2., 2.)),
            ("c", DataType::float()),
            ("d", DataType::float_interval(0., 1.)),
        ]
        .into_iter()
        .collect();
        let table: Rc<Relation> = Rc::new(
            Relation::table()
                .name("table")
                .schema(schema.clone())
                .size(100)
                .build(),
        );
        let map: Rc<Relation> = Rc::new(
            Relation::map()
                .name("map_1")
                .with(Expr::exp(Expr::col("a")))
                .input(table.clone())
                .with(Expr::col("b") + Expr::col("d"))
                .build(),
        );
        let join: Rc<Relation> = Rc::new(
            Relation::join()
                .name("join")
                .cross()
                .left(table.clone())
                .right(map.clone())
                .build(),
        );
        let map_2: Rc<Relation> = Rc::new(
            Relation::map()
                .name("map_2")
                .with(Expr::exp(Expr::col(join[4].name())))
                .input(join.clone())
                .with(Expr::col(join[0].name()) + Expr::col(join[1].name()))
                .limit(100)
                .build(),
        );
        let join_2: Rc<Relation> = Rc::new(
            Relation::join()
                .name("join_2")
                .cross()
                .left(join.clone())
                .right(map_2.clone())
                .build(),
        );
        join_2
    }

    #[test]
    fn test_from_table_relation() {
        // let relation = build_complex_relation();
        let schema: Schema = vec![
            ("a", DataType::float()),
            ("b", DataType::float_interval(-2., 2.)),
            ("c", DataType::float()),
            ("d", DataType::float_interval(0., 1.)),
        ]
        .into_iter()
        .collect();
        let table: Relation = Relation::table()
            .name("Name")
            .schema(schema.clone())
            .build();
        let query = ast::Query::from(&table);
        println!("query = {query}");
    }

    #[test]
    fn test_from_complex_relation() {
        let relation = build_complex_relation();
        let relation = relation.as_ref();
        relation.display_dot().unwrap();
        let query = ast::Query::from(relation);
        println!("query = {query}");
    }

    #[test]
    fn test_display_join() {
        namer::reset();
        let schema: Schema = vec![("b", DataType::float_interval(-2., 2.))]
            .into_iter()
            .collect();
        let left: Relation = Relation::table()
            .name("left")
            .schema(schema.clone())
            .size(1000)
            .build();
        let right: Relation = Relation::table()
            .name("right")
            .schema(schema.clone())
            .size(1000)
            .build();

        let join: Relation = Relation::join()
            .name("join")
            .left_outer()
            //.using("a")
            .on(Expr::eq(Expr::qcol("left", "b"), Expr::qcol("right", "b")))
            .left(left)
            .right(right)
            .build();

        let query = ast::Query::from(&join);
        println!("query = {}", query.to_string());
    }
}
