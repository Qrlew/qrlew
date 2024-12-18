use super::{Error, Field, JoinOperator, Relation, Variant as _, Visitor};
use crate::{
    data_type::DataTyped,
    display::{self, colors},
    expr::Expr,
    namer,
    relation::Join,
    visitor::Acceptor,
};
use itertools::Itertools;
use std::{borrow::Cow, fmt, io, str, string};

impl From<string::FromUtf8Error> for Error {
    fn from(err: string::FromUtf8Error) -> Self {
        Error::Other(err.to_string())
    }
}

#[derive(Clone, Debug)]
pub struct Node<'a, T: Clone + fmt::Display>(&'a Relation, T);
#[derive(Clone, Debug)]
pub struct Edge<'a, T: Clone + fmt::Display>(&'a Relation, &'a Relation, T);
#[derive(Clone, Debug)]
pub struct VisitedRelation<'a, V>(&'a Relation, V);

#[derive(Clone, Debug)]
pub struct DotVisitor;

#[derive(Clone, Debug)]
pub struct FieldDataTypes(Vec<(Field, Expr)>);

impl fmt::Display for FieldDataTypes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            self.0
                .iter()
                .map(|(field, expr)| {
                    let formated = format!(
                        "{}",
                        shorten_string(&if let Some(c) = field.constraint() {
                            format!(
                                "{} = {} ∈ {} {}",
                                dot::escape_html(&field.name().to_string()),
                                dot::escape_html(&expr.to_string()),
                                dot::escape_html(&field.data_type().to_string()),
                                c
                            )
                        } else {
                            format!(
                                "{} = {} ∈ {}",
                                dot::escape_html(&field.name().to_string()),
                                dot::escape_html(&expr.to_string()),
                                dot::escape_html(&field.data_type().to_string()),
                            )
                        })
                    );
                    formated
                })
                .join("<br/>")
        )
    }
}

impl<'a> Visitor<'a, FieldDataTypes> for DotVisitor {
    fn table(&self, table: &'a super::Table) -> FieldDataTypes {
        FieldDataTypes(
            table
                .schema()
                .fields()
                .iter()
                .map(|field| (field.clone(), Expr::col(field.name())))
                .collect(),
        )
    }

    fn map(&self, map: &'a super::Map, _input: FieldDataTypes) -> FieldDataTypes {
        FieldDataTypes(
            map.schema()
                .fields()
                .iter()
                .zip(&map.projection)
                .map(|(field, expr)| (field.clone(), expr.clone()))
                .collect(),
        )
    }

    fn reduce(&self, reduce: &'a super::Reduce, _input: FieldDataTypes) -> FieldDataTypes {
        FieldDataTypes(
            reduce
                .schema()
                .fields()
                .iter()
                .zip(&reduce.aggregate)
                .map(|(field, aggregate)| (field.clone(), aggregate.clone().into()))
                .collect(),
        )
    }

    fn join(
        &self,
        join: &'a super::Join,
        _left: FieldDataTypes,
        _right: FieldDataTypes,
    ) -> FieldDataTypes {
        FieldDataTypes(
            join.left()
                .schema()
                .iter()
                .map(|f| vec![Join::left_name(), f.name()])
                .chain(
                    join.right()
                        .schema()
                        .iter()
                        .map(|f| vec![Join::right_name(), f.name()]),
                )
                .zip(join.schema().iter())
                .map(|(p, field)| {
                    (
                        field.clone(),
                        Expr::qcol(p[0].to_string(), p[1].to_string()),
                    )
                })
                .collect(),
        )
    }

    fn set(
        &self,
        set: &'a super::Set,
        _left: FieldDataTypes,
        _right: FieldDataTypes,
    ) -> FieldDataTypes {
        FieldDataTypes(
            set.schema()
                .fields()
                .iter()
                .map(|field| (field.clone(), Expr::col(field.name())))
                .collect(),
        )
    }

    fn values(&self, values: &'a super::Values) -> FieldDataTypes {
        FieldDataTypes(
            values
                .schema()
                .fields()
                .iter()
                .map(|field| (field.clone(), Expr::col(field.name())))
                .collect(),
        )
    }
}

/// Clip a str
fn truncate(s: &str, max_chars: usize) -> &str {
    match s.char_indices().nth(max_chars) {
        None => s,
        Some((idx, _)) => &s[..idx],
    }
}

fn shorten_string(s: &str) -> Cow<str> {
    const MAX_STR_LEN: usize = 128;
    if s.len() > MAX_STR_LEN {
        format!("{}...", truncate(s, MAX_STR_LEN - 3)).into()
    } else {
        s.into()
    }
}

impl<'a, T: Clone + fmt::Display, V: Visitor<'a, T>> dot::Labeller<'a, Node<'a, T>, Edge<'a, T>>
    for VisitedRelation<'a, V>
{
    fn graph_id(&'a self) -> dot::Id<'a> {
        dot::Id::new(namer::name_from_content("graph", self.0)).unwrap()
    }

    fn node_id(&'a self, node: &Node<'a, T>) -> dot::Id<'a> {
        dot::Id::new(namer::name_from_content("graph", node.0)).unwrap()
    }

    fn node_label(&'a self, node: &Node<'a, T>) -> dot::LabelText<'a> {
        dot::LabelText::html(match &node.0 {
            Relation::Table(table) => format!(
                "<b>{} size ∈ {}</b><br/>{}",
                table.name().to_uppercase(),
                table.size(),
                &node.1
            ),
            Relation::Map(map) => {
                let filter = (map.filter.as_ref()).map_or(format!(""), |f| {
                    format!("<br/>WHERE {}", dot::escape_html(&f.to_string()))
                });
                let order_by = if map.order_by.is_empty() {
                    "".to_string()
                } else {
                    format!(
                        "<br/>ORDER BY ({})",
                        dot::escape_html(
                            &map.order_by
                                .iter()
                                .map(|o| format!(
                                    "{} {}",
                                    o.expr,
                                    if o.asc { "ASC" } else { "DESC" }
                                ))
                                .join(", ")
                        )
                    )
                };
                let limit = match map.limit {
                    Some(limit) => format!(
                        "<br/>LIMIT {}",
                        dot::escape_html(limit.to_string().as_str())
                    ),
                    _ => "".to_string(),
                };
                let offset = match map.offset {
                    Some(offset) => format!(
                        "<br/>OFFSET {}",
                        dot::escape_html(offset.to_string().as_str())
                    ),
                    _ => "".to_string(),
                };
                format!(
                    "<b>{} size ∈ {}</b><br/>{}{filter}{order_by}{limit}{offset}",
                    map.name().to_uppercase(),
                    map.size(),
                    &node.1
                )
            }
            Relation::Reduce(reduce) => {
                let group_by = if reduce.group_by.is_empty() {
                    "".to_string()
                } else {
                    format!(
                        "<br/>GROUP BY ({})",
                        dot::escape_html(&reduce.group_by.iter().map(|e| e.to_string()).join(", "))
                    )
                };
                format!(
                    "<b>{} size ∈ {}</b><br/>{}{group_by}",
                    reduce.name().to_uppercase(),
                    reduce.size(),
                    &node.1
                )
            }
            Relation::Join(join) => {
                let operator = match &join.operator {
                    JoinOperator::Inner(expr)
                    | JoinOperator::LeftOuter(expr)
                    | JoinOperator::RightOuter(expr)
                    | JoinOperator::FullOuter(expr) => {
                        format!(
                            "<br/>{} ON {}",
                            join.operator.to_string(),
                            dot::escape_html(&expr.to_string())
                        )
                    }
                    JoinOperator::Cross => format!("<br/>{}", join.operator.to_string()),
                };
                format!(
                    "<b>{} size ∈ {}</b><br/>{}{}",
                    join.name().to_uppercase(),
                    join.size(),
                    &node.1,
                    operator,
                )
            }
            Relation::Set(set) => format!(
                "<b>{} size ∈ {}</b><br/>{}",
                set.name().to_uppercase(),
                set.size(),
                &node.1
            ),
            Relation::Values(values) => format!(
                "<b>{} size ∈ {}</b><br/>[{}]",
                values.name().to_uppercase(),
                values.size(),
                values.values.iter().map(|v| v.to_string()).join(", "),
            ),
        })
    }

    fn node_color(&'a self, node: &Node<'a, T>) -> Option<dot::LabelText<'a>> {
        Some(dot::LabelText::label(match &node.0 {
            Relation::Table(_) => colors::MEDIUM_RED,
            Relation::Map(_) => colors::LIGHT_GREEN,
            Relation::Reduce(_) => colors::DARK_GREEN,
            Relation::Join(_) => colors::LIGHT_RED,
            Relation::Set(_) => colors::LIGHTER_GREEN,
            Relation::Values(_) => colors::MEDIUM_GREEN,
        }))
    }
}

impl<'a, T: Clone + fmt::Display, V: Visitor<'a, T> + Clone>
    dot::GraphWalk<'a, Node<'a, T>, Edge<'a, T>> for VisitedRelation<'a, V>
{
    fn nodes(&'a self) -> dot::Nodes<'a, Node<'a, T>> {
        self.0
            .iter_with(self.1.clone())
            .collect_vec()
            .into_iter()
            .rev()
            .map(|(relation, t)| Node(relation, t))
            .collect()
    }

    fn edges(&'a self) -> dot::Edges<'a, Edge<'a, T>> {
        self.0
            .iter_with(self.1.clone())
            .flat_map(|(relation, t)| match relation {
                Relation::Table(_) => vec![],
                Relation::Map(map) => vec![Edge(relation, &map.input, t)],
                Relation::Reduce(reduce) => vec![Edge(relation, &reduce.input, t)],
                Relation::Join(join) => vec![
                    Edge(relation, &join.left, t.clone()),
                    Edge(relation, &join.right, t),
                ],
                Relation::Set(set) => vec![
                    Edge(relation, &set.left, t.clone()),
                    Edge(relation, &set.right, t),
                ],
                Relation::Values(_) => vec![],
            })
            .collect()
    }

    fn source(&'a self, edge: &Edge<'a, T>) -> Node<'a, T> {
        Node(edge.0, edge.2.clone())
    }

    fn target(&'a self, edge: &Edge<'a, T>) -> Node<'a, T> {
        Node(edge.1, edge.2.clone())
    }
}

impl Relation {
    /// Render the Relation to dot
    pub fn dot<W: io::Write>(&self, w: &mut W, opts: &[&str]) -> io::Result<()> {
        display::dot::render(&VisitedRelation(self, DotVisitor), w, opts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        builder::{Ready, With},
        data_type::{DataType, Value},
        display::Dot,
        expr::Expr,
        relation::{schema::Schema, Relation},
    };

    #[test]
    fn test_dot() {
        namer::reset();
        let schema: Schema = vec![
            ("a", DataType::float()),
            ("b", DataType::float_interval(-2., 2.)),
            ("c", DataType::float()),
            ("d", DataType::float_interval(0., 1.)),
        ]
        .into_iter()
        .collect();
        let table: Relation = Relation::table()
            .name("table")
            .schema(schema.clone())
            .size(1000)
            .build();
        println!("table = {}", table);
        let map: Relation = Relation::map()
            .name("map_1")
            .with(("exp_a", Expr::exp(Expr::col("a"))))
            .input(table.clone())
            .with(("alias", Expr::col("b") + Expr::col("d")))
            .build();
        println!("map = {}", map);
        println!("map[0] = {}", map[0]);
        let join: Relation = Relation::join()
            .name("join")
            .cross()
            .left(table.clone().with_name("left_relation".to_string()))
            .right(map.clone().with_name("right_relation".to_string()))
            .build();
        println!("join = {}", join);
        let map_2: Relation = Relation::map()
            .name("map_2")
            .with(("a", Expr::exp(Expr::col(join[4].name()))))
            .input(join.clone())
            .with(Expr::col(join[0].name()) + Expr::col(join[1].name()))
            .build();
        println!("map_2 = {}", map_2);
        let join_2: Relation = Relation::join()
            .name("join_2")
            .cross()
            .left(join.clone())
            .right(map_2.clone())
            .build();
        println!("join_2 = {}", join_2);
        join_2.display_dot().unwrap();
    }

    #[test]
    fn test_escape_html() {
        namer::reset();
        let schema: Schema = vec![
            ("a", DataType::float()),
            ("b", DataType::text_values(&["A&B".into(), "C>D".into()])),
        ]
        .into_iter()
        .collect();
        let table: Relation = Relation::table()
            .name("table")
            .schema(schema.clone())
            .size(1000)
            .build();
        println!("table = {}", table);
        let map: Relation = Relation::map()
            .name("map_1")
            .with((
                "case_expr",
                Expr::case(
                    Expr::gt(Expr::col("a"), Expr::val(0)),
                    Expr::val(1),
                    Expr::val(-1),
                ),
            ))
            .input(table.clone())
            .build();
        map.display_dot().unwrap();
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
            .cross()
            .left(left)
            .right(right)
            .build();
        join.display_dot().unwrap();
    }

    #[test]
    fn test_display_reduce() {
        namer::reset();
        let schema: Schema = vec![
            (
                "a",
                DataType::integer_interval(1, 5),
                Some(crate::relation::Constraint::PrimaryKey),
            ),
            ("b", DataType::float_interval(-2., 2.), None),
        ]
        .into_iter()
        .collect();
        let table: Relation = Relation::table()
            .name("table")
            .schema(schema.clone())
            .size(1000)
            .build();
        let reduce: Relation = Relation::reduce()
            .name("reduce")
            .input(table)
            .with_group_by_column("a")
            .with(Expr::sum(Expr::col("b")))
            .build();
        reduce.display_dot().unwrap();
    }

    #[test]
    fn test_display_values() {
        let values: Relation = Relation::values().name("Float").values(vec![5.]).build();
        values.display_dot().unwrap();

        let values: Relation = Relation::values()
            .name("List_of_floats")
            .values(vec![Value::float(10.), Value::float(4.0)])
            .build();
        values.display_dot().unwrap();
    }
}
