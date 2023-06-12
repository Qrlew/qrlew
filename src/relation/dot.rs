use super::{Error, Field, JoinConstraint, JoinOperator, Relation, Variant as _, Visitor};
use crate::{
    data_type::DataTyped,
    display::{self, colors},
    expr::Expr,
    namer,
    visitor::Acceptor,
};
use itertools::Itertools;
use std::{borrow::Cow, fmt, fs::File, io, process::Command, str, string};

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
                .map(|(field, expr)| format!(
                    "{} = {} ∈ {}",
                    field.name(),
                    dot::escape_html(&expr.to_string()),
                    field.data_type()
                ))
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
                .map(|(field, expr)| (field.clone(), expr.clone()))
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
            join.field_inputs()
                .map(|(f, i)| {
                    (
                        join.field_from_qualified_name(&f).unwrap().clone(),
                        Expr::from(i),
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
}

#[allow(dead_code)]
fn shorten_string(s: &str) -> Cow<str> {
    const MAX_STR_LEN: usize = 16;
    if s.len() > MAX_STR_LEN {
        let mut ms: String = s.into();
        ms.truncate(MAX_STR_LEN - 3);
        format!("{}...", ms).into()
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
                format!(
                    "<b>{} size ∈ {}</b><br/>{}{filter}{order_by}",
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
                    JoinOperator::Inner(JoinConstraint::On(expr))
                    | JoinOperator::LeftOuter(JoinConstraint::On(expr))
                    | JoinOperator::RightOuter(JoinConstraint::On(expr))
                    | JoinOperator::FullOuter(JoinConstraint::On(expr)) => {
                        format!("<br/>{} ON {}", join.operator.to_string(), expr)
                    }
                    JoinOperator::Inner(JoinConstraint::Using(identifiers))
                    | JoinOperator::LeftOuter(JoinConstraint::Using(identifiers))
                    | JoinOperator::RightOuter(JoinConstraint::Using(identifiers))
                    | JoinOperator::FullOuter(JoinConstraint::Using(identifiers)) => format!(
                        "<br/>{} USING ({})",
                        join.operator.to_string(),
                        identifiers.iter().join(", ")
                    ),
                    JoinOperator::Inner(JoinConstraint::Natural)
                    | JoinOperator::LeftOuter(JoinConstraint::Natural)
                    | JoinOperator::RightOuter(JoinConstraint::Natural)
                    | JoinOperator::FullOuter(JoinConstraint::Natural) => {
                        format!("<br/>NATURAL {}", join.operator.to_string())
                    }
                    JoinOperator::Inner(JoinConstraint::None)
                    | JoinOperator::LeftOuter(JoinConstraint::None)
                    | JoinOperator::RightOuter(JoinConstraint::None)
                    | JoinOperator::FullOuter(JoinConstraint::None)
                    | JoinOperator::Cross => format!("<br/>{}", join.operator.to_string()),
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
        })
    }

    fn node_color(&'a self, node: &Node<'a, T>) -> Option<dot::LabelText<'a>> {
        Some(dot::LabelText::label(match &node.0 {
            Relation::Table(_) => colors::MEDIUM_RED,
            Relation::Map(_) => colors::LIGHT_GREEN,
            Relation::Reduce(_) => colors::DARK_GREEN,
            Relation::Join(_) => colors::LIGHT_RED,
            Relation::Set(_) => colors::LIGHTER_GREEN,
        }))
    }
}

impl<'a, T: Clone + fmt::Display, V: Visitor<'a, T> + Clone>
    dot::GraphWalk<'a, Node<'a, T>, Edge<'a, T>> for VisitedRelation<'a, V>
{
    fn nodes(&'a self) -> dot::Nodes<'a, Node<'a, T>> {
        self.0
            .iter_with(self.1.clone())
            .map(|(relation, t)| Node(relation, t))
            .collect()
    }

    fn edges(&'a self) -> dot::Edges<'a, Edge<'a, T>> {
        self.0
            .iter_with(self.1.clone())
            .flat_map(|(relation, t)| match relation {
                Relation::Table(_) => Vec::new(),
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
        data_type::DataType,
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
        println!("table[a] = {}", table[&"a".into()]);
        let map: Relation = Relation::map()
            .name("map_1")
            .with(("exp_a", Expr::exp(Expr::col("a"))))
            .input(table.clone())
            .with(("alias", Expr::col("b") + Expr::col("d")))
            .build();
        println!("map = {}", map);
        println!("map[0] = {}", map[0]);
        println!("map[table.a] = {}", map[&["table", "a"].into()]);
        let join: Relation = Relation::join()
            .name("join")
            .cross()
            .left(table.clone())
            .right(map.clone())
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
        join_2.display_dot();
    }

    #[test]
    fn test_escape_html() {
        namer::reset();
        let schema: Schema = vec![("a", DataType::float())].into_iter().collect();
        let table: Relation = Relation::table()
            .name("table")
            .schema(schema.clone())
            .size(1000)
            .build();
        println!("table = {}", table);
        println!("table[a] = {}", table[&"a".into()]);
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
        map.display_dot();
    }

    #[ignore]
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
            //.using("a")
            //.on(Expr::eq(Expr::qcol("left", "b"), Expr::qcol("right", "b")))
            .left(left)
            .right(right)
            .build();
        join.display_dot();
    }

    #[test]
    fn test_display_reduce() {
        namer::reset();
        let schema: Schema = vec![
            ("a", DataType::integer_interval(1, 5)),
            ("b", DataType::float_interval(-2., 2.)),
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
        reduce.display_dot();
    }
}
