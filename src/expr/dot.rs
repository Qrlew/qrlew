//! Plot the dot graph of an expression to debug

use std::{fmt, fs::File, io, process::Command, string};

use super::{aggregate, function, Column, Error, Expr, Value, Visitor};
use crate::{
    builder::{WithContext, WithoutContext},
    data_type::{DataType, DataTyped},
    display::{self, colors},
    namer,
    visitor::Acceptor,
};

impl From<string::FromUtf8Error> for Error {
    fn from(err: string::FromUtf8Error) -> Self {
        Error::Other(err.to_string())
    }
}

#[derive(Clone, Debug)]
pub struct Node<'a, T: Clone + fmt::Display>(&'a Expr, T);
#[derive(Clone, Debug)]
pub struct Edge<'a, T: Clone + fmt::Display>(&'a Expr, &'a Expr, T);
#[derive(Clone, Debug)]
pub struct VisitedExpr<'a, V>(&'a Expr, V);

#[derive(Clone, Debug)]
pub struct DotVisitor<'a>(pub &'a DataType);

impl<'a> Visitor<'a, DataType> for DotVisitor<'a> {
    fn column(&self, column: &'a Column) -> DataType {
        self.0[column.clone()].clone()
    }

    fn value(&self, value: &'a Value) -> DataType {
        value.data_type()
    }

    fn function(&self, function: &'a function::Function, arguments: Vec<DataType>) -> DataType {
        function.clone().super_image(&arguments).unwrap()
    }

    fn aggregate(
        &self,
        aggregate: &'a aggregate::Aggregate,
        _distinct: &'a bool,
        argument: DataType,
    ) -> DataType {
        aggregate.clone().super_image(&argument).unwrap()
    }

    fn structured(&self, fields: Vec<(super::identifier::Identifier, DataType)>) -> DataType {
        let fields: Vec<(String, DataType)> = fields
            .into_iter()
            .map(|(i, t)| (i.split_last().unwrap().0, t))
            .collect();
        DataType::structured(fields)
    }
}

#[derive(Clone, Debug)]
pub struct DotValueVisitor<'a>(pub &'a Value);

impl<'a> Visitor<'a, Value> for DotValueVisitor<'a> {
    fn column(&self, column: &'a Column) -> Value {
        self.0[column.clone()].clone()
    }

    fn value(&self, value: &'a Value) -> Value {
        value.clone()
    }

    fn function(&self, function: &'a function::Function, arguments: Vec<Value>) -> Value {
        function.clone().value(&arguments).unwrap()
    }

    fn aggregate(
        &self,
        aggregate: &'a aggregate::Aggregate,
        _distinct: &'a bool,
        argument: Value,
    ) -> Value {
        aggregate.clone().value(&argument).unwrap()
    }

    fn structured(&self, fields: Vec<(super::identifier::Identifier, Value)>) -> Value {
        let fields: Vec<(String, Value)> = fields
            .into_iter()
            .map(|(i, v)| (i.split_last().unwrap().0, v))
            .collect();
        Value::structured(fields)
    }
}

impl<'a, T: Clone + fmt::Display, V: Visitor<'a, T>> dot::Labeller<'a, Node<'a, T>, Edge<'a, T>>
    for VisitedExpr<'a, V>
{
    fn graph_id(&'a self) -> dot::Id<'a> {
        dot::Id::new(namer::name_from_content("graph", self.0)).unwrap()
    }

    fn node_id(&'a self, node: &Node<'a, T>) -> dot::Id<'a> {
        dot::Id::new(namer::name_from_content("graph", node.0)).unwrap()
    }

    fn node_label(&'a self, node: &Node<'a, T>) -> dot::LabelText<'a> {
        dot::LabelText::html(match &node.0 {
            Expr::Column(col) => format!(
                "<b>{}</b><br/>{}",
                dot::escape_html(&col.to_string()),
                &node.1
            ),
            Expr::Value(val) => format!(
                "<b>{}</b><br/>{}",
                dot::escape_html(&val.to_string()),
                &node.1
            ),
            Expr::Function(fun) => {
                format!(
                    "<b>{}</b><br/>{}",
                    dot::escape_html(&fun.function.to_string()),
                    &node.1
                )
            }
            Expr::Aggregate(agg) => format!(
                "<b>{}</b><br/>{}",
                dot::escape_html(&agg.aggregate.to_string()),
                &node.1
            ),
            Expr::Struct(s) => format!(
                "<b>{}</b><br/>{}",
                dot::escape_html(&s.to_string()),
                &node.1
            ),
        })
    }

    fn node_color(&'a self, node: &Node<'a, T>) -> Option<dot::LabelText<'a>> {
        Some(dot::LabelText::label(match &node.0 {
            Expr::Column(_) => colors::MEDIUM_RED,
            Expr::Value(_) => colors::LIGHT_RED,
            Expr::Function(_) => colors::LIGHT_GREEN,
            Expr::Aggregate(_) => colors::DARK_GREEN,
            Expr::Struct(_) => colors::LIGHTER_GREEN,
        }))
    }
}

impl<'a, T: Clone + fmt::Display, V: Visitor<'a, T> + Clone>
    dot::GraphWalk<'a, Node<'a, T>, Edge<'a, T>> for VisitedExpr<'a, V>
{
    fn nodes(&'a self) -> dot::Nodes<'a, Node<'a, T>> {
        self.0
            .iter_with(self.1.clone())
            .map(|(expr, t)| Node(expr, t))
            .collect()
    }

    fn edges(&'a self) -> dot::Edges<'a, Edge<'a, T>> {
        self.0
            .iter_with(self.1.clone())
            .flat_map(|(expr, t)| match expr {
                Expr::Column(_) | Expr::Value(_) => vec![],
                Expr::Function(fun) => fun
                    .arguments
                    .iter()
                    .map(|arg| Edge(expr, arg, t.clone()))
                    .collect(),
                Expr::Aggregate(agg) => vec![Edge(expr, &agg.argument, t)],
                Expr::Struct(s) => s
                    .fields
                    .iter()
                    .map(|(_i, e)| Edge(expr, e, t.clone()))
                    .collect(),
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

impl Expr {
    /// Render the Expr to dot
    pub fn dot<W: io::Write>(
        &self,
        data_type: DataType,
        w: &mut W,
        opts: &[&str],
    ) -> io::Result<()> {
        display::dot::render(&VisitedExpr(self, DotVisitor(&data_type)), w, opts)
    }

    /// Render the Expr to dot
    pub fn dot_value<W: io::Write>(&self, val: Value, w: &mut W, opts: &[&str]) -> io::Result<()> {
        display::dot::render(&VisitedExpr(self, DotValueVisitor(&val)), w, opts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        builder::{Ready, With},
        data_type::DataType,
        display::Dot,
        relation::{schema::Schema, Relation},
    };
    use std::sync::Arc;

    #[test]
    fn test_dot() {
        // Create an expr
        let a = Expr::col("a");
        let b = Expr::col("b");
        let x = Expr::col("x");
        let expr = Expr::exp(Expr::sin(Expr::plus(Expr::multiply(a, x), b)));
        expr.with(DataType::Any).display_dot().unwrap();
    }

    #[test]
    fn test_dot_dsl() {
        let rel: Arc<Relation> = Arc::new(
            Relation::table()
                .schema(
                    Schema::builder()
                        .with(("a", DataType::float_range(1.0..=1.1)))
                        .with(("b", DataType::float_values([0.1, 1.0, 5.0, -1.0, -5.0])))
                        .with(("c", DataType::float_range(0.0..=5.0)))
                        .with(("d", DataType::float_values([0.0, 1.0, 2.0, -1.0])))
                        .with(("x", DataType::float_range(0.0..=2.0)))
                        .with(("y", DataType::float_range(0.0..=5.0)))
                        .with(("z", DataType::float_range(9.0..=11.)))
                        .with(("t", DataType::float_range(0.9..=1.1)))
                        .build(),
                )
                .build(),
        );
        // Create an expr
        expr!(exp(a * b) + cos(1. * z) * x - 0.2 * (y + 3.) + b + t * sin(c + 4. * (d + 5. + x)))
            .with(rel.data_type())
            .display_dot()
            .unwrap();
    }

    #[test]
    fn test_dot_dsl_squared() {
        let rel: Arc<Relation> = Arc::new(
            Relation::table()
                .schema(
                    Schema::builder()
                        .with(("a", DataType::float_range(1.0..=1.1)))
                        .with(("b", DataType::float_values([0.1, 1.0, 5.0, -1.0, -5.0])))
                        .with(("c", DataType::float_range(0.0..=5.0)))
                        .with(("d", DataType::float_values([0.0, 1.0, 2.0, -1.0])))
                        .with(("x", DataType::float_range(0.0..=2.0)))
                        .with(("y", DataType::float_range(0.0..=5.0)))
                        .with(("z", DataType::float_range(9.0..=11.)))
                        .with(("t", DataType::float_range(0.9..=1.1)))
                        .build(),
                )
                .build(),
        );
        // Create an expr
        let e = expr!(
            exp(a * b) + cos(1. * z) * x - 0.2 * (y + 3.) + b + t * sin(c + 4. * (d + 5. + x))
        );
        let e = Expr::multiply(e.clone(), e);
        e.with(rel.data_type()).display_dot().unwrap();
    }

    #[ignore]
    #[test]
    fn test_dot_distributivity_dsl() {
        let val = Value::structured([
            ("a", Value::float(1.)),
            ("b", Value::float(2.)),
            ("c", Value::float(3.)),
            ("d", Value::integer(4)),
        ]);
        &expr! { a*b+d }.with(val.clone()).display_dot().unwrap();
        &expr! { d+a*b }.with(val.clone()).display_dot().unwrap();
        &expr! { (a*b+d) }.with(val).display_dot().unwrap();
    }

    #[ignore]
    #[test]
    fn test_dot_plus_minus_dsl() {
        let val = Value::structured([
            ("a", Value::float(1.)),
            ("b", Value::float(2.)),
            ("c", Value::float(3.)),
            ("d", Value::integer(4)),
        ]);
        expr! { a+b-c+d }.with(val).display_dot().unwrap();
    }

    #[ignore]
    #[test]
    fn test_dot_simple_value_dsl() {
        let val = Value::structured([
            ("a", Value::float(0.1)),
            ("b", Value::float(0.1)),
            ("z", Value::float(0.1)),
            ("d", Value::integer(0)),
            ("t", Value::float(0.1)),
            ("c", Value::float(0.0)),
            ("x", Value::float(0.0)),
        ]);
        expr! { exp(a*b + cos(2*z)*d - 2*z + t*sin(c+3*x)) }
            .with(val)
            .display_dot()
            .unwrap();
    }

    #[ignore]
    #[test]
    fn test_dot_value_dsl() {
        let val = Value::structured([
            ("a", Value::float(0.1)),
            ("b", Value::float(0.1)),
            ("c", Value::float(0.1)),
            ("d", Value::float(0.1)),
            ("x", Value::float(0.1)),
            ("y", Value::integer(0)),
            ("z", Value::float(0.1)),
            ("t", Value::float(0.0)),
        ]);
        // Create an expr
        expr!(exp(a * b) + cos(1. * z) * x - 0.2 * (y + 3.) + b + t * sin(c + 4. * (d + 5. + x)))
            .with(val)
            .display_dot()
            .unwrap();
    }

    #[ignore]
    #[test]
    fn test_dot_aggregate_dsl() {
        let data_types = DataType::structured([
            ("a", DataType::list(DataType::Any, 1, 10)),
            (
                "b",
                DataType::list(DataType::integer_interval(2, 18), 1, 10),
            ),
            ("c", DataType::list(DataType::float_interval(5., 7.), 1, 10)),
            ("d", DataType::float_interval(5., 7.)),
        ]);
        println!("data_types = {data_types}");
        let x = expr!((exp(d) + 2 + sum(b) * count(a) + sum(c)) / (1 + count(a)));
        println!("x = {x}");
        for (x, t) in x.iter_with(DotVisitor(&data_types)) {
            println!("({x}, {t})");
        }
        println!("END ITER");
        // Create an expr
        x.with(data_types).display_dot().unwrap();
    }

    #[ignore]
    #[test]
    fn test_dot_aggregate_any_dsl() {
        let data_types = DataType::structured([
            ("a", DataType::Any),
            (
                "b",
                DataType::list(DataType::integer_interval(2, 18), 1, 10),
            ),
            ("c", DataType::Any),
            ("d", DataType::Any),
        ]);
        // Create an expr
        expr!(sum(sum(a) + count(b)) * count(c))
            .with(data_types)
            .display_dot()
            .unwrap();
    }

    #[test]
    fn test_dot_escape_html() {
        let data_types = DataType::structured([("a", DataType::integer_interval(1, 10))]);

        let my_expr = expr!(lt_eq(a, 5));
        my_expr.with(data_types.clone()).display_dot().unwrap();
        assert_eq!(my_expr.to_string(), "(a <= 5)".to_string());

        let my_expr = expr!(gt(a, 5));
        my_expr.with(data_types.clone()).display_dot().unwrap();
        assert_eq!(my_expr.to_string(), "(a > 5)".to_string());

        let my_expr = expr!(modulo(a, 2));
        my_expr.with(data_types).display_dot().unwrap();
        assert_eq!(my_expr.to_string(), "(a % 2)".to_string());
    }

    #[ignore]
    #[test]
    fn test_max() {
        let data_types = DataType::structured([("a", DataType::float_interval(0., 4.))]);

        let my_expr = expr!((a + 1 + abs(a - 1)) / 2);
        my_expr.with(data_types.clone()).display_dot().unwrap();

        let my_expr = expr!(1 - gt(a, 1) * (1 - a));
        my_expr.with(data_types).display_dot().unwrap();
    }

    #[ignore]
    #[test]
    fn test_dot_struct_dsl() {
        let rel: Arc<Relation> = Arc::new(
            Relation::table()
                .schema(
                    Schema::builder()
                        .with(("a", DataType::float_range(1.0..=1.1)))
                        .with(("b", DataType::float_values([0.1, 1.0, 5.0, -1.0, -5.0])))
                        .with(("c", DataType::float_range(0.0..=5.0)))
                        .with(("d", DataType::float_values([0.0, 1.0, 2.0, -1.0])))
                        .with(("x", DataType::float_range(0.0..=2.0)))
                        .with(("y", DataType::float_range(0.0..=5.0)))
                        .with(("z", DataType::float_range(9.0..=11.)))
                        .with(("t", DataType::float_range(0.9..=1.1)))
                        .build(),
                )
                .build(),
        );
        // Create an expr
        Expr::structured([
            ("a", Arc::new(expr!(exp(a * b)))),
            (
                "b",
                Arc::new(expr!(
                    cos(1. * z) * x - 0.2 * (y + 3.) + b + t * sin(c + 4. * (d + 5. + x))
                )),
            ),
        ])
        .with(rel.data_type())
        .display_dot()
        .unwrap();
    }

    #[ignore]
    #[test]
    fn test_dot_case() {
        let data_types = DataType::structured([(
            "a",
            DataType::list(DataType::integer_interval(2, 18), 1, 10),
        )]);
        // Create an expr
        expr!(case(eq(a, 5), 5, a))
            .with(data_types)
            .display_dot()
            .unwrap();
    }
}
