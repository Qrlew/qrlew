//! Plot the dot graph of an expression to debug

use std::{fmt, fs::File, process::Command, string};

use super::{aggregate, function, Column, Error, Expr, Result, Value, Visitor};
use crate::{
    data_type::{DataType, DataTyped},
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
        function.super_image(&arguments).unwrap()
    }

    fn aggregate(&self, aggregate: &'a aggregate::Aggregate, argument: DataType) -> DataType {
        aggregate.super_image(&argument).unwrap()
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
        function.value(&arguments).unwrap()
    }

    fn aggregate(&self, aggregate: &'a aggregate::Aggregate, argument: Value) -> Value {
        aggregate.value(&argument).unwrap()
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
            Expr::Column(col) => format!("<b>{}</b><br/>{}", col, &node.1),
            Expr::Value(val) => format!("<b>{}</b><br/>{}", val, &node.1),
            Expr::Function(fun) => format!("<b>{}</b><br/>{}", fun.function, &node.1),
            Expr::Aggregate(agg) => format!("<b>{}</b><br/>{}", agg.aggregate, &node.1),
            Expr::Struct(s) => format!("<b>{}</b><br/>{}", s, &node.1),
        })
    }

    fn node_shape(&'a self, node: &Node<'a, T>) -> Option<dot::LabelText<'a>> {
        Some(dot::LabelText::label(match &node.0 {
            Expr::Column(_) => format!("box"),
            Expr::Value(_) => format!("box"),
            Expr::Function(_) => format!("box"),
            Expr::Aggregate(_) => format!("box"),
            Expr::Struct(_) => format!("box"),
        }))
    }

    fn node_style(&'a self, _node: &Node<'a, T>) -> dot::Style {
        dot::Style::Filled
    }

    fn node_color(&'a self, node: &Node<'a, T>) -> Option<dot::LabelText<'a>> {
        Some(dot::LabelText::label(match &node.0 {
            Expr::Column(_) => format!("aquamarine3"),
            Expr::Value(_) => format!("goldenrod3"),
            Expr::Function(_) => format!("cornsilk1"),
            Expr::Aggregate(_) => format!("deeppink"),
            Expr::Struct(_) => format!("darkslategray2"),
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
                Expr::Column(_) | Expr::Value(_) => Vec::new(),
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
    pub fn dot(&self, data_type: DataType) -> Result<String> {
        let mut buffer: Vec<u8> = Vec::new();
        dot::render(&VisitedExpr(self, DotVisitor(&data_type)), &mut buffer).unwrap();
        Ok(String::from_utf8(buffer)?)
    }
}

/// A simple MacOS specific function to display `Expr`s as graphs
pub fn display(expr: &Expr, data_type: DataType) {
    let name = namer::name_from_content("expr", expr);
    let dot_path = &format!("/tmp/{name}.dot");
    let pdf_path = &format!("/tmp/{name}.pdf");
    let mut output = File::create(dot_path).unwrap();
    dot::render(&VisitedExpr(expr, DotVisitor(&data_type)), &mut output).unwrap();
    Command::new("dot")
        .arg(dot_path)
        .arg("-Tpdf")
        .arg("-o")
        .arg(pdf_path)
        .output()
        .expect("Error: you need graphviz installed (and dot on the PATH)");
    Command::new("open")
        .arg(pdf_path)
        .output()
        .expect("Error: this works on MacOS only");
}

/// A simple MacOS specific function to display `Expr`s as graphs
pub fn display_value(expr: &Expr, val: Value) {
    let name = namer::name_from_content("expr", &(expr, &val));
    let dot_path = &format!("/tmp/{name}.dot");
    let pdf_path = &format!("/tmp/{name}.pdf");
    let mut output = File::create(dot_path).unwrap();
    dot::render(&VisitedExpr(expr, DotValueVisitor(&val)), &mut output).unwrap();
    Command::new("dot")
        .arg(dot_path)
        .arg("-Tpdf")
        .arg("-o")
        .arg(pdf_path)
        .output()
        .expect("Error: you need graphviz installed (and dot on the PATH)");
    Command::new("open")
        .arg(pdf_path)
        .output()
        .expect("Error: this works on MacOS only");
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        builder::{Ready, With},
        data_type::DataType,
        relation::{schema::Schema, Relation},
    };
    use std::rc::Rc;

    #[ignore]
    #[test]
    fn test_dot() {
        // Create an expr
        let a = Expr::col("a");
        let b = Expr::col("b");
        let x = Expr::col("x");
        let expr = Expr::exp(Expr::sin(Expr::plus(Expr::multiply(a, x), b)));
        display(&expr, DataType::Any);
    }

    #[ignore]
    #[test]
    fn test_dot_dsl() {
        let rel: Rc<Relation> = Rc::new(
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
        display(
            &expr!(
                exp(a * b) + cos(1. * z) * x - 0.2 * (y + 3.) + b + t * sin(c + 4. * (d + 5. + x))
            ),
            rel.data_type(),
        );
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
        display_value(&expr! { a*b+d }, val.clone());
        display_value(&expr! { d+a*b }, val.clone());
        display_value(&expr! { (a*b+d) }, val);
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
        display_value(&expr! { a+b-c+d }, val);
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
        display_value(&expr! { exp(a*b + cos(2*z)*d - 2*z + t*sin(c+3*x)) }, val);
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
        display_value(
            &expr!(
                exp(a * b) + cos(1. * z) * x - 0.2 * (y + 3.) + b + t * sin(c + 4. * (d + 5. + x))
            ),
            val,
        );
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
        display(&x, data_types);
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
        display(&expr!(sum(sum(a) + count(b)) * count(c)), data_types);
    }

    #[ignore]
    #[test]
    fn test_dot_struct_dsl() {
        let rel: Rc<Relation> = Rc::new(
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
        display(
            &Expr::structured([
                ("a", Rc::new(expr!(exp(a * b)))),
                (
                    "b",
                    Rc::new(expr!(
                        cos(1. * z) * x - 0.2 * (y + 3.) + b + t * sin(c + 4. * (d + 5. + x))
                    )),
                ),
            ]),
            rel.data_type(),
        );
    }
}
