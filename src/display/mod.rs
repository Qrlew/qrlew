//! # Methods to display representations of `Relation`s and `Expr`s
//!
//! This is experimental and little tested yet.
//!

pub mod colors;
pub mod dot;

use crate::{
    builder::{WithContext, WithoutContext},
    data_type::DataTyped,
    namer, DataType, Expr, Relation, Value,
    rewriting::{RelationWithRewritingRules, RelationWithRewritingRule},
};
use std::{
    fs::File,
    io::{Result, Write},
    process::Command,
    sync::Arc,
};

pub trait Dot {
    fn display_dot(&self) -> Result<()>;
}

const HTML_HEADER: &str = r##"<!DOCTYPE html>
<!-- Inspired from https://gist.github.com/magjac/a23d1f1405c2334f288a9cca4c0ef05b -->
<meta charset="utf-8">
"##;
const HTML_STYLE: &str = r##"<style>
#graph {
    height: 100%;
    width: 100%;
}
#graph svg {
    height: 100%;
    width: 100%;
}
</style>
"##;
const HTML_DARK_STYLE: &str = r##"<style>
#graph {
    background-color: #2b303a;
    height: 100%;
    width: 100%;
}
#graph svg {
    height: 100%;
    width: 100%;
}
</style>
"##;
const HTML_BODY: &str = r##"<body>
<script src="https://d3js.org/d3.v5.min.js"></script>
<script src="https://unpkg.com/@hpcc-js/wasm@0.3.11/dist/index.min.js"></script>
<script src="https://unpkg.com/d3-graphviz@3.0.5/build/d3-graphviz.js"></script>
<div id="graph" style="text-align: center; display: block; position: absolute;"></div>
<script>
d3.select("#graph").graphviz().engine("dot")
.renderDot(`"##;
const HTML_FOOTER: &str = r##"`);
</script>
"##;

impl Dot for Relation {
    fn display_dot(&self) -> Result<()> {
        let name = namer::name_from_content("relation", self);
        let mut output = File::create(format!("/tmp/{name}.html")).unwrap();
        output.write(HTML_HEADER.as_bytes())?;
        output.write(HTML_DARK_STYLE.as_bytes())?;
        output.write(HTML_BODY.as_bytes())?;
        self.dot(&mut output, &["dark"])?;
        output.write(HTML_FOOTER.as_bytes())?;
        #[cfg(feature = "graphviz_display")]
        Command::new("open")
            .arg(format!("/tmp/{name}.html"))
            .output()
            .expect("Error: this works on MacOS");
        Ok(())
    }
}

impl Dot for WithContext<&Expr, DataType> {
    fn display_dot(&self) -> Result<()> {
        let name = namer::name_from_content("expr", &self.object);
        let mut output = File::create(format!("/tmp/{name}.html")).unwrap();
        output.write(HTML_HEADER.as_bytes())?;
        output.write(HTML_STYLE.as_bytes())?;
        output.write(HTML_BODY.as_bytes())?;
        self.dot(self.context.clone(), &mut output, &[])?;
        output.write(HTML_FOOTER.as_bytes())?;
        #[cfg(feature = "graphviz_display")]
        Command::new("open")
            .arg(format!("/tmp/{name}.html"))
            .output()
            .expect("Error: this works on MacOS");
        Ok(())
    }
}

impl Dot for WithContext<&Expr, Value> {
    fn display_dot(&self) -> Result<()> {
        let name = namer::name_from_content("expr_value", &self.object);
        let mut output = File::create(format!("/tmp/{name}.html")).unwrap();
        output.write(HTML_HEADER.as_bytes())?;
        output.write(HTML_HEADER.as_bytes())?;
        output.write(HTML_STYLE.as_bytes())?;
        output.write(HTML_BODY.as_bytes())?;
        self.dot_value(self.context.clone(), &mut output, &[])?;
        output.write(HTML_FOOTER.as_bytes())?;
        #[cfg(feature = "graphviz_display")]
        Command::new("open")
            .arg(format!("/tmp/{name}.html"))
            .output()
            .expect("Error: this works on MacOS");
        Ok(())
    }
}

impl<'a> Dot for RelationWithRewritingRules<'a> {
    fn display_dot(&self) -> Result<()> {
        let name = namer::name_from_content("relation_with_rewriting_rules", self);
        let mut output = File::create(format!("/tmp/{name}.html")).unwrap();
        output.write(HTML_HEADER.as_bytes())?;
        output.write(HTML_DARK_STYLE.as_bytes())?;
        output.write(HTML_BODY.as_bytes())?;
        self.dot(&mut output, &["dark"])?;
        output.write(HTML_FOOTER.as_bytes())?;
        #[cfg(feature = "graphviz_display")]
        Command::new("open")
            .arg(format!("/tmp/{name}.html"))
            .output()
            .expect("Error: this works on MacOS");
        Ok(())
    }
}

impl<'a> Dot for RelationWithRewritingRule<'a> {
    fn display_dot(&self) -> Result<()> {
        let name = namer::name_from_content("relation_with_rewriting_rules", self);
        let mut output = File::create(format!("/tmp/{name}.html")).unwrap();
        output.write(HTML_HEADER.as_bytes())?;
        output.write(HTML_DARK_STYLE.as_bytes())?;
        output.write(HTML_BODY.as_bytes())?;
        self.dot(&mut output, &["dark"])?;
        output.write(HTML_FOOTER.as_bytes())?;
        #[cfg(feature = "graphviz_display")]
        Command::new("open")
            .arg(format!("/tmp/{name}.html"))
            .output()
            .expect("Error: this works on MacOS");
        Ok(())
    }
}

pub mod macos {
    use super::*;
    /// A simple MacOS specific function to display `Expr`s as graphs
    pub fn relation_display_dot(relation: &Relation) {
        let name = namer::name_from_content("relation", &relation);
        let mut output = File::create(format!("/tmp/{name}.dot")).unwrap();
        relation.dot(&mut output, &[]).unwrap();
        Command::new("dot")
            .arg(format!("/tmp/{name}.dot"))
            .arg("-Tpdf")
            .arg("-o")
            .arg(format!("/tmp/{name}.pdf"))
            .output()
            .expect("Error: you need graphviz installed (and dot on the PATH)");
        #[cfg(feature = "graphviz_display")]
        Command::new("open")
            .arg(format!("/tmp/{name}.pdf"))
            .output()
            .expect("Error: this works on MacOS only");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        builder::{Ready, With},
        data_type::DataType,
        expr::Expr,
        relation::{schema::Schema, Relation},
    };

    #[ignore]
    #[test]
    fn test_relation() {
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
        join_2.display_dot().unwrap();
    }

    #[ignore]
    #[test]
    fn test_expr() {
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
        let expr = expr!(
            exp(a * b) + cos(1. * z) * x - 0.2 * (y + 3.) + b + t * sin(c + 4. * (d + 5. + x))
        );
        expr.with(rel.as_ref().data_type().clone())
            .display_dot()
            .unwrap();
    }
}
