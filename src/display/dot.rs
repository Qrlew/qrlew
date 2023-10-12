use super::colors::*;
use dot::{GraphWalk, Labeller, Style};
use std::io::{self, Write};

/// Renders graph `g` into the writer `w` in DOT syntax.
/// (Main entry point for the library.)
pub fn render<
    'a,
    N: Clone + 'a,
    E: Clone + 'a,
    G: Labeller<'a, N, E> + GraphWalk<'a, N, E>,
    W: Write,
>(
    g: &'a G,
    w: &mut W,
    opts: &[&str],
) -> io::Result<()> {
    fn writeln<W: Write>(w: &mut W, arg: &[&str]) -> io::Result<()> {
        for &s in arg {
            w.write_all(s.as_bytes())?;
        }
        write!(w, "\n")
    }

    fn indent<W: Write>(w: &mut W) -> io::Result<()> {
        w.write_all(b"    ")
    }

    writeln(w, &["digraph ", g.graph_id().as_slice(), " {"])?;
    // Add base Styling
    writeln(
        w,
        &[r##"
        rankdir="TB";
        splines=true;
        overlap=false;
        nodesep="0.2";
        ranksep="0.4";
        labelloc="t";
        fontname="Ovo,Red Hat Text";
        fontsize="11pt"
        bgcolor="#00000000""##],
    )?;
    if opts.contains(&"dark") {
        writeln(
            w,
            &[r##"
            node [ shape="box" style="filled,rounded" margin=0.2, fontname="Red Hat Display,sans-serif", fontsize="11pt", color="#ffffffbb" ]
            edge [ fontname="Red Hat Text" color="#ffffffbb" ]
            "##],
        )?;
    } else {
        writeln(
            w,
            &[r##"
            node [ shape="box" style="filled,rounded" margin=0.2, fontname="Red Hat Display,sans-serif", fontsize="11pt", color="#00000055" ]
            edge [ fontname="Red Hat Text" color="#2B303A" ]
            "##],
        )?;
    }
    for n in g.nodes().iter() {
        let mut colorstring;

        indent(w)?;
        let id = g.node_id(n);

        let escaped = &g.node_label(n).to_dot_string();
        let shape;

        let mut text = vec![id.as_slice()];
        // Add node label
        text.push("[label=");
        text.push(escaped);
        text.push("]");
        // Add node style
        let style = g.node_style(n);
        if style != Style::None {
            text.push("[style=\"");
            text.push(style.as_slice());
            text.push("\"]");
        }
        // Add node color
        let color = g.node_color(n);
        if let Some(c) = color {
            colorstring = c.to_dot_string();
            text.push("[fillcolor=");
            text.push(&colorstring);
            text.push("]");
        }
        let color = g.node_color(n);
        if let Some(dot::LabelText::LabelStr(c)) = color {
            text.push("[fontcolor=\"");
            match c.as_ref().to_lowercase().as_str() {
                DARK_GREEN | MEDIUM_GREEN | MEDIUM_RED | DARK_RED => text.push("#ffffffbb"),
                LIGHT_GREEN | LIGHTER_GREEN | LIGHT_RED => text.push("#000000bb"),
                _ => text.push("black"),
            }
            text.push("\"]");
        }
        // Add node shape
        if let Some(s) = g.node_shape(n) {
            shape = s.to_dot_string();
            text.push("[shape=");
            text.push(&shape);
            text.push("]");
            // Remove margin for circles
            if shape==r#""circle""# {
                text.push("[margin=0.1]");
                text.push(r#"[fontsize="8pt"]"#);
            }
        }

        text.push(";");
        writeln(w, &text)?;
    }

    for e in g.edges().iter() {
        let mut colorstring;
        let escaped_label = &g.edge_label(e).to_dot_string();
        let start_arrow = g.edge_start_arrow(e);
        let end_arrow = g.edge_end_arrow(e);
        let start_arrow_s = start_arrow.to_dot_string();
        let end_arrow_s = end_arrow.to_dot_string();

        indent(w)?;
        let source = g.source(e);
        let target = g.target(e);
        let source_id = g.node_id(&source);
        let target_id = g.node_id(&target);

        let mut text = vec![source_id.as_slice(), " -> ", target_id.as_slice()];
        // Add edge labels
        text.push("[label=");
        text.push(escaped_label);
        text.push("]");
        // Add edge style
        let style = g.edge_style(e);
        if style != Style::None {
            text.push("[style=\"");
            text.push(style.as_slice());
            text.push("\"]");
        }
        // Add edge color
        let color = g.edge_color(e);
        if let Some(c) = color {
            colorstring = c.to_dot_string();
            text.push("[color=");
            text.push(&colorstring);
            text.push("]");
        }

        if !start_arrow.arrows.is_empty() || !end_arrow.arrows.is_empty() {
            text.push("[");
            if !end_arrow.arrows.is_empty() {
                text.push("arrowhead=\"");
                text.push(&end_arrow_s);
                text.push("\"");
            }
            if !start_arrow.arrows.is_empty() {
                text.push(" dir=\"both\" arrowtail=\"");
                text.push(&start_arrow_s);
                text.push("\"");
            }

            text.push("]");
        }

        text.push(";");
        writeln(w, &text)?;
    }

    writeln(w, &["}"])
}
