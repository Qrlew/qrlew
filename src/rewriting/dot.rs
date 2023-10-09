use std::{fmt, io, string, iter};

use itertools::Itertools;

use super::{Property, RewritingRule, RelationWithRewritingRules, rewriting_rule};
use crate::{
    relation::{Relation, Variant},
    namer,
    visitor::Acceptor,
    display::{self, colors}, expr::{Reduce, rewriting},
};

/// A node in the RelationWithRewritingRules representation
#[derive(Clone, Copy, Debug, Hash)]
enum Node<'a> {
    Relation(&'a Relation),
    RewritingRule(&'a RewritingRule, &'a Relation),
}

/// An edge in the RelationWithRewritingRules representation
#[derive(Clone, Copy, Debug, Hash)]
enum Edge<'a> {
    RelationInput(&'a Relation, &'a Relation),
    RelationRewritingRule(&'a Relation, &'a RewritingRule),
}

impl<'a> dot::Labeller<'a, Node<'a>, Edge<'a>> for RelationWithRewritingRules<'a> {
    fn graph_id(&'a self) -> dot::Id<'a> {
        dot::Id::new(namer::name_from_content("graph", self)).unwrap()
    }

    fn node_id(&'a self, node: &Node<'a>) -> dot::Id<'a> {
        dot::Id::new(namer::name_from_content("graph", node)).unwrap()
    }

    fn node_label(&'a self, node: &Node<'a>) -> dot::LabelText<'a> {
        dot::LabelText::html(match node {
            Node::Relation(relation) => format!("<b>{}</b><br/>{}", relation.name().to_uppercase(), relation.schema().iter().map(|f| f.name()).join(", ")),
            Node::RewritingRule(rewriting_rule, _) => match rewriting_rule.output() {
                Property::Public => format!("P"),
                Property::Published => format!("p"),
                Property::ProtectedEntityPreserving => format!("PEP"),
                Property::DifferentiallyPrivate => format!("DP"),
            },
        })
    }

    fn node_color(&'a self, node: &Node<'a>) -> Option<dot::LabelText<'a>> {
        Some(dot::LabelText::label(match node {
            Node::Relation(relation) => match relation {
                Relation::Table(_) => colors::MEDIUM_RED,
                Relation::Map(_) => colors::LIGHT_GREEN,
                Relation::Reduce(_) => colors::DARK_GREEN,
                Relation::Join(_) => colors::LIGHT_RED,
                Relation::Set(_) => colors::LIGHTER_GREEN,
                Relation::Values(_) => colors::MEDIUM_GREEN,
            },
            Node::RewritingRule(rewriting_rule, _) => match rewriting_rule.output() {
                Property::Public => colors::TABLEAU_CYAN,
                Property::Published => colors::TABLEAU_BLUE,
                Property::ProtectedEntityPreserving => colors::TABLEAU_ORANGE,
                Property::DifferentiallyPrivate => colors::TABLEAU_RED,
            },
        }))
    }

    fn node_shape(&'a self, node: &Node<'a>) -> Option<dot::LabelText<'a>> {
        match node {
            Node::Relation(_) => None,
            Node::RewritingRule(_, _) => Some(dot::LabelText::label("circle")),
        }
    }

    fn edge_label(&'a self, edge: &Edge<'a>) -> dot::LabelText<'a> {
        dot::LabelText::LabelStr("".into())
    }

    fn edge_style(&'a self, edge: &Edge<'a>) -> dot::Style {
        match edge {
            Edge::RelationInput(r, i) => dot::Style::None,
            Edge::RelationRewritingRule(r, rr) => dot::Style::Dotted,
        }
    }
}

impl<'a> dot::GraphWalk<'a, Node<'a>, Edge<'a>> for RelationWithRewritingRules<'a> {
    fn nodes(&'a self) -> dot::Nodes<'a, Node<'a>> {
        self.iter().flat_map(|rwrr|
            iter::once(rwrr.relation()).map(Node::Relation).chain(
                rwrr.attributes().iter().map(|rewriting_rule| Node::RewritingRule(rewriting_rule, rwrr.relation()))
            )
        ).collect()
    }

    fn edges(&'a self) -> dot::Edges<'a, Edge<'a>> {
        self.iter().flat_map(|rwrr| {
            rwrr.relation().inputs().into_iter().map(|input| Edge::RelationInput(rwrr.relation(), input)).chain(
                rwrr.attributes().into_iter().map(|rewriting_rule| Edge::RelationRewritingRule(rwrr.relation(), rewriting_rule))
            )
        }).collect()
    }

    fn source(&'a self, edge: &Edge<'a>) -> Node<'a> {
        match edge {
            Edge::RelationInput(relation, _input) => Node::Relation(relation),
            Edge::RelationRewritingRule(relation, _rewriting_rule) => Node::Relation(relation),
        }
    }

    fn target(&'a self, edge: &Edge<'a>) -> Node<'a> {
        match edge {
            Edge::RelationInput(_relation, input) => Node::Relation(input),
            Edge::RelationRewritingRule(relation, rewriting_rule) => Node::RewritingRule(rewriting_rule, relation),
        }
    }
}

impl<'a> RelationWithRewritingRules<'a> {
    /// Render the Relation to dot
    pub fn dot<W: io::Write>(&self, w: &mut W, opts: &[&str]) -> io::Result<()> {
        display::dot::render(self, w, opts)
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::*;
    use crate::{
        ast,
        builder::With,
        display::Dot,
        io::{postgresql, Database},
        sql::parse,
        Relation,
    };

    #[test]
    fn test_set_rewriting_rules() {
        let mut database = postgresql::test_database();
        let relations = database.relations();
        let query = parse(
            "SELECT order_id, sum(price) AS sum_price,
        count(price) AS count_price,
        avg(price) AS mean_price
        FROM item_table WHERE order_id IN (1,2,3,4,5,6,7,8,9,10) GROUP BY order_id",
        )
        .unwrap();
        let relation = Relation::try_from(query.with(&relations)).unwrap();
        relation.display_dot().unwrap();
        // Add rewritting rules
        let relation_with_rules = relation.with_attributes(vec![RewritingRule::new(vec![], Property::Public)]);
        relation_with_rules.display_dot().unwrap();
    }
}