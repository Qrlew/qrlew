use std::{fmt, io, string};

use super::{Property, RewritingRule, RelationWithRewritingRules, rewriting_rule};
use crate::{
    relation::Relation,
    namer,
    visitor::Acceptor,
    display::{self, colors}, expr::{Reduce, rewriting},
};

/// A node in the RelationWithRewritingRules representation
#[derive(Clone, Copy, Debug, Hash)]
enum Node<'a> {
    Relation(&'a Relation),
    RewritingRule(&'a RewritingRule),
}

/// An edge in the RelationWithRewritingRules representation
#[derive(Clone, Copy, Debug, Hash)]
enum Edge<'a> {
    InputRelation(&'a Relation, &'a Relation),
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
            Node::Relation(relation) => format!("{relation}"),
            Node::RewritingRule(rewriting_rule) => format!("{rewriting_rule}"),
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
            Node::RewritingRule(rewriting_rule) => match rewriting_rule.output() {
                Property::Public => colors::TABLEAU_DARK_BLUE,
                Property::Published => colors::TABLEAU_LIGHT_BLUE,
                Property::ProtectedEntityPreserving => colors::TABLEAU_ORANGE,
                Property::DifferentiallyPrivate => colors::TABLEAU_RED,
            },
        }))
    }
}

impl<'a> dot::GraphWalk<'a, Node<'a>, Edge<'a>> for RelationWithRewritingRules<'a> {
    fn nodes(&'a self) -> dot::Nodes<'a, Node<'a>> {
        self.relation().iter().map(Node::Relation).chain(
            self.attributes().iter().map(Node::RewritingRule)
        ).collect()
    }

    fn edges(&'a self) -> dot::Edges<'a, Edge<'a>> {
        self.iter().flat_map(|relation_with_rewriting_rules| {
            relation_with_rewriting_rules.relation().inputs().into_iter().map(|input| Edge::InputRelation(input, relation_with_rewriting_rules.relation())).chain(
                relation_with_rewriting_rules.attributes().into_iter().map(|rewriting_rule| Edge::RelationRewritingRule(relation_with_rewriting_rules.relation(), rewriting_rule))
            )
        }).collect()
    }

    fn source(&'a self, edge: &Edge<'a>) -> Node<'a> {
        match edge {
            Edge::InputRelation(input, _relation) => Node::Relation(input),
            Edge::RelationRewritingRule(relation, _rewriting_rule) => Node::Relation(relation),
        }
    }

    fn target(&'a self, edge: &Edge<'a>) -> Node<'a> {
        match edge {
            Edge::InputRelation(_input, relation) => Node::Relation(relation),
            Edge::RelationRewritingRule(_relation, rewriting_rule) => Node::RewritingRule(rewriting_rule),
        }
    }
}

impl<'a> RelationWithRewritingRules<'a> {
    /// Render the Relation to dot
    pub fn dot<W: io::Write>(&self, w: &mut W, opts: &[&str]) -> io::Result<()> {
        display::dot::render(self, w, opts)
    }
}