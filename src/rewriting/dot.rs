use std::{io, iter};

use itertools::Itertools;

use super::{Property, RelationWithRewritingRule, RelationWithRewritingRules, RewritingRule};
use crate::{
    display::{self, colors},
    namer,
    relation::{Relation, Variant},
    visitor::Acceptor,
};

/// A node in the RelationWithRewritingRules representation
#[derive(Clone, Copy, Debug, Hash)]
enum Node<'a> {
    Relation(&'a RelationWithRewritingRules<'a>),
    RewritingRule(&'a RewritingRule, &'a RelationWithRewritingRules<'a>),
}

/// An edge in the RelationWithRewritingRules representation
#[derive(Clone, Copy, Debug, Hash)]
enum Edge<'a> {
    RelationInput(
        &'a RelationWithRewritingRules<'a>,
        &'a RelationWithRewritingRules<'a>,
    ),
    RelationRewritingRule(&'a RelationWithRewritingRules<'a>, &'a RewritingRule),
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
            Node::Relation(relation) => format!(
                "<b>{}</b><br/>{}",
                relation.name().to_uppercase(),
                relation.schema().iter().map(|f| f.name()).join(", ")
            ),
            Node::RewritingRule(rewriting_rule, _) => {
                format!("{rewriting_rule}").replace(" → ", "<br/>→<br/>")
            }
        })
    }

    fn node_color(&'a self, node: &Node<'a>) -> Option<dot::LabelText<'a>> {
        Some(dot::LabelText::label(match node {
            Node::Relation(rwrr) => match rwrr.relation() {
                Relation::Table(_) => colors::MEDIUM_RED,
                Relation::Map(_) => colors::LIGHT_GREEN,
                Relation::Reduce(_) => colors::DARK_GREEN,
                Relation::Join(_) => colors::LIGHT_RED,
                Relation::Set(_) => colors::LIGHTER_GREEN,
                Relation::Values(_) => colors::MEDIUM_GREEN,
            },
            Node::RewritingRule(rewriting_rule, _) => match rewriting_rule.output() {
                Property::Private => colors::TABLEAU_BROWN,
                Property::SyntheticData => colors::TABLEAU_GREEN,
                Property::PrivacyUnitPreserving => colors::TABLEAU_ORANGE,
                Property::DifferentiallyPrivate => colors::TABLEAU_RED,
                Property::Published => colors::TABLEAU_BLUE,
                Property::Public => colors::TABLEAU_CYAN,
            },
        }))
    }

    fn node_shape(&'a self, node: &Node<'a>) -> Option<dot::LabelText<'a>> {
        match node {
            Node::Relation(_) => None,
            Node::RewritingRule(_, _) => Some(dot::LabelText::label("circle")),
        }
    }

    fn edge_label(&'a self, _edge: &Edge<'a>) -> dot::LabelText<'a> {
        dot::LabelText::LabelStr("".into())
    }

    fn edge_style(&'a self, edge: &Edge<'a>) -> dot::Style {
        match edge {
            Edge::RelationInput(_r, _i) => dot::Style::None,
            Edge::RelationRewritingRule(_r, _rr) => dot::Style::Dotted,
        }
    }
}

impl<'a> dot::GraphWalk<'a, Node<'a>, Edge<'a>> for RelationWithRewritingRules<'a> {
    fn nodes(&'a self) -> dot::Nodes<'a, Node<'a>> {
        self.iter()
            .collect_vec()
            .into_iter()
            .rev()
            .flat_map(|rwrr| {
                iter::once(rwrr).map(Node::Relation).chain(
                    rwrr.attributes()
                        .iter()
                        .map(|rewriting_rule| Node::RewritingRule(rewriting_rule, rwrr)),
                )
            })
            .collect()
    }

    fn edges(&'a self) -> dot::Edges<'a, Edge<'a>> {
        self.iter()
            .flat_map(|rwrr| {
                rwrr.inputs()
                    .into_iter()
                    .map(|input| Edge::RelationInput(rwrr, input))
                    .chain(
                        rwrr.attributes().into_iter().map(|rewriting_rule| {
                            Edge::RelationRewritingRule(rwrr, rewriting_rule)
                        }),
                    )
            })
            .collect()
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
            Edge::RelationRewritingRule(relation, rewriting_rule) => {
                Node::RewritingRule(rewriting_rule, relation)
            }
        }
    }
}

impl<'a> RelationWithRewritingRules<'a> {
    /// Render the Relation to dot
    pub fn dot<W: io::Write>(&self, w: &mut W, opts: &[&str]) -> io::Result<()> {
        display::dot::render(self, w, opts)
    }
}

impl<'a> RelationWithRewritingRule<'a> {
    /// Render the Relation to dot
    pub fn dot<W: io::Write>(&self, w: &mut W, opts: &[&str]) -> io::Result<()> {
        RelationWithRewritingRules::from(self).dot(w, opts)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::{
        builder::With,
        display::Dot,
        io::{postgresql, Database},
        rewriting::rewriting_rule::Parameters,
        sql::parse,
        Relation,
    };

    #[test]
    fn test_query() {
        let database = postgresql::test_database();
        let relations = database.relations();
        println!("{relations}");
        let query = parse(
            "SELECT a, count(abs(10*a+b)) AS x FROM table_1 WHERE b>-0.1 AND a IN (1,2,3) GROUP BY a",
        )
        .unwrap();
        let relation = Relation::try_from(query.with(&relations)).unwrap();
        relation.display_dot().unwrap();
    }

    #[test]
    fn test_set_rewriting_rules() {
        let database = postgresql::test_database();
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
        let relation_with_rules = relation.with_attributes(vec![RewritingRule::new(
            vec![],
            Property::Public,
            Parameters::None,
        )]);
        relation_with_rules.display_dot().unwrap();
    }
}
