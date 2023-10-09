pub mod relation_with_attributes;
pub mod rewriting_rule;
pub mod dot;

pub use relation_with_attributes::RelationWithAttributes;
pub use rewriting_rule::{Property, RewritingRule, RelationWithRewritingRules};
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
    fn test_compile() {
        let mut database = postgresql::test_database();
        let relations = database.relations();

        for (p, r) in relations.iter() {
            println!("{} -> {r}", p.into_iter().join("."))
        }

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
        let relation_with_rules: rewriting_rule::RelationWithRewritingRules = relation.with_default_attributes();
        println!("{:#?}", relation_with_rules);
    }
}