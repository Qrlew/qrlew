fn rewrite() {
    use qrlew::ast::Query;
    use qrlew::display::Dot;
    use qrlew::io::{postgresql, Database};
    use qrlew::With;
    use qrlew::{sql::parse, Relation};

    let database = postgresql::test_database();
    let relations = database.relations();
    let relation = Relation::try_from(
        parse(
            "SELECT * FROM order_table JOIN
            item_table ON id=order_id;",
        )
        .unwrap()
        .with(&relations),
    )
    .unwrap();
    println!("relation = {relation}");
    relation.display_dot().unwrap();
    let query = Query::from(&relation);
    println!("query = {query}");
}

fn ranges() {
    use qrlew::ast::Query;
    use qrlew::display::Dot;
    use qrlew::io::{postgresql, Database};
    use qrlew::With;
    use qrlew::{sql::parse, Relation};

    let database = postgresql::test_database();
    let relations = database.relations();
    let relation = Relation::try_from(
        parse("SELECT price, cos(price/100) FROM item_table;")
            .unwrap()
            .with(&relations),
    )
    .unwrap();
    println!("relation = {relation}");
    relation.display_dot().unwrap();
    let query = Query::from(&relation);
    println!("query = {query}");
}

fn main() {
    // rewrite();
    ranges();
}
