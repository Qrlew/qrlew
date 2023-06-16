fn rewrite() {
    use qrlew::display::Dot;
    use qrlew::io::{postgresql, Database};
    use qrlew::With;
    use qrlew::{sql::parse, Relation};
    use qrlew::ast::Query;

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
    use qrlew::display::Dot;
    use qrlew::io::{postgresql, Database};
    use qrlew::With;
    use qrlew::{sql::parse, Relation};
    use qrlew::ast::Query;

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

fn protect() {
    use qrlew::display::Dot;
    use qrlew::io::{postgresql, Database};
    use qrlew::With;
    use qrlew::{sql::parse, Relation};
    use qrlew::ast::Query;

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
    let relation = relation.force_protect_from_field_paths(
        &relations,
        &[
            (
                "item_table",
                &[
                    ("order_id", "order_table", "id"),
                    ("user_id", "user_table", "id"),
                ],
                "name",
            ),
            ("order_table", &[("user_id", "user_table", "id")], "name"),
            ("user_table", &[], "name"),
        ],
    );
    println!("relation = {relation}");
    relation.display_dot().unwrap();
    let query = Query::from(&relation);
    println!("query = {query}");
}

fn compile() {
    use qrlew::display::Dot;
    use qrlew::io::{postgresql, Database};
    use qrlew::With;
    use qrlew::{sql::parse, Relation};
    use qrlew::ast::Query;

    let database = postgresql::test_database();
    let relations = database.relations();
    let relation = Relation::try_from(
        parse("SELECT sum(price) FROM item_table;")
            .unwrap()
            .with(&relations),
    )
    .unwrap();
    println!("relation = {relation}");
    let relation = relation.dp_compilation(
        &relations,
        &[
            (
                "item_table",
                &[
                    ("order_id", "order_table", "id"),
                    ("user_id", "user_table", "id"),
                ],
                "name",
            ),
            ("order_table", &[("user_id", "user_table", "id")], "name"),
            ("user_table", &[], "name"),
        ],
        1.,   // epsilon
        1e-5, // delta
    );
    println!("relation = {relation}");
    relation.display_dot().unwrap();
    let query = Query::from(&relation);
    println!("query = {query}");
}

fn main() {
    // rewrite();
    // ranges();
    // protect();
    compile();
}
