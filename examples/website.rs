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

fn protect() {
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
    let relation: Relation = relation
        .force_protect_from_field_paths(
            &relations,
            vec![
                (
                    "item_table",
                    vec![
                        ("order_id", "order_table", "id"),
                        ("user_id", "user_table", "id"),
                    ],
                    "name",
                ),
                ("order_table", vec![("user_id", "user_table", "id")], "name"),
                ("user_table", vec![], "name"),
            ],
        )
        .into();
    println!("relation = {relation}");
    relation.display_dot().unwrap();
    let query = Query::from(&relation);
    println!("query = {query}");
}

fn compile() {
    use qrlew::ast::Query;
    use qrlew::display::Dot;
    use qrlew::io::{postgresql, Database};
    use qrlew::With;
    use qrlew::{sql::parse, Relation};

    let database = postgresql::test_database();
    let relations = database.relations();
    let relation = Relation::try_from(
        parse("SELECT sum(price) FROM item_table;")
            .unwrap()
            .with(&relations),
    )
    .unwrap();
    println!("relation = {relation}");
    let pep_relation = relation.force_protect_from_field_paths(
        &relations,
        vec![
            (
                "item_table",
                vec![
                    ("order_id", "order_table", "id"),
                    ("user_id", "user_table", "id"),
                ],
                "name",
            ),
            ("order_table", vec![("user_id", "user_table", "id")], "name"),
            ("user_table", vec![], "name"),
        ],
    );
    pep_relation.display_dot().unwrap();

    let epsilon = 1.;
    let delta = 1e-3;
    let dp_relation = pep_relation.dp_compile(epsilon, delta).unwrap();
    let relation = dp_relation.0;
    relation.display_dot().unwrap();
    println!("relation = {relation}");
    let query = Query::from(&relation);
    println!("query = {query}");
}

fn main() {
    // rewrite();
    // ranges();
    // protect();
    compile();
}
