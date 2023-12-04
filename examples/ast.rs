use qrlew::{ast::*, sql::relation::parse};

fn build_ast() -> Result<(), &'static str> {
    // A query
    let query = Query {
        with: None,
        body: Box::new(SetExpr::Select(Box::new(Select {
            distinct: None,
            top: None,
            projection: vec![
                SelectItem::UnnamedExpr(Expr::Identifier(Ident::from("a"))),
                SelectItem::ExprWithAlias {
                    expr: Expr::Identifier(Ident::from("b")),
                    alias: Ident::from("B"),
                },
            ],
            into: None,
            from: vec![
                TableWithJoins {
                    relation: TableFactor::Table {
                        name: ObjectName(vec!["tab_1".into()]),
                        alias: None,
                        args: None,
                        with_hints: vec![],
                        version: None,
                        partitions: vec![],
                    },
                    joins: vec![],
                },
                TableWithJoins {
                    relation: TableFactor::Table {
                        name: ObjectName(vec!["tab_2".into()]),
                        alias: None,
                        args: None,
                        with_hints: vec![],
                        version: None,
                        partitions: vec![],
                    },
                    joins: vec![Join {
                        relation: TableFactor::Table {
                            name: ObjectName(vec!["path".into(), "tab_3".into()]),
                            alias: None,
                            args: None,
                            with_hints: vec![],
                            version: None,
                            partitions: vec![],
                        },
                        join_operator: JoinOperator::LeftOuter(JoinConstraint::Using(vec![
                            "a".into(),
                            "b".into(),
                            "c".into(),
                        ])),
                    }],
                },
            ],
            lateral_views: vec![],
            selection: None,
            group_by: GroupByExpr::Expressions(vec![]),
            cluster_by: vec![],
            distribute_by: vec![],
            sort_by: vec![],
            having: None,
            qualify: None,
            named_window: vec![],
        }))),
        order_by: vec![],
        limit: None,
        offset: None,
        fetch: None,
        locks: vec![],
        limit_by: vec![],
        for_clause: None,
    };
    println!("{}\n", query);
    // A CTE
    let cte = Cte {
        alias: TableAlias {
            name: Ident::new("table"),
            columns: vec![Ident::new("a"), Ident::new("B")],
        },
        query: Box::new(query.clone()),
        from: None,
    };
    println!("{}", cte);
    let cte = Cte {
        alias: TableAlias {
            name: Ident::new("table"),
            columns: vec![Ident::new("a"), Ident::new("B")],
        },
        query: Box::new(query),
        from: Some(Ident::new("fro")),
    };
    println!("{}", cte);

    Ok(())
}

fn print_ast(query: &str) -> Result<(), &'static str> {
    let query = parse(query).unwrap();
    println!("Printing the tree of {query}");
    println!("Tree = {:#?}", query);

    Ok(())
}

fn main() -> Result<(), &'static str> {
    build_ast()?;

    // Print an AST with a subquery
    // print_ast("select * from (select count(a) as c from sch.tbl)");

    // Print an AST with a JOIN
    print_ast("select * from sch.tbl1 LEFT OUTER JOIN sch.tbl2 ON tbl1.id = tbl2.id LIMIT 100")?;

    // Print an AST with CTEs
    // print_ast("WITH cte_tbl AS (select a, b FROM sch.tbl1) select * from cte_tbl");

    // Print an AST with Insert
    print_ast("INSERT INTO person (name, data) VALUES (?1, ?2)")?;

    // Print a UNION
    print_ast("SELECT * FROM table_1 10 UNION SELECT * FROM table_2")?;

    // Print an expression with nesting
    print_ast("SELECT (a+b)*(c+d) FROM table_1")?;

    // Print an AST with Values
    print_ast("SELECT a FROM (VALUES (1), (2), (3)) AS t1 (possible_values);")?;

    // Print an AST with Values
    print_ast("(VALUES (1), (2), (3)) AS t1")?;

    // Print an AST with count(*)
    print_ast("SELECT COUNT(*) FROM table_1")?;

    Ok(())
}
