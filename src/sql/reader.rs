use super::Result;
use sqlparser::{
    ast::{Expr, Statement},
    dialect::{Dialect, GenericDialect, PostgreSqlDialect},
    parser::Parser,
    tokenizer::Tokenizer,
};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Reader {
    sql_dialect: Arc<dyn Dialect + Sync>,
}

impl Reader {
    pub fn new() -> Self {
        Reader {
            sql_dialect: Arc::new(GenericDialect::default()),
        }
    }

    pub fn parse(&self, sql: &str) -> Result<Vec<Statement>> {
        Ok(Parser::parse_sql(self.sql_dialect.as_ref(), sql)?)
    }
}

impl Default for Reader {
    fn default() -> Self {
        Reader {
            sql_dialect: Arc::new(GenericDialect::default()),
        }
    }
}

pub fn parse_expr(expr: &str) -> Result<Expr> {
    let dialect = &GenericDialect {};
    let mut tokenizer = Tokenizer::new(dialect, expr);
    let tokens = tokenizer.tokenize()?;
    let mut parser = Parser::new(dialect).with_tokens(tokens);
    Ok(parser.parse_expr()?)
}

pub fn parse_postgres_expr(expr: &str) -> Result<Expr> {
    let dialect = &PostgreSqlDialect {};
    let mut tokenizer = Tokenizer::new(dialect, expr);
    let tokens = tokenizer.tokenize()?;
    let mut parser = Parser::new(dialect).with_tokens(tokens);
    Ok(parser.parse_expr()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expr() -> Result<()> {
        let a = "sum((-t.s.b + sqrt(pow(t.s.b,2)-4*t.s.a*t.r.c))/(2*t.s.a))";
        let expr = parse_expr(a)?;
        dbg!(expr);
        //assert!(true);
        Ok(())
    }

    #[test]
    fn test_reader() -> Result<()> {
        let sql = "SELECT a, b+c FROM tab;";
        let reader = Reader::new();
        let stmts = reader.parse(sql)?;
        dbg!(stmts);
        assert!(true);
        Ok(())
    }
}
