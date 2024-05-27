use super::{Column, Expr, Function};

impl Expr {
    /// Reduce the expression modulo a bijection
    pub fn reduce_modulo_bijection(&self) -> &Expr {
        match self {
            Expr::Function(Function {
                function,
                arguments,
            }) => {
                if function.is_bijection() {
                    arguments
                        .get(0)
                        .map(|arg| arg.reduce_modulo_bijection())
                        .unwrap_or_else(|| self)
                } else {
                    self
                }
            }
            _expr => self,
        }
    }

    /// Some column if it reduces to a column, None else.
    pub fn into_column_modulo_bijection(&self) -> Option<Column> {
        let expr = self.reduce_modulo_bijection();
        match expr {
            Expr::Column(column) => Some(column.clone()),
            _ => None,
        }
    }

    /// True if reduces into a unique 0-ary function.
    pub fn is_unique(&self) -> bool {
        let expr = self.reduce_modulo_bijection();
        match expr {
            Expr::Function(Function {
                function,
                arguments,
            }) => {
                if function.is_bijection() {
                    arguments
                        .get(0)
                        .map(|arg| arg.is_unique())
                        .unwrap_or_else(|| false)
                } else {
                    function.is_unique()
                }
            }
            _ => false,
        }
    }

    /// True if 2 expressions are equal modulo a bijection
    pub fn eq_modulo_bijection(&self, expr: &Expr) -> bool {
        self.reduce_modulo_bijection() == expr.reduce_modulo_bijection()
    }
}

#[cfg(test)]
mod tests {
    use crate::expr::identifier::Identifier;

    use super::*;

    #[test]
    fn test_into_column_modulo_bijection() {
        let a = expr!(md5(cast_as_text(exp(a))));
        let b = expr!(md5(cast_as_text(sin(a))));
        println!(
            "a.into_column_modulo_bijection() {:?}",
            a.into_column_modulo_bijection()
        );
        println!(
            "b.into_column_modulo_bijection() {:?}",
            b.into_column_modulo_bijection()
        );
        assert!(a.into_column_modulo_bijection() == Some(Identifier::from_name("a")));
        assert!(b.into_column_modulo_bijection() == None);
    }

    #[test]
    fn test_eq_modulo_bijection() {
        let a = expr!(a + b);
        let b = expr!(exp(a + b));
        assert!(a.eq_modulo_bijection(&b));
        let a = expr!(a + b);
        let b = expr!(exp(sin(a + b)));
        assert!(!a.eq_modulo_bijection(&b));
    }

    #[test]
    fn test_is_unique() {
        assert!(Expr::md5(Expr::cast_as_text(Expr::exp(Expr::newid()))).is_unique());
        assert!(!Expr::md5(Expr::cast_as_text(Expr::exp(Expr::col("a")))).is_unique());
    }
}
