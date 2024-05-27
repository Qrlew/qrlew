use crate::{
    expr::{Expr},
    namer,
};
use std::f64::consts::PI;

impl Expr {
    /// Gaussian noise based on [Box Muller transform](https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform)
    pub fn gaussian_noise() -> Self {
        Expr::multiply(
            Expr::sqrt(Expr::multiply(
                Expr::val(-2.0),
                Expr::ln(Expr::random(namer::new_id("GAUSSIAN_NOISE"))),
            )),
            Expr::cos(Expr::multiply(
                Expr::val(2.0 * PI),
                Expr::random(namer::new_id("GAUSSIAN_NOISE")),
            )),
        )
    }
    /// Gaussian noise based on [Box Muller transform](https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform)
    pub fn add_gaussian_noise(self, sigma: f64) -> Self {
        Expr::plus(
            self,
            Expr::multiply(Expr::val(sigma), Expr::gaussian_noise()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        builder::WithoutContext,
        data_type::{function::Function as _, value::Value},
        display::Dot,
    };

    #[test]
    fn test_gaussian_noise() {
        //TODO not great to have stateful functions, fix it
        let x = Expr::gaussian_noise();
        println!("gaussian noise = {x}");
        println!(
            "gaussian noise value = {}",
            x.value(&Value::structured_from_values(vec![])).unwrap()
        );
        x.with(Value::structured_from_values(vec![]))
            .display_dot()
            .unwrap();
    }

    #[test]
    fn test_add_gaussian_noise() {
        //TODO not great to have stateful functions, fix it
        let g = Expr::col("mu").add_gaussian_noise(1.);
        println!("mu plus gaussian noise = {g}");
        g.with(Value::structured(vec![("mu", Value::from(1.0))]))
            .display_dot()
            .unwrap();
    }
}
