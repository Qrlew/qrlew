/// A Private Query
#[derive(Clone, Debug, PartialEq)]
pub enum PrivateQuery {
    Gaussian(f64),
    Laplace(f64),
    EpsilonDelta(f64, f64),
    Composed(Vec<PrivateQuery>),
}

impl PrivateQuery {
    pub fn compose(self, other: Self) -> Self {
        if other.is_null() {
            self
        } else if self.is_null() {
            other
        } else {
            let (v1, v2) = match (self, other) {
                (PrivateQuery::Composed(v1), PrivateQuery::Composed(v2)) => (v1, v2),
                (PrivateQuery::Composed(v), other) => (v, vec![other]),
                (current, PrivateQuery::Composed(v)) => (vec![current], v),
                (current, other) => (vec![current], vec![other]),
            };
            PrivateQuery::Composed(v1.into_iter().chain(v2.into_iter()).collect())
        }
    }

    pub fn null() -> Self {
        Self::EpsilonDelta(0., 0.)
    }

    pub fn is_null(&self) -> bool {
        match self {
            PrivateQuery::Gaussian(n) | PrivateQuery::Laplace(n) => n == &0.0,
            PrivateQuery::EpsilonDelta(e, d) => e == &0. && d == &0.,
            PrivateQuery::Composed(v) => v.iter().all(|q| q.is_null()),
        }
    }
}

impl From<Vec<PrivateQuery>> for PrivateQuery {
    fn from(v: Vec<PrivateQuery>) -> Self {
        v.into_iter().reduce(|c, q| c.compose(q)).unwrap()
    }
}
