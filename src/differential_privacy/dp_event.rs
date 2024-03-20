use itertools::Itertools;
use statrs::{
    distribution::{ContinuousCDF, Normal},
    prec::F64_PREC,
};
use std::fmt;

/// An object inspired by Google's [DPEvent](https://github.com/google/differential-privacy/blob/main/python/dp_accounting/dp_event.py)
/// to represent a Private Query
#[derive(Clone, Debug, PartialEq)]
pub enum DpEvent {
    /// Represents application of an operation with no privacy impact.
    /// 
    /// A `NoOp` is generally never required, but it can be useful as a
    /// placeholder where a `DpEvent` is expected, such as in tests or some live
    /// accounting pipelines.
    NoOp,
    /// Represents an application of the Gaussian mechanism.
    /// 
    /// For values v_i and noise z ~ N(0, s^2I), this mechanism returns sum_i v_i + z.
    /// If the norms of the values are bounded ||v_i|| <= C, the noise_multiplier is
    /// defined as s / C.
    Gaussian {
        noise_multiplier: f64,
    },
    /// Represents an application of the Laplace mechanism.
    /// 
    /// For values v_i and noise z sampled coordinate-wise from the Laplace
    /// distribution L(0, s), this mechanism returns sum_i v_i + z.
    /// The probability density function of the Laplace distribution L(0, s) with
    /// parameter s is given as exp(-|x|/s) * (0.5/s) at x for any real value x.
    /// If the L_1 norm of the values are bounded ||v_i||_1 <= C, the noise_multiplier
    /// is defined as s / C.
    Laplace{
        noise_multiplier: f64,
    },
    /// Represents the application of a mechanism which is epsilon-delta approximate DP
    EpsilonDelta {
        epsilon: f64,
        delta: f64,
    },
    /// Represents application of a series of composed mechanisms.
    /// 
    /// The composition may be adaptive, where the query producing each event depends
    /// on the results of prior queries.
    Composed {
        events: Vec<DpEvent>,
    },
    /// Represents an application of Poisson subsampling.
    /// 
    /// Each record in the dataset is included in the sample independently with
    /// probability `sampling_probability`. Then the `DpEvent` `event` is applied
    /// to the sample of records.
    PoissonSampled {
        sampling_probability: f64,
        event: Box<DpEvent>,
    },
    /// Represents sampling a fixed sized batch of records with replacement.
    /// 
    /// A sample of `sample_size` (possibly repeated) records is drawn uniformly at
    /// random from the set of possible samples of a source dataset of size
    /// `source_dataset_size`. Then the `DpEvent` `event` is applied to the sample of
    /// records.
    SampledWithReplacement {
        source_dataset_size: i64,
        sample_size: i64,
        event: Box<DpEvent>,
    },
    /// Represents sampling a fixed sized batch of records without replacement.
    /// 
    /// A sample of `sample_size` unique records is drawn uniformly at random from the
    /// set of possible samples of a source dataset of size `source_dataset_size`.
    /// Then the `DpEvent` `event` is applied to the sample of records.
    SampledWithoutReplacement {
        source_dataset_size: i64,
        sample_size: i64,
        event: Box<DpEvent>,
    },
}

impl DpEvent {
    pub fn no_op() -> Self {
        Self::NoOp
    }

    pub fn gaussian(noise_multiplier: f64) -> Self {
        Self::Gaussian { noise_multiplier }
    }

    pub fn laplace(noise_multiplier: f64) -> Self {
        Self::Laplace { noise_multiplier }
    }

    pub fn epsilon_delta(epsilon: f64, delta: f64) -> Self {
        Self::EpsilonDelta { epsilon, delta }
    }

    // pub fn composed(events: &[DpEvent]) -> Self {
    //     events
    // }

    pub fn compose(self, other: Self) -> Self {
        if other.is_no_op() {
            self
        } else if self.is_no_op() {
            other
        } else {
            let (v1, v2) = match (self, other) {
                (DpEvent::Composed {events: v1}, DpEvent::Composed {events: v2}) => (v1, v2),
                (DpEvent::Composed {events: v}, other) => (v, vec![other]),
                (current, DpEvent::Composed {events: v}) => (vec![current], v),
                (current, other) => (vec![current], vec![other]),
            };
            DpEvent::Composed {events: v1.into_iter().chain(v2.into_iter()).collect()}
        }
    }

    pub fn is_no_op(&self) -> bool {
        match self {
            DpEvent::NoOp => true,
            DpEvent::Gaussian {noise_multiplier} | DpEvent::Laplace {noise_multiplier} => noise_multiplier == &0.0,
            DpEvent::EpsilonDelta {epsilon, delta} => epsilon == &0. && delta == &0.,
            DpEvent::Composed {events} => events.iter().all(|q| q.is_no_op()),
            _ => todo!(),
        }
    }

    pub fn gaussian_from_epsilon_delta_sensitivity(
        epsilon: f64,
        delta: f64,
        sensitivity: f64,
    ) -> Self {
        DpEvent::Gaussian {noise_multiplier: gaussian_noise(epsilon, delta, sensitivity)}
    }
}

impl fmt::Display for DpEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DpEvent::NoOp => writeln!(f, "NoOp"),
            DpEvent::Gaussian {noise_multiplier} => writeln!(f, "Gaussian ({noise_multiplier})"),
            DpEvent::Laplace {noise_multiplier} => writeln!(f, "Laplace ({noise_multiplier})"),
            DpEvent::EpsilonDelta {epsilon, delta} => writeln!(f, "EpsilonDelta ({epsilon}, {delta})"),
            DpEvent::Composed {events} => write!(
                f,
                "Composed ({})",
                events.iter().map(|dpe| format!("{}", dpe)).join(", ")
            ),
            _ => todo!(),
        }
    }
}

impl FromIterator<DpEvent> for DpEvent {
    fn from_iter<T: IntoIterator<Item = DpEvent>>(iter: T) -> Self {
        iter.into_iter().fold(DpEvent::NoOp, |composed, event| composed.compose(event))
    }
}

impl From<Vec<DpEvent>> for DpEvent {
    fn from(v: Vec<DpEvent>) -> Self {
        v.into_iter().collect()
    }
}

pub fn gaussian_noise(epsilon: f64, delta: f64, sensitivity: f64) -> f64 {
    // it can be inf so we clamp the results between 0 and f64::MAX
    ((2. * (1.25_f64 / delta).ln()).sqrt() * sensitivity / epsilon).clamp(0, f64::MAX)
}

pub fn gaussian_tau(epsilon: f64, delta: f64, sensitivity: f64) -> f64 {
    let dist = Normal::new(0.0, 1.0).unwrap();
    let scale = gaussian_noise(epsilon, delta, sensitivity);
    // TODO: we want to overestimate tau
    1. + scale * dist.inverse_cdf((1. - delta / 2.).powf(1. / sensitivity)) + F64_PREC
}
