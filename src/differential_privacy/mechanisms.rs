use statrs::distribution::{ContinuousCDF, Normal};

pub fn gaussian_noise(epsilon: f64, delta: f64, sensitivity: f64) -> f64 {
    (2. * (1.25_f64.ln() / delta)).sqrt() * sensitivity / epsilon
}

pub fn gaussian_tau(epsilon: f64, delta: f64, sensitivity: f64) -> f64 {
    let dist = Normal::new(0.0, 1.0).unwrap();
    let scale = gaussian_noise(epsilon, delta, sensitivity);
    1. + scale * dist.inverse_cdf((1. - delta / 2.).powf(1. / sensitivity))
}
