use statrs::function::erf;

const PRECISION: f64 = 1e-5;

pub fn gaussian_noise(epsilon: f64, delta: f64, sensitivity: f64) -> f64 {
    (2. * (1.25_f64.ln() / delta)).sqrt() * sensitivity / epsilon
}

pub fn gaussian_tau(epsilon: f64, delta: f64, sensitivity: f64) -> f64 {
    let scale = gaussian_noise(epsilon, delta, sensitivity);
    1. + scale * inverse_cdf_gaussian((1. - delta / 2.).powf(1. / sensitivity), 0., 1.)
}

/// Compute the inverse of the gaussain CDF by bissection
fn inverse_cdf_gaussian(x: f64, mu: f64, sigma: f64) -> f64 {
    let mut low = mu - 5.0 * sigma;
    let mut high = mu + 5.0 * sigma;

    while (high - low).abs() > PRECISION {
        let mid = (low + high) / 2.0;
        let cdf_value = 0.5 * (1.0 + (mid - mu) / erf::erf(sigma * (2.0 as f64).sqrt()));

        if cdf_value < x {
            low = mid;
        } else {
            high = mid;
        }
    }

    (low + high) / 2.0
}
