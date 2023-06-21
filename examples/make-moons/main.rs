use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;
use ugradrs::nn::{Mlp, SizedLayer};
use ugradrs::value::Value;

#[derive(Copy, Clone, Eq, PartialEq)]
enum Moon {
    Upper,
    Lower,
}

impl From<&Moon> for Value {
    fn from(value: &Moon) -> Self {
        match value {
            Moon::Upper => 1.0,
            Moon::Lower => -1.0,
        }
        .into()
    }
}

impl From<&Value> for Moon {
    fn from(value: &Value) -> Self {
        if value.data() > 0. {
            Moon::Upper
        } else {
            Moon::Lower
        }
    }
}

/// Create two interwoven half-circles
///
/// Based on the scikit-learn `make_moons` method
///
/// # Arguments
///
/// * `n_samples` - Number of samples to include in each moon
/// * `noise_stddev` - Standard deviation of normal distribution noise added on top of the crescent values
/// * `rng` - Random number generator used to create the noise
fn make_moons(n_samples: usize, noise_stddev: f64, rng: &mut ThreadRng) -> Vec<(Moon, (f64, f64))> {
    let noise = Normal::new(0., noise_stddev).unwrap();
    let outer = (0..n_samples).map(|s| {
        let r = s as f64 * PI / n_samples as f64;
        (Moon::Lower, (r.cos(), r.sin()))
    });
    let inner = (0..n_samples).map(|s| {
        let r = s as f64 * PI / n_samples as f64;
        (Moon::Upper, (1.0 - r.cos(), 1.0 - r.sin() - 0.5))
    });
    let mut outer: Vec<_> = outer
        .chain(inner)
        .map(|(m, (mut x, mut y))| {
            x += noise.sample(rng);
            y += noise.sample(rng);
            (m, (x, y))
        })
        .collect();
    outer.shuffle(rng);
    outer
}

/// Calculate the loss function by using an SVM "max-margin" loss and L2 regularization
fn calculate_loss(mlp: &Mlp<2, 1>, data: &[(Moon, (f64, f64))]) -> (Value, f64) {
    // Estimates of label
    let scores: Vec<(Moon, Value)> = data
        .iter()
        .map(|(m, (x, y))| (*m, mlp.forward([(*x).into(), (*y).into()])[0].clone()))
        .collect();
    let mut loss = scores
        .iter()
        .map(|(label, estimate)| (Value::from(1.0) - estimate.clone() * Value::from(label)).relu())
        .sum::<Value>()
        / Value::from(data.len() as f64);

    // L2 Regularization
    loss += Value::from(1e-4)
        * mlp
            .parameters()
            .into_iter()
            .map(|p| p.powf(2.0.into()))
            .sum::<Value>();

    // Accuracy prediction
    let acc = scores
        .iter()
        .map(|(label, estimate)| {
            if &Moon::from(estimate) == label {
                1.0
            } else {
                0.
            }
        })
        .sum::<f64>()
        / scores.len() as f64;
    (loss, acc)
}

fn draw_decision_boundary(mlp: &Mlp<2, 1>) {
    let steps = 15;
    for i in (-steps..=steps).rev() {
        let y = 2.0 * i as f64 / steps as f64;
        (-steps..=steps).for_each(|s| {
            let x = 2.0 * s as f64 / steps as f64;
            print!(
                "{} ",
                match Moon::from(&mlp.forward([x.into(), y.into()])[0]) {
                    Moon::Lower => "*",
                    Moon::Upper => "-",
                }
            )
        });
        println!()
    }
}

fn main() {
    let moons = make_moons(50, 0.1, &mut rand::thread_rng());
    let mlp: Mlp<2, 1> = Mlp::from_layer(SizedLayer::new(false))
        .add_layer(SizedLayer::<16, 16>::new(false))
        .add_layer(SizedLayer::new(true));

    for k in 0..100 {
        let (total_loss, acc) = calculate_loss(&mlp, &moons);
        mlp.zero_grad();
        total_loss.backward();

        // SGD
        let learning_rate = 1.0 - 0.9 * (k as f64) / 100.;
        for p in mlp.parameters() {
            p.set_data(p.data() - learning_rate * p.gradient())
        }
        println!("Step {k}, loss {}, accuracy {acc}", total_loss.data());
    }
    draw_decision_boundary(&mlp)
}
