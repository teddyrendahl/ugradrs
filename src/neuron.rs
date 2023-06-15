use rand::{thread_rng, Rng};

use crate::value::Value;

#[derive(Debug)]
pub struct Neuron<const N: usize> {
    weights: [Value; N],
    bias: Value,
}

impl<const N: usize> Neuron<N> {
    pub fn new() -> Self {
        let mut rng = thread_rng();
        Neuron {
            weights: (0..N)
                .map(|_| Value::from(rng.gen_range(-1.0..=1.0)))
                .collect::<Vec<Value>>()
                .try_into()
                .unwrap(),
            bias: Value::from(rng.gen_range(-1.0..=1.0)),
        }
    }

    pub fn forward(&self, x: [Value; N]) -> Value {
        self.weights
            .clone()
            .into_iter()
            .zip(x.into_iter())
            .map(|(a, b)| a * b)
            .sum::<Value>()
            .tanh()
    }
}

pub struct Layer<const I: usize, const O: usize> {
    neurons: [Neuron<I>; O],
}

impl<const I: usize, const O: usize> Layer<I, O> {
    pub fn new() -> Self {
        Self {
            neurons: (0..O)
                .map(|_| Neuron::new())
                .collect::<Vec<Neuron<I>>>()
                .try_into()
                .unwrap(),
        }
    }

    pub fn forward(&self, x: [Value; I]) -> [Value; O] {
        self.neurons
            .iter()
            .map(|n| n.forward(x.clone()))
            .collect::<Vec<Value>>()
            .try_into()
            .unwrap()
    }
}