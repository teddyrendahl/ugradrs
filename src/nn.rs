use rand::{thread_rng, Rng};
use std::ops::Add;

use crate::value::Value;

#[derive(Debug)]
struct Neuron<const N: usize> {
    weights: [Value; N],
    bias: Value,
}

impl<const N: usize> Neuron<N> {
    fn new() -> Self {
        let mut rng = thread_rng();
        Neuron {
            weights: (0..N)
                .map(|_| Value::from(rng.gen_range(-1.0..1.0)))
                .collect::<Vec<Value>>()
                .try_into()
                .unwrap(),
            bias: Value::from(0.),
        }
    }

    fn forward(&self, x: [Value; N]) -> Value {
        self.weights
            .clone()
            .into_iter()
            .zip(x.into_iter())
            .map(|(a, b)| a * b)
            .sum::<Value>()
            .add(self.bias.clone())
            .tanh()
    }

    fn parameters(&self) -> Vec<Value> {
        let mut p = self.weights.clone().to_vec();
        p.push(self.bias.clone());
        p
    }
}

pub trait Layer {
    fn forward(&self, x: Vec<Value>) -> Vec<Value>;
    fn parameters(&self) -> Vec<Value>;
}

/// A Layer with the input and output dimensions as generics
pub struct SizedLayer<const I: usize, const O: usize> {
    neurons: [Neuron<I>; O],
}

impl<const I: usize, const O: usize> Default for SizedLayer<I, O> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const I: usize, const O: usize> SizedLayer<I, O> {
    /// Create a layer of the provided size, initialized with random weights
    pub fn new() -> Self {
        Self {
            neurons: (0..O)
                .map(|_| Neuron::new())
                .collect::<Vec<Neuron<I>>>()
                .try_into()
                .unwrap(),
        }
    }
}

impl<const I: usize, const O: usize> Layer for SizedLayer<I, O> {
    fn forward(&self, x: Vec<Value>) -> Vec<Value> {
        self.neurons
            .iter()
            .map(|n| n.forward(x.clone().try_into().unwrap()))
            .collect::<Vec<Value>>()
    }

    fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}

pub struct Mlp<const I: usize, const O: usize> {
    layers: Vec<Box<dyn Layer>>,
}

impl<const I: usize, const O: usize> Mlp<I, O> {
    /// Create a new Mlp from an initial layer
    pub fn from_layer(layer: SizedLayer<I, O>) -> Mlp<I, O> {
        Self {
            layers: vec![Box::new(layer)],
        }
    }

    /// Add a layer to the Mlp
    ///
    /// Consumes the current Mlp re-defining the type to have the same number
    /// of outputs as the added layer.
    pub fn add_layer<const OUT: usize>(mut self, layer: SizedLayer<O, OUT>) -> Mlp<I, OUT> {
        self.layers.push(Box::new(layer));
        Mlp {
            layers: self.layers,
        }
    }

    /// Create a prediction by evaluating an input through a forward pass of each layer
    pub fn forward(&self, x: [Value; I]) -> [Value; O] {
        let mut x = x.to_vec();
        for layer in &self.layers {
            x = layer.forward(x)
        }
        x.try_into().unwrap()
    }

    /// Complete list of parameters in the Mlp graph
    pub fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    /// Set all parameter gradients back to zero
    pub fn zero_grad(&self) {
        for p in self.parameters() {
            p.zero_grad()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::nn::{Layer, Mlp, SizedLayer};
    use crate::value::Value;
    use rstest::{fixture, rstest};

    #[test]
    fn test_layer_forward() {
        let l: SizedLayer<2, 3> = SizedLayer::new();
        let o = l.forward([2.0, 3.0].into_iter().map(Value::from).collect());
        assert_eq!(o.len(), 3);
    }

    #[fixture]
    fn mlp() -> Mlp<3, 1> {
        Mlp::from_layer(SizedLayer::<3, 4>::new())
            .add_layer(SizedLayer::<4, 4>::new())
            .add_layer(SizedLayer::new())
    }

    #[rstest]
    fn test_mlp_forward(mlp: Mlp<3, 1>) {
        let o = mlp.forward([Value::from(2.0), Value::from(3.0), Value::from(-1.0)]);
        assert_eq!(o.len(), 1);
    }

    #[rstest]
    fn test_mlp_parameters(mlp: Mlp<3, 1>) {
        let p = mlp.parameters();
        assert_eq!(p.len(), 41);
    }

    #[rstest]
    fn test_mpl_train(mlp: Mlp<3, 1>) {
        let dataset = [
            [Value::from(2.0), Value::from(3.0), Value::from(-1.0)],
            [Value::from(3.0), Value::from(-1.0), Value::from(0.5)],
            [Value::from(0.5), Value::from(1.0), Value::from(1.0)],
            [Value::from(1.0), Value::from(1.0), Value::from(-1.0)],
        ];

        let truth = [
            Value::from(1.0),
            Value::from(-1.0),
            Value::from(-1.0),
            Value::from(1.0),
        ];
        for _ in 0..15 {
            let loss: Value = dataset
                .clone()
                .into_iter()
                .zip(truth.clone().into_iter())
                .map(|(d, t)| (t - mlp.forward(d)[0].clone()).powf(Value::from(2.0)))
                .sum();

            mlp.zero_grad();
            loss.backward();
            for p in mlp.parameters() {
                p.set_data(p.data() + p.gradient() * -0.1)
            }
        }
    }
}
