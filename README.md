# ugradrs
![ugradrs](https://github.com/teddyrendahl/ugradrs/blob/assets/micrograd.jpeg)

`ugradrs` is a tiny Autograd engine inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) implemented in Rust. Intended as a personal exploration 
of the mechanics of neural networks. The repository allows for the creation of a
DAG of scalar value operations with a small Pytorch-like API wrapper.

## Example Usage
Calculate the gradient for a convoluted calculation
```rust
use ugradrs::value::Value;

let a = Value::from(-4.0);
let b = Value::from(2.0);
let mut c = a.clone() + b.clone();
let mut d = a.clone() * b.clone() + b.clone().powf(3.0.into());
c += c.clone() + 1.0;
c += c.clone() + 1.0 - a.clone();
d += d.clone() * 2.0 + (b.clone() + a.clone()).relu();
d += d.clone() * 3.0 + (b.clone() - a.clone()).relu();
let e = c.clone() - d.clone();
let f = e.clone().powf(2.0.into());
let mut g = f.clone() / 2.0;
g += Value::from(10.0) / f;

let eps = 10.0_f64.powi(-4);
assert!((g.data() - 24.7041).abs() < eps); // The outcome of the forward pass

// Perform backward propagation of gradient calculation
g.backward();

assert!((a.gradient() - 138.8338).abs() < eps); // dg/da
assert!((b.gradient() - 645.5773).abs() < eps); // dg/db
```

The library houses a small API for building simple Multi-Layer Perceptrons (MLPs). Both the `Mlp` and the `SizedLayer`
have their input and output dimensions captured in the Rust typing system. This means that mismatches in input data
and consecutive layer sizes are caught at compile time!
```rust
use ugradrs::nn::{Mlp, SizedLayer};

// By specifying the size of the hidden layer and the dimensions we ultimately want for the perceptron,
// the correct size of the input and output layers can be determined via the typing system.
let mlp: Mlp<2, 1> = Mlp::from_layer(SizedLayer::new(false)) // Adds a non-linear SizeLayer::<2, 16>
    .add_layer(SizedLayer::<16, 16>::new(false)) 
    .add_layer(SizedLayer::new(true)); // Creates a linear SizeLayer::<16, 1>
```

To see it in action, look at the `make-moons` example
```shell
$ cargo run --example make-moons --release
```

```shell
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
* * * * * * * * * * * * * * - - * * * * * * * * * * * * * - - 
* * * * * * * * * * * * * - - - - - * * * * * * * * * - - - - 
* * * * * * * * * * * * * - - - - - - * * * * * * * - - - - - 
* * * * * * * * * * * * - - - - - - - - * * * * * - - - - - - 
* * * * * * * * * * * - - - - - - - - - - * * * * - - - - - - 
* * * * * * * * * - - - - - - - - - - - - - * * - - - - - - - 
* * * * * * * - - - - - - - - - - - - - - - - - - - - - - - - 
* * * * * - - - - - - - - - - - - - - - - - - - - - - - - - - 
* * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
```

You can also visualize created graphs with the `draw_graph` optional feature. 
```rust
    use ugradrs::draw_dot::draw_dot;
    use ugradrs::value::Value;

    let x = Value::from(1.0);
    let y = (x * 2.0 + 1.0).relu();
    y.backward(); // Perform back-propagation to populate gradient fields
    draw_dot(y, "relu.dot").expect("Failed to create graph");
```
![ugradrs](https://github.com/teddyrendahl/ugradrs/blob/assets/relu.svg)
Or draw an entire Neuron:
```rust
    use ugradrs::draw_dot::draw_dot;
    use ugradrs::nn::Neuron;
    use ugradrs::value::Value;

    let n: Neuron<2> = Neuron::new(false);
    let y = n.forward([1.0.into(), (-2.0).into()]);
    y.backward(); // Perform back-propagation to populate gradient fields
    draw_dot(y, "neuron.dot").expect("Failed to create graph");
```
![ugradrs](https://github.com/teddyrendahl/ugradrs/blob/assets/neuron.svg)

To try it yourself run:
```shell
cargo run --example draw_dot --features draw_graph
```

## Run Tests
```shell
cargo test
```
