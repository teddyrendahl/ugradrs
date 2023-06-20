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


## Run Tests
```shell
cargo test
```