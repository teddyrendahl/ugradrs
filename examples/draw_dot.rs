use ugradrs::draw_dot::draw_dot;
use ugradrs::nn::Neuron;
use ugradrs::value::Value;

fn main() {
    let x = Value::from(1.0);
    let y = (x * 2.0 + 1.0).relu();
    y.backward();
    draw_dot(y, "relu.dot").expect("Failed to create graph");

    let n: Neuron<2> = Neuron::new(false);
    let y = n.forward([1.0.into(), (-2.0).into()]);
    y.backward();
    draw_dot(y, "neuron.dot").expect("Failed to create graph");
}
