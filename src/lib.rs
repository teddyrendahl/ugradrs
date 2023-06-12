use std::{
    fmt::Display,
    ops::{Add, Mul},
};
pub enum Operation {
    Add,
    Multiply,
    Tanh,
}
pub struct Value {
    data: f64,
    children: Vec<Value>,
    operation: Option<Operation>,
    gradient: f64,
    label: Option<String>,
}

impl Value {
    pub fn new(data: f64, label: Option<String>) -> Self {
        Self {
            data,
            children: vec![],
            operation: None,
            gradient: 0.,
            label,
        }
    }

    pub fn tanh(self) -> Self {
        Self {
            data: self.data.tanh(),
            children: vec![self],
            operation: Some(Operation::Tanh),
            gradient: 0.,
            label: None,
        }
    }

    pub fn backward(&mut self) {
        match self.operation {
            Some(Operation::Add) => {
                for child in self.children.iter_mut() {
                    child.gradient += self.gradient;
                }
            }
            Some(Operation::Multiply) => {
                let first = self.children[0].data;
                let second = self.children[1].data;

                self.children[0].gradient = second * self.gradient;
                self.children[1].gradient = first * self.gradient;
            }
            Some(Operation::Tanh) => {
                for child in self.children.iter_mut() {
                    child.gradient += (1.0 - self.data.powi(2)) * self.gradient;
                }
            }
            None => (),
        };
        for child in self.children.iter_mut() {
            child.backward();
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value(data={})", self.data)
    }
}

impl Add for Value {
    type Output = Value;

    fn add(self, rhs: Self) -> Self::Output {
        Value {
            data: self.data + rhs.data,
            children: vec![rhs, self],
            operation: Some(Operation::Add),
            gradient: 0.,
            label: None,
        }
    }
}

impl Mul for Value {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Value {
            data: self.data * rhs.data,
            children: vec![self, rhs],
            operation: Some(Operation::Multiply),
            gradient: 0.,
            label: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::Value;
    use approx::assert_abs_diff_eq;
    use std::collections::HashMap;

    fn find_gradients(v: &Value) -> HashMap<String, f64> {
        let mut gradients = HashMap::new();

        fn find_gradient(v: &Value, g: &mut HashMap<String, f64>) {
            g.insert(v.label.clone().unwrap(), v.gradient);
            for child in v.children.iter() {
                find_gradient(child, g)
            }
        }

        find_gradient(v, &mut gradients);
        gradients
    }

    #[test]
    fn test_backprop_add_and_mul() {
        let a = Value::new(2.0, Some("a".into()));
        let b = Value::new(-3.0, Some("b".into()));
        let c = Value::new(10.0, Some("c".into()));
        let mut e = a * b;
        e.label = Some("e".into());
        let mut d = e + c;
        d.label = Some("d".into());
        let f = Value::new(-2.0, Some("f".into()));
        let mut l = d * f;
        l.label = Some("l".into());

        l.gradient = 1.0;
        l.backward();

        let gradients = find_gradients(&l);

        assert_eq!(gradients["l".into()], 1.0);
        assert_eq!(gradients["d".into()], -2.0);
        assert_eq!(gradients["f".into()], 4.0);
        assert_eq!(gradients["c".into()], -2.0);
        assert_eq!(gradients["e".into()], -2.0);
        assert_eq!(gradients["a".into()], 6.0);
        assert_eq!(gradients["b".into()], -4.0);
    }

    #[test]
    fn test_backprop_neuron() {
        let x1 = Value::new(2.0, Some("x1".into()));
        let x2 = Value::new(0.0, Some("x2".into()));
        let w1 = Value::new(-3.0, Some("w1".into()));
        let w2 = Value::new(1.0, Some("w2".into()));
        let b = Value::new(6.8813735870195432, Some("b".into()));

        let mut x1w1 = x1 * w1;
        x1w1.label = Some("x1w1".into());

        let mut x2w2 = x2 * w2;
        x2w2.label = Some("x2w2".into());

        let mut x1w1x2w2 = x1w1 + x2w2;
        x1w1x2w2.label = Some("x1w1x2w2".into());
        let mut n = x1w1x2w2 + b;
        n.label = Some("n".into());
        let mut o = n.tanh();
        o.label = Some("o".into());
        o.gradient = 1.0;
        o.backward();
        let gradients = find_gradients(&o);

        assert_abs_diff_eq!(gradients["o".into()], 1.0, epsilon = 0.001);
        assert_abs_diff_eq!(gradients["w1".into()], 1.0, epsilon = 0.001);
        assert_eq!(gradients["w2".into()], 0.0);
        assert_abs_diff_eq!(gradients["x1".into()], -1.5, epsilon = 0.001);
        assert_abs_diff_eq!(gradients["x2".into()], 0.5, epsilon = 0.001);
        assert_abs_diff_eq!(gradients["n".into()], 0.5, epsilon = 0.001);
        assert_abs_diff_eq!(gradients["b".into()], 0.5, epsilon = 0.001);
        assert_abs_diff_eq!(gradients["x2w2".into()], 0.5, epsilon = 0.001);
        assert_abs_diff_eq!(gradients["x1w1".into()], 0.5, epsilon = 0.001);
    }
}
