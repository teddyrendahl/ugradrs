use std::{
    cell::RefCell,
    collections::HashSet,
    fmt::Display,
    hash::Hash,
    ops::{Add, Deref, Div, Mul, Sub},
    rc::Rc,
};

#[derive(Hash, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    Add,
    Multiply,
    Tanh,
    Exponent,
    Pow(i32),
}

#[derive(Clone, PartialEq, Eq)]
pub struct Value(Rc<RefCell<ValueInternal>>);

pub struct ValueInternal {
    data: f64,
    children: Vec<Value>,
    operation: Option<Operation>,
    gradient: f64,
    label: Option<String>,
}

impl Deref for Value {
    type Target = Rc<RefCell<ValueInternal>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.borrow().hash(state)
    }
}
impl Value {
    pub fn new(value: ValueInternal) -> Self {
        Value(Rc::new(RefCell::new(value)))
    }

    pub fn data(&self) -> f64 {
        self.borrow().data
    }

    pub fn gradient(&self) -> f64 {
        self.borrow().gradient
    }

    pub fn children(&self) -> Vec<Value> {
        self.borrow().children.clone()
    }

    pub fn operation(&self) -> Option<Operation> {
        self.borrow().operation
    }

    fn tanh(&self) -> Value {
        Value::new(ValueInternal::new(
            self.borrow().data.tanh(),
            vec![self.clone()],
            Some(Operation::Tanh),
            None,
        ))
    }

    pub fn exp(&self) -> Self {
        Value::new(ValueInternal::new(
            self.borrow().data.exp(),
            vec![self.clone()],
            Some(Operation::Exponent),
            None,
        ))
    }

    pub fn powi(&self, value: i32) -> Self {
        Value::new(ValueInternal::new(
            self.borrow().data.powi(value),
            vec![self.clone()],
            Some(Operation::Pow(value)),
            None,
        ))
    }

    pub fn backward(&self) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();

        fn build_topo(node: Value, visited: &mut HashSet<Value>, topo: &mut Vec<Value>) {
            if !visited.contains(&node) {
                visited.insert(node.clone());
                for child in node.children() {
                    build_topo(child, visited, topo)
                }
                topo.push(node)
            }
        }
        build_topo(self.clone(), &mut visited, &mut topo);
        self.borrow_mut().gradient = 1.0;
        for node in topo.into_iter().rev() {
            node.backward_internal()
        }
    }
    fn backward_internal(&self) {
        match self.operation() {
            Some(Operation::Add) => {
                for child in self.children().iter_mut() {
                    child.borrow_mut().gradient += self.gradient();
                }
            }
            Some(Operation::Multiply) => {
                let first = self.children()[0].data();
                let second = self.children()[1].data();

                self.children()[0].borrow_mut().gradient = second * self.gradient();
                self.children()[1].borrow_mut().gradient = first * self.gradient();
            }
            Some(Operation::Tanh) => {
                for child in self.children().iter_mut() {
                    child.borrow_mut().gradient += (1.0 - self.data().powi(2)) * self.gradient();
                }
            }
            Some(Operation::Exponent) => {
                for child in self.children().iter_mut() {
                    child.borrow_mut().gradient += self.data() * self.gradient()
                }
            }
            Some(Operation::Pow(v)) => {
                for child in self.children().iter_mut() {
                    child.borrow_mut().gradient +=
                        v as f64 * child.data().powf(v as f64 - 1.0) * self.gradient();
                }
            }
            None => (),
        };
    }
}

impl Display for ValueInternal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value(data={})", self.data)
    }
}

impl Hash for ValueInternal {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.to_bits().hash(state);
        self.gradient.to_bits().hash(state);
        self.label.hash(state);
        self.operation.hash(state);
        self.children.hash(state);
    }
}
impl ValueInternal {
    pub fn new(
        data: f64,
        children: Vec<Value>,
        operation: Option<Operation>,
        label: Option<String>,
    ) -> Self {
        Self {
            data,
            children,
            operation,
            gradient: 0.,
            label,
        }
    }
}
impl From<f64> for Value {
    fn from(data: f64) -> Self {
        Value::new(ValueInternal::new(data, vec![], None, None))
    }
}

impl Add for Value {
    type Output = Value;

    fn add(self, rhs: Self) -> Self::Output {
        Value::new(ValueInternal {
            data: self.data() + rhs.data(),
            children: vec![rhs, self],
            operation: Some(Operation::Add),
            gradient: 0.,
            label: None,
        })
    }
}

impl Sub for Value {
    type Output = Value;

    fn sub(self, rhs: Self) -> Self::Output {
        self + rhs * Value::from(-1.0)
    }
}

impl Mul for Value {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Value::new(ValueInternal::new(
            self.data() * rhs.data(),
            vec![self, rhs],
            Some(Operation::Multiply),
            None,
        ))
    }
}

impl Div for Value {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.powi(-1)
    }
}

impl PartialEq for ValueInternal {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
            && self.gradient == other.gradient
            && self.label == other.label
            && self.operation == other.operation
            && self.children == other.children
    }
}

impl Eq for ValueInternal {}
#[cfg(test)]
mod tests {
    use crate::Value;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_backprop_add_and_mul() {
        let a = Value::from(2.0);
        let b = Value::from(-3.0);
        let c = Value::from(10.0);
        let e = a.clone() * b.clone();
        let d = e.clone() + c.clone();
        let f = Value::from(-2.0);
        let l = d.clone() * f.clone();
        l.backward();

        assert_eq!(l.gradient(), 1.0);
        assert_eq!(d.gradient(), -2.0);
        assert_eq!(f.gradient(), 4.0);
        assert_eq!(c.gradient(), -2.0);
        assert_eq!(e.gradient(), -2.0);
        assert_eq!(a.gradient(), 6.0);
        assert_eq!(b.gradient(), -4.0);
    }

    #[test]
    fn test_backprop_neuron() {
        let x1 = Value::from(2.0);
        let x2 = Value::from(0.0);
        let w1 = Value::from(-3.0);
        let w2 = Value::from(1.0);
        let b = Value::from(6.881_373_587_019_543);

        let x1w1 = x1.clone() * w1.clone();
        let x2w2 = x2.clone() * w2.clone();
        let x1w1x2w2 = x1w1.clone() + x2w2.clone();
        let n = x1w1x2w2 + b.clone();
        let o = n.tanh();
        o.backward();

        assert_abs_diff_eq!(o.gradient(), 1.0, epsilon = 0.001);
        assert_abs_diff_eq!(w1.gradient(), 1.0, epsilon = 0.001);
        assert_eq!(w2.gradient(), 0.0);
        assert_abs_diff_eq!(x1.gradient(), -1.5, epsilon = 0.001);
        assert_abs_diff_eq!(x2.gradient(), 0.5, epsilon = 0.001);
        assert_abs_diff_eq!(n.gradient(), 0.5, epsilon = 0.001);
        assert_abs_diff_eq!(b.gradient(), 0.5, epsilon = 0.001);
        assert_abs_diff_eq!(x2w2.gradient(), 0.5, epsilon = 0.001);
        assert_abs_diff_eq!(x1w1.gradient(), 0.5, epsilon = 0.001);
    }

    #[test]
    fn test_backprop_neuron_from_components() {
        let x1 = Value::from(2.0);
        let x2 = Value::from(0.0);
        let w1 = Value::from(-3.0);
        let w2 = Value::from(1.0);
        let b = Value::from(6.881_373_587_019_543);

        let x1w1 = x1.clone() * w1.clone();
        let x2w2 = x2.clone() * w2.clone();
        let x1w1x2w2 = x1w1.clone() + x2w2.clone();
        let n = x1w1x2w2 + b.clone();
        let e = (Value::from(2.0) * n.clone()).exp();
        let o = (e.clone() - Value::from(1.0)) / (e + Value::from(1.0));
        o.backward();

        assert_abs_diff_eq!(o.gradient(), 1.0, epsilon = 0.001);
        assert_abs_diff_eq!(n.gradient(), 0.5, epsilon = 0.001);
        assert_abs_diff_eq!(w1.gradient(), 1.0, epsilon = 0.001);
        assert_eq!(w2.gradient(), 0.0);
        assert_abs_diff_eq!(x1.gradient(), -1.5, epsilon = 0.001);
        assert_abs_diff_eq!(x2.gradient(), 0.5, epsilon = 0.001);
        assert_abs_diff_eq!(n.gradient(), 0.5, epsilon = 0.001);
        assert_abs_diff_eq!(b.gradient(), 0.5, epsilon = 0.001);
        assert_abs_diff_eq!(x2w2.gradient(), 0.5, epsilon = 0.001);
        assert_abs_diff_eq!(x1w1.gradient(), 0.5, epsilon = 0.001);
    }
}
