use std::ops::AddAssign;
use std::{
    cell::RefCell,
    collections::HashSet,
    fmt::Debug,
    fmt::Display,
    hash::Hash,
    iter::Sum,
    ops::{Add, Deref, Div, Mul, Sub},
    rc::Rc,
};

#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    Add,
    Multiply,
    Tanh,
    Exponent,
    Pow,
    Relu,
}

/// Implementation of an equation value
///
/// Actual is kept internally so that Value can be freely cloned
/// without actually copying data
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Value(Rc<RefCell<ValueInternal>>);

/// Internal holder of Value information
pub struct ValueInternal {
    pub data: f64,
    pub children: Vec<Value>,
    pub operation: Option<Operation>,
    pub gradient: f64,
    pub label: Option<String>,
    pub uuid: String,
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

impl Debug for ValueInternal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValueInternal")
            .field("data", &self.data)
            .field("gradient", &self.gradient)
            .field("label", &self.label)
            .field("operation", &self.operation)
            .finish()
    }
}

impl Value {
    fn new(value: ValueInternal) -> Self {
        Value(Rc::new(RefCell::new(value)))
    }

    /// Access the internal f64 of the Value
    pub fn data(&self) -> f64 {
        self.borrow().data
    }

    /// Set the internal f64 of the Value
    pub fn set_data(&self, value: f64) {
        self.borrow_mut().data = value;
    }

    /// The gradient of the current value
    ///
    /// Will be 0. until the value is included as part of the `backward` call
    pub fn gradient(&self) -> f64 {
        self.borrow().gradient
    }

    /// Zero the gradient value
    pub fn zero_grad(&self) {
        self.borrow_mut().gradient = 0.;
    }

    /// The child nodes of the Value
    pub fn children(&self) -> Vec<Value> {
        self.borrow().children.clone()
    }

    /// The operation that created this node (if any)
    pub fn operation(&self) -> Option<Operation> {
        self.borrow().operation
    }

    /// Apply the tanh operation to the node, creating a new Value
    pub fn tanh(self) -> Value {
        let d = self.borrow().data.tanh();
        Value::new(ValueInternal::new(
            d,
            vec![self],
            Some(Operation::Tanh),
            None,
        ))
    }

    /// Apply the exp operation to the node, creating a new Value
    pub fn exp(self) -> Self {
        let d = self.borrow().data.exp();
        Value::new(ValueInternal::new(
            d,
            vec![self],
            Some(Operation::Exponent),
            None,
        ))
    }

    /// Apply the powf operation to the node, creating a new Value
    pub fn powf(self, value: Value) -> Self {
        let d = self.data().powf(value.data());
        Value::new(ValueInternal::new(
            d,
            vec![self, value],
            Some(Operation::Pow),
            None,
        ))
    }

    /// Apply the relu operation to the node
    pub fn relu(self) -> Self {
        let d = self.data().max(0.0);
        Value::new(ValueInternal::new(
            d,
            vec![self],
            Some(Operation::Relu),
            None,
        ))
    }
    /// Apply backward propagation of the gradient for this Value and all children in our graph
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

                self.children()[0].borrow_mut().gradient += second * self.gradient();
                self.children()[1].borrow_mut().gradient += first * self.gradient();
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
            Some(Operation::Pow) => {
                let base = &self.children()[0];
                let power = &self.children()[1];
                base.borrow_mut().gradient +=
                    power.data() * base.data().powf(power.data() - 1.0) * self.gradient();
            }
            Some(Operation::Relu) => {
                for child in self.children().iter_mut() {
                    child.borrow_mut().gradient += if self.data() > 0. {
                        self.gradient()
                    } else {
                        0.
                    }
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
        self.uuid.hash(state)
    }
}
impl ValueInternal {
    fn new(
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
            uuid: uuid::Uuid::new_v4().to_string(),
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
        Value::new(ValueInternal::new(
            self.data() + rhs.data(),
            vec![rhs, self],
            Some(Operation::Add),
            None,
        ))
    }
}

impl AddAssign for Value {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}
impl Add<f64> for Value {
    type Output = Value;

    fn add(self, rhs: f64) -> Self::Output {
        self + Value::from(rhs)
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

impl Mul<f64> for Value {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        self * Value::from(rhs)
    }
}

impl Div for Value {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.powf(Value::from(-1.0))
    }
}

impl Div<f64> for Value {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        self / Value::from(rhs)
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
impl Sum for Value {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut sum = Value::from(0.0);
        for value in iter {
            sum += value
        }
        sum
    }
}

impl Eq for ValueInternal {}
#[cfg(test)]
mod tests {
    use crate::value::Value;
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
        let o = n.clone().tanh();
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
        assert_abs_diff_eq!(b.gradient(), 0.5, epsilon = 0.001);
        assert_abs_diff_eq!(x2w2.gradient(), 0.5, epsilon = 0.001);
        assert_abs_diff_eq!(x1w1.gradient(), 0.5, epsilon = 0.001);
    }
}
