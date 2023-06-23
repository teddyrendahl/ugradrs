pub mod nn;
pub mod value;

#[cfg(feature ="draw_graph")]
pub mod draw_dot {
    use crate::value::Value;
    use petgraph::dot::{Config, Dot};
    use petgraph::graph::DiGraph;
    use std::collections::{HashMap, HashSet};
    use std::fs::File;
    use std::io;
    use std::io::Write;

    /// Build a set of Node and Edges for the Graph
    fn trace_graph(v: Value) -> (HashSet<Value>, HashSet<(Value, Value)>) {
        let mut nodes = HashSet::new();
        let mut edges = HashSet::new();

        fn build_graph(v: Value, nodes: &mut HashSet<Value>, edges: &mut HashSet<(Value, Value)>) {
            if !nodes.contains(&v) {
                nodes.insert(v.clone());
                for child in v.children() {
                    edges.insert((child.clone(), v.clone()));
                    build_graph(child, nodes, edges)
                }
            }
        }
        build_graph(v, &mut nodes, &mut edges);
        (nodes, edges)
    }

    /// Create a DiGraph based on DAG leading to a Value
    fn create_graph(v: Value) -> DiGraph<String, ()> {
        let (nodes, edges) = trace_graph(v);
        let mut g = DiGraph::new();
        let mut op_graph = HashMap::new();
        let mut node_graph = HashMap::new();

        for n in nodes {
            let idx = g.add_node(format!("{{ data {:.4} | grad {:.4} }}", n.data(), n.gradient()));
            node_graph.insert(n.clone(), idx);
            // A node that is the result of an operation has a separate bubble to connect to
            if let Some(op) = n.operation() {
                let op_idx = g.add_node(op.into());
                g.add_edge(op_idx, idx, ());
                op_graph.insert(n, op_idx);
            }
        }
        for (child, parent) in edges {
            let idx = *op_graph.get(&parent).unwrap();
            g.add_edge(*node_graph.get(&child).unwrap(), idx, ());
        }
        g
    }

    /// Create a dot file description of the DAG that leads to Value
    pub fn draw_dot(v: Value, filename: &str) -> Result<(), io::Error> {
        let g = create_graph(v);
        let mut dot = format!("{:?}", Dot::with_config(&g, &[Config::EdgeNoLabel]));
        // Hack output dot file for options not availabel in petgraph
        dot = dot.replace("\\\"", "");
        dot.insert_str(10, "    rankdir=\"LR\"");
        dot.insert_str(10, "    node [shape=record]\n");
        let mut f = File::create(filename)?;
        f.write_all(dot.as_bytes())
    }
}

#[cfg(doctest)]
mod test_readme {
    macro_rules! external_doc_test {
        ($x:expr) => {
            #[doc = $x]
            extern "C" {}
        };
    }

    external_doc_test!(include_str!("../README.md"));
}
