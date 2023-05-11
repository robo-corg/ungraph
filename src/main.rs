use petgraph::dot::{self, Dot};
use petgraph::graph;
use petgraph::prelude::DiGraphMap;
use prost::Message;
use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use std::{fs, os};

pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

use clap::Parser;

use crate::onnx::{NodeProto, TensorProto, TypeProto, ValueInfoProto};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Model file to load
    model_file: PathBuf,
}

type ValueId = usize;
type NodeId = usize;
type InitId = usize;

struct IdMapper<T> {
    mapping: HashMap<String, usize>,
    values: Vec<T>,
}

impl<T> IdMapper<T> {
    fn insert(&mut self, name: &str, value: T) -> usize {
        let id = self.values.len();
        self.values.push(value);
        self.mapping.insert(name.to_string(), id);
        id
    }

    fn get_by_name_mut(&mut self, name: &str) -> Option<&mut T> {
        self.mapping.get(name).map(|id| &mut self.values[*id])
    }

    fn new() -> Self {
        IdMapper {
            mapping: Default::default(),
            values: Default::default(),
        }
    }

    fn get_by_name(&self, name: &str) -> Option<&T> {
        self.mapping.get(name).map(|id| &self.values[*id])
    }

    fn get_id_by_name(&self, name: &str) -> Option<usize> {
        self.mapping.get(name).copied()
    }

    fn get_by_id(&self, value_id: usize) -> &T {
        &self.values[value_id]
    }
}

enum ValueSource {
    Node(NodeId),
    Initializer(InitId),
}

struct ValueInfo {
    proto: ValueInfoProto,
    source: Option<ValueSource>,
}

impl ValueInfo {
    fn name(&self) -> &str {
        &self.proto.name
    }

    fn type_info(&self) -> TypeInfo {
        TypeInfo(self.proto.r#type.as_ref().unwrap())
    }
}

#[derive(Debug, Copy, Clone)]
struct DataTypeDisplay(onnx::tensor_proto::DataType);

impl fmt::Display for DataTypeDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            onnx::tensor_proto::DataType::Undefined => write!(f, "undefined"),
            onnx::tensor_proto::DataType::Float => write!(f, "f32"),
            onnx::tensor_proto::DataType::Uint8 => write!(f, "u8"),
            onnx::tensor_proto::DataType::Int8 => write!(f, "i8"),
            onnx::tensor_proto::DataType::Uint16 => write!(f, "u16"),
            onnx::tensor_proto::DataType::Int16 => write!(f, "i16"),
            onnx::tensor_proto::DataType::Int32 => write!(f, "i32"),
            onnx::tensor_proto::DataType::Int64 => write!(f, "i64"),
            onnx::tensor_proto::DataType::String => write!(f, "string"),
            onnx::tensor_proto::DataType::Bool => write!(f, "bool"),
            onnx::tensor_proto::DataType::Float16 => write!(f, "f16"),
            onnx::tensor_proto::DataType::Double => write!(f, "f64"),
            onnx::tensor_proto::DataType::Uint32 => write!(f, "u32"),
            onnx::tensor_proto::DataType::Uint64 => write!(f, "u64"),
            onnx::tensor_proto::DataType::Complex64 => write!(f, "complex64"),
            onnx::tensor_proto::DataType::Complex128 => write!(f, "complex128"),
            onnx::tensor_proto::DataType::Bfloat16 => write!(f, "bfloat16"),
        }
    }
}

struct TypeInfo<'a>(&'a TypeProto);

impl<'a> fmt::Display for TypeInfo<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0.value.as_ref().unwrap() {
            onnx::type_proto::Value::TensorType(tensor) => {
                let elem_type = onnx::tensor_proto::DataType::from_i32(tensor.elem_type).unwrap();

                write!(f, "{}", DataTypeDisplay(elem_type))?;

                let shape = &tensor.shape.as_ref().unwrap().dim;

                if shape.len() > 0 {
                    write!(f, "[")?;

                    let mut dim_iter = tensor.shape.as_ref().unwrap().dim.iter().peekable();

                    while let Some(d) = dim_iter.next() {
                        match d.value.as_ref().unwrap() {
                            onnx::tensor_shape_proto::dimension::Value::DimValue(val) => {
                                write!(f, "{}", val)?
                            }
                            onnx::tensor_shape_proto::dimension::Value::DimParam(name) => {
                                write!(f, "{}", name)?
                            }
                        }

                        if dim_iter.peek().is_some() {
                            write!(f, ",")?;
                        }
                    }

                    write!(f, "]")?;
                }

                Ok(())
            }
            onnx::type_proto::Value::SequenceType(_) => todo!(),
            onnx::type_proto::Value::MapType(_) => todo!(),
            onnx::type_proto::Value::OptionalType(_) => todo!(),
            onnx::type_proto::Value::SparseTensorType(_) => todo!(),
        }
    }
}

struct NodeInfo {
    proto: NodeProto,
}

struct OnnxModel {
    proto: onnx::ModelProto,
    values: IdMapper<ValueInfo>,
    nodes: Vec<NodeInfo>,
    node_graph: DiGraphMap<usize, usize>,
    inputs: Vec<ValueId>,
    outputs: Vec<ValueId>,
}

impl OnnxModel {
    fn graph_proto(&self) -> &onnx::GraphProto {
        self.proto.graph.as_ref().unwrap()
    }

    fn inputs(&self) -> impl Iterator<Item = &ValueInfo> {
        self.inputs
            .iter()
            .copied()
            .map(|value_id| self.values.get_by_id(value_id))
    }

    fn outputs(&self) -> impl Iterator<Item = &ValueInfo> {
        self.outputs
            .iter()
            .copied()
            .map(|value_id| self.values.get_by_id(value_id))
    }

    fn from_proto(proto: onnx::ModelProto) -> Self {
        let model_graph = proto.graph.as_ref().expect("Model must have graph");

        let mut values = IdMapper::new();
        let mut nodes = Vec::<NodeInfo>::new();
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();

        let mut node_graph: DiGraphMap<usize, usize> = DiGraphMap::new();

        let init_map: HashMap<&str, InitId> = model_graph
            .initializer
            .iter()
            .enumerate()
            .map(|(index, init)| (init.name.as_str(), index))
            .collect();

        for value_info in model_graph.value_info.iter() {
            values.insert(
                &value_info.name,
                ValueInfo {
                    proto: value_info.clone(),
                    source: None,
                },
            );
        }

        for graph_input in model_graph.input.iter() {
            let source = init_map
                .get(graph_input.name.as_str())
                .copied()
                .map(ValueSource::Initializer);

            let value_id = values.insert(
                &graph_input.name,
                ValueInfo {
                    proto: graph_input.clone(),
                    source,
                },
            );

            inputs.push(value_id);
        }

        for graph_output in model_graph.output.iter() {
            let value_id = values.insert(
                &graph_output.name,
                ValueInfo {
                    proto: graph_output.clone(),
                    source: None,
                },
            );

            outputs.push(value_id);
        }

        for (node_index, node) in model_graph.node.iter().enumerate() {
            let node_id = nodes.len();

            nodes.push(NodeInfo {
                proto: node.clone(),
            });

            node_graph.add_node(node_id);

            // for input in node.input.iter() {

            // }

            for input in node.output.iter() {
                if let Some(value_info) = values.get_by_name_mut(input) {
                    value_info.source = Some(ValueSource::Node(node_index));
                }
            }
        }

        for (node_index, node) in model_graph.node.iter().enumerate() {
            for input in node.input.iter() {
                if let Some(value_id) = values.get_id_by_name(input) {
                    let value_info = values.get_by_id(value_id);

                    let maybe_dep_node = value_info.source.as_ref().and_then(|src| match src {
                        ValueSource::Node(node_id) => Some(*node_id),
                        _ => None,
                    });

                    if let Some(dep_node) = maybe_dep_node {
                        node_graph.add_edge(dep_node, node_index, value_id);
                    }
                }
            }
        }

        OnnxModel {
            proto,
            inputs,
            outputs,
            values,
            nodes,
            node_graph,
        }
    }

    fn to_dot(&self) -> impl fmt::Display {
        let dot_config = [dot::Config::NodeNoLabel];

        let node_attr_getter =
            |g, n: (usize, &usize)| format!("label = \"{}\"", &self.nodes[*n.1].proto.name);

        format!(
            "{}",
            Dot::with_attr_getters(
                &self.node_graph,
                &dot_config,
                &|g, e| String::new(),
                &node_attr_getter,
            )
        )
    }
}

fn main() {
    let args = Args::parse();

    let model_bytes = fs::read(&args.model_file).unwrap();

    let model_proto = onnx::ModelProto::decode(model_bytes.as_slice()).unwrap();

    let model = OnnxModel::from_proto(model_proto);

    println!(
        "ONNX Model: {} {} (v{})",
        &model.proto.domain,
        &model.graph_proto().name,
        model.proto.model_version
    );
    if model.proto.doc_string.len() > 0 {
        println!("{}", model.proto.doc_string);
    }
    println!("");

    println!(
        "Producer: {} {}",
        model.proto.producer_name, model.proto.producer_version
    );
    println!("");

    println!("IR Version: {}", model.proto.ir_version);
    if model.proto.opset_import.len() == 1 {
        print!("Opset: ");
    }

    for opset in model.proto.opset_import.iter() {
        let domain_name = if opset.domain == "" {
            "ai.onnx"
        } else {
            opset.domain.as_str()
        };
        println!("{} {}", &domain_name, opset.version);
    }

    println!("");
    println!("Inputs:");

    for input in model.inputs() {
        if input.source.is_none() {
            println!("    {}: {}", input.name(), input.type_info());
        }
    }

    println!("Outputs:");

    for output in model.outputs() {
        println!("    {}: {}", output.name(), output.type_info());
    }

    //dbg!(&model.proto.opset_import);

    //println!("{}", model.to_dot());

    //dbg!(model);

    //dbg!(&model.graph_proto().node);

    //dbg!(&model.graph_proto().initializer);

    // for init in model.graph_proto().initializer.iter() {
    //     println!("{}", init.name);
    // }
}
