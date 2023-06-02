use petgraph::dot::{self, Dot};
use petgraph::graph;
use petgraph::prelude::DiGraphMap;
use prost::Message;
use std::cmp::Reverse;
use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use std::{fs, os};


pub mod onnx_proto {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}


use onnx_proto::{NodeProto, TensorProto, TypeProto, ValueInfoProto};

use crate::summary::{OnnxSummary, self, OperatorUsageSummary, OperatorUsage};

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

pub enum ValueSource {
    Node(NodeId),
    Initializer(InitId),
}

pub struct ValueInfo {
    pub proto: ValueInfoProto,
    pub source: Option<ValueSource>,
}

impl ValueInfo {
    pub fn name(&self) -> &str {
        &self.proto.name
    }

    pub fn type_info(&self) -> TypeInfo {
        TypeInfo(self.proto.r#type.as_ref().unwrap())
    }
}

#[derive(Debug, Copy, Clone)]
struct DataTypeDisplay(onnx_proto::tensor_proto::DataType);

impl fmt::Display for DataTypeDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            onnx_proto::tensor_proto::DataType::Undefined => write!(f, "undefined"),
            onnx_proto::tensor_proto::DataType::Float => write!(f, "f32"),
            onnx_proto::tensor_proto::DataType::Uint8 => write!(f, "u8"),
            onnx_proto::tensor_proto::DataType::Int8 => write!(f, "i8"),
            onnx_proto::tensor_proto::DataType::Uint16 => write!(f, "u16"),
            onnx_proto::tensor_proto::DataType::Int16 => write!(f, "i16"),
            onnx_proto::tensor_proto::DataType::Int32 => write!(f, "i32"),
            onnx_proto::tensor_proto::DataType::Int64 => write!(f, "i64"),
            onnx_proto::tensor_proto::DataType::String => write!(f, "string"),
            onnx_proto::tensor_proto::DataType::Bool => write!(f, "bool"),
            onnx_proto::tensor_proto::DataType::Float16 => write!(f, "f16"),
            onnx_proto::tensor_proto::DataType::Double => write!(f, "f64"),
            onnx_proto::tensor_proto::DataType::Uint32 => write!(f, "u32"),
            onnx_proto::tensor_proto::DataType::Uint64 => write!(f, "u64"),
            onnx_proto::tensor_proto::DataType::Complex64 => write!(f, "complex64"),
            onnx_proto::tensor_proto::DataType::Complex128 => write!(f, "complex128"),
            onnx_proto::tensor_proto::DataType::Bfloat16 => write!(f, "bfloat16"),
            onnx_proto::tensor_proto::DataType::Float8e4m3fn => write!(f, "f8e4m3fn"),
            onnx_proto::tensor_proto::DataType::Float8e4m3fnuz => write!(f, "f8e4m3fnuz"),
            onnx_proto::tensor_proto::DataType::Float8e5m2 => write!(f, "f8e5m2"),
            onnx_proto::tensor_proto::DataType::Float8e5m2fnuz => write!(f, "f8e5m2fnuz"),
        }
    }
}

pub struct TypeInfo<'a>(&'a TypeProto);

impl<'a> fmt::Display for TypeInfo<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0.value.as_ref().unwrap() {
            onnx_proto::type_proto::Value::TensorType(tensor) => {
                let elem_type = onnx_proto::tensor_proto::DataType::from_i32(tensor.elem_type).unwrap();

                write!(f, "{}", DataTypeDisplay(elem_type))?;

                let empty_shape = Vec::new();
                let shape = tensor
                    .shape
                    .as_ref()
                    .map_or(&empty_shape, |shape| &shape.dim);

                if shape.len() > 0 {
                    write!(f, "[")?;

                    let mut dim_iter = tensor.shape.as_ref().unwrap().dim.iter().peekable();

                    while let Some(d) = dim_iter.next() {
                        match d.value.as_ref() {
                            Some(onnx_proto::tensor_shape_proto::dimension::Value::DimValue(val)) => {
                                write!(f, "{}", val)?
                            }
                            Some(onnx_proto::tensor_shape_proto::dimension::Value::DimParam(name)) => {
                                write!(f, "{}", name)?
                            }
                            None => {
                                write!(f, "?")?
                            },
                        }

                        if dim_iter.peek().is_some() {
                            write!(f, ",")?;
                        }
                    }

                    write!(f, "]")?;
                }

                Ok(())
            }
            onnx_proto::type_proto::Value::SequenceType(seq) => {
                write!(f, "sequence<")?;

                if let Some(elem) = seq.elem_type.as_deref() {
                    write!(f, "{}", TypeInfo(elem))?;
                } else {
                    write!(f, "??")?;
                }

                write!(f, ">")
            }
            onnx_proto::type_proto::Value::MapType(map) => {
                let key =
                    DataTypeDisplay(onnx_proto::tensor_proto::DataType::from_i32(map.key_type).unwrap());

                write!(f, "map<{},", key)?;

                if let Some(elem) = map.value_type.as_deref() {
                    write!(f, "{}", TypeInfo(elem))?;
                } else {
                    write!(f, "??")?;
                }

                write!(f, ">")
            }
            onnx_proto::type_proto::Value::OptionalType(opt) => {
                write!(f, "optional<")?;
                if let Some(elem) = opt.elem_type.as_deref() {
                    write!(f, "{}", TypeInfo(elem))?;
                }
                else {
                    write!(f, "??")?;
                }
                write!(f, ">")
            }
            onnx_proto::type_proto::Value::SparseTensorType(_) => todo!(),
        }
    }
}

struct NodeInfo {
    proto: NodeProto,
}

pub struct OnnxModel {
    pub proto: onnx_proto::ModelProto,
    values: IdMapper<ValueInfo>,
    nodes: Vec<NodeInfo>,
    node_graph: DiGraphMap<usize, usize>,
    inputs: Vec<ValueId>,
    outputs: Vec<ValueId>,
}

impl <'a> From<&'a ValueInfo> for summary::Value<'a> {
    fn from(value: &'a ValueInfo) -> Self {
        summary::Value { name: value.name(), ty: value.type_info() }
    }
}

impl OnnxModel {
    pub fn graph_proto(&self) -> &onnx_proto::GraphProto {
        self.proto.graph.as_ref().unwrap()
    }

    pub fn inputs(&self) -> impl Iterator<Item = &ValueInfo> {
        self.inputs
            .iter()
            .copied()
            .map(|value_id| self.values.get_by_id(value_id))
    }

    pub fn outputs(&self) -> impl Iterator<Item = &ValueInfo> {
        self.outputs
            .iter()
            .copied()
            .map(|value_id| self.values.get_by_id(value_id))
    }

    pub fn from_bytes<B>(model_bytes: B) -> anyhow::Result<Self>
        where B: prost::bytes::Buf
    {
        let model_proto = onnx_proto::ModelProto::decode(model_bytes)?;

        let model = OnnxModel::from_proto(model_proto);

        Ok(model)
    }

    fn from_proto(proto: onnx_proto::ModelProto) -> Self {
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

    pub fn summary<'a>(&'a self) -> OnnxSummary<'a> {
        let mut node_counts = HashMap::new();

        for node in self.nodes.iter() {
            let count = node_counts.entry((&node.proto.domain, node.proto.op_type.as_str())).or_default();
            *count += 1;
        }

        let mut operators: Vec<OperatorUsage> = node_counts.into_iter().map(|((domain, name), count)| OperatorUsage {
            domain:  if domain == "" { "ai.onnx" } else { domain },
            name,
            count,
        }).collect();

        operators.sort_by_key(|op| Reverse(op.count));

        let operator_summary = OperatorUsageSummary {
            operators
        };

        OnnxSummary {
            domain: &self.proto.domain,
            name: &self.graph_proto().name,
            version: self.proto.model_version,
            doc_string: &self.proto.doc_string,
            producer_name: &self.proto.producer_name,
            producer_version: &self.proto.producer_version,
            ir_version: self.proto.ir_version,
            opsets: self.proto.opset_import.iter().map(|opset| {
                summary::OnnxOpset {
                    name: if opset.domain == "" { "ai.onnx" } else { &opset.domain },
                    version: opset.version,
                }
            }).collect(),
            // Filter out inputs that have initializers or node inputs etc...
            inputs: self.inputs().filter(|v| v.source.is_none()).map(summary::Value::from).collect(),
            outputs: self.outputs().map(summary::Value::from).collect(),
            operator_summary
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