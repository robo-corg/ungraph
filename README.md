# Ungraph

Work in progress ml model inspector. Currently only supports onnx and safetensors.

## Example

```
> cargo run gpt2-10.onnx

ONNX Model:  torch-jit-export (v0)

Producer: pytorch 1.4

IR Version: 6
Opset: ai.onnx 10

Inputs:
    input1: i64[input1_dynamic_axes_1,input1_dynamic_axes_2,input1_dynamic_axes_3]
Outputs:
    output1: f32[1,1,8,768]
    output2: f32[2,1,12,8,64]
    output3: f32[2,1,12,8,64]
    output4: f32[2,1,12,8,64]
    output5: f32[2,1,12,8,64]
    output6: f32[2,1,12,8,64]
    output7: f32[2,1,12,8,64]
    output8: f32[2,1,12,8,64]
    output9: f32[2,1,12,8,64]
    output10: f32[2,1,12,8,64]
    output11: f32[2,1,12,8,64]
    output12: f32[2,1,12,8,64]
    output13: f32[2,1,12,8,64]
```
