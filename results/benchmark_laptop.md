CPU: AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD (16 threads), batch 1, 536 test images, imgsz 640

| Format | mAP@50 | mAP@50-95 | Δ mAP@50 | p50 ms | p95 ms | FPS | Size MB | Peak RAM MB |
|---|---|---|---|---|---|---|---|---|
| PyTorch FP32 | 0.359 | 0.154 | -0.000 | 24.1 | 29.6 | 40.6 | 6.5 | 792 |
| ONNX fp16 | 0.358 | 0.154 | -0.001 | 28.6 | 33.9 | 35.4 | 6.4 | 171 |
| ONNX fp32 | 0.358 | 0.153 | -0.001 | 25.4 | 30.1 | 39.9 | 12.8 | 153 |
| ONNX int8 | 0.000 | 0.000 | -0.359 | 15.1 | 18.7 | 63.7 | 3.6 | 112 |
| ONNX int8_head | 0.305 | 0.130 | -0.053 | 21.7 | 26.7 | 45.3 | 6.2 | 123 |
