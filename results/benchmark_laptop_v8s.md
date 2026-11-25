CPU: AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD (16 threads), batch 1, 536 test images, imgsz 640

| Format | mAP@50 | mAP@50-95 | Δ mAP@50 | p50 ms | p95 ms | FPS | Size MB | Peak RAM MB |
|---|---|---|---|---|---|---|---|---|
| PyTorch FP32 | 0.355 | 0.145 | -0.000 | 67.8 | 81.0 | 14.2 | 22.5 | 899 |
| ONNX fp16 | 0.354 | 0.144 | -0.000 | 70.6 | 79.7 | 14.1 | 22.4 | 273 |
| ONNX fp32 | 0.354 | 0.144 | -0.001 | 65.7 | 74.1 | 15.2 | 44.8 | 236 |
| ONNX int8 | 0.000 | 0.000 | -0.355 | 27.3 | 35.3 | 35.6 | 11.6 | 137 |
| ONNX int8_head | 0.336 | 0.134 | -0.019 | 41.1 | 50.2 | 23.8 | 18.0 | 152 |
