| Run | Change | mAP@50 | mAP@50-95 | AP50 D00 | AP50 D10 | AP50 D20 | AP50 D40 | Train min |
|---|---|---|---|---|---|---|---|---|
| baseline | Phase 2 baseline: YOLOv8n, 640px, Ultralytics defaults except epochs | 0.342 | 0.140 | 0.279 | 0.036 | 0.605 | 0.448 | 188.5 |
| exp_v8s | Exp 1 model size: v8s vs baseline v8n | 0.360 | 0.148 | 0.352 | 0.004 | 0.621 | 0.463 | 54.5 |
| exp_960 | Exp 2 image size: 960 vs 640 (thin cracks). Source images are 720px, so 960 upsamples. | 0.326 | 0.132 | 0.286 | 0.026 | 0.572 | 0.420 | 75.7 |
| exp_aug | Exp 3 augmentations: road-specific albumentations + darker HSV | 0.322 | 0.137 | 0.272 | 0.023 | 0.600 | 0.394 | 103.4 |
| exp_rfs | Exp 4 class imbalance: repeat-factor oversampling of minority classes | 0.339 | 0.142 | 0.294 | 0.005 | 0.624 | 0.434 | 31.8 |
| exp_multi | Exp 5 data: India + Japan/Czech/US train; val/test India only | 0.376 | 0.164 | 0.384 | 0.080 | 0.615 | 0.425 | 196.3 |
