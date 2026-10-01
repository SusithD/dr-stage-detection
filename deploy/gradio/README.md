---
title: RetinaScan
emoji: 👁️
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.29.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Diabetic retinopathy screening from fundus photographs
---

# RetinaScan

Research prototype that detects diabetic retinopathy and grades it on the five-level ICDR
scale from colour fundus photographs, built for a university Computer Vision coursework.

- Model: ConvNeXt-Base fine-tuned on the DDR dataset (512 px, circle-cropped input).
- Held-out DDR test set: QWK 0.938, accuracy 90.4%, DR sensitivity 95.4%, specificity 96.3%.
- Each photograph gets a quality check, a grade with per-grade probabilities and a Grad-CAM
  heatmap; the worse of the two eyes sets the screening outcome.

This Space uses the Gradio SDK as a Python runtime for the RetinaScan FastAPI app and runs
inference on the CPU, so analysis takes a few seconds per photograph and test-time augmentation
is switched off. **Research prototype only. Not a medical device and not for
clinical use.**

Source code: https://github.com/SusithD/dr-stage-detection
