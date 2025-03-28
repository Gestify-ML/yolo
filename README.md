# Yolo
Gestify's hand recognition model is based on Ultralytics' detection model, and was trained on a subset of HaGRID's gestures dataset. The trained model was then pruned and quantized for running on an Android device.

## Architecture
* Architecture diagrams
* Appropriate model size for device constraints
* Efficient layer organization
* Memory optimization considerations

## Implementation
* Quantization implementation
* Pruning techniques applied
* Model compression methods
* Successful deployment on target device
* Hardware-specific optimizations
* Error handling implementation

## Optimization Decision Reasoning

## Setup and Training
* Explain source code organization, Hagrid, Ultralytics, yolo, Gestify
* Dataset prep, and env prop for training
* Running scripts for pruning and quantization
* Running demo on PC
* Running demo on Phone

## Benchmark Results
* Inference Speed
  * Latency measurements
  * Throughput analysis
* Resource Utilization
  * Memory usage monitoring
  * CPU/GPU efficiency
  * Battery impact assessment
* Accuracy Metrics (Initial Accuracy of the model will not be evaluated)
  * Model accuracy comparison
  * Performance degradation analysis

## Trade-off Analysis