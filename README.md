# Yolo
Gestify's hand recognition model is based on Ultralytics' detection model, and was trained on a subset of HaGRID's gestures dataset. The trained model was then pruned and quantized for running on an Android device.

## Architecture
* Architecture diagrams
* Appropriate model size for device constraints
* Efficient layer organization
* Memory optimization considerations

## Implementation
The trained model is based on Ultralytics's Yolo11n detection model, and is trained on Hagrid's hand gesture  dataset.

### Quantization
Quantization is important as a post processing step to increase inference speed and to reduce memory usage. Following the workflow from Ultralytics training framework, quantization is done when the model is exported to other formats.

The model originally has elements in 32-bit floating point format. We use post-training quantization to convert the model into 16-bit floating point, and 8-bit integer variants. When quantizing to 8-bit integers, we need a calibration dataset which is representative of the real-world usage. We construct this dataset by taking a subset of the images in each gesture class from the hagrid dataset. By default we randomly select 10 images from each gesture class.

TODO:
* Pruning techniques applied
* Model compression methods
* Successful deployment on target device
* Hardware-specific optimizations
* Error handling implementation

## Optimization Decision Reasoning

## Setup and Training
The code base is split into 4 distinct repositories. The [Hagrid](https://github.com/Gestify-ML/hagrid) repo contains the dataset acquisition and preparation scripts. It was forked and modified to allow it to covert from the hagrid dataset format to the yolo dataset format. The [Ultralytics](https://github.com/Gestify-ML/ultralytics) repo contains the training framework. It was forked and modified to use the dataset from the converted hagrid dataset. The [Gestify](https://github.com/Gestify-ML/gestify) repo contains the Gestify app which runs the trained model. And lastly, the [Yolo](https://github.com/Gestify-ML/yolo) repo contains the scripts to train, prune, quantize, and export the hand detection model. It also contains a desktop demo to run a model.

The yolo repo has the following scripts:
* `train.py`, trains or resumes training for a model. The dataset and base model are configurable through command line arguments. By default, the base model is the `yolo11n.pt` detection model from Ultralytics.
* `quantization.py`, quantizes the trained model into several sizes and format. Formats being, ONNX in float32, float16, and int8. And tensorflow lite in float32 and float16. When quantizing to int8, 10 images from each class is selected as the calibration dataset.
* TODO Fine prune and coarse prune scripts

### Dataset and Training Environment Setup
To use the training scripts in this repo, the dataset needs to be setup first.
1. Download and unzip the lightweight version of the HaGRID dataset from the repo link, along with the annotations. 
2. Clone the Hagrid repo from the organization and cd into it.
3. Create the virtual environment
   1. uv venv -p 3.11.11
   2. source .venv/bin/activate
   3. uv install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   4. uv pip install omegaconf mediapipe albumentations torchmetrics tensorflow pandas tdqm pycocotools
4. Modify the "converters/converter_config.yaml" file.
   1. Change "dataset_annotations" and "dataset_folder" to the unzipped archive directories.
   2. Select which targets are desired.
      * More targets take longer to train.
      * Order matters, when trying to add more targets after converting. Make sure to maintain strict order when adding more. Removing targets from an already converted dataset requires extensive fixups, it’s probably easier to just convert from scratch.
5. Run the conversion script
   1. python -m converters.hagrid_to_yolo
      * Check script arguments with "-h" if needed,
      * --cfg, if you want to use a different config file than "converters/converter_config.yaml"
      * --out, if you want to change the output directory
      * --mode, should be left as the default, "gestures"

Next to set up the environment for this repo, follow the following steps:
1. Clone yolo training repo from the organization and cd into it
2. Create virtual environment
   1. uv venv -p 3.12
   2. Source ./venv/bin/activate
3. Install customized Ultralytics from the organization
   1. uv pip install git+https://github.com/Gestify-ML/ultralytics.git
4. Enter screen session with "tmux", take note of which login node (login-01, login-02, etc) you are logged into as you can only reconnect on the same one. If you ever get disconnected use "tmux a" to reconnect. To disconnect from a session, use "ctrl-b” and then "d"
5. Enter an interactive job. The following requests one node, with 16 cores, 128GB of ram, 24hrs, and 2 gpus
   1. srun -N 1 -n 16 --mem=131072 --time=1440 --partition=academic --gres=gpu:2 --pty /usr/bin/bash
6. Enable virtual environment again, make sure you’re in the yolo directory,
   1. Source ./.venv/bin/activate
7. Run training script train.py, check options available with "-h"

### Running Scripts
`train.py`, is used for training a model, it has the following arguments,
* `name`, A name for the model
* `--dataset`, The dataset to use, this must be given if not resuming.
* `--project`, The directory to store the model under.
* `--base-model`, The base model to train from. This can be one from Ultralytics like `yolo11n.pt` or a local model, say one that was pruned.
* `--resume`, resume training a model if it was stopped or interrupted. This should not be used if training a model that was just pruned, in that case treat it as a base model.

---

`quantization.py`, is used for quantizing and exporting a model into several format. It has the following arguments,
* `model`, path to the model to quantize. The outputs from this script will be placed into the same directory.
* `dataset`, path to the dataset to use for calibration data.

The artifacts from the script are as follows,
* `[model_name]_full.onnx`, float32 in ONNX format
* `[model_name]_half.onnx`, float16 in ONNX format
* `[model_name]_int8.onnx`, int8 in ONNX format
* `[model_name]_full.tflite`, float32 in tensorflow lite format
* `[model_name]_half.tflite`, float16 in tensorflow lite format

---
TODO pruning scripts descriptions

TODO
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