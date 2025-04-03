#!/usr/bin/env python3

import argparse
import pathlib
import random
import shutil
import subprocess

import cv2
import numpy as np
from onnxruntime.quantization import (  # type: ignore
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,  # type: ignore
)
from ultralytics import YOLO  # type: ignore

# MODEL_PATH = pathlib.Path(R"models/ten_gestures.pt")
# DATASET = pathlib.Path(R"/home/hwang6/data/hagrid_yolo_format")


class ProgramArguments(argparse.Namespace):
    model: pathlib.Path
    dataset: pathlib.Path


class DataReader(CalibrationDataReader):
    def __init__(self, dataset: pathlib.Path):
        self.idx = 0
        self.input_name = "images"

        self.image_paths: list[pathlib.Path] = []
        for c in (dataset / "val").iterdir():
            self.image_paths.extend(random.sample(list(c.iterdir()), 10))

    def preprocess(self, imgPath: pathlib.Path):
        # Same preprocessing that you do before feeding it to the model
        frame = cv2.imread(str(imgPath))
        X = cv2.resize(frame, (640, 640))
        image_data = np.array(X).astype(np.float32) / 255.0  # Normalize to [0, 1] range
        image_data = np.transpose(image_data, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        image_data = np.expand_dims(image_data, axis=0)  # Add batch dimension
        return image_data

    def get_next(self):  # type: ignore
        # method to iterate through the data set
        if self.idx >= len(self.image_paths):
            return None

        image_path = self.image_paths[self.idx]
        print(f"{self.idx}: {image_path}")
        input_data = self.preprocess(image_path)
        self.idx += 1
        return {self.input_name: input_data}


def getArgs() -> ProgramArguments:
    parser = argparse.ArgumentParser("quantization.py")
    parser.add_argument(
        "model", type=pathlib.Path, help="Path to model for quantization"
    )
    parser.add_argument(
        "dataset",
        type=pathlib.Path,
        help="Path to dataset root to create calibration dataset.",
    )
    return parser.parse_args(namespace=ProgramArguments())


def integerQuantize(args: ProgramArguments, model: YOLO) -> None:
    onnxPath = args.model.with_suffix(".onnx")
    preprocessPath = onnxPath.with_name(onnxPath.stem + "_preprocessed.onnx")
    quantizedPath = onnxPath.with_name(onnxPath.stem + "_int8.onnx")

    if quantizedPath.exists():
        return

    # Preprocess model if needed
    if not preprocessPath.exists():
        model.export(format="onnx")

        subprocess.check_call(
            [
                "python",
                "-m",
                "onnxruntime.quantization.preprocess",
                "--input",
                onnxPath,
                "--output",
                preprocessPath,
            ]
        )

    # Quantize
    print("Quantizing to int8")
    quantize_static(
        preprocessPath,
        quantizedPath,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        calibration_data_reader=DataReader(args.dataset),
        quant_format=QuantFormat.QDQ,
        nodes_to_exclude=[
            "/model.23/dfl/Reshape",
            "/model.23/dfl/Transpose",
            "/model.23/dfl/Softmax",
            "/model.23/dfl/conv/Conv",
            "/model.23/dfl/Reshape_1",
            "/model.23/Slice",
            "/model.23/Slice_1",
            "/model.23/Sub",
            "/model.23/Add_1",
            "/model.23/Sub_1",
            "/model.23/Add_2",
            "/model.23/Div_1",
            "/model.23/Concat_4",
            "/model.23/Mul_2",
            "/model.23/Sigmoid",
            "/model.23/Concat_5",
        ],
        per_channel=False,
        reduce_range=True,
    )
    pass


def main() -> None:
    args = getArgs()
    model = YOLO(args.model)

    if not pathlib.Path(R"models/ten_gestures_full.onnx").exists():
        model.export(
            format="onnx",
            device=0,
            data=args.dataset / "hagrid.yaml",
        )
        shutil.move("models/ten_gestures.onnx", "models/ten_gestures_full.onnx")

    if not pathlib.Path(R"models/ten_gestures_half.onnx").exists():
        model.export(
            format="onnx",
            half=True,
            device=0,
            data=args.dataset / "hagrid.yaml",
        )
        shutil.move("models/ten_gestures.onnx", "models/ten_gestures_half.onnx")

    if (
        not pathlib.Path(R"models/ten_gestures_full.tflite").exists()
        or not pathlib.Path(R"models/ten_gestures_half.tflite").exists()
    ):
        model.export(
            format="tflite",
            device=0,
            data=args.dataset / "hagrid.yaml",
        )
        shutil.move(
            "models/ten_gestures_saved_model/ten_gestures_float32.tflite",
            "models/ten_gestures_full.tflite",
        )
        shutil.move(
            "models/ten_gestures_saved_model/ten_gestures_float16.tflite",
            "models/ten_gestures_half.tflite",
        )

    integerQuantize(args, model)


if __name__ == "__main__":
    main()
