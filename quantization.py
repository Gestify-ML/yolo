#!/usr/bin/env python3

import pathlib
import random
import shutil
import subprocess

import cv2
import numpy as np
import numpy.typing as npt
from onnxruntime.quantization import (  # type: ignore
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,  # type: ignore
)
from ultralytics import YOLO  # type: ignore

MODEL_PATH = pathlib.Path(R"models/ten_gestures.pt")
DATASET = pathlib.Path(R"/home/hwang6/data/hagrid_yolo_format")


class ImageCalibrationDataReader(CalibrationDataReader):
    def __init__(self):
        self.idx = 0
        self.inputName = "images"

        # Select 100 random images from each class
        classes = (DATASET / "test").iterdir()
        self.imagePaths: list[pathlib.Path] = []
        for c in classes:
            self.imagePaths.extend(random.sample(list(c.iterdir()), 100))

    def preprocess(self, path: pathlib.Path) -> npt.NDArray[np.float32]:
        # Same preprocessing that you do before feeding it to the model
        frame = cv2.imread(str(path))
        x = cv2.resize(frame, (640, 640))
        imageData = np.array(x).astype(np.float32) / 255.0  # Normalize to [0, 1] range
        imageData = np.transpose(imageData, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        imageData = np.expand_dims(imageData, axis=0)  # Add batch dimension
        return imageData

    def get_next(self) -> dict[str, npt.NDArray[np.float32]]:
        # method to iterate through the data set
        if self.idx >= len(self.imagePaths):
            return None  # type: ignore

        imagePath = self.imagePaths[self.idx]
        inputData = self.preprocess(imagePath)
        self.idx += 1
        print(f"Done: {self.idx}")
        return {self.inputName: inputData}

    def __len__(self):
        return len(self.imagePaths)


def main() -> None:
    random.seed("315e395e-e4f5-444d-9b7e-c98c82f0b0b6")

    # Preprocess model if needed
    onnxPath = MODEL_PATH.with_suffix(".onnx")
    preprocessPath = onnxPath.with_name(onnxPath.stem + "_preprocessed")
    quantizedPath = onnxPath.with_name(onnxPath.stem + "_quantized")
    if not preprocessPath.exists():
        model = YOLO(MODEL_PATH)
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
    quantize_static(
        preprocessPath,
        quantizedPath,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        calibration_data_reader=ImageCalibrationDataReader(),
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


def main2() -> None:
    model = YOLO(MODEL_PATH)

    if not pathlib.Path(R"models/ten_gestures_full.onnx").exists():
        model.export(
            format="onnx",
            device=0,
            data=DATASET / "hagrid.yaml",
        )
        shutil.move("models/ten_gestures.onnx", "models/ten_gestures_full.onnx")

    if not pathlib.Path(R"models/ten_gestures_half.onnx").exists():
        model.export(
            format="onnx",
            half=True,
            device=0,
            data=DATASET / "hagrid.yaml",
        )
        shutil.move("models/ten_gestures.onnx", "models/ten_gestures_half.onnx")

    if (
        not pathlib.Path(R"models/ten_gestures_full.tflite").exists()
        or not pathlib.Path(R"models/ten_gestures_half.tflite").exists()
    ):
        model.export(
            format="tflite",
            device=0,
            data=DATASET / "hagrid.yaml",
        )
        shutil.move(
            "models/ten_gestures_saved_model/ten_gestures_float32.tflite",
            "models/ten_gestures_full.tflite",
        )
        shutil.move(
            "models/ten_gestures_saved_model/ten_gestures_float16.tflite",
            "models/ten_gestures_half.tflite",
        )

    # if not pathlib.Path(R"models/ten_gestures_full.onnx").exists() and True:
    # model.export(
    #     format="tflite",
    #     int8=True,
    #     device=0,
    #     data=DATASET / "hagrid.yaml",
    # )
    # shutil.move("models/ten_gestures.tflite", "models/ten_gestures_int8.tflite")
    pass


if __name__ == "__main__":
    main2()
