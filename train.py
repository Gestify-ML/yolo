#!/usr/bin/env python3

import argparse
import pathlib

from ultralytics import YOLO  # type: ignore


class Arguments(argparse.Namespace):
    name: str
    dataset: pathlib.Path | None
    project: pathlib.Path
    base_model: str
    resume: bool


def main() -> None:
    scriptDir = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser("train")
    parser.add_argument("name", type=str, help="Custom model name")
    parser.add_argument(
        "--dataset", type=pathlib.Path, help="Dataset to use if training a new model."
    )
    parser.add_argument(
        "--project",
        type=pathlib.Path,
        default=scriptDir / "runs/detect",
        help="Workspace to store training files.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="yolo11n.pt",
        help="Base model to start training with.",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training if model exists."
    )
    args = parser.parse_args(namespace=Arguments())

    # Check if training already started
    modelPath = args.project / args.name
    modelExists = modelPath.exists()
    if modelExists and not args.resume:
        print(
            "Model already exists and resume was not enabled. Either enable resume or delete model directory."
        )
        exit(1)

    if modelExists:
        model = YOLO(modelPath / "weights/last.pt")
        model.train(cfg=modelPath / "args.yaml", resume=True)  # type: ignore
    else:
        if args.dataset is None:
            print("--dataset parameter is needed to train a new model.")
            exit(-1)

        model = YOLO(args.base_model)
        model.train(  # type: ignore
            data=args.dataset,
            epochs=100,
            imgsz=640,  # training image size
            device=[
                0,
                1,
            ],  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
            project=args.project,
            name=args.name,
        )


if __name__ == "__main__":
    main()
