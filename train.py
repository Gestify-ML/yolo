#!/usr/bin/env python3

import argparse
import os
import pathlib

from ultralytics import YOLO  # type: ignore

from custom.trainer import PrunedDetectionTrainer


class Arguments(argparse.Namespace):
    name: str
    dataset: pathlib.Path | None
    project: pathlib.Path
    base_model: str
    resume: bool
    pruned_model: bool
    epochs: int
    patience: int


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
        help="Base model to start training with, can be a base model name like 'yolo11n.pt' or a path to a local one.",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training if model exists."
    )
    parser.add_argument(
        "--pruned-model",
        action="store_true",
        help="If the model is a pruned model, this option should be enabled to maintain the pruning. When this option is used, the repo directory needs to be added to the PYTHON path, i.e. 'export PYTHONPATH=\"${PYTHONPATH}:/full/path/to/yolo/repo\"'",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Max number of epochs to train for.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="How many epochs to wait without validation improvement to stop early.",
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

    # Check that python path is set if using pruned trainer
    if args.pruned_model and os.environ.get("PYTHONPATH") is None:
        print(
            "PYTHONPATH is not set, add the repo directory to it when using the --pruned-model option."
        )

    if modelExists:
        model = YOLO(modelPath / "weights/last.pt")
        model.train(  # type: ignore
            trainer=PrunedDetectionTrainer if args.pruned_model else None,
            cfg=modelPath / "args.yaml",
            resume=True,
        )
    else:
        if args.dataset is None:
            print("--dataset parameter is needed to train a new model.")
            exit(-1)

        model = YOLO(args.base_model)
        model.train(  # type: ignore
            trainer=PrunedDetectionTrainer if args.pruned_model else None,
            data=args.dataset,
            epochs=args.epochs,
            patience=args.patience,
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
