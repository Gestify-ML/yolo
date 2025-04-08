#!/usr/bin/env python3

import argparse
import pathlib

import torch
from ultralytics import YOLO  # type: ignore


class Arguments(argparse.Namespace):
    pruned: pathlib.Path
    tuned: pathlib.Path
    output: pathlib.Path


def parseArgs():
    parser = argparse.ArgumentParser("reapply_prune.py")
    parser.add_argument("pruned", type=pathlib.Path)
    parser.add_argument("tuned", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    return parser.parse_args(namespace=Arguments())


def main() -> None:
    args = parseArgs()
    pruned = torch.load(args.pruned, map_location="cpu", weights_only=False)  # type: ignore
    tuned = YOLO(args.tuned)

    # Compute masks
    masks: dict[str, torch.Tensor] = {}
    model = pruned["ema"] if "ema" in pruned else pruned["model"]
    for name, module in model.named_modules():  # type: ignore
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            # Create a mask where weights are non-zero
            mask = (torch.abs(module.weight) > 1e-6).float()
            masks[name] = mask

    # Apply mask
    for name, module in tuned.model.named_modules():  # type: ignore
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)) and name in masks:
            module.weight *= masks[name]

    torch.save(  # type: ignore
        {"model": tuned.model, "state_dict": tuned.model.state_dict()},  # type: ignore
        args.output,
    )


if __name__ == "__main__":
    main()
