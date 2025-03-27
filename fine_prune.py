#!/usr/bin/env python3

import argparse

import torch
import torch.nn.utils.prune as prune
from ultralytics import YOLO  # type: ignore


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-Grained L1 Pruning for YOLOV11")
    parser.add_argument(
        "model_path", type=str, help="Path to the trained YOLO model (.pt)"
    )
    parser.add_argument(
        "--prune-ratio",
        type=float,
        default=0.3,
        help="Percentage of weights to prune (0.0 - 1.0)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="fine_pruned_model.pt",
        help="Path to save the pruned model",
    )
    return parser.parse_args()


def fine_grained_prune_model(model, prune_ratio):
    print("Performing fine-grained L1 pruning...")

    # Iterate over all convolutional and linear layers
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            print(f"Pruning {name} with {prune_ratio * 100}% sparsity...")

            # Perform fine-grained L1 pruning using PyTorch's pruning API
            prune.l1_unstructured(module, name="weight", amount=prune_ratio)

            # Optionally prune bias if it exists
            if module.bias is not None:
                prune.l1_unstructured(module, name="bias", amount=prune_ratio)

            # Remove the pruning reparameterization to save clean model
            prune.remove(module, "weight")
            if module.bias is not None:
                prune.remove(module, "bias")

    print("Fine-grained pruning complete.")
    return model


def main():
    args = parse_args()

    # Load model
    print("Loading model...")
    model = YOLO(args.model_path)
    model.info(detailed=True)

    # Perform fine-grained pruning
    pruned_model = fine_grained_prune_model(model.model, args.prune_ratio)

    # Save pruned model
    print(f"Saving pruned model to {args.output_path}...")
    model.model.load_state_dict(pruned_model.state_dict())
    model.info(detailed=True)
    model.save(args.output_path)


if __name__ == "__main__":
    main()
