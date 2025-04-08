import argparse
import pathlib
import torch
import torch.nn.utils.prune as prune
from ultralytics import YOLO 


class Arguments(argparse.Namespace):
    name: str
    project: pathlib.Path
    base_model: str
    prune_ratio: float


def parse_args():
    parser = argparse.ArgumentParser("unstructured_prune")

    parser.add_argument(
        "name", type=str, help="Custom model name for the pruned model."
    )
    parser.add_argument(
        "--project",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent / "runs/pruned",
        help="Directory to store pruned model files.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="yolo11n.pt",
        help="Path to the base YOLO model.",
    )
    parser.add_argument(
        "--prune-ratio",
        type=float,
        default=0.3,
        help="Percentage of weights in each layer to prune (0.0 - 1.0).",
    )

    return parser.parse_args(namespace=Arguments())


def unstructured_prune_model(model, prune_ratio):
    """
    Performs unstructured pruning on convolutional and linear layers of a model.
    Prunes weights (not entire channels/neurons) based on L1 magnitude.
    """
    print(f"Performing unstructured pruning with {prune_ratio * 100:.1f}% sparsity...")

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print(f"Pruning {prune_ratio * 100:.1f}% channels in layer: {name}")
            prune.l1_unstructured(module, name="weight", amount=prune_ratio)
            prune.remove(module, "weight")

        elif isinstance(module, torch.nn.Linear):
            print(f"Pruning {prune_ratio * 100:.1f}% neurons in layer: {name}")
            prune.l1_unstructured(module, name="weight", amount=prune_ratio)
            prune.remove(module, "weight")

    print("Unstructured pruning completed.")
    return model

def main():
    args = parse_args()

    # Ensure project directory exists
    output_dir = args.project / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load YOLO model
    print(f"Loading model from {args.base_model}...")
    model = YOLO(args.base_model)

    # Perform unstructured pruning
    pruned_model = unstructured_prune_model(model.model, args.prune_ratio)

    # Save pruned model
    output_path = output_dir / "fine_grain_pruned_model.pt"
    print(f"Saving pruned model to {output_path}...")
    torch.save(
        {"model": pruned_model, "state_dict": pruned_model.state_dict()}, output_path
    )
    print(f"✅ Pruned model saved successfully at {output_path}")


if __name__ == "__main__":
    main()
