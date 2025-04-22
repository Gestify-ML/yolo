import argparse
import pathlib

from ultralytics import YOLO  # type: ignore


class ProgramArguments(argparse.Namespace):
    model: pathlib.Path
    video: pathlib.Path


def getArgs() -> ProgramArguments:
    parser = argparse.ArgumentParser("measure_inference.py")
    parser.add_argument("model", type=pathlib.Path, help="Path to model")
    parser.add_argument("video", type=pathlib.Path, help="Path to video")
    return parser.parse_args(namespace=ProgramArguments())


def main() -> None:
    args = getArgs()
    model = YOLO(args.model, task="detect")

    results = model.predict(args.video, stream=True)  # type: ignore
    times: list[float] = []
    for r in results:  # type: ignore
        times.append(r.speed["inference"])  # type: ignore
    print(f"Average Inference Time (ms): {sum(times) / len(times):.03f}")


if __name__ == "__main__":
    main()
