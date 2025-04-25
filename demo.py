import argparse
import functools
import pathlib
from typing import Any

import gradio as gr
from ultralytics import YOLO  # type: ignore


def inference(model: YOLO, img: Any) -> Any:
    results = model.predict(img, verbose=True)  # type: ignore
    resultsPlotted = results[0].plot()  # type: ignore
    return resultsPlotted  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser("demo.py")
    parser.add_argument("model", type=pathlib.Path, help="Path to model.")
    args = parser.parse_args()
    modelPath = args.model

    model = YOLO(modelPath)

    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column():
                imgStream = gr.Image(sources=["webcam"])

            with gr.Column():
                outputImg = gr.Image(streaming=True, show_label=False)

        imgStream.stream(
            fn=functools.partial(inference, model),
            inputs=[imgStream],
            outputs=outputImg,
            stream_every=0.001,
            concurrency_limit=30,
        )

        app.launch()


if __name__ == "__main__":
    main()
