import pathlib
from typing import Any

import gradio as gr
from ultralytics import YOLO  # type: ignore

MODEL_PATH = pathlib.Path(R"models/ten_gestures.pt")
model = YOLO(MODEL_PATH)


def inference(img: Any) -> Any:
    results = model.predict(img, verbose=True)  # type: ignore
    resultsPlotted = results[0].plot()  # type: ignore
    return resultsPlotted  # type: ignore


def main() -> None:
    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column():
                imgStream = gr.Image(sources=["webcam"])

            with gr.Column():
                outputImg = gr.Image(streaming=True, show_label=False)

        imgStream.stream(
            fn=inference,
            inputs=[imgStream],
            outputs=outputImg,
            stream_every=0.001,
            concurrency_limit=30,
        )

        app.launch()


if __name__ == "__main__":
    main()
