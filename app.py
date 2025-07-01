import gradio as gr
from ultralytics import YOLO
from pathlib import Path
import cv2
import tempfile
import os

# Load model
model = YOLO("best.pt")

def detect_elephant(file_obj):
    # Get file path and suffix
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file_obj.name)
    with open(file_path, "wb") as f:
        f.write(file_obj.read())
    suffix = Path(file_path).suffix.lower()

    if suffix in [".jpg", ".jpeg", ".png"]:
        # Image detection
        results = model.predict(source=file_path, save=False, conf=0.25)
        result_img = results[0].plot()
        return result_img
    elif suffix == ".mp4":
        # Video detection
        save_dir = os.path.join(temp_dir, "output")
        results = model.predict(source=file_path, save=True, project=save_dir, name="gradio", conf=0.25)
        output_video_path = os.path.join(save_dir, "gradio", Path(file_path).name)
        return output_video_path
    else:
        return None

# Gradio interface
demo = gr.Interface(
    detect_elephant,
    inputs=gr.File(label="Upload Image or Video"),
    outputs=gr.outputs.Auto(gr.Image(label="Prediction"), gr.Video(label="Prediction")),
    title="🐘 Elephant Detector using YOLOv8",
    description="Upload an image or video to detect elephants using your trained model."
)

demo.launch()
