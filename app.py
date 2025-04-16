import gradio as gr
from datetime import datetime
import os
from model.Final_Model_User import run_model
import shutil

def analyze_video(video):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create separate folders for inputs and outputs
    input_path = f"results/uploads/input_{timestamp}.mp4"
    output_path = f"results/outputs/output_{timestamp}.mp4"

    os.makedirs(os.path.dirname(input_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save uploaded video
    shutil.copy(video, input_path)

    # Run your model
    stats = run_model(input_path, output_path)

    return output_path, stats


# Set up Gradio interface
app = gr.Interface(
    fn=analyze_video,
    inputs=gr.Video(label="Upload Match Clip"),
    outputs=[
        gr.Video(label="Analyzed Clip"),
        gr.JSON(label="Player Stats (Distance Run)")
    ],
    title="Juega Vision",
    description="Upload a match clip and get automatic player tracking and stats."
)

if __name__ == "__main__":
    app.launch()