import gradio as gr
from app.ui import build_interface

# Build our high-fidelity 3D UI
demo = build_interface()

if __name__ == "__main__":
    demo.launch()