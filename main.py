from fastapi import FastAPI
import gradio as gr
from app.ui import build_interface

app = FastAPI()

# Build our high-fidelity 3D UI
demo = build_interface()

# Mount it to the root path for Vercel/Railway
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)