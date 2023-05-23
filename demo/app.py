import gradio as gr
import subprocess as sp

def download_testdata():
    sp.check_call(["wget", "https://github.com/gradio-app/gradio/raw/main/demo/model3D/files/Duck.glb"])

def load_mesh(mesh_file_name):
    return mesh_file_name

demo = gr.Interface(
    fn=load_mesh,
    inputs=gr.Model3D(),
    outputs=gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="3D Model"),
    examples=[
        ["Duck.glb"],
    ],
    cache_examples=True,
)

if __name__ == "__main__":
    download_testdata()
    demo.launch()
