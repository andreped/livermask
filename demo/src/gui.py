import gradio as gr
from .utils import load_ct_to_numpy, load_pred_volume_to_numpy
from .compute import run_model
from .convert import nifti_to_glb


class WebUI:
    def __init__(self, model_name, class_name):
        # global states
        self.images = []
        self.pred_images = []

        self.nb_slider_items = 100

        self.model_name = model_name
        self.class_name = class_name

        # define widgets not to be rendered immediantly, but later on
        self.slider = gr.Slider(1, self.nb_slider_items, value=1, step=1, label="Which 2D slice to show")
        self.volume_renderer = gr.Model3D(
            clear_color=[0.0, 0.0, 0.0, 0.0],
            label="3D Model",
            visible=True
        ).style(height=512)

    def combine_ct_and_seg(self, img, pred):
        return (img, [(pred, self.class_name)])
    
    def upload_file(self, file):
        return file.name
    
    def load_mesh(self, mesh_file_name, model_name="/home/user/app/model.h5"):
        path = mesh_file_name.name
        run_model(path, model_name)
        nifti_to_glb("prediction-livermask.nii")
        self.images = load_ct_to_numpy("./files/test_ct.nii")
        self.pred_images = load_pred_volume_to_numpy("./prediction-livermask.nii")
        self.slider = self.slider.update(value=2)
        return "./prediction.obj"
    
    def get_img_pred_pair(self, k):
        k = int(k) - 1
        out = [gr.AnnotatedImage.update(visible=False)] * self.nb_slider_items
        out[k] = gr.AnnotatedImage.update(self.combine_ct_and_seg(self.images[k], self.pred_images[k]), visible=True)
        return out

    def run(self):
        with gr.Blocks() as demo:

            with gr.Row().style(equal_height=True):
                file_output = gr.File(file_types=[".nii", ".nii.nz"], file_count="single").style(full_width=False, size="sm")
                file_output.upload(self.upload_file, file_output, file_output)

                run_btn = gr.Button("Run analysis").style(full_width=False, size="sm")
                run_btn.click(fn=lambda x: self.load_mesh(x, model_name=self.model_name), inputs=file_output, outputs=self.volume_renderer)
            
            with gr.Row().style(equal_height=True):
                with gr.Box():
                    image_boxes = []
                    for i in range(self.nb_slider_items):
                        visibility = True if i == 1 else False
                        t = gr.AnnotatedImage(visible=visibility)\
                            .style(color_map={self.class_name: "#ffae00"}, height=512, width=512)
                        image_boxes.append(t)

                    self.slider.change(self.get_img_pred_pair, self.slider, image_boxes)
                
                with gr.Box():
                    self.volume_renderer.render()
            
            with gr.Row():
                self.slider.render()

        # sharing app publicly -> share=True: https://gradio.app/sharing-your-app/
        # inference times > 60 seconds -> need queue(): https://github.com/tloen/alpaca-lora/issues/60#issuecomment-1510006062
        demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)
