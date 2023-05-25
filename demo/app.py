import gradio as gr
import subprocess as sp
from skimage.measure import marching_cubes
import nibabel as nib
from nibabel.processing import resample_to_output
import numpy as np
import random


def nifti_to_glb(path):
    # load NIFTI into numpy array
    image = nib.load(path)
    resampled = resample_to_output(image, [1, 1, 1], order=1)
    data = resampled.get_fdata().astype("uint8")

    # extract surface
    verts, faces, normals, values = marching_cubes(data, 0)
    faces += 1

    with open('prediction.obj', 'w') as thefile:
        for item in verts:
            thefile.write("v {0} {1} {2}\n".format(item[0],item[1],item[2]))

        for item in normals:
            thefile.write("vn {0} {1} {2}\n".format(item[0],item[1],item[2]))

        for item in faces:
            thefile.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(item[0],item[1],item[2]))


def run_model(input_path):
    from livermask.utils.run import run_analysis
    
    run_analysis(cpu=True, extension='.nii', path=input_path, output='prediction', verbose=True, vessels=False, name="/home/user/app/model.h5", mp_enabled=False)


def load_mesh(mesh_file_name):
    path = mesh_file_name.name
    run_model(path)
    nifti_to_glb("prediction-livermask.nii")
    return "./prediction.obj"


def setup_gallery(data_path, pred_path):
    image = nib.load(data_path)
    resampled = resample_to_output(image, [1, 1, 1], order=1)
    data = resampled.get_fdata().astype("uint8")

    image = nib.load(pred_path)
    resampled = resample_to_output(image, [1, 1, 1], order=0)
    pred = resampled.get_fdata().astype("uint8")


def load_ct_to_numpy(data_path):
    if type(data_path) != str:
        data_path = data_path.name

    image = nib.load(data_path)
    data = image.get_fdata()

    data = np.rot90(data, k=1, axes=(0, 1))

    data[data < -150] = -150
    data[data > 250] = 250

    data = data - np.amin(data)
    data = data / np.amax(data) * 255
    data = data.astype("uint8")

    print(data.shape)
    return [data[..., i] for i in range(data.shape[-1])]


def upload_file(file):
    return file.name

#def select_section(evt: gr.SelectData):
#    return section_labels[evt.index]


if __name__ == "__main__":
    print("Launching demo...")
    with gr.Blocks() as demo:
        """
        with gr.Blocks() as demo:
            with gr.Row():
                text1 = gr.Textbox(label="t1")
                slider2 = gr.Textbox(label="slide")
                drop3 = gr.Dropdown(["a", "b", "c"], label="d3")
            with gr.Row():
                with gr.Column(scale=1, min_width=600):
                    text1 = gr.Textbox(label="prompt 1")
                    text2 = gr.Textbox(label="prompt 2")
                    inbtw = gr.Button("Between")
                    text4 = gr.Textbox(label="prompt 1")
                    text5 = gr.Textbox(label="prompt 2")
                with gr.Column(scale=2, min_width=600):
                    img1 = gr.Image("images/cheetah.jpg")
                    btn = gr.Button("Go").style(full_width=True)
        
        greeter_1 = gr.Interface(lambda name: f"Hello {name}!", inputs="textbox", outputs=gr.Textbox(label="Greeter 1"))
        greeter_2 = gr.Interface(lambda name: f"Greetings {name}!", inputs="textbox", outputs=gr.Textbox(label="Greeter 2"))
        demo = gr.Parallel(greeter_1, greeter_2)

        volume_renderer = gr.Interface(
            fn=load_mesh,
            inputs=gr.UploadButton(label="Click to Upload a File", file_types=[".nii", ".nii.nz"], file_count="single"),
            outputs=gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="3D Model"),
            title="livermask: Automatic Liver Parenchyma segmentation in CT",
            description="Using pretrained deep learning model trained on the LiTS17 dataset",
        )
        """

        with gr.Row():
            # file_output = gr.File()
            upload_button = gr.UploadButton(label="Click to Upload a File", file_types=[".nii", ".nii.nz"], file_count="single")
            # upload_button.upload(upload_file, upload_button, file_output)

            #select_btn = gr.Button("Run analysis")
            #select_btn.click(fn=upload_file, inputs=upload_button, outputs=output, api_name="Analysis")
        
            #upload_button.click(section, [img_input, num_boxes, num_segments], img_output)
        
        #print("file output:", file_output)

        images = load_ct_to_numpy("./test-volume.nii")

        def variable_outputs(k):
            k = int(k) - 1
            out = [gr.AnnotatedImage.update(visible=False)] * len(images)
            out[k] = gr.AnnotatedImage.update(visible=True)
            return out
        
        def section(img, num_segments):
            sections = []
            for b in range(num_segments):
                x = random.randint(0, img.shape[1])
                y = random.randint(0, img.shape[0])
                r = random.randint(0, min(x, y, img.shape[1] - x, img.shape[0] - y))
                mask = np.zeros(img.shape[:2])
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        dist_square = (i - y) ** 2 + (j - x) ** 2
                        if dist_square < r**2:
                            mask[i, j] = round((r**2 - dist_square) / r**2 * 4) / 4
                sections.append((mask, "parenchyma"))
            return (img, sections)
        
        with gr.Row():
            s = gr.Slider(1, len(images), value=1, step=1, label="Which 2D slice to show")
        
        with gr.Row():
            with gr.Box():
                images_boxes = []
                for i, image in enumerate(images):
                    visibility = True if i == 1 else False  # only first slide visible - change slide through slider
                    t = gr.AnnotatedImage(value=section(image, 1), visible=visibility).style(color_map={"parenchyma": "#ffae00"}, width=image.shape[1])
                    images_boxes.append(t)

                s.change(variable_outputs, s, images_boxes)

        
        #upload_button.upload(upload_file, upload_button, file_output)
        
        #section_btn.click(section, [images[40], num_boxes, num_segments], img_output)
        #ct_images.upload(section, [images[40], num_boxes, num_segments], img_output)

        #demo = gr.Interface(
        #    fn=load_ct_to_numpy,
        #    inputs=gr.UploadButton(label="Click to Upload a File", file_types=[".nii", ".nii.nz"], file_count="single"),
        #    outputs=gr.Gallery(label="CT slices").style(columns=[4], rows=[4], object_fit="contain", height="auto"),
        #    title="livermask: Automatic Liver Parenchyma segmentation in CT",
        #    description="Using pretrained deep learning model trained on the LiTS17 dataset",
        #)

    # sharing app publicly -> share=True: https://gradio.app/sharing-your-app/
    # inference times > 60 seconds -> need queue(): https://github.com/tloen/alpaca-lora/issues/60#issuecomment-1510006062
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)
