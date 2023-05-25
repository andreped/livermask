---
title: 'livermask: Automatic Liver Parenchyma and vessel segmentation in CT'
colorFrom: indigo
colorTo: indigo
sdk: docker
app_port: 7860
emoji: 🚀
pinned: false
license: mit
app_file: demo/app.py
---

![Screenshot 2023-05-24 at 19 14 15](https://github.com/andreped/livermask/assets/29090665/6083a28a-9250-4f86-a9b2-8d4e9823c836)

# livermask Hugging Face demo - through docker SDK

Deploying simple models in a gradio-based web interface in Hugging Face spaces is easy.
For any other custom pipeline, with various dependencies and challenging behaviour, it
might be necessary to use Docker containers instead.

Deployment through a custom Docker image over the existing Gradio image was
necessary in this case due to `tensorflow` and `gradio` having colliding
versions. As `livermask` depends on `tf`, the only way to get around it was
fixing the broken dependency, which was handled by reinstalling and changing
the `typing_extensions` with a version that `gradio` required for the widgets
we used. Luckily, this did not break anything in `tf`, even though `tf` has a
very strict versioning criteria for this dependency.

Anyways, everything works as intended now. For every new push to the main branch,
continuous deployment to the Hugging Face `livermask` space is performed through
GitHub Actions.

When the space is updated, the Docker image is rebuilt/updated (caching if possible).
Then when finished, the end users can test the app as they please.

Right now, the functionality of the app is extremely limited, only offering a widget
for uploading a NIfTI file (`.nii` or `.nii.gz`) and visualizing the produced surface
of the predicted liver parenchyma 3D volume when finished processing.

Analysis process can be monitored from the `Logs` tab next to the `Running` button
in the Hugging Face `livermask` space.

It is also possible to build the app as a docker image and deploy it. To do so follow these steps:

```
docker build -t livermask ..
docker run -it -p 7860:7860 livermask
```

Then open `http://127.0.0.1:7860` in your favourite internet browser to view the demo.

Natural future TODOs include:
- [ ] Add gallery widget to enable scrolling through 2D slices
- [ ] Render segmentation for individual 2D slices as overlays
