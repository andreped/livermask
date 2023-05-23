import gradio as gr

def greet(name):
    return "Hello" + name + "!!"

iface = gr.Interface(fn=greet, inputs="text", output="text")
iface.launch()
