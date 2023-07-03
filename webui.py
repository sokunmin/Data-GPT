import gradio as gr
import numpy as np


def main():
    with gr.Blocks() as ui:
        with gr.Row():
            # left panel
            with gr.Column(scale=1):
                with gr.Tab("CSV"):
                    text_input = gr.Textbox(label="")
                with gr.Tab("SQL"):
                    text_input = gr.Textbox()
                with gr.Tab("Settings"):
                    model_temp = gr.Slider(0, 10, label="Temperature")
                    gr.Dropdown(
                        ["ChatGPT-3.5 turbo", "ChatGPT-4", "Bing", "Poe"], label="Model", info="Choose backend chat bot"
                    )
                    gr.Markdown("***")  # horizontal line
                    gr.Textbox(label="User")
                    gr.Textbox(label="Password", type='password')
                    gr.Textbox(label="Database")

            # right panel
            with gr.Column(scale=4):
                with gr.Tab("Result"):
                    text_input = gr.Textbox("")
                with gr.Tab("History"):
                    text_input = gr.Textbox()
                with gr.Tab("Debug Msg"):
                    text_input = gr.Textbox()
    ui.launch()


if __name__ == '__main__':
    main()