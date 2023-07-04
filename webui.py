import gradio as gr
import numpy as np
import pandas as pd

import shared


def process_csv():
    pass

def main():
    preview_data = [
        ['Alice', 25, 'F'],
        ['Bob', 30, 'M'],
        ['Chun-Ming', 38, 'M']
    ]

    with gr.Blocks() as ui:
        with gr.Row().style(equal_height=True):
            # left panel
            with gr.Column(scale=2):
                with gr.Tab("CSV"):
                    shared.gradio['csv_file'] = gr.File(file_count='multiple', file_types=['.csv'])
                    with gr.Accordion("Preview & Edit"):
                        # [1] use Markdown to show a table
                        # shared.gradio['csv_preview'] = gr.Markdown(
                        #     """
                        #     | Player Name | Team | Position |
                        #     | --- | --- | --- |
                        #     | Shohei Ohtani | Los Angeles Angels | Pitcher/DH |
                        #     | Fernando Tatis Jr. | San Diego Padres | Shortstop |
                        #     | Jacob deGrom | New York Mets | Pitcher |
                        #     """
                        # )
                        # [2] use DataFrame to show a table
                        shared.gradio['csv_preview'] = gr.DataFrame(
                            headers=["Name", "Age", "Gender"],
                            value=preview_data,
                            datatype=["str", "str", "str"],
                            col_count=3
                        )
                with gr.Tab("SQL"):
                    shared.gradio["sql_query"] = gr.TextArea(placeholder="Enter the SQL query here ...")
                    shared.gradio['query_button'] = gr.Button(value="Query")
                with gr.Tab("Settings"):
                    gr.Markdown("## Model settings")
                    shared.gradio['temp_slider'] = gr.Slider(0, 10, step=1, label="Temperature")
                    shared.gradio['model_dropdown'] = gr.Dropdown(
                        ["ChatGPT-3.5 turbo", "ChatGPT-4", "Bing", "Poe"],
                        label="Model", info="Choose the backend of chat bot"
                    )
                    gr.Markdown("***")  # horizontal line
                    gr.Markdown("## Database settings")  # horizontal line
                    with gr.Row():
                        shared.gradio['db_user_textbox'] = gr.Textbox(label="User")
                        shared.gradio['db_passwd_textbox'] = gr.Textbox(label="Password", type='password')
                        shared.gradio['db_name_textbox'] = gr.Textbox(label="Database")
                    shared.gradio['db_save_button'] = gr.Button(value="Test & Save")

                shared.gradio['ask_textbox'] = gr.Textbox(label="Prompt", placeholder="Ask questions here ...", lines=3)
                with gr.Row():
                    shared.gradio['clear_button'] = gr.Button(value="Clear")
                    shared.gradio['submit_button'] = gr.Button(value="Submit")
            # right panel
            with gr.Column(scale=3):
                with gr.Tab("Result"):
                    text_input = gr.Textbox("")
                with gr.Tab("History"):
                    text_input = gr.Textbox()
                with gr.Tab("Debug Msg"):
                    text_input = gr.Textbox()
    ui.launch()

if __name__ == '__main__':
    main()