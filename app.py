import re
import uuid
import chromadb
import gradio as gr
import numpy as np
import pandas as pd
import random
import shared
from io import StringIO
from os.path import basename
from langchain.vectorstores import Chroma
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
import fitz
from PIL import Image


def feed_csv_into_collector(files):
    dfs = []
    if isinstance(files, list):
        for file in files:
            df = pd.read_csv(file.name)
            cols = list(df.columns)
            dfs.append((basename(file.name), cols, file.name))
    return dfs


def feed_pdf_into_collector(files):
    return


def on_csv_selected(df, evt: gr.SelectData):
    assert isinstance(evt.target, gr.components.Dataframe)
    row, col = evt.index
    file_path = df['File path'][row]
    print(evt)
    return pd.read_csv(file_path)


def on_pdf_selected(pdf, evt: gr.SelectData):
    pass


def output_result_dataframe():
    pass


def output_result_chart():
    pass


def output_prediction():
    pass


def main():
    with gr.Blocks() as ui:
        with gr.Row().style(equal_height=True):
            # left panel
            with gr.Column(scale=2):
                with gr.Tab("CSV"):
                    shared.gradio['csv_button'] = gr.UploadButton(label="Upload CSVs",
                                                                  file_types=['.csv'],
                                                                  file_count="multiple")
                    shared.gradio['csv_file_list'] = gr.Dataframe(headers=["Filename", "Categories", "File path"],
                                                                  type="pandas",
                                                                  value=[],
                                                                  interactive=False)
                    shared.gradio['csv_button'].upload(fn=feed_csv_into_collector,
                                                       inputs=shared.gradio['csv_button'],
                                                       outputs=shared.gradio['csv_file_list'],
                                                       api_name="upload_csv")
                    with gr.Accordion("Drop here to preview"):
                        shared.gradio['csv_preview'] = gr.DataFrame(overflow_row_behaviour='paginate',
                                                                    max_rows=12,
                                                                    interactive=True)
                    shared.gradio['csv_file_list'].select(on_csv_selected, shared.gradio['csv_file_list'],
                                                          shared.gradio['csv_preview'])
                with gr.Tab("SQL"):
                    shared.gradio["query_textarea"] = gr.TextArea(placeholder="Enter the SQL query here ...")
                    shared.gradio['query_button'] = gr.Button(value="Query")
                    with gr.Accordion("Query Result"):
                        shared.gradio['query_dataframe'] = gr.DataFrame(
                            headers=["Name", "Age", "Gender"],
                        )

                with gr.Tab("PDF"):
                    shared.gradio['pdf_button'] = gr.UploadButton(label="Upload PDFs",
                                                                  file_types=['.pdf'],
                                                                  file_count="multiple")
                    shared.gradio['pdf_file_list'] = gr.Dataframe(headers=["Filename", "File path"],
                                                                  type="pandas",
                                                                  value=[],
                                                                  interactive=False)
                    shared.gradio['pdf_file_list'].select(on_pdf_selected, shared.gradio['pdf_file_list'])

                with gr.Tab("Settings"):
                    gr.Markdown("## Model settings")
                    shared.gradio['temp_slider'] = gr.Slider(0, 10, step=1, label="Temperature")
                    shared.gradio['model_dropdown'] = gr.Dropdown(
                        ["ChatGPT-3.5 turbo", "ChatGPT-4", "Bing", "Poe"],
                        label="Model", info="Choose the backend of chat bot"
                    )
                    gr.Markdown("***")  # horizontal line
                    gr.Markdown("## Database settings")  # horizontal line
                    shared.gradio['db_url_textbox'] = gr.Textbox(label="URL", value="http://localhost:5432")
                    with gr.Row():
                        shared.gradio['db_user_textbox'] = gr.Textbox(label="User", value="postgres")
                        shared.gradio['db_passwd_textbox'] = gr.Textbox(label="Password", type='password',
                                                                        value="postgres")
                        shared.gradio['db_name_textbox'] = gr.Textbox(label="Database", value="postgres-db")
                    shared.gradio['db_save_button'] = gr.Button(value="Test & Save")
                shared.gradio['ask_textbox'] = gr.Textbox(label="Prompt", placeholder="Ask questions here ...", lines=3)
                with gr.Row():
                    shared.gradio['clear_button'] = gr.Button(value="Clear")
                    shared.gradio['submit_button'] = gr.Button(value="Submit")
            # right panel
            with gr.Column(scale=3):
                with gr.Tab("Chat"):
                    shared.gradio['chatbot_text'] = gr.Chatbot(value=[])
                    shared.gradio['chatbot_image'] = gr.Image(label='Upload PDF', tool='select')
                with gr.Tab("Result"):
                    shared.gradio['result_dataframe'] = gr.Dataframe(label="Output", headers=["1", "2", "3"],
                                                                     max_rows=30)
                    shared.gradio['result_timeseries'] = gr.Timeseries(label="Chart", x="time",
                                                                       y=['retail', 'food', 'other'])
                    shared.gradio['result_label'] = gr.Label(label="Prediction")
                with gr.Tab("History"):
                    shared.gradio['history_label'] = gr.Label()
                with gr.Tab("Debug Msg"):
                    shared.gradio['debug_label'] = gr.Label()

    ui.launch()


if __name__ == '__main__':
    main()
