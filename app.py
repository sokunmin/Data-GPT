import os

import fitz
import shared
import gradio as gr
import pandas as pd
from shared import update_file_list
from shared import load_file_and_split_chunks
from typing import List, Set
from PIL import Image
from os.path import basename, join, exists
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.config import Settings


def load_csv_to_table(files: List[str]):
    dfs = []
    for file in files:
        df = pd.read_csv(file.name)
        cols = list(df.columns)
        dfs.append((basename(file.name), cols, file.name))
    return dfs


def load_pdf_to_table(files: List[str]):
    db_files = list(map(lambda x: (x,), shared.db['files'].keys()))
    pdf_list = db_files
    for file in files:
        filename = basename(file.name)
        if filename not in shared.db['files']:
            load_file_and_split_chunks(file.name)
            pdf_list.append((filename,))
    return pdf_list


def on_csv_selected(df, evt: gr.SelectData):
    assert isinstance(evt.target, gr.components.Dataframe)
    row, col = evt.index
    file_path = df['File path'][row]
    docs = load_file_and_split_chunks(file_path)
    return docs #pd.read_csv(file_path)


def on_pdf_selected(files, pdf, evt: gr.SelectData):
    row, col = evt.index
    filename = pdf[row][0]
    file_path = shared.db['files'][filename]
    image, page_count = render_pdf(file_path)
    return image, gr.update(minimum=1, maximum=page_count, value=1)


def render_pdf(pdf_path: str):
    doc = fitz.open(pdf_path)
    page = doc[0]
    # Render the page as a PNG image with a resolution of 300 DPI
    pix = page.get_pixmap(matrix=fitz.Matrix(300 / 150, 300 / 150))
    image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    return image, doc.page_count


def ask_chatbot_questions(history, question):
    if not question:
        raise gr.Error('You haven\'t asked a question yet!')
    history += [(question, '')]
    return history


def get_chatbot_response():
    pass


def output_pdf_pages():
    pass


def output_result_dataframe():
    pass


def output_result_chart():
    pass


def output_prediction():
    pass


def create_embeddings():
    model_name = os.environ['EMBEDDINGS_MODEL_NAME']
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings


def load_db():
    embeddings = create_embeddings()
    if exists(join(shared.PERSIST_DB_DIR, 'chroma.sqlite3')):
        # Update and store locally vectorstore
        shared.db['db'] = Chroma(
            embedding_function=embeddings,
            client_settings=shared.CHROMA_SETTINGS
        )
        update_file_list()
    else:
        shared.db['db'] = Chroma.from_documents(
            [],
            embedding=embeddings,
            client_settings=shared.CHROMA_SETTINGS
        )


def main():
    with gr.Blocks() as ui:
        with gr.Row().style(equal_height=True):
            # left panel
            with gr.Column(scale=5):
                # PDF tab
                with gr.Tab("PDF"):
                    shared.gr['pdf_button'] = gr.UploadButton(
                        label="Upload PDFs", file_types=['.pdf'], file_count="multiple")
                    shared.gr['pdf_file_list'] = gr.Dataframe(
                        headers=["Filename"],
                        label='Files would be ignored if they already exist in database',
                        type="array",
                        value=list(map(lambda x: (x,), shared.db['files'].keys())),
                        interactive=False)
                    with gr.Accordion("Preview"):
                        shared.gr['pdf_preview'] = gr.Image(
                            label='PDF', tool='select', elem_id='preview_box'
                        ).style(height=680)
                    shared.gr['pdf_slider'] = gr.Slider(
                        label='Page', step=1, interactive=True)
                    shared.gr['pdf_pages'] = gr.HighlightedText(
                        label='Found at', combine_adjacent=True, show_legend=True
                    ).style(color_map={"Chapter": "green", "Page#": 'blue'})
                # CSV tab
                with gr.Tab("CSV"):
                    shared.gr['csv_button'] = gr.UploadButton(
                        label="Upload CSVs", file_types=['.csv'], file_count="multiple")
                    shared.gr['csv_file_list'] = gr.Dataframe(
                        headers=["Filename", "Categories", "File path"],
                        type="pandas", value=[], interactive=False)
                    with gr.Accordion("Drop here to preview"):
                        shared.gr['csv_preview'] = gr.DataFrame(
                            overflow_row_behaviour='paginate', max_rows=12, interactive=True)
                # SQL tab
                with gr.Tab("SQL"):
                    shared.gr["query_textarea"] = gr.TextArea(placeholder="Enter the SQL query here ...")
                    shared.gr['query_button'] = gr.Button(value="Query")
                    with gr.Accordion("Query Result"):
                        shared.gr['query_dataframe'] = gr.DataFrame(headers=["Name", "Age", "Gender"], )

                # Settings tab
                with gr.Tab("Settings"):
                    gr.Markdown("## Model settings")
                    shared.gr['temp_slider'] = gr.Slider(0, 10, step=1, label="Temperature")
                    shared.gr['model_dropdown'] = gr.Dropdown(
                        ["ChatGPT-3.5 turbo", "ChatGPT-4", "Bing", "Poe"],
                        label="Model", info="Choose the backend of chat bot")
                    gr.Markdown("***")  # horizontal line
                    gr.Markdown("## Database settings")  # horizontal line
                    shared.gr['db_url_textbox'] = gr.Textbox(label="URL", value="http://localhost:5432")
                    with gr.Row():
                        shared.gr['db_user_textbox'] = gr.Textbox(label="User", value="postgres")
                        shared.gr['db_passwd_textbox'] = gr.Textbox(
                            label="Password", type='password', value="postgres")
                        shared.gr['db_name_textbox'] = gr.Textbox(label="Database", value="postgres-db")
                    shared.gr['db_save_button'] = gr.Button(value="Test & Save")
                shared.gr['ask_textbox'] = gr.Textbox(
                    label="Prompt", placeholder="Ask questions here ...", lines=3)
                with gr.Row():
                    shared.gr['clear_button'] = gr.Button(value="Clear")
                    shared.gr['submit_button'] = gr.Button(value="Submit")
            # right panel
            with gr.Column(scale=5):
                with gr.Tab("Chat"):
                    shared.gr['chatbot_history'] = gr.Chatbot(value=[])
                with gr.Tab("Result"):
                    shared.gr['result_dataframe'] = gr.Dataframe(
                        label="Output", headers=["1", "2", "3"], max_rows=30)
                    shared.gr['result_timeseries'] = gr.Timeseries(
                        label="Chart", x="time", y=['retail', 'food', 'other'])
                    shared.gr['result_label'] = gr.Label(label="Prediction")
                with gr.Tab("History"):
                    shared.gr['history_label'] = gr.Label()
                with gr.Tab("Debug Msg"):
                    shared.gr['debug_label'] = gr.Label()
        shared.gr['pdf_button'].upload(
            fn=load_pdf_to_table,
            inputs=shared.gr['pdf_button'],
            outputs=shared.gr['pdf_file_list'],
            api_name='upload_pdf'
        )
        shared.gr['pdf_file_list'].select(
            fn=on_pdf_selected,
            inputs=[shared.gr['pdf_button'], shared.gr['pdf_file_list']],
            outputs=[shared.gr['pdf_preview'], shared.gr['pdf_slider']]
        )
        shared.gr['csv_button'].upload(
            fn=load_csv_to_table,
            inputs=shared.gr['csv_button'],
            outputs=shared.gr['csv_file_list'],
            api_name="upload_csv")
        shared.gr['csv_file_list'].select(
            fn=on_csv_selected,
            inputs=shared.gr['csv_file_list'],
            outputs=shared.gr['csv_preview'])
        shared.gr['submit_button'].click(
            fn=ask_chatbot_questions,
            inputs=[shared.gr['chatbot_history'], shared.gr['ask_textbox']],
            outputs=[shared.gr['chatbot_history']],
            queue=False
        ).success(
            fn=get_chatbot_response,
            inputs=[shared.gr['chatbot_history'], shared.gr['ask_textbox']],
            # inputs=[shared.gr['chatbot_history'], shared.gr['ask_textbox'], llm],
            outputs=[shared.gr['chatbot_history']]
        ).success(
            fn=render_pdf,
            inputs=[],
            outputs=[shared.gr['pdf_preview'], shared.gr['pdf_slider'], shared.gr['pdf_pages']]
        )

    ui.launch()


if __name__ == '__main__':
    load_db()
    main()
