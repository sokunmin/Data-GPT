import gradio as gr
import numpy as np
import pandas as pd
import random
import shared


def feed_csv_into_collector(card_activity, categories=["retail", "food", "other"]):
    activity_range = random.randint(0, 100)
    return (
        card_activity,
        card_activity,
        {"fraud": activity_range / 100.0, "not fraud": 1 - activity_range / 100.0},
    )


def output_result_dataframe():
    pass


def output_result_chart():
    pass


def output_prediction():
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
                    shared.gradio['csv_file'] = gr.Timeseries(x="time", y=["retail", "food", "other"])
                    # shared.gradio['csv_file'] = gr.Files(file_count='multiple', file_types=['.csv'])
                    shared.gradio['csv_button'] = gr.Button(value="Upload")
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
                    shared.gradio["query_textarea"] = gr.TextArea(placeholder="Enter the SQL query here ...")
                    shared.gradio['query_button'] = gr.Button(value="Query")
                    with gr.Accordion("Query Result"):
                        shared.gradio['query_dataframe'] = gr.DataFrame(
                            headers=["Name", "Age", "Gender"],
                            value=preview_data,
                            datatype=["str", "str", "str"],
                            col_count=3
                        )

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
                with gr.Tab("Result"):
                    shared.gradio['result_dataframe'] = gr.Dataframe(label="Output", headers=["1", "2", "3"])
                    shared.gradio['result_timeseries'] = gr.Timeseries(label="Chart", x="time",
                                                                       y=['retail', 'food', 'other'])
                    shared.gradio['result_label'] = gr.Label(label="Prediction")
                with gr.Tab("History"):
                    shared.gradio['history_label'] = gr.Label()
                with gr.Tab("Debug Msg"):
                    shared.gradio['debug_label'] = gr.Label()
        shared.gradio['csv_button'].click(feed_csv_into_collector,
                                          [shared.gradio['csv_file']],
                                          [shared.gradio['result_dataframe'], shared.gradio['result_timeseries'], shared.gradio['result_label']],
                                          show_progress=False)
    ui.launch()


if __name__ == '__main__':
    main()