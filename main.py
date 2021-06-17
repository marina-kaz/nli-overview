import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State


import pandas as pd
import numpy as np
import plotly.express as px

import plotly.graph_objects as go
from matplotlib import pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import json


DATASETS = {
    "snli": pd.read_csv(r"datasets\snli.csv"),
    "mnli": pd.read_csv(r"datasets\mnli.csv"),
    "xnli": pd.read_csv(r"datasets\xnli.csv"),
    "validation": pd.read_csv(r"datasets\validation.csv"),
}

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

print("STARTING PROJECT")


def pie_by_langs(data):
    title_text = f"Samples distribution by languages in {data.upper()} set"
    data = DATASETS[data]
    fig = px.pie(
        values=data.language.value_counts().values,
        names=data.language.value_counts().index,
        color_discrete_sequence=px.colors.sequential.matter,
    )
    fig.update_layout(height=500, width=500, title_text=title_text)
    fig.update_traces(
        hoverinfo="label+percent",
        textinfo="percent",
        textfont_size=20,
        marker=dict(line=dict(color="#2f243a", width=0.5)),
    )
    return fig


def word_count_langwise(data):
    title_text = f"Word count distibution in {data.upper()} set"
    data = DATASETS[data]
    if data.language.unique().size != 1:
        fig = make_subplots(
            rows=5,
            cols=3,
            subplot_titles=data.language.unique(),
            specs=[[{"type": "bar"}] * 3] * 5,
            horizontal_spacing=0.12,
            vertical_spacing=0.1,
        )
        for i, lang in enumerate(data.language.unique()):
            fig.add_trace(
                go.Histogram(
                    histnorm="density",
                    x=data[data.language == lang].hypothesis.apply(
                        lambda x: len(x.split())
                    ),
                    name="hypothesis",
                    marker_color="#ef767a",
                    showlegend=(i == 0),
                ),
                row=divmod(i, 3)[0] + 1,
                col=divmod(i, 3)[1] + 1,
            )
            fig.add_trace(
                go.Histogram(
                    histnorm="density",
                    x=data[data.language == lang].premise.apply(
                        lambda x: len(x.split())
                    ),
                    marker_color="#9f87af",
                    name="premise",
                    showlegend=(i == 0),
                ),
                row=divmod(i, 3)[0] + 1,
                col=divmod(i, 3)[1] + 1,
            )
            fig.update_layout(barmode="overlay")
            fig.update_traces(opacity=0.75)
    else:
        wc = pd.DataFrame(
            {
                "word_count": data.hypothesis.apply(lambda x: len(x.split())),
                "type": "hypothesis",
            }
        ).append(
            pd.DataFrame(
                {
                    "word_count": data.premise.apply(lambda x: len(x.split())),
                    "type": "premise",
                }
            )
        )
        fig = px.histogram(
            wc,
            x="word_count",
            color="type",
            barmode="overlay",
            histnorm="density",
            color_discrete_map={"hypothesis": "#ef767a", "premise": "#9f87af"},
        )
    fig.update_layout(height=600, width=600, title_text=title_text)
    fig.update_annotations(font_size=12)
    return fig


def label_count_langwise(data):
    title_text = f"Labels distribution in {data.upper()} set"
    data = DATASETS[data]
    if data.language.unique().size != 1:
        fig = make_subplots(
            rows=5,
            cols=3,
            specs=[[{"type": "domain"}] * 3] * 5,
            horizontal_spacing=0.005,
            vertical_spacing=0.01,
        )
        for i, lang in enumerate(list(data.language.unique())):
            label_cnt = data[data.language == lang]["label"].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=label_cnt.index,
                    values=label_cnt.values,
                    hoverinfo="percent+value+label",
                    textinfo="percent",
                    textposition="inside",
                    textfont=dict(size=14),
                    title=lang,
                    titlefont=dict(size=15),
                    hole=0.7,
                    showlegend=True,
                    marker=dict(
                        line=dict(color="#2f243a", width=0.5),
                        colors=["#a2c5ac", "#9f87af", "#ef767a"],
                    ),
                ),
                row=divmod(i, 3)[0] + 1,
                col=divmod(i, 3)[1] + 1,
            )
    else:

        fig = px.pie(
            values=data.label.value_counts().values,
            names=data.label.value_counts().index,
            color_discrete_map={
                "entailment": "#a2c5ac",
                "neutral": "#9f87af",
                "contradiction": "#ef767a",
            },
        )
        fig.update_traces(
            hoverinfo="label+percent",
            textinfo="percent",
            textfont_size=20,
            marker=dict(
                colors=["#a2c5ac", "#9f87af", "#ef767a"],
                line=dict(color="#2f243a", width=2),
            ),
        )
    fig.update_layout(height=500, width=600, title_text=title_text)
    return fig


def get_accs(validation, preds):
    lang_counts = validation.language.value_counts().sort_index()
    tp_per_lang = (
        validation[validation["label"] == preds]
        .groupby("language")
        .agg({"language": ["count"]})
        .sort_index()
    )
    lang_names = lang_counts.index.tolist()
    lang_tuples = list(
        zip(
            lang_names,
            lang_counts.values.tolist(),
            tp_per_lang.iloc[:, 0].values.tolist(),
        )
    )
    names = [i[0] for i in lang_tuples]
    values = list(map(lambda x: round(float(x[2] / x[1]), 2) * 100, lang_tuples))
    return names, values


def model_report(data):

    with open(f"modeling\\history\\bert_{data}.json") as file:
        bert = json.load(file)
    with open(f"modeling\\history\\roberta_{data}.json") as file:
        roberta = json.load(file)
    validation = DATASETS["validation"]

    fig = make_subplots(
        rows=2,
        cols=2,
        shared_yaxes=True,
        subplot_titles=[
            f"BERT accuracy by each epoch\n(trained on {data})",
            f"ROBERTA accuracy by each epoch\n(trained on {data})",
            "BERT accuracy distribution by each language",
            "ROBERTA accuracy distibution by each language",
        ],
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(bert["accuracy"])),
            y=bert["accuracy"],
            mode="lines+markers",
            line=dict(color="#2f243a", width=4),
            name="train",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(bert["val_accuracy"])),
            y=bert["val_accuracy"],
            mode="lines+markers",
            name="validation",
            line=dict(color="#ef767a", width=4),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(roberta["accuracy"])),
            y=roberta["accuracy"],
            mode="lines+markers",
            line=dict(color="#69dc9e", width=4),
            name="train",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(roberta["val_accuracy"])),
            y=roberta["val_accuracy"],
            mode="lines+markers",
            name="validation",
            showlegend=False,
            line=dict(color="#9f87af", width=4),
        ),
        row=1,
        col=2,
    )

    names, values = get_accs(
        validation,
        pd.read_csv(f"modeling\\predictions\\bert_{data}_preds.csv").label.apply(
            lambda x: {0: "entailment", 1: "neutral", 2: "contradition"}[int(x)]
        ),
    )
    fig.add_trace(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker=dict(color="#9f87af", line=dict(color="black", width=0.2)),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    names, values = get_accs(
        validation,
        pd.read_csv(f"modeling\\predictions\\roberta_{data}_preds.csv").label.apply(
            lambda x: {0: "entailment", 1: "neutral", 2: "contradition"}[int(x)]
        ),
    )
    fig.add_trace(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker=dict(color="#ef767a", line=dict(color="black", width=0.2)),
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    return fig


app.layout = html.Div(
    [
        html.Div(
            [
                # 1 dash title, descr
                html.H1(
                    children="Natural Language Inference",
                    style={
                        "color": "#2f243a",
                        #  'fontSize': 14,
                        "text-align": "center",
                    },
                ),
                html.H3(
                    children="comprehensive task overview",
                    style={"text-align": "center", "color": "#444054"},
                ),
                html.P(
                    children="Natural Language Inference (or Recognizing Textual Entailment), \
is the task of determining the inference relation between two (short, ordered) texts: \
entailment, neutral, or contradition",
                    style={"text-align": "center", "fontSize": 20, "color": "#2f243a"},
                ),
                html.P(
                    children="Select an NLI corpus you would like to inspect",
                    style={"text-align": "center", "fontSize": 14, "color": "#2f243a"},
                )
                # 1 end
            ]
        ),
        html.Div(
            [
                dcc.Dropdown(
                    id="data",
                    options=[
                        {"label": "Stanford NLI", "value": "snli"},
                        {"label": "Cross-lingual NLI", "value": "xnli"},
                        {"label": "Multi-Genre NLI", "value": "mnli"},
                    ],
                    value="xnli",
                    multi=False,
                ),
                html.Div(
                    [
                        html.Div(
                            dcc.Graph(figure=pie_by_langs("validation")),
                            className="six columns",
                        ),
                        html.Div(dcc.Graph(id="eda_langs"), className="six columns"),
                    ],
                    className="row",
                ),
                html.Div(
                    [
                        html.Div(
                            dcc.Graph(figure=word_count_langwise("validation")),
                            className="six columns",
                        ),
                        html.Div(dcc.Graph(id="word_count"), className="six columns"),
                    ],
                    className="row",
                ),
                html.Div(
                    [
                        html.Div(
                            dcc.Graph(figure=label_count_langwise("validation")),
                            className="six columns",
                        ),
                        html.Div(dcc.Graph(id="label_count"), className="six columns"),
                    ],
                    className="row",
                ),
            ]
        ),
        html.H3(
            "BERT and XML-RoBERTa comparison",
            style={"text-align": "center", "color": "#444054"},
        ),
        dcc.Graph(id="model_report"),
    ]
)


@app.callback(
    Output(component_id="eda_langs", component_property="figure"),
    Output(component_id="word_count", component_property="figure"),
    Output(component_id="label_count", component_property="figure"),
    Output(component_id="model_report", component_property="figure"),
    Input(component_id="data", component_property="value"),
)
def train_eda(data):
    fig1 = pie_by_langs(data)
    fig2 = word_count_langwise(data)
    fig3 = label_count_langwise(data)
    fig4 = model_report(data)
    return fig1, fig2, fig3, fig4


if __name__ == "__main__":
    app.run_server(debug=True)
