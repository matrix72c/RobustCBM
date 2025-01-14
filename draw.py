import os

import pandas as pd
import plotly.graph_objects as go
import wandb
from plotly.subplots import make_subplots


def plot_by_mode(source, adv_mode, output_folder="figs"):
    source = source[source['adv_mode'] == adv_mode]

    datasets_metrics = [("CUB", "Acc@1"), ("CUB", "ASR@1"), ("AwA", "Acc@1"), ("AwA", "ASR@1")]
    base_models = ["vgg16", "resnet50", "vit"]

    subplot_titles = []
    for base in base_models:
        for dataset, metric in datasets_metrics:
            subplot_titles.append(f"{base} - {dataset} - {metric}")

    fig = make_subplots(
        rows=3,
        cols=4,
        subplot_titles=subplot_titles,
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
    )

    model_colors = {}
    legends = set()
    for i, base in enumerate(base_models):
        for j, (dataset, metric) in enumerate(datasets_metrics):
            subset = source[
                (source['dataset'] == dataset) &
                (source['base'] == base)
                ]
            subset = subset.dropna(subset=[metric])

            for run in subset['model'].unique():
                data = subset[subset['model'] == run].iloc[0]
                if run not in model_colors:
                    model_colors[run] = \
                        f'rgb({(len(model_colors) * 100) % 255}, {(len(model_colors) * 200) % 255}, {(len(model_colors) * 300) % 255})'
                color = model_colors[run]

                if run not in legends:
                    legends.add(run)
                    showlegend = True
                else:
                    showlegend = False
                fig.add_trace(
                    go.Scatter(
                        x=list(range(5)),
                        y=data[metric],
                        mode='lines+markers',
                        name=run,
                        legendgroup=run,
                        marker=dict(color=color),
                        showlegend=showlegend
                    ),
                    row=i + 1,
                    col=j + 1
                )

    fig.update_layout(
        height=1500,
        width=2000,
        title_text=f"adv_mode = {adv_mode}",
        showlegend=True
    )

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"pic_{adv_mode}.png")
    fig.write_image(output_path, engine="orca")

    fig.show()


wandb.login(key="872d67256b614f408c84eb1138d8a2acd073d911")
api = wandb.Api()

sweep = api.sweep("matrix72c-jesse/RobustCBM/iaykwf0h")
sweep_runs = sweep.runs
tb = []

for r in sweep_runs:
    cfg = r.config
    cfg.update(r.summary)
    tb.append(cfg)

df = pd.DataFrame(tb)

res = df.copy()
res['name'] = res['adv_mode'] + '_' + res['base'] + '_' + res['dataset'] + '_' + res['model']

plot_by_mode(res, adv_mode='adv')
plot_by_mode(res, adv_mode='std')
