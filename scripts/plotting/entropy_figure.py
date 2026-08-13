
# from 

















# Copyright (c) Meta Platforms, Inc. and affiliates.
import json
import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
from omegaconf import OmegaConf
from pydantic import BaseModel


class PlotEntropiesConfig(BaseModel):
    data_path: str | None
    chart_path: str
    score_override_path: str | None = None
    threshold_override: float | None = None

    class Config:
        extra = "forbid"


class PlotEntropiesData(BaseModel):
    text: str
    threshold: float = 1.335442066192627
    dataframe_json: str | None

    class Config:
        extra = "forbid"


def main():
    config_path = sys.argv[1]
    file_config = OmegaConf.load(config_path)
    # Omit program name and config file name
    cli_conf = OmegaConf.from_cli(sys.argv[2:])
    conf_dict = OmegaConf.to_container(
        OmegaConf.merge(file_config, cli_conf), resolve=True, throw_on_missing=True
    )
    plot_config = PlotEntropiesConfig(**conf_dict)
    with open(plot_config.data_path) as f:
        json_data = f.read()

    plot_data = PlotEntropiesData.model_validate_json(json_data)
    df = pd.read_json(plot_data.dataframe_json)
    print("LEN", len(df))
    if plot_config.threshold_override is None:
        threshold = plot_data.threshold
    else:
        threshold = plot_config.threshold_override
    if plot_config.score_override_path is not None:
        with open(plot_config.score_override_path) as f:
            scores = json.load(f)["score"]
            assert len(scores) == len(df)
            df["entropies"] = scores
            df["start"] = [1] + (df["entropies"] > threshold).values.tolist()[:-1]

    x_ticks = []
    for row in df.itertuples():
        position = row.position
        token = row.tokens
        x_ticks.append(f"{str(position).zfill(3)}|{token}")
    df["position_with_token"] = x_ticks
    print(df)

    x_axis = alt.Axis(
        labelExpr="split(datum.label, '|')[1]",
        grid=False,
        labelOverlap=False,
        labelAngle=0,
    )
    width = 1200
    height = 150
    base = alt.Chart(df).properties(width=width, height=height)
    points = base.mark_line(point=True).encode(
        x=alt.X("position_with_token:O", title=None, axis=x_axis),
        y=alt.Y(
            "entropies",
            title="Entropy of Next Byte",
        ),
    )
    rule = base.mark_rule(color="red", strokeDash=[4, 4]).encode(
        y=alt.datum(threshold),
    )
    patch_rules = (
        alt.Chart(df[df["start"] > 0])
        .properties(width=width, height=height)
        .mark_rule(color="#474747", strokeDash=[4, 2])
        .encode(x=alt.X("position_with_token:O", axis=x_axis))
    )

    chart = patch_rules + rule + points
    chart = chart.configure_axis(labelFontSize=15, titleFontSize=15)
    path = Path(plot_config.chart_path)
    path.parent.mkdir(exist_ok=True)
    chart.save(path)


if __name__ == "__main__":
    main()





















# -------------------------------------------------------








def plot_training(history, save_path=None):
    epochs = range(1, len(history['train_loss']) + 1)
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    def _ax(row, col, title, ylabel='Loss'):
        ax = fig.add_subplot(gs[row, col])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        return ax

    # 1. Total loss
    ax = _ax(0, 0, 'Total Loss')
    ax.plot(epochs, history['train_loss'], 'b-o', label='train', ms=4)
    ax.plot(epochs, history['val_loss'],   'r-o', label='val',   ms=4)
    ax.legend()

    # 2. BCS (Gaussianity) loss
    ax = _ax(0, 1, 'BCS Loss  (↓ = more Gaussian, no collapse)')
    ax.plot(epochs, history['train_bcs'], 'b-o', label='train', ms=4)
    ax.plot(epochs, history['val_bcs'],   'r-o', label='val',   ms=4)
    ax.axhline(0.0, color='k', ls='--', alpha=0.4)
    ax.legend()

    # 3. Invariance loss
    ax = _ax(0, 2, 'Invariance Loss  (↓ = ctx/tgt closer)')
    ax.plot(epochs, history['train_inv'], 'b-o', label='train', ms=4)
    ax.plot(epochs, history['val_inv'],   'r-o', label='val',   ms=4)
    ax.legend()

    # 4. Train/val gap
    ax = _ax(1, 0, 'Val − Train Gap  (red=overfit, green=underfit)')
    gap    = [v - t for t, v in zip(history['train_loss'], history['val_loss'])]
    colors = ['red' if g > 0 else 'green' for g in gap]
    ax.bar(epochs, gap, color=colors, alpha=0.7)
    ax.axhline(0, color='k', ls='--', alpha=0.5)

    # 5. Learning rate
    ax = _ax(1, 1, 'Learning Rate Schedule', ylabel='LR')
    ax.plot(epochs, history['lr'], color='orange', lw=1.5)
    ax.set_yscale('log')

    # 6. Config summary
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    ax.text(0.5, 0.5,
            f'Loss     : BCS  (lmbd=10.0)\n'
            f'EMA decay: {CFG.ema_decay}\n'
            f'Spans×len: {CFG.num_target_spans}×{CFG.target_span_length}\n'
            f'Hidden   : {CFG.hidden_size}  Layers: {CFG.num_layers}\n'
            f'LR       : {CFG.lr}  Epochs: {CFG.n_epochs}',
            ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))

    plt.suptitle(
        f'Text JEPA Stage-1 — {len(epochs)} epochs', fontsize=13, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f'Plot saved → {save_path}')

    plt.show()
    plt.close()


plot_training(history, save_path='/content/drive/MyDrive/text_jepa_training.png')