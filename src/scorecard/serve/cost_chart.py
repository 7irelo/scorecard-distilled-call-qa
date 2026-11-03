"""Break-even chart: monthly cost against monthly call volume, one line per way of running this."""

from __future__ import annotations

from pathlib import Path

from scorecard.serve.cost_model import (
    CostInputs,
    always_on_monthly_usd,
    scheduled_monthly_usd,
    serverless_monthly_usd,
)

# Fixed order, so a line's colour follows the option rather than its rank.
_SERIES_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
_INK, _MUTED, _GRID, _SURFACE = "#1f1f1d", "#6b6a63", "#e6e5e0", "#fcfcfb"
_WARNING = "PLACEHOLDER INPUTS: do not quote"


def chart_series(inputs: CostInputs) -> dict[str, list[float]]:
    """Monthly USD per option. API lines use batch prices, the fair comparison for post-call QA."""
    series: dict[str, list[float]] = {}
    volumes = inputs.volumes_per_month
    for option in inputs.self_hosted:
        series[f"{option.name}, always on"] = [
            always_on_monthly_usd(option, volume, inputs.fixed) for volume in volumes
        ]
        series[f"{option.name}, scheduled batches"] = [
            scheduled_monthly_usd(option, volume, inputs.fixed) for volume in volumes
        ]
        if option.serverless_usd_per_hour is not None:
            series[f"{option.name}, serverless"] = [
                float(serverless_monthly_usd(option, volume, inputs.fixed) or 0)
                for volume in volumes
            ]
    for api in inputs.apis:
        series[f"{api.name}, batch API"] = [
            api.usd_per_1k(batch=True) * volume / 1000 for volume in volumes
        ]
    return series


def render_chart(inputs: CostInputs, output: Path) -> Path:
    """Needs the `report` extra. The volume table holds the same numbers as a text alternative."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import FuncFormatter

    series = chart_series(inputs)
    if len(series) > len(_SERIES_COLOURS):
        raise ValueError(f"at most {len(_SERIES_COLOURS)} options fit one chart; use two charts")

    figure, axes = plt.subplots(figsize=(9, 5.2), dpi=160, facecolor=_SURFACE)
    axes.set_facecolor(_SURFACE)
    for (name, values), colour in zip(series.items(), _SERIES_COLOURS, strict=False):
        axes.plot(
            inputs.volumes_per_month,
            values,
            label=name,
            color=colour,
            linewidth=2,
            marker="o",
            markersize=4,
            markeredgecolor=_SURFACE,
            markeredgewidth=1,
        )
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axes.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
    axes.set_xlabel("Calls per month", color=_MUTED)
    axes.set_ylabel("Cost per month (USD)", color=_MUTED)
    axes.grid(True, which="major", color=_GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.tick_params(which="both", colors=_MUTED, length=0)
    for side in ("top", "right", "left", "bottom"):
        axes.spines[side].set_visible(False)
    title = "Self-hosting pays off only at enough volume"
    if not inputs.fully_measured:
        title = _WARNING
    axes.set_title(title, loc="left", color=_INK, fontsize=13, fontweight="bold", pad=12)
    legend = axes.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=_INK)
    legend.set_title(None)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, facecolor=_SURFACE)
    plt.close(figure)
    return output
