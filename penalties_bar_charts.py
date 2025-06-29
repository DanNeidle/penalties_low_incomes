#!/usr/bin/env python
"""
Generate static PNG and interactive HTML penalty charts by income decile.

Reads the Excel data, processes multiple tax penalties datasets, and outputs interactive HTML files with dropdowns for year and summary views.
"""

import os
from plotly.graph_objects import Figure
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from PIL import Image

# --- Configuration ---
DIRECTORY = "penalties_charts/"
DATA_FILE = os.path.join(DIRECTORY, "penalties_data.xlsx")
# Path to the logo image embedded in each chart
logo_file = "logo_full_white_on_blue.jpg"

datasets = [
    "£100 late filing penalty",
    "late payment penalties",
    "£300 late filing penalty",
]
years = ["2018/19", "2019/20", "2020/21", "2021/22", "2022/23"]

color_penalties_charged = "#D9534F"
color_penalties_appealed = "#4DB6AC"


standard_legend = dict(
    x=0.5, y=1.05, xanchor="center", yanchor="top", orientation="h", font=dict(size=14)
)

# Chart layout constants
NUM_TRACES_PER_YEAR_STACKED = 2
NUM_TOTAL_TRACES_STACKED = 2
CLUSTER_WIDTH = 0.15


def save_static_chart(fig: Figure, filename: str) -> None:
    """
    Saves a figure as a static PNG image at the given path.
    """
    fig.write_image(filename, scale=1, width=1200, height=800)
    print(f"  -> Saved static chart: {filename}")


def save_interactive_chart(fig: Figure, filename: str) -> None:
    """
    Saves a figure as an interactive HTML file at the given path.
    """
    fig.write_html(filename)
    print(f"  -> Saved interactive chart: {filename}")


def load_logo(logo_path: str) -> dict | None:
    """
    Load a logo image for embedding in all charts; returns Plotly image dict or None.
    """
    if not os.path.exists(logo_path):
        print(f"Warning: Logo file not found at {logo_path}")
        return None
    return dict(
        source=Image.open(logo_path),
        xref="paper",
        yref="paper",
        x=1,
        y=1.02,
        sizex=0.1,
        sizey=0.1,
        xanchor="right",
        yanchor="bottom",
    )


def read_all_data(dataset_path: str) -> pd.ExcelFile:
    """
    Read the Excel workbook once and return the ExcelFile for reuse.
    """
    if not os.path.exists(dataset_path):
        print(f"Error: The file {dataset_path} was not found.")
        exit()
    return pd.ExcelFile(dataset_path)


def read_personal_allowances(xls: pd.ExcelFile, sheet_name: str) -> dict[str, int]:
    """
    Read the 'personal allowance' sheet into a Year->allowance dict.
    """
    df = pd.read_excel(xls, sheet_name=sheet_name)
    df.set_index("Year", inplace=True)
    return df["personal allowance"].to_dict()


def read_dataset(
    xls: pd.ExcelFile, sheet_name: str
) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    """
    Read a penalties sheet into a DataFrame and extract decile starting incomes.
    """
    try:
        df = pd.read_excel(xls, header=[0, 1], sheet_name=sheet_name, nrows=10)
        decile_starting_incomes: dict[str, list[int]] = {}
        for year in years:
            key = (year, "decile_starting_income")
            if key in df.columns:
                decile_starting_incomes[year] = [v * 1000 for v in df[key].tolist()]
        return df, decile_starting_incomes
    except FileNotFoundError:
        print(f"Error: The file {xls} was not found.")
        exit()
    except Exception as e:
        print(f"Error reading {sheet_name} from {xls}: {e}")
        exit()


def ordinal(n: int) -> str:
    """
    Convert an integer to its ordinal representation (1st, 2nd, etc.).
    """
    return str(n) + {1: "st", 2: "nd", 3: "rd"}.get(
        n % 10 if n % 100 not in (11, 12, 13) else 0, "th"
    )


def prepare_penalties_data(
    df: pd.DataFrame, years: list[str]
) -> dict[str, tuple[list[int], list[int], list[str]]]:
    """
    Build per-year (charged, appealed, labels) tuples from DataFrame.
    """
    data: dict[str, tuple[list[int], list[int], list[str]]] = {}
    for year in years:
        cols = [
            (year, "decile_starting_income"),
            (year, "Penalty charged"),
            (year, "Penalty initially charged, but cancelled after appeal"),
        ]
        if all(c in df.columns for c in cols):
            incomes = df[year, "decile_starting_income"].tolist()
            labels = [
                f"{ordinal(i+1)} (£{incomes[i]}k - £{incomes[i+1]}k)"
                for i in range(len(incomes) - 1)
            ] + [f"{ordinal(len(incomes))} (£{incomes[-1]}k+)"]
            charged = df[year, "Penalty charged"].tolist()
            appealed = df[
                year, "Penalty initially charged, but cancelled after appeal"
            ].tolist()
            data[year] = (charged, appealed, labels)
    return data


def calculate_allowance_position(
    year: str, allowance: int, decile_starting_incomes: dict[str, list[int]]
) -> float:
    """
    Find allowance position within decile boundaries as a float index.
    """
    boundaries = decile_starting_incomes.get(year)
    if not boundaries:
        print("Error calculating allowance position")
        exit()
    for i in range(1, len(boundaries)):
        low, high = boundaries[i - 1], boundaries[i]
        if low <= allowance < high:
            span = high - low
            return i + (allowance - low) / span if span > 0 else float(i)
    if allowance >= boundaries[-1]:
        return float(len(boundaries))
    print("Error calculating allowance position")
    exit()


def sum_penalties_by_decile(df: pd.DataFrame) -> tuple[list[int], list[int]]:
    """
    Sum 'Penalty charged' and 'appealed' across all years per decile.
    """
    charged_slice = df.loc[:, pd.IndexSlice[:, "Penalty charged"]]  # type: ignore
    total_charged = (
        charged_slice.apply(pd.to_numeric, errors="coerce")
        .sum(axis=1)
        .fillna(0)
        .tolist()
    )

    appealed_slice = df.loc[:, pd.IndexSlice[:, "Penalty initially charged, but cancelled after appeal"]]  # type: ignore
    total_appealed = (
        appealed_slice.apply(pd.to_numeric, errors="coerce")
        .sum(axis=1)
        .fillna(0)
        .tolist()
    )
    return total_charged, total_appealed


def make_mask(on_indices: list[int], total: int) -> list[bool]:
    """Return a boolean mask of length *total* with True at *on_indices*."""
    mask = [False] * total  # type: ignore[operator]
    for idx in on_indices:
        mask[idx] = True
    return mask


def calculate_total_numbers_affected(
    valid_years: list[str],
    penalties: dict[str, tuple[list[int], list[int], list[str]]],
    decile_starting_incomes: dict[str, list[int]],
    personal_allowances: dict[str, int],
) -> tuple[int, int]:
    """
    Calculate total charged and appealed numbers for taxpayers below each year's personal allowance,
    summed across given years.
    """
    total_charged = 0.0
    total_appealed = 0.0
    for year in valid_years:
        charged, appealed, _ = penalties[year]
        boundaries = decile_starting_incomes[year]
        allowance = personal_allowances[year]
        for i, low in enumerate(boundaries):
            high = boundaries[i + 1] if i + 1 < len(boundaries) else None
            if high is None or allowance >= high:
                total_charged += charged[i]
                total_appealed += appealed[i]
            elif allowance <= low:
                break
            else:
                fraction = (allowance - low) / (high - low) if high > low else 1
                total_charged += charged[i] * fraction
                total_appealed += appealed[i] * fraction
                break
    return int(total_charged), int(total_appealed)
def _generate_non_taxpayer_annotations(
    charged: int, appealed: int, pos: float, y_level: float, label_prefix: str
) -> list[dict]:
    """
    Generates the text box and arrow annotations for non-taxpayer stats.
    """
    # The arrow will now span from just inside the y-axis (0.6)
    # to just before the personal allowance line (pos - 0.6).
    arrow_start_x = 0.6
    arrow_end_x = pos - 0.6
    
    # The text box should be centered in the middle of the new arrow span.
    text_center_x = (arrow_start_x + arrow_end_x) / 2

    text_annotation = dict(
        x=text_center_x,  # <-- UPDATED to be in the new center
        y=y_level,
        text=f"{charged+appealed:,} {label_prefix} sent penalties<br><br>of which {appealed:,} successfully appealed",
        showarrow=False,
        font=dict(size=16),
        align="center",
        xanchor="center",
        yanchor="middle",
        bgcolor="rgba(240, 240, 240, 0.85)",
        bordercolor="rgba(0, 0, 0, 0.3)",
        borderwidth=1,
        borderpad=10,
    )
    arrow_annotation = dict(
        ay=y_level,
        y=y_level,
        ax=arrow_start_x, 
        x=arrow_end_x,    
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        text="",
        arrowcolor="red",
        arrowwidth=4,
        arrowhead=2,
        startarrowhead=2,
        arrowside="end+start",
    )
    return [text_annotation, arrow_annotation]

def _add_stacked_traces(
    fig: Figure,
    valid_years: list[str],
    penalties: dict[str, tuple[list[int], list[int], list[str]]],
    deciles: np.ndarray,
) -> None:
    """Add per-year stacked bar traces for charged and appealed penalties."""
    for year in valid_years:
        charged, appealed, labels = penalties[year]
        fig.add_trace(
            go.Bar(
                x=deciles,
                y=charged,
                name="Penalty charged",
                marker=dict(color=color_penalties_charged, line_width=0),
                customdata=labels,
                hovertemplate=(
                    "Penalty charged<br>Decile: %{customdata}<br>Taxpayers: %{y:,}<extra></extra>"
                ),
                visible=False,
            )
        )
        
        fig.add_trace(
            go.Bar(
                x=deciles,
                y=appealed,
                name="Penalty appealed",
                marker=dict(color=color_penalties_appealed, line_width=0),
                customdata=labels,
                text=appealed,
                textposition='none', # we need this because we added the text to make the tooltip work, but don't want it displayed!
                hovertemplate=(
                    "Penalty appealed<br>Decile: %{customdata}<br>Taxpayers: %{text:,}<extra></extra>"
                ),
                visible=False,
                base=charged,
            )
        )


def _add_total_traces(
    fig: Figure,
    deciles: np.ndarray,
    total_charged: list[int],
    total_appealed: list[int],
    decile_labels: list[str],
) -> None:
    """Add traces representing totals across all years."""
    fig.add_trace(
        go.Bar(
            x=deciles,
            y=total_charged,
            name="Total charged",
            marker_color=color_penalties_charged,
            visible=False,
            customdata=decile_labels,
            hovertemplate=(
                "Total charged<br>Decile: %{customdata}<br>Taxpayers: %{y:,}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            x=deciles,
            y=total_appealed,
            name="Total appealed",
            marker_color=color_penalties_appealed,
            visible=False,
            base=total_charged,
            customdata=decile_labels,
            text=total_appealed,
            textposition='none', # we need this because we added the text to make the tooltip work, but don't want it displayed!
            hovertemplate=(
                "Total appealed<br>Decile: %{customdata}<br>Taxpayers: %{text:,}<extra></extra>"
            ),
        )
    )


def _add_cluster_traces(
    fig: Figure,
    valid_years: list[str],
    penalties: dict[str, tuple[list[int], list[int], list[str]]],
    deciles: np.ndarray,
) -> None:
    """Add clustered bar traces showing totals by year."""
    qualitative = px.colors.qualitative.Plotly
    for idx, year in enumerate(valid_years):
        charged, appealed, labels = penalties[year]
        totals = [c + a for c, a in zip(charged, appealed)]
        fig.add_trace(
            go.Bar(
                
                x=deciles,
                y=totals,
                name=year,
                marker_color=qualitative[idx % len(qualitative)],
                visible=False,
                customdata=labels,
                hovertemplate="Taxpayers: %{y:,}<extra></extra>",
            )
        )


def _create_year_button(
    idx: int,
    year: str,
    total_traces: int,
    penalties: dict[str, tuple[list[int], list[int], list[str]]],
    personal_allowances: dict[str, int],
    decile_starting_incomes: dict[str, list[int]],
    dataset: str,
) -> dict:
    """Return a dropdown button for a single year's stacked view."""
    on = [idx * NUM_TRACES_PER_YEAR_STACKED, idx * NUM_TRACES_PER_YEAR_STACKED + 1]
    mask = make_mask(on, total_traces)
    charged, appealed, labels = penalties[year]
    max_h = max(a + b for a, b in zip(charged, appealed))
    pos = calculate_allowance_position(year, personal_allowances[year], decile_starting_incomes)

    shape = dict(
        type="line",
        x0=pos - 0.5,
        x1=pos - 0.5,
        y0=0,
        y1=max_h * 1.1,
        line=dict(color="Black", width=3),
    )

    allowance_anno = dict(
        x=pos - 0.6,
        y=max_h * 1.05,
        text=f"Personal allowance<br>£{personal_allowances[year]:,}",
        showarrow=False,
        font=dict(size=16),
        align="right",
        xanchor="right",
        yanchor="bottom",
    )

    non_tax_charged, non_tax_appealed = calculate_total_numbers_affected(
        [year], penalties, decile_starting_incomes, personal_allowances
    )
    y_pos_anno = max_h * 0.75
    non_tax_annos = _generate_non_taxpayer_annotations(
        non_tax_charged, non_tax_appealed, pos, y_pos_anno, "non-taxpayers"
    )

    return dict(
        label=year,
        method="update",
        args=[
            {"visible": mask},
            {
                "title.text": f"{dataset} - {year}",
                "barmode": "stack",
                "shapes": [shape],
                "annotations": [allowance_anno] + non_tax_annos,
                "xaxis.ticktext": labels,
                "xaxis.tickfont": dict(size=14),
            },
        ],
    )


def _create_total_button(
    start: int,
    valid_years: list[str],
    total_traces: int,
    penalties: dict[str, tuple[list[int], list[int], list[str]]],
    decile_starting_incomes: dict[str, list[int]],
    personal_allowances: dict[str, int],
    deciles: np.ndarray,
    avg_allowance: float,
    dataset: str,
    max_total: int,
    years: list[str],
) -> dict:
    """Return the button definition for the total view."""
    mask_total = make_mask([start, start + 1], total_traces)
    avg_pos = calculate_allowance_position(valid_years[0], avg_allowance, decile_starting_incomes) # type: ignore
    total_below_charged, total_below_appealed = calculate_total_numbers_affected(
        valid_years, penalties, decile_starting_incomes, personal_allowances
    )
    total_people_affected_y = max_total * 0.75

    avg_allowance_anno = dict(
        x=avg_pos - 0.6,
        y=max_total * 1.05,
        text=f"Average personal allowance<br>£{int(avg_allowance):,}",
        showarrow=False,
        font=dict(size=16),
        align="right",
        xanchor="right",
        yanchor="bottom",
    )

    total_non_tax_annos = _generate_non_taxpayer_annotations(
        total_below_charged,
        total_below_appealed,
        avg_pos,
        total_people_affected_y,
        "total non-taxpayers",
    )

    return dict(
        label="Total (All Years)",
        method="update",
        args=[
            {"visible": mask_total},
            {
                "title.text": f"{dataset.replace('penalty', 'penalties')}, {years[0][:4]} to 20{years[-1][-2:]}",
                "barmode": "stack",
                "shapes": [
                    dict(
                        type="line",
                        x0=avg_pos - 0.5,
                        x1=avg_pos - 0.5,
                        y0=0,
                        y1=max_total * 1.1,
                        line=dict(color="Black", width=3),
                    )
                ],
                "annotations": [avg_allowance_anno] + total_non_tax_annos,
                "xaxis.ticktext": [ordinal(i + 1) for i in range(len(deciles))],
                "xaxis.tickfont": dict(size=18)
            },
        ],
    )


def _create_cluster_button(
    start: int,
    valid_years: list[str],
    total_traces: int,
    deciles: np.ndarray,
    dataset: str,
) -> dict:
    """Return the button for the clustered totals view."""
    cluster_start = start + NUM_TOTAL_TRACES_STACKED
    mask_cluster = make_mask(list(range(cluster_start, cluster_start + len(valid_years))), total_traces)
    return dict(
        label="Totals by Year (Clustered)",
        method="update",
        args=[
            {"visible": mask_cluster},
            {
                "title.text": f"Total Penalties for {dataset} by Year",
                "barmode": "group",
                "shapes": [],
                "annotations": [],
                "xaxis.ticktext": [ordinal(i + 1) for i in range(len(deciles))],
                "xaxis.tickfont": dict(size=18)
            },
        ],
    )


def _base_layout(active: int, buttons: list[dict], deciles: np.ndarray, logo: dict | None) -> dict:
    """Return the common chart layout."""
    return dict(
        font=dict(family="Arial, Helvetica, sans-serif"),
        title_font_size=40,
        updatemenus=[
            dict(
                active=active,
                buttons=buttons,
                direction="down",
                x=0.89,
                xanchor="right",
                y=1.01,
                yanchor="bottom",
                pad=dict(l=20, r=20, t=10, b=10),
                font=dict(size=18),
            )
        ],
        xaxis=dict(title_text="Income Decile", title_font_size=32, tickvals=deciles, range=[0.5, 10.5]),
        yaxis=dict(title_text="Number of Taxpayers", title_font_size=32, tickformat=",", tickfont=dict(size=18)),
        legend=standard_legend,
        template="seaborn",
        images=[logo] if logo else [],
        margin=dict(t=100, b=100),
    )


def create_main_interactive_chart(
    years: list[str],
    penalties: dict[str, tuple[list[int], list[int], list[str]]],
    deciles: np.ndarray,
    dataset: str,
    logo: dict | None,
    personal_allowances: dict[str, int],
    decile_starting_incomes: dict[str, list[int]],
    total_charged: list[int],
    total_appealed: list[int],
) -> Figure:
    """Create the interactive penalty chart with dropdown controls."""
    fig: Figure = go.Figure()
    
    buttons: list[dict] = []

    decile_labels_base = [ordinal(int(i)) for i in deciles]
    valid_years = [y for y in years if y in penalties]

    avg_allowance = sum(personal_allowances[yr] for yr in valid_years) / len(valid_years)
    max_total = max(c + a for c, a in zip(total_charged, total_appealed))

    _add_stacked_traces(fig, valid_years, penalties, deciles)
    _add_total_traces(fig, deciles, total_charged, total_appealed, decile_labels_base)
    _add_cluster_traces(fig, valid_years, penalties, deciles)

    total_traces = len(fig.data)  # type: ignore[operator]

    for i, year in enumerate(valid_years):
        buttons.append(
            _create_year_button(
                i,
                year,
                total_traces,
                penalties,
                personal_allowances,
                decile_starting_incomes,
                dataset,
            )
        )

    start = len(valid_years) * NUM_TRACES_PER_YEAR_STACKED
    buttons.append(
        _create_total_button(
            start,
            valid_years,
            total_traces,
            penalties,
            decile_starting_incomes,
            personal_allowances,
            deciles,
            avg_allowance,
            dataset,
            max_total,
            years,
        )
    )
    buttons.append(_create_cluster_button(start, valid_years, total_traces, deciles, dataset))

    active = len(valid_years) - 1 if valid_years else 0
    init_layout = buttons[active]["args"][1]
    base_layout = _base_layout(active, buttons, deciles, logo)
    fig.update_layout({**base_layout, **init_layout})

    init_vis = buttons[active]["args"][0]["visible"]
    for idx, trace in enumerate(fig.data):
        trace.visible = init_vis[idx]  # type: ignore[attr-defined]

    return fig

def main() -> None:
    """
    Entry point: read data once, loop datasets, generate & save charts.
    """
    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)
        print(f"Created directory: {DIRECTORY}")

    xls = read_all_data(DATA_FILE)
    logo = load_logo(logo_file)
    personal_allowances = read_personal_allowances(xls, "personal allowance")

    deciles = np.arange(1, 11)
    # Each dataset corresponds to a sheet in the workbook
    # and produces an independent chart.
    for dataset in datasets:
        print(f"\n--- Processing dataset: {dataset} ---")
        df, decile_starting_incomes = read_dataset(xls, dataset)
        penalties = prepare_penalties_data(df, years)

        if not penalties:
            print(f"No data found for dataset '{dataset}'. Skipping.")
            continue

        total_charged, total_appealed = sum_penalties_by_decile(df)
        interactive_fig = create_main_interactive_chart(
            years,
            penalties,
            deciles,
            dataset,
            logo,
            personal_allowances,
            decile_starting_incomes,
            total_charged,
            total_appealed,
        )
        save_interactive_chart(
            interactive_fig,
            os.path.join(DIRECTORY, f"interactive_{dataset.replace(" ", "_")}_analysis.html"),
        )


if __name__ == "__main__":
    main()
