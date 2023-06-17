#!/usr/bin/env python

import plotly.graph_objects as go
import numpy as np
import pandas as pd
from PIL import Image

datasets = ["late filing", "late payment"]

logo_file = "logo_full_white_on_blue.jpg"
colors_charged = ['rgba(220, 0, 0, 0.8)', 'rgba(204, 0, 0, 0.8)', 'rgba(187, 0, 0, 0.8)', 'rgba(170, 0, 0, 0.8)']
colors_appealed = ['rgba(0, 128, 0, 0.8)', 'rgba(0, 115, 0, 0.8)', 'rgba(0, 102, 0, 0.8)', 'rgba(0, 90, 0, 0.8)']
years = ['2018/19', '2019/20', '2020/21', '2021/22']

def load_logo(logo_path):
    return dict(
        source=Image.open(logo_path),
        xref="paper", yref="paper",
        x=1, y=1.05,
        sizex=0.1, sizey=0.1,
        xanchor="right", yanchor="bottom"
    )

def read_dataset(dataset_path, sheet_name):
    df = pd.read_excel(dataset_path, header=[0, 1], sheet_name=sheet_name)
    decile_starting_incomes = {}
    for year in years:
        decile_starting_incomes[year] = [income * 1000 for income in df[year, 'decile_starting_income'].tolist()]
    return df, decile_starting_incomes


def read_personal_allowances(dataset_path, sheet_name):
    df = pd.read_excel(dataset_path, sheet_name=sheet_name)
    df.set_index('Year', inplace=True)
    return df['personal allowance'].to_dict()

def ordinal(n):
    if 10 <= n <= 20:
        return str(n) + 'th'
    else:
        return  str(n) + {1 : 'st', 2 : 'nd', 3 : 'rd'}.get(n % 10, "th")



def prepare_penalties_data(df, years):
    labels = {}
    for year in years:
        decile_starting_incomes = df[year, 'decile_starting_income'].tolist()
        labels[year] = [f"{ordinal(i+1)} (£{decile_starting_incomes[i]}k - £{decile_starting_incomes[i+1]}k)" for i in range(len(decile_starting_incomes)-1)] + [f"{ordinal(len(decile_starting_incomes))} (£{decile_starting_incomes[-1]}k+)"]
    return {year: (df[year, 'Penalty charged'].tolist(),
                   df[year, 'Penalty initially charged, but cancelled after appeal'].tolist(),
                   labels[year]) for year in years}

def calculate_allowance_position(year, allowance, decile_starting_incomes):
    decile_boundaries = decile_starting_incomes[year]
    for i in range(1, len(decile_boundaries)):
        if decile_boundaries[i - 1] <= allowance < decile_boundaries[i]:
            return i + (allowance - decile_boundaries[i - 1]) / (decile_boundaries[i] - decile_boundaries[i - 1])
    if allowance >= decile_boundaries[-1]:
        return len(decile_boundaries)


def add_clustered_bars(fig, years, penalties, colors_charged, colors_appealed, deciles, width):
    for i, year in enumerate(years):
        charged, appealed, _ = penalties[year]
        for y, color, name, legendgroup in zip([charged, appealed], [colors_charged, colors_appealed],
                                               ['Penalty charged', 'Penalty appealed'], ['Penalty charged', 'Penalty appealed']):
            fig.add_trace(
                go.Bar(x=deciles + i * width, y=y, width=width, name=name if i == 0 else '',
                       legendgroup=legendgroup, marker_color=color[i], marker_line_width=1, showlegend=i == 0))


def add_year_annotations(fig, years, deciles, width):
    for j in range(1, 11):
        for i, year in enumerate(years):
            fig.add_annotation(
                x=j + i * width, y=0, text=year, showarrow=False,
                font=dict(size=14, color="black", family="Arial, bold"),
                textangle=90, xshift=-1, yshift=33)

def create_clustered_barchart(title, years, width, logo, decile_labels):
    return go.Figure().update_layout(
        images=[logo],
        barmode='stack', title=title,
        xaxis=dict(
            title='Income Decile',
            tickvals=[i + (len(years) - 1) * width / 2 for i in range(1, 11)],
            ticktext=decile_labels),
        yaxis=dict(tickformat=','),
        legend=dict(x=0.85, y=-0.10),
        template="seaborn")

# this plots a barchart for particular years (currently the two most recent ones, i.e. range starts -2)
def create_yearly_barcharts(years, penalties, deciles, dataset, logo, colors_charged, colors_appealed, personal_allowances, decile_starting_incomes):
    yearly_barchart_fig = {}
    for i in range(-2, 0):
        year, (charged, appealed, decile_labels) = years[i], penalties[years[i]]
        fig_year = go.Figure()

        for y, color, name in zip([charged, appealed], [colors_charged[0], colors_appealed[0]],
                                  ['Penalty charged', 'Penalty appealed']):
            fig_year.add_trace(
                go.Bar(x=deciles, y=y, name=name, marker_color=color, offsetgroup=0,
                       base=charged if name == 'Penalty appealed' else None))

        max_height = max(a + b for a, b in zip(charged, appealed))

        # Add the vertical line at the position of the personal allowance
        allowance_position = calculate_allowance_position(year, personal_allowances[year], decile_starting_incomes) - 0.5

        fig_year.add_shape(
            dict(type="line", x0=allowance_position, x1=allowance_position, y0=0, y1=max_height * 1.1,
                line=dict(color="Black", width=3))
        )
        
        fig_year.add_annotation(
            dict(
                x=allowance_position - 0.1,
                y=max_height * 1,
                text=f'Personal allowance<br>£{personal_allowances[year]} - incomes<br>under this are not taxed',
                showarrow=False,
                font=dict(
                    size=12,
                    color="black"
                ),
                align="right",
                xanchor="right",
                yanchor="bottom"
            )
        )

        
        fig_year.update_layout(
            images=[logo],
            title_text=f"Self-assessment taxpayers in each income decile assessed with {dataset} penalties - {year}",
            xaxis=dict(title="Income Decile", tickvals=list(range(1, 11)), ticktext=decile_labels),
            yaxis=dict(title="Number of Taxpayers", titlefont_size=16, tickfont_size=14, tickformat=',',
                       range=[0, max_height * 1.1]),
            legend=dict(x=0.85, y=-0.10), barmode='stack', bargap=0.15, bargroupgap=0.1,
            template="seaborn")

        fig_year.update_traces(
            hovertemplate='Decile: %{x}<br>Count: %{y}<extra></extra>',
            marker_line_color='rgba(0, 0, 0, 0)',
            marker_line_width=1.5, opacity=1)

        yearly_barchart_fig[year] = fig_year
    return yearly_barchart_fig


def sum_penalties_by_decile(df):
    total_charged = df.loc[:, pd.IndexSlice[:, 'Penalty charged']].sum(axis=1).tolist()
    total_appealed = df.loc[:, pd.IndexSlice[:, 'Penalty initially charged, but cancelled after appeal']].sum(axis=1).tolist()
    return total_charged, total_appealed

def create_total_barchart(deciles, decile_names, total_charged, total_appealed, logo, dataset, personal_allowances, decile_starting_incomes):
    fig = go.Figure()
    for y, color, name in zip([total_charged, total_appealed], [colors_charged[0], colors_appealed[0]],
                              ['Total Penalty charged', 'Total Penalty appealed']):
        fig.add_trace(
            go.Bar(x=deciles, y=y, name=name, marker_color=color, offsetgroup=0,
                   base=total_charged if name == 'Total Penalty appealed' else None))

    max_height = max(a + b for a, b in zip(total_charged, total_appealed))

    fig.update_layout(
        images=[logo],
        title_text=f"Total number of {dataset} penalties on taxpayers in each income decile - 2018/19  to 2021/22",
        xaxis=dict(title="Income Decile", tickvals=list(range(1, 11)), ticktext=decile_names),
        yaxis=dict(title="Number of Taxpayers", titlefont_size=16, tickfont_size=14, tickformat=',',
                   range=[0, max_height * 1.1]),
        legend=dict(x=0.80, y=-0.10), barmode='stack', bargap=0.15, bargroupgap=0.1,
        template="seaborn")
    
    # Add the vertical line at the position of the personal allowance
    allowance_positions = 0
    for year in years:
        allowance_positions += calculate_allowance_position(year, personal_allowances[year], decile_starting_incomes) - 0.5
        
    average_allowance_position = allowance_positions / len(years)

    fig.add_shape(
        dict(type="line", x0=average_allowance_position, x1=average_allowance_position, y0=0, y1=max_height * 1.1,
            line=dict(color="Black", width=3))
    )
    
    fig.add_annotation(
        dict(
            x=average_allowance_position - 0.1,
            y=max_height * 1,
            text=f'Personal allowance - incomes<br>under this are not taxed',
            showarrow=False,
            font=dict(
                size=12,
                color="black"
            ),
            align="right",
            xanchor="right",
            yanchor="bottom"
        )
    )

    fig.update_traces(
        hovertemplate='Decile: %{x}<br>Count: %{y}<extra></extra>',
        marker_line_color='rgba(0, 0, 0, 0)',
        marker_line_width=1.5, opacity=1)

    return fig



def main():
        
    deciles = np.array(list(range(1, 11)))
    decile_names = [ordinal(n) for n in range(1, 11)]
    
    width = 0.2
    logo = load_logo(logo_file)
    personal_allowances = read_personal_allowances('penalties_data.xlsx', 'personal allowance')

    # this runs first for the late filing dataset, then late payment
    for dataset in datasets:
        title = f"Number of {dataset} penalties on taxpayers in each income decile - 2018/19  to 2021/22"
        df, decile_starting_incomes = read_dataset('penalties_data.xlsx', dataset)
        penalties = prepare_penalties_data(df, years)
        
        clustered_barchart_fig = create_clustered_barchart(title, years, width, logo, decile_names)
        add_clustered_bars(clustered_barchart_fig, years, penalties, colors_charged, colors_appealed, deciles, width)
        add_year_annotations(clustered_barchart_fig, years, deciles, width)
        
        clustered_barchart_fig.write_image(f"penalties_{dataset}_all.png", scale=1, width=1200, height=800)
        
        total_charged, total_appealed = sum_penalties_by_decile(df)
        total_barchart_fig = create_total_barchart(deciles, decile_names, total_charged, total_appealed, logo, dataset, personal_allowances, decile_starting_incomes)
        total_barchart_fig.write_image(f"total_penalties_{dataset}.png", scale=1, width=1200, height=800)

        yearly_barchart_figs = create_yearly_barcharts(years, penalties, deciles, dataset, logo, colors_charged, colors_appealed, personal_allowances, decile_starting_incomes)
        for year in yearly_barchart_figs:
            yearly_barchart_figs[year].write_image(f"penalties_{dataset}_{year.replace('/','-')}.png", scale=1, width=1200, height=800)
            


if __name__ == '__main__':
    main()
