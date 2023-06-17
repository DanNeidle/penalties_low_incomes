#!/usr/bin/env python

import numpy as np
import plotly.graph_objects as go
from PIL import Image

logo_file = "logo_full_white_on_blue.jpg"

def load_logo(logo_path):
    return dict(
        source=Image.open(logo_path),
        xref="paper", yref="paper",
        x=1, y=1.05,
        sizex=0.1, sizey=0.1,
        xanchor="right", yanchor="bottom"
    )

def main():
  # Create x values: pre-tax income from £1,000 to £120,000
  x = np.linspace(1000, 120000, 1000)

  # Calculate y values based on the provided formula
  y = 100 * 100/(x/52)
  y = np.where(y > 100, None, y)



  logo = load_logo(logo_file)

  # Create a line chart
  fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines', line=dict(color='red', width=4)))

  # Decile boundaries
  deciles = [0, 7000, 11000, 15000, 20000, 26000, 34000, 45000, 57000, 100000, 120000]

  # Set chart's layout
  fig.update_layout(title='What % of weekly income is a £100 fixed penalty, for various levels of gross income?<br>(deciles shaded)',
                    images=[logo],
                    xaxis_title='Pre-tax income (£)',
                    yaxis_title='% of weekly income',
                    yaxis=dict(
                      title='% of weekly income',
                      ticksuffix='%', # Add % suffix to y-axis labels
                      range=[0,100],
                      showgrid=True, 
                      zeroline=True, 
                      zerolinewidth=1, 
                      zerolinecolor='black'
                  ),
                    xaxis=dict(
                          title='Pre-tax income',
                          tickprefix='£', # Add £ prefix to x-axis labels
                          range=[0,120000],
                          showgrid=True, 
                          zeroline=True, 
                          zerolinewidth=1, 
                          zerolinecolor='black'
                      ),
                    shapes=[
                      # Shade deciles with alternate colors
                      *[dict(type="rect",
                            xref="x",
                            yref="paper",
                            x0=deciles[i],
                            y0=0,
                            x1=deciles[i+1],
                            y1=1,
                            fillcolor="lightblue" if i % 2 == 0 else "grey",
                            opacity=0.2,
                            layer="below",
                            line_width=0,) for i in range(len(deciles)-1)]
                    ])


  # Show the chart
  fig.write_image(f"penalties_vs_weekly_income.png", scale=1, width=1200, height=800)


if __name__ == '__main__':
    main()