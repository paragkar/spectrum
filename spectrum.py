
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import plotly
import pandas as pd
import plotly.figure_factory as ff

import streamlit as st

st.set_page_config(layout="wide")

xl = pd.ExcelFile('spectrum_map.xlsx')
sheet = xl.sheet_names
df = pd.read_excel('spectrum_map.xlsx', sheet_name=sheet)

price = df["Master_Price_Sheet"]


st.sidebar.title('Navigation')

# x_axis_val = st.sidebar.selectbox('Select X-Axis Value', options = price.columns)
# y_axis_val = st.sidebar.selectbox('Select Y-Axis Value', options = price.columns)

Type = st.sidebar.selectbox('Select Price Type', options = ["FP","DP"])

Band = st.sidebar.selectbox('Select a Band', sorted(list(set(price["Band"]))))

price = price[(price["Band"]==Band) & (price["Year"] != 2018)]

price["Year"] = sorted([str(x) for x in price["Year"].values])


data = [go.Heatmap(
              	z = round(price[Type],1),
              	y = price["LSA"],
              	x = price["Year"],
              	xgap = 1,
            	ygap = 1,
            	hoverinfo ='text',
            	# text = hovertext1,
            	colorscale='Hot',
           		texttemplate="%{z}", 
    			textfont={"size":8},
    			reversescale=True,
                	),

       ]

fig = go.Figure(data=data )


fig.update_layout(uniformtext_minsize=8, 
                  uniformtext_mode='hide', 
                  xaxis_title=None, 
                  yaxis_title=None, 
                  yaxis_autorange='reversed',
                  font=dict(size=8),
                  template='simple_white',
                  paper_bgcolor=None,
                  height=550, width=1000,
                  # title="<b>"+title_fig1[Band]+"<b>",
                  margin=dict(t=80, b=50, l=50, r=50, pad=0),
                  title_x=0.51, title_y=0.99,
                  title_font=dict(size=14),
                  xaxis = dict(
                  side = 'top',
                  tickmode = 'linear',
                  tickangle=0,)
	               # tick0 =703,ß
                  # dtick = dtick[Band]),
                  # showlegend=True
                )


fig.update_xaxes(fixedrange=True,showline=True,linewidth=1.2,linecolor='black', mirror=True)
fig.update_yaxes(fixedrange=True,showline=True, linewidth=1.2, linecolor='black', mirror=True)

fig


