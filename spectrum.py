
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import plotly
import pandas as pd
import plotly.figure_factory as ff

import streamlit as st

st.set_page_config(layout="wide")

operators = {700: {'Vacant':0,'Railways':1,'Govt':2,'RJIO':3,'BSNL':4},
             800: {'Vacant':0,'RCOM':1,'Govt':2,'RJIO':3,'Bharti':4, 'MTS':5, 'BSNL':6},
             900:{'Vacant':0,'RCOM':1,'Govt':2,'Railways':3,'Bharti':4, 'AircelU':5, 
                  'BSNLU':6,'MTNLU':7,'BhartiU':8,'VI':9,'VIU':10},
             1800: {'Vacant':0,'RCOM':1,'Govt':2,'RJIO':3,'Bharti':4,
                    'BhartiU':5, 'AircelR':6, 'BSNL':7,'MTNL':8,'VI':9,'VIU':10,'AircelU':11, 'Aircel':12},
             2100: {'Vacant':0,'RCOM':1,'Govt':2,'Bharti':3, 'BSNL':4,'MTNL':5,'VI':6, 'Aircel':7},
             2300: {'Vacant':0,'RJIO':1,'Govt':2,'Bharti':3, 'VI':4},
             2500: {'Vacant':0,'Govt':1,'BSNL':2, 'VI':3},
             3500: {'Vacant':0,'Bharti':1,'RJIO':2,'BSNL':3, 'MTNL':4,'VI':5},
             26000: {'Vacant':0,'Bharti':1,'RJIO':2,'BSNL':3, 'MTNL':4,'VI':5,'Adani':6}
            }

xgap = {700:1, 
        800:1, 
        900:0.5, 
        1800:0, 
        2100:1, 
        2300:1, 
        2500:1, 
        3500:1, 
        26000:1}


layout = go.Layout(uniformtext_minsize=8, 
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
#                   tickangle=tickangle,
		  )
	               # tick0 =703,ß
                  # dtick = dtick[Band]),
                  # showlegend=True
		  )

#preparing color scale for heatmap which chnages in steps (discrete)
def stepcolscale(operators, colcodes):
    scale = [round(x/(len(operators)),2) for x in range(len(operators)+1)]
    colors =[]
    for k, v  in operators.items():
        colors.append(colcodes.loc[k,:].values[0])

    col= pd.concat([pd.DataFrame(scale),pd.DataFrame(colors)], axis=1)

    col.columns =["colscale", "colors"]
    col["colscaleshift"] = col.iloc[:,0].shift(-1)
    # col = col.fillna(1)
    lst=[]

    for line in col.values:
        lst.append((line[0],line[1])),
        lst.append((line[2],line[1])),
    lst = lst[:-2]
    return lst


xl = pd.ExcelFile('spectrum_map.xlsx')
sheet = xl.sheet_names
df = pd.read_excel('spectrum_map.xlsx', sheet_name=sheet)

colcodes = df["ColorCodes"]
colcodes=colcodes.set_index("Description")


# st.sidebar.title('Navigation')

price = df["Master_Price_Sheet"]

df = df[df["Band"] != 600]

Bands = sorted(list(set(price["Band"])))

Feature = st.sidebar.selectbox('Select a Feature', options = ["Map","Price"])

output = st.empty()
if Feature == "Price":
	Band = st.sidebar.selectbox('Select a Band', Bands)
	price = price[(price["Band"]==Band) & (price["Year"] != 2018)]
	price["Year"] = sorted([str(x) for x in price["Year"].values])
	Type = st.sidebar.selectbox('Select Price Type', options = ["FP","DP"])
	tickangle=0

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
	fig = go.Figure(data=data, layout =layout)
	st.write(fig)

output = st.empty()
if Feature == "Map":
	Band = st.sidebar.selectbox('Select a Band', Bands)
	freqtab = str(Band)+"MHz"
	sf = df[freqtab]
	sf = sf.set_index("LSA")
	operators =operators[Band]
	sf[sf.columns] = sf[sf.columns].replace(operators)
	colorscalestep = stepcolscale(operators, colcodes)
	tickangle = -90
	
	data = [go.Heatmap(
              z = sf.values,
              y = sf.index,
              x = sf.columns,
              xgap = xgap[Band],
              ygap = 1,
#               hoverinfo ='text',
#               text = hovertext1,
              colorscale=colorscalestep,
              colorbar=dict(
              tickvals = list(operators.values()),
              # tickvals = tickval,
              ticktext = list(operators.keys()),
              dtick=1,
              tickmode="array"),
                ),

       		]	

	fig = go.Figure(data=data, layout = layout )
	st.write(fig)


# layout = layout(uniformtext_minsize=8, 
#                   uniformtext_mode='hide', 
#                   xaxis_title=None, 
#                   yaxis_title=None, 
#                   yaxis_autorange='reversed',
#                   font=dict(size=8),
#                   template='simple_white',
#                   paper_bgcolor=None,
#                   height=550, width=1000,
#                   # title="<b>"+title_fig1[Band]+"<b>",
#                   margin=dict(t=80, b=50, l=50, r=50, pad=0),
#                   title_x=0.51, title_y=0.99,
#                   title_font=dict(size=14),
#                   xaxis = dict(
#                   side = 'top',
#                   tickmode = 'linear',
#                   tickangle=tickangle,)
# 	               # tick0 =703,ß
#                   # dtick = dtick[Band]),
#                   # showlegend=True
#                 )


# fig.update_xaxes(fixedrange=True,showline=True,linewidth=1.2,linecolor='black', mirror=True)
# fig.update_yaxes(fixedrange=True,showline=True, linewidth=1.2, linecolor='black', mirror=True)

# st.write(fig)


