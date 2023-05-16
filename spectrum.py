
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

#if "1" the expiry tab is present and if "0" then not present
ExpTab = {700:1, 
          800:1, 
          900:1,
          1800:1,
          2100:1, 
          2300:1, 
          2500:1,
          3500:1,
          26000:1}

ChannelSize = {700:2.5, 
               800:0.625, 
               900:0.2,
               1800:0.2,
               2100:2.5, 
               2300:2.5, 
               2500:5,
               3500:5,
               26000:25}

# scale of the x axis plots

dtick = {700:1, 
         800:0.25, 
         900:0.4, 
         1800:1, 
         2100:1, 
         2300:1, 
         2500:2, 
         3500:5, 
         26000:50}

# vertical line widths

xgap = {700:1, 
        800:1, 
        900:0.5, 
        1800:0, 
        2100:1, 
        2300:1, 
        2500:1, 
        3500:1, 
        26000:1}

# adjustment need for tool tip display data for channel frequency

xaxisadj = {700:1, 
            800:0.25, 
            900:0, 
            1800:0, 
            2100:1,
            2300:1,
            2500:2,
            3500:0,
            26000:0}

BandType = {700:"FDD", 
            800:"FDD", 
            900:"FDD", 
            1800:"FDD", 
            2100:"FDD",
            2300:"TDD",
            2500:"TDD",
            3500:"TDD",
            26000:"TDD"}

auctionfailyears = {700:["2016","2021"], #when all auction prices are zero and there are no takers 
        800:["2012"], 
        900:["2013","2016"], 
        1800:["2013"], 
        2100:[], 
        2300:["2022"], 
        2500:["2021"], 
        3500:[], 
        26000:[]}

auctionsucessyears = {700:[2022], #these are years where at least in one circle there was a winner
        800:[2013, 2015, 2016, 2021, 2022], 
        900:[2014, 2015, 2021, 2022], 
        1800:[2012, 2014, 2015, 2016, 2021, 2022], 
        2100:[2010, 2015, 2016, 2021, 2022], 
        2300:[2010, 2016, 2021], 
        2500:[2010, 2016, 2022], 
        3500:[2022], 
        26000:[2022]}


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

#preparing color scale for hoverbox
def hovercolscale(operators, colcodes):
    scale = [round(x/(len(operators)-1),2) for x in range(len(operators))]
    colors =[]
    for k, v  in operators.items():
        colors.append(colcodes.loc[k,:].values[0])

    colorscale=[]
    for i in range(len(scale)):
        colorscale.append([scale[i],colors[i]])
    return colorscale


#loading data 
xl = pd.ExcelFile('spectrum_map.xlsx')
sheet = xl.sheet_names
df = pd.read_excel('spectrum_map.xlsx', sheet_name=sheet)
colcodes = df["ColorCodes"]
colcodes=colcodes.set_index("Description")


#processing data 
# st.sidebar.title('Navigation')

price = df["Master_Price_Sheet"]

price.rename(columns = {"FP" : "Auction Price", "DP": "Reserve Price"}, inplace = True)

price = price[price["Band"] != 600]

Bands = sorted(list(set(price["Band"])))

#menu for selecting features 
Feature = st.sidebar.selectbox('Select a Feature', options = ["Price","Map"])


if Feature == "Map":
	Band = st.sidebar.selectbox('Select a Band', options = Bands)
	freqtab = str(Band)+"MHz"
	sf = df[freqtab]
	sf = sf.set_index("LSA")
	operators_mapping =operators[Band]
	sf[sf.columns] = sf[sf.columns].replace(operators_mapping)
	tickangle = -90

	data1 = [go.Heatmap(
	      z = sf.values,
	      y = sf.index,
	      x = sf.columns,
	      xgap = xgap[Band],
	      ygap = 1,
#               hoverinfo ='text',
#               text = hovertext1,
	      colorscale=hovercolscale(operators_mapping, colcodes),
	      colorbar=dict(
	      tickvals = list(operators_mapping.values()),
	      # tickvals = tickval,
	      ticktext = list(operators_mapping.keys()),
	      dtick=1,
	      tickmode="array"),
			    ),
		]

	fig = go.Figure(data=data1)


if Feature == "Price":
	Band = st.sidebar.selectbox('Select a Band', options = Bands)
	price = price[(price["Band"]==Band) & (price["Year"] != 2018)]
	price["Year"] = sorted([str(x) for x in price["Year"].values])
	Type = st.sidebar.selectbox('Select Price Type', options = ["Auction Price","Reserve Price"])
	tickangle=0

	data2 = [go.Heatmap(
			z = round(price[Type],1),
			y = price["LSA"],
			x = price["Year"],
			xgap = 1,
			ygap = 1,
			hoverinfo ='text',
			# text = hovertext1,
			colorscale='Hot',
				texttemplate="%{z}", 
				textfont={"size":10},
				reversescale=True,
				),
		]
	fig = go.Figure(data=data2)

	

#updating figure layouts
fig.update_layout(uniformtext_minsize=10, 
		  uniformtext_mode='hide', 
		  xaxis_title=None, 
		  yaxis_title=None, 
		  yaxis_autorange='reversed',
		  font=dict(size=10),
		  template='simple_white',
		  paper_bgcolor=None,
		  height=600, width=1200,
		  title="<b>"+"Spectrum "+[Type if Feature=="Price"]+" for "+str(Band)+" MHz Band"+"<b>",
		  margin=dict(t=80, b=50, l=50, r=50, pad=0),
		  title_x=0.35, title_y=0.99,
		  title_font=dict(size=20),
		  xaxis = dict(
		  side = 'top',
		  tickmode = 'linear',
		  tickangle=tickangle,
		  dtick = dtick[Band]),
		  # showlegend=True
		)


fig.update_xaxes(fixedrange=True,showline=True,linewidth=1.2,linecolor='black', mirror=True)
fig.update_yaxes(fixedrange=True,showline=True, linewidth=1.2, linecolor='black', mirror=True)

st.write(fig)



