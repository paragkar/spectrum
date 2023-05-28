#importing libraries
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from collections import OrderedDict
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu
import plotly
import pandas as pd
import plotly.figure_factory as ff
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt
from datetime import datetime
import calendar

from dateutil.relativedelta import relativedelta

import io
import msoffcrypto



#Setting Page layout
st.set_page_config(layout="wide")

#set summary chart flag
flag = False # It will toggle to True when we what summary chart to show



#hide streamlit style

hide_st_style = '''
				<style>
				#MainMenu {visibility : hidden;}
				footer {visibility : hidder;}
				header {visibility :hidden;}
				<style>
				'''
st.markdown(hide_st_style, unsafe_allow_html =True)


password = st.secrets["db_password"]

excel_content = io.BytesIO()

with open("spectrum_map_protected.xlsx", 'rb') as f:
	excel = msoffcrypto.OfficeFile(f)
	excel.load_key(password)
	excel.decrypt(excel_content)

#loading data from excel file
xl = pd.ExcelFile(excel_content)
sheet = xl.sheet_names
df = pd.read_excel(excel_content, sheet_name=sheet)

  
#Defining Dictionaries	
state_dict = {'AP': 'Andhra Pradesh', 'AS': 'Assam', 'BH': 'Bihar', 'DL': 'Delhi', 'GU': 'Gujarat',
    'HA': 'Haryana','HP': 'Himachal Pradesh','JK': 'Jammu & Kashmir','KA': 'Karnataka',
    'KE': 'Kerala','KO': 'Kolkata','MP': 'Madhya Pradesh','MA': 'Maharashtra','MU': 'Mumbai',
    'NE': 'Northeast','OR': 'Odisha','PU': 'Punjab','RA': 'Rajasthan','TN': 'Tamil Nadu',
    'UPE': 'Uttar Pradesh (East)','UPW': 'Uttar Pradesh (West)','WB': 'West Bengal' }

#defining all dictionaries here with data linked to a specific band
subtitle_freqlayout_dict = {700:"FDD: Uplink - 703-748 MHz(shown); Downlink - 758-803(notshown); ",
         800:"Uplink - 824-844 MHz(shown); Downlink - 869-889 MHz(not shown); ", 
         900:"Uplink - 890-915 MHz(shown); Downlink - 935-960 MHz(not shown); ", 
         1800:"Uplink - 1710-1785 MHz(shown); Downlink - 1805-1880 MHz(notshown); ", 
         2100:"Uplink - 1919-1979 MHz(shown); Downlink - 2109-2169 MHz(notshown); ",
         2300:"Up & Downlinks - 2300-2400 MHz(shown); ",
         2500:"Up & Downlinks - 2500-2690 MHz(shown); ",
         3500:"Up & Downlinks - 3300-3670 MHz(shown); ",
         26000:"Up & Downlinks - 24250-27500 MHz(shown); "}

#operator dict for dimension - spectrum bands and subfeatures - freq and exp maps
newoperators_dict = {700: {'Vacant':0,'Railways':1,'Govt':2,'RJIO':3,'BSNL':4},
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

#operators dict for dimension - calendar years
oldoperators_dict = {2010 : ["Bharti", "QCOM", "Augere", "Vodafone", "Idea", "RJIO", "RCOM", "STel", "Tata", "Aircel", "Tikona"],
		    2012 : ["Bharti", "Vodafone", "Idea", "Telenor", "Videocon"],
		    2013 : ["MTS"],
		    2014 : ["Bharti", "Vodafone", "Idea", "RJIO", "RCOM", "Aircel", "Telenor"],
		    2015 : ["Bharti", "Vodafone", "Idea", "RJIO", "RCOM", "Tata", "Aircel"],
		    2016 : ["Bharti", "Vodafone", "Idea", "RJIO", "RCOM", "Tata", "Aircel"],
		    2021 : ["Bharti", "RJIO", "VodaIdea"],
		    2022 : ["Bharti", "RJIO", "VodaIdea", "Adani"] }

#band dicts for dimension calendar year and sub feature operator wise
bands_auctioned_dict = {2010 : [2100, 2300],
	       2012 : [800, 1800],
	       2013 : [800, 900, 1800],
	       2014 : [900, 1800],
	       2015 : [800, 900, 1800, 2100],
	       2016 : [700, 800, 900, 1800, 2100, 2300, 2500],
	       2021 : [700, 800, 900, 1800, 2100, 2300, 2500],
	       2022 : [600, 700, 800, 900, 2100, 2300, 2500, 3500, 26000]}
		    

#if "1" the expiry tab is present and if "0" then not present
exptab_dict = {700:1, 800:1, 900:1, 1800:1, 2100:1, 2300:1, 2500:1, 3500:1, 26000:1}

#Setting the channel sizes for respective frequency maps
channelsize_dict = {700:2.5, 800:0.625, 900:0.2, 1800:0.2, 2100:2.5, 2300:2.5, 2500:5, 3500:5, 26000:25}

# scale of the x axis plots
xdtickfreq_dict = {700:1, 800:0.25, 900:0.4, 1800:1, 2100:1, 2300:1, 2500:2, 3500:5, 26000:50}

# used to control the number of ticks on xaxis for chosen feature = AuctionMap
dtickauction_dict = {700:1, 800:1, 900:1, 1800:1, 2100:1, 2300:1, 2500:1, 3500:1, 26000:1}

# vertical line widths
xgap_dict = {700:1, 800:1, 900:0.5, 1800:0, 2100:1, 2300:1, 2500:1, 3500:1, 26000:1}

# adjustment need for tool tip display data for channel frequency
xaxisadj_dict = {700:1, 800:0.25, 900:0, 1800:0, 2100:1, 2300:1, 2500:2, 3500:0, 26000:0}

#describing the type of band TDD/FDD
bandtype_dict = {700:"FDD", 800:"FDD", 900:"FDD", 1800:"FDD", 2100:"FDD", 2300:"TDD", 2500:"TDD", 3500:"TDD", 26000:"TDD"}

#auctionfailyears when all auction prices are zero and there are no takers 
auctionfailyears_dict = {700:["2016","2021"], 800:["2012"], 900:["2013","2016"], 1800:["2013"], 
        2100:[], 2300:["2022"], 2500:["2021"], 3500:[], 26000:[]}

#auction sucess years are years where at least in one circle there was a winner
auctionsucessyears_dict = {700:[2022], 
        800:[2013, 2015, 2016, 2021, 2022], 
        900:[2014, 2015, 2021, 2022], 
        1800:[2012, 2014, 2015, 2016, 2021, 2022], 
        2100:[2010, 2015, 2016, 2021, 2022], 
        2300:[2010, 2016, 2021, 2022],  #added 2022 an as exception (due to error) need to revist the logic of succes and failure
        2500:[2010, 2016, 2022], 
        3500:[2022], 
        26000:[2022]}

#Error is added to auction closing date so that freq assignment dates fall within the window.
#This helps to identify which expiry year is linked to which operators
errors_dict= {700:0.25, 800:1, 900:1, 1800:1, 2100:1.5, 2300:1.25, 2500:1, 3500:0.1, 26000:0.5}

list_of_circles_codes = ['AP','AS', 'BH', 'DL', 'GU', 'HA', 'HP', 'JK', 'KA', 'KE', 'KO', 'MA', 'MP',
       	   'MU', 'NE', 'OR', 'PU', 'RA', 'TN', 'UPE', 'UPW', 'WB']

#defining various functions 
#preparing color scale for freqmap
@st.cache_resource
def colscalefreqlayout(operators, colcodes):
	operators = dict(sorted(operators.items(), key=lambda x:x[1]))
	operator_names = list(operators.keys())
	operator_codes = list(operators.values())
	scale = [round(x/(len(operators)),2) for x in range(len(operator_names)+1)]
	colorscale =[]
	for i, op in enumerate(operator_names):
		if op in colcodes.index:
			colorscale.append([scale[i],colcodes.loc[op,:][0]])
	colorscale.append([1, np.nan])
	col= pd.DataFrame(colorscale)
	col.columns =["colscale", "colors"]
	col["colscaleshift"] = col.iloc[:,0].shift(-1)
	col = col.iloc[:-1,:]
	colorscale=[]
	for line in col.values:
		colorscale.append((line[0],line[1]))
		colorscale.append((line[2],line[1]))
	
	return colorscale

#function for calculating expiry year heatmap for yearwise
@st.cache_resource
def exp_year_cal_yearly_trends(ef, selected_operator):
	lst1 =[]
	for i, line1 in enumerate(ef.values):
		explst = list(set(line1))
		l1 = [[ef.index[i],round(list(line1).count(x)*channelsize_dict[Band],2), round(x,2)] for x in explst]
		lst1.append(l1)

	lst2 =[]
	for i, val in enumerate(lst1):
		for item in val:
			lst2.append(item)
	df = pd.DataFrame(lst2)
	df.columns = ["LSA", "Spectrum", "ExpYrs"]
	df = df.groupby(['LSA','ExpYrs']).sum()
	df = df.reset_index()
	df = df.pivot(index ='LSA', columns ='ExpYrs', values ='Spectrum') 
	df.columns = [str(x) for x in df.columns]
	if selected_operator == "All":
		df = df.iloc[:,1:]
	else:
		pass
	df = df.fillna(0)
	return df

#function for calculating quantum of spectrum expiring mapped to LSA and Years for expiry map yearwise
@st.cache_resource
def bw_exp_cal_yearly_trends(sff,ef):
	lst=[]
	for j, index in enumerate(ef.index):
		for i, col in enumerate(ef.columns):
			l= [index, sff.iloc[j,i],ef.iloc[j,i]]
			lst.append(l)
			
	df = pd.DataFrame(lst)
	df.columns = ["LSA","Operators", "ExpYear"]
	df = df.groupby(["ExpYear"])[["LSA","Operators"]].value_counts()*channelsize_dict[Band]
	df = df.reset_index()
	df.columns =["ExpYear","LSA", "Operators","BW"]
	
	return df

#funtion to process pricing datframe for hovertext for auction map
# @st.cache_resource
def cal_bw_mapped_to_operators_auctionmap(dff):
	dff = dff.replace(0,np.nan).fillna(0)
	dff = dff.applymap(lambda x: round(x,2) if type(x)!=str else x)
	dff = dff[(dff["Band"]==Band) & (dff["Cat"]=="L") & (dff["OperatorOld"] != "Free") & (dff["Year"] >= 2010)]
	dff = dff.drop(['OperatorNew', 'Band','Cat'], axis = 1)
	for col in dff.columns[3:]:
		dff[col]=dff[col].astype(float)
	dff = dff.groupby(["OperatorOld", "Year"]).sum()
	dff = dff.drop(['Batch No',], axis = 1) 
	if bandtype_dict[Band]=="TDD": #doubling the TDD spectrum for aligning with normal convention 
		dff = (dff*2).round(2)
	dff = dff.replace(0,"")
	dff= dff.reset_index().set_index("Year")
	dff =dff.replace("Voda Idea","VI")
	dff = dff.replace("Vodafone", "Voda")
	dff = dff.astype(str)
	lst =[]
	for index, row in zip(dff.index,dff.values):
		lst.append([index]+[row[0]+" "+x+" MHz, " for x in row[1:]])
	temp = pd.DataFrame(lst)
	col = dff.reset_index().columns
	col = list(col)
	col.pop(1)
	temp.columns = col
	temp = temp.replace('[a-zA-Z]+\s+MHz, ',"", regex = True)
	dff = temp.groupby("Year").sum()
	dff =dff.T
	dff = dff.reset_index()
	dff.columns = ["LSA"]+auctionsucessyears_dict[Band]
	dff = dff.set_index("LSA")
	return dff

#convert columns of dataframe into string
@st.cache_resource
def coltostr(df):
	lst =[]
	for col in df.columns:
		lst.append(str(col))
	df.columns=lst
	return df

#add dummy columns for auction failed years
@st.cache_resource
def adddummycols(df,col):
    df[col]="NA  " # space with NA is delibelitratly added.
    cols = sorted(df.columns)
    df =df[cols]
    return df

#function to calculate the year in which the spectrum was acquired
@st.cache_resource
def cal_year_spectrum_acquired(ef,excepf,pf1):
	lst=[]
	for col in ef.columns:
		for i, (efval,excepfval) in enumerate(zip(ef[col].values, excepf[col].values)):
			for j, pf1val in enumerate(pf1.values):
				if excepfval == 0:
					error = abs(efval-pf1val[6]) #orignal
				else:
					error = 0
				if (ef.index[i] == pf1val[0]) and error <= errors_dict[Band]:
					lst.append([ef.index[i],col-xaxisadj_dict[Band],pf1val[1],pf1val[2], pf1val[3], pf1val[4], error]) 
				
	df_final = pd.DataFrame(lst)
	df_final.columns = ["LSA", "StartFreq", "TP", "RP", "AP", "Year", "Error"]
	df_final["Year"] = df_final["Year"].astype(int)
	ayear = df_final.pivot_table(index=["LSA"], columns='StartFreq', values="Year", aggfunc='first').fillna("NA")
	return ayear
  
#processing for hovertext for freq map, band wise
@st.cache_resource
def htext_specmap_freq_layout(sf):  
	hovertext = []
	for yi, yy in enumerate(sf.index):
		hovertext.append([])
		for xi, xx in enumerate(sf.columns):
			if exptab_dict[Band]==1: #1 means that the expiry table in the excel sheet has been set and working 
				expiry = round(ef.values[yi][xi],2)
			else:
				expiry = "NA"
			try:
			    auction_year = round(ayear.loc[yy,round(xx-xaxisadj_dict[Band],3)])
			except:
			    auction_year ="NA"
				
			operatornew = sff.values[yi][xi]
			operatorold = of.values[yi][xi]
			bandwidth = bandf.values[yi][xi]
			hovertext[-1].append(
					    'StartFreq: {} MHz\
					     <br>Channel Size : {} MHz\
					     <br>Circle : {}\
				             <br>Operator: {}\
					     <br>Total BW: {} MHz\
					     <br>ChExp In: {} Years\
					     <br>Acquired In: {} by {}'

				     .format(
					    round(xx-xaxisadj_dict[Band],2),
					    channelsize_dict[Band],
					    state_dict.get(yy),
					    operatornew,
					    bandwidth,
					    expiry,
					    auction_year,
					    operatorold,
					    )
					    )
	return hovertext

#processing for hovertext for expiry map, freq wise
@st.cache_resource
def htext_expmap_freq_layout(sf):
	hovertext = []
	for yi, yy in enumerate(sf.index):
		hovertext.append([])
		for xi, xx in enumerate(sf.columns):
			if exptab_dict[Band]==1: #1 means that the expiry table in the excel sheet has been set and working 
				expiry = round(ef.values[yi][xi],2)
			else:
				expiry = "NA"
			try:
			    auction_year = round(ayear.loc[yy,round(xx-xaxisadj_dict[Band],3)])
			except:
			    auction_year ="NA"
			operatornew = sff.values[yi][xi]
			operatorold = of.values[yi][xi]
			bandwidthexpiring = bandexpf.values[yi][xi]
			bandwidth = bandf.values[yi][xi]
			hovertext[-1].append(
					    'StartFreq: {} MHz\
					     <br>Channel Size : {} MHz\
					     <br>Circle : {}\
				             <br>Operator: {}\
					     <br>Expiring BW: {} of {} MHz\
					     <br>Expiring In: {} Years\
					     <br>Acquired In: {} by {}'

				     .format(
					    round(xx-xaxisadj_dict[Band],2),
					    channelsize_dict[Band],
					    state_dict.get(yy),
					    operatornew,
					    bandwidthexpiring,
					    bandwidth,
					    expiry,
					    auction_year,
					    operatorold,
					    )
					    )
	return hovertext

#processing for hovertext for expiry map, year wise operator selection "All"
@st.cache_resource
def htext_expmap_yearly_trends_with_all_select(bwf,eff): 
	bwf["Op&BW"] = bwf["Operators"]+" - "+round(bwf["BW"],2).astype(str)+" MHz"
	bwff = bwf.set_index("LSA").drop(['Operators'], axis=1)
	xaxisyears = sorted(list(set(bwff["ExpYear"])))[1:]
	hovertext = []
	for yi, yy in enumerate(eff.index):
		hovertext.append([])
		for xi, xx in enumerate(xaxisyears):
			opwiseexpMHz = list(bwff[(bwff["ExpYear"]==xx) & (bwff.index ==yy)]["Op&BW"].values)
			if opwiseexpMHz==[]:
				opwiseexpMHz="NA"
			else:
				opwiseexpMHz = ', '.join(str(e) for e in opwiseexpMHz) #converting a list into string

			TotalBW = list(bwff[(bwff["ExpYear"]==xx) & (bwff.index ==yy)]["BW"].values)
			
			if TotalBW==[]:
				TotalBW="NA"
			else:
				TotalBW = round(sum([float(x) for x in TotalBW]),2)

			hovertext[-1].append(
					    '{} : Expiry in {} Years\
					    <br />Break Up : {}'

				     .format(
					    state_dict.get(yy),
					    xx, 
					    opwiseexpMHz,
					    )
					    )
	return hovertext


#processing for hovertext for expiry map, year wise along with operator menue
@st.cache_resource
def htext_expmap_yearly_trends_with_op_select(eff): 
	hovertext = []
	for yi, yy in enumerate(eff.index):
		hovertext.append([])
		for xi, xx in enumerate(eff.columns):

			hovertext[-1].append(
					    'Circle: {}\
					    <br />Expiring In: {} Years'

				     .format(
					    state_dict.get(yy),
					    xx, 
					    )
					    )
	return hovertext
	
#processing for hovertext for Auction Map
@st.cache_resource
def htext_auctionmap(dff): 
	hovertext=[]
	for yi, yy in enumerate(dff.index):
		hovertext.append([])
		for xi, xx in enumerate(dff.columns):
			winners = dff.values[yi][xi][:-2] #removing comma in the end
			resprice = reserveprice.values[yi][xi]
			aucprice = auctionprice.values[yi][xi]
			offmhz = offeredspectrum.values[yi][xi]
			soldmhz = soldspectrum.values[yi][xi]
			unsoldmhz = unsoldspectrum.values[yi][xi]

			hovertext[-1].append(
					    '{} , {}\
					     <br / >RP/AP: Rs {}/ {} Cr/MHz\
					     <br / >Offered/Sold/Unsold: {} / {} / {} MHz\
					     <br>Winners: {}'

				     .format( 
					    state_dict.get(yy),
					    xx,
					    resprice,
					    aucprice,
					    round(offmhz,2),
					    round(soldmhz,2),
					    round(unsoldmhz,2),
					    winners,
					    )
					    )
	return hovertext


#processing for hovertext and colormatrix for Spectrum Band, Features- Freq Map, SubFeature - Operator Wise 
@st.cache_resource
def htext_colmatrix_spec_map_op_hold_share(dfff, selected_operators, operatorlist):

	if len(selected_operators)==0:
		operators_to_process = operatorlist
	else:
		operators_to_process = selected_operators

	dfffcopy =dfff.copy()
	
	dfffcopy["Total"] = dfffcopy.sum(axis=1)
		
	lst =[]

	dfffshare = pd.DataFrame()
	for op in operators_to_process:
		dfffcopy[op+"1"] = dfffcopy[op]/dfffcopy["Total"]
		lst.append(op+"1")
	
	dfffshare = dfffcopy[lst]
	for col in dfffshare.columns:
		dfffshare.rename(columns = {col:col[:-1]}, inplace = True) #stripping the last digit "1"

	
	hovertext=[]
	lst = []
	for yi, yy in enumerate(dfffshare.index):
		hovertext.append([])
		for xi, xx in enumerate(dfffshare.columns):
			share = dfffshare.values[yi][xi]
			holdings = dfff.values[yi][xi]
			
			if share >= 0.4 :
				ccode = '#008000' #% spectrum share more than 40% (green)
			elif (share < 0.4) & (share >= 0.2):
				ccode = '#808080' # spectrum share between 40 to 20% (grey)
			else:
				ccode = '#FF0000' # spectrum share less than 20% (red)
			lst.append([yy,xx,ccode])
			temp = pd.DataFrame(lst)
			temp.columns = ["Circle", "Operator", "Color"]
			colormatrix = temp.pivot(index='Circle', columns='Operator', values="Color")
			colormatrix = list(colormatrix.values)
			
			hovertext[-1].append(
					    'Circle: {}\
					     <br>Operator: {}\
					     <br>Holdings: {} MHz\
					     <br>Market Share: {} %'

				     .format( 
					    state_dict.get(yy),
					    xx,
					    round(holdings,2),
					    round(share*100,2),
					    )
					    )
	return hovertext, colormatrix


#processing for hovertext and colormatrix for Calendar Year, Band Wise, SubFeatures Reserve Price etc
@st.cache_resource
def htext_colmatrix_auction_year_band_metric(df1):
	auctionprice =  df1.pivot(index="Circle", columns='Band', values=subfeature_dict["Auction Price"])
	reserveprice =  df1.pivot(index="Circle", columns='Band', values=subfeature_dict["Reserve Price"])
	qtyoffered = df1.pivot(index="Circle", columns='Band', values=subfeature_dict["Quantum Offered"])
	qtysold = df1.pivot(index="Circle", columns='Band', values=subfeature_dict["Quantum Sold"])
	qtyunsold = df1.pivot(index="Circle", columns='Band', values=subfeature_dict["Quantum Unsold"])
	
	hovertext=[]
	lst = []
	for yi, yy in enumerate(reserveprice.index):
		hovertext.append([])
		for xi, xx in enumerate(reserveprice.columns):
			resprice = reserveprice.values[yi][xi]
			aucprice = auctionprice.values[yi][xi]
			offered = qtyoffered.values[yi][xi]
			sold = qtysold.values[yi][xi]
			unsold = qtyunsold.values[yi][xi]
			delta = round(aucprice - resprice,0)
			if delta < 0 :
				ccode = '#000000' #auction failed (black)
			elif delta == 0:
				ccode = '#008000' #auction price = reserve price (green)
			elif delta > 0:
				ccode = '#FF0000' #auction price > reserve price (red)
			else:
				ccode = '#C0C0C0' #No Auction (silver)
			lst.append([yy,xx,ccode])
			temp = pd.DataFrame(lst)
			temp.columns = ["Circle", "Year", "Color"]
			colormatrix = temp.pivot(index='Circle', columns='Year', values="Color")
			colormatrix = list(colormatrix.values)
			
			hovertext[-1].append(
					    'Circle: {}\
					     <br>Band: {} MHz\
					     <br>Reserve Price: {} Rs Cr/MHz\
					     <br>Auction Price: {} Rs Cr/MHz\
					     <br>Offered: {} MHz\
					     <br>Sold: {} MHz\
					     <br>Unsold: {} MHz'

				     .format( 
					    state_dict.get(yy),
					    xx,
					    round(resprice,1),
					    round(aucprice,1),
					    round(offered,2),
					    round(sold,2),
					    round(unsold,2),
					    )
					    )
	return hovertext, colormatrix

#processing for hovertext and colormatrix for Calendar Year, Operator Wise, SubFeatures - Total Outflow, Total Purchase
@st.cache_resource
def htext_colmatrix_auction_year_operator_metric(df1, selectedbands, SubFeature, df_subfeature):	
	temp1 = pd.DataFrame()
	if selectedbands != []:
		for band in selectedbands:
			temp2= df1[df1["Band"]==band]
			temp1 = pd.concat([temp2,temp1], axis =0)
		df1  = temp1
	
	if SubFeature == "Total Purchase": #then process for total purchase
		df_purchase = df_subfeature
	else: 
		columnstoextract = ["Circle", "Band"]+oldoperators_dict[Year]
		df2_temp2 = df1[columnstoextract]
		df2_temp2.drop("Band", inplace = True, axis =1)
		df2_temp2 = df2_temp2.groupby(["Circle"]).sum().round(2)
		df2_temp2 = df2_temp2.reindex(sorted(df2_temp2.columns), axis=1)
		df_purchase = df2_temp2
	
	if SubFeature == "Total Ouflow": #then process for total outflow
		df_outflow = df_subfeature
	else:
		operators_dim_cy_new=[]
		for op in oldoperators_dict[Year]:
			df1[op+"1"] = df1["Auction Price/MHz"]*df1[op]
			operators_dim_cy_new.append(op+"1")
		columnstoextract = ["Circle", "Band"]+operators_dim_cy_new
		df2_temp1 = df1[columnstoextract]
		operators_dim_cy_new = [x[:-1] for x in operators_dim_cy_new] # removing the last letter "1" from operator name
		df2_temp1.columns = ["Circle", "Band"]+ operators_dim_cy_new
		df2_temp1.drop("Band", inplace = True, axis =1)
		df2_temp1 = df2_temp1.groupby(["Circle"]).sum().round(0)
		df2_temp1 = df2_temp1.reindex(sorted(df2_temp1.columns), axis=1)
		df_outflow = df2_temp1
	
	hovertext=[]
	lst = []
	for yi, yy in enumerate(df_subfeature.index): #dataframe of total outflow (any one of them can be used)
		hovertext.append([])
		for xi, xx in enumerate(df_subfeature.columns): #dataframe of total outflow (any one of them can be used)
			outflow = df_outflow.values[yi][xi]
			purchase = df_purchase.values[yi][xi]
			if outflow > 0 :
				ccode = '#008000' # Purchased (green)
			else:
				ccode = '#C0C0C0' #No Purchase (silver)
			lst.append([yy,xx,ccode])
			temp = pd.DataFrame(lst)
			temp.columns = ["Circle", "Operator", "Color"]
			colormatrix = temp.pivot(index='Circle', columns='Operator', values="Color")
			colormatrix = list(colormatrix.values)
			
			hovertext[-1].append(
					    'Circle: {}\
					     <br>Operator: {}\
					     <br>Outflow: {} Rs Cr\
					     <br>Purchase: {} MHz'

				     .format( 
					    state_dict.get(yy),
					    xx,
					    round(outflow,0),
					    round(purchase,2),
					    )
					    )
	return hovertext, colormatrix


#processing for hovertext for Telecom Data and 5G BTS Trends
@st.cache_resource
def htext_telecomdata_5gbts(df5gbtsf): 

	summarydf = df5gbtsf.sum(axis=0)
	df5gbtsfPercent = round((df5gbtsf/summarydf)*100,2)

	lst =[]
	for row in df5gbtsf.values:

		increments = np.diff(row)
		lst.append(increments)

	df5gbtsincf = pd.DataFrame(lst)

	df5gbtsincf.index = df5gbtsf.index 

	df5gbtsincf.columns = df5gbtsf.columns[1:]

	lastcolumn = df5gbtsincf.columns[-1]
	df5gbtsincf = df5gbtsincf.sort_values(lastcolumn, ascending = False) #sort by the last column

	hovertext=[]

	for yi,yy in enumerate(df5gbtsf.index):
		hovertext.append([])
		for xi,xx in enumerate(df5gbtsf.columns):

			# btscum = df5gbtsf.values[yi][xi]
			# btspercent = df5gbtsfPercent.values[yi][xi]
			btscum = df5gbtsf.loc[yy,xx]
			btspercent = df5gbtsfPercent.loc[yy,xx]

			try:
				btsinc = df5gbtsincf.loc[yy,xx]
			except:
				btsinc = np.nan


			hovertext[-1].append(
					    'State: {}\
					    <br>Date: {}\
					    <br>BTS Cum: {} K Nos\
					    <br>BTS Inc: {} K Nos\
					    <br>BTS Cum: {} % of Total'

				     .format( 
					    yy,
					    xx,
					    btscum,
					    round(btsinc,2),
					    btspercent,
					    )
					    )
	return hovertext


#preparing color scale for hoverbox for freq and exp maps
@st.cache_resource
def colscale_hbox_spectrum_expiry_maps(operators, colcodes):
    scale = [round(x/(len(operators)-1),2) for x in range(len(operators))]
    colors =[]
    for k, v  in operators.items():
        colors.append(colcodes.loc[k,:].values[0])
    colorscale=[]
    for i in range(len(scale)):
        colorscale.append([scale[i],colors[i]])
    return colorscale

#shaping colorscale for driving the color of hoverbox of freq and exp maps
@st.cache_resource
def transform_colscale_for_spec_exp_maps(colorscale, sf):
	hlabel_bgcolor = [[x[1] for x in colorscale if x[0] == round(value/(len(colorscale) - 1),2)] 
			      for row in sf.values for value in row]
	hlabel_bgcolor = list(np.array(hlabel_bgcolor).reshape(22,int(len(hlabel_bgcolor)/22)))
	return hlabel_bgcolor

#preparing and shaping the colors for hoverbox for auction map
@st.cache_resource
def transform_colscale_for_hbox_auction_map(dff,reserveprice, auctionprice): 
	lst =[]
	for yi, yy in enumerate(dff.index):
		reserveprice = reserveprice.replace("NA\s*", np.nan, regex = True)
		auctionprice = auctionprice.replace("NA\s*", np.nan, regex = True)
		delta = auctionprice-reserveprice
		delta = delta.replace(np.nan, "NA")
		for xi, xx in enumerate(dff.columns):
			delval = delta.values[yi][xi]
			if delval =="NA":
				ccode = '#000000' #auction failed #black
			elif delval == 0:
				ccode = '#008000' #auction price = reserve price #green
			else:
				ccode = '#FF0000' #auction price > reserve price #red
			lst.append([yy,xx,ccode])
			temp = pd.DataFrame(lst)
			temp.columns = ["LSA", "Year", "Color"]
			colormatrix = temp.pivot(index='LSA', columns='Year', values="Color")
			colormatrix = list(colormatrix.values)
	return colormatrix


#function for preparing the summary chart 


def summarychart(summarydf, xcolumn, ycolumn):
	bar = alt.Chart(summarydf).mark_bar().encode(
	y = alt.Y(ycolumn+':Q', axis=alt.Axis(labels=False)),
	x = alt.X(xcolumn+':O', axis=alt.Axis(labels=False)),
	color = alt.Color(xcolumn+':N', legend=None))
	
	text = bar.mark_text(size = 12, dx=0, dy=-7, color = 'white').encode(text=ycolumn+':Q')
	
	chart = (bar + text).properties(width=1120, height =150)
	chart = chart.configure_title(fontSize = 20, font ='Arial', anchor = 'middle', color ='black')
	return chart


#**********  Main Program Starts here ***************

with st.sidebar:
	selected_dimension = option_menu(
		menu_title = "Select a Menu",
		options = ["Spectrum Bands", "Auction Years", "Business Data"],
		icons = ["1-circle-fill", "2-circle-fill", "3-circle-fill"],
		menu_icon = "arrow-down-circle-fill",
		default_index =0,
		)



# #Choose a dimension
# selected_dimension = st.sidebar.selectbox('Select a Dimension', ["Spectrum Band", "Auction Year"],0)

if selected_dimension == "Spectrum Bands":
	#selecting a Spectrum band
	Band = st.sidebar.selectbox('Select a Band', list(exptab_dict.keys()), 3) #default index 1800 MHz Band
	
	#setting up excel file tabs for reading data
	freqtab = str(Band)+"MHz"
	bandwidthtab = str(Band)+"MHzBW"
	bandwithexptab = str(Band)+"MHzExpBW"
	freqexpbwcalsheet = str(Band)+"MHzExpBWCalSheet"
	freqtabori = str(Band)+"MHzOriginal"
	pricetab = str(Band)+"MHzPrice"
	exptab = str(Band)+"MHzExpCorrected"
	expexceptab = str(Band)+"MHzExpException"
	spectrumall = "Spectrum_All"
	spectrumofferedvssold = "Spectrum_Offered_vs_Sold"
	masterall = "MasterAll-TDDValueConventional" #all auction related information 

	#processing colorcode excel data tab
	colcodes = df["ColorCodes"]
	colcodes=colcodes.set_index("Description")

	#processing excel tabs into various dataframes
	sf = df[freqtab]
	bandf = df[bandwidthtab]
	bandexpf = df[bandwithexptab]
	bandexpcalsheetf = df[freqexpbwcalsheet]
	of = df[freqtabori]
	sff = sf.copy() #create a copy for further processing, not used now.
	sff = sff.set_index("LSA")
	pf =df[pricetab]
	pf1 = pf.copy()
	pf = pf[pf["Year"]==2022]
	if exptab_dict[Band]==1:
	    ef = df[exptab]
	    ef = ef.set_index("LSA")
	    excepf = df[expexceptab]
	    excepf = excepf.set_index("LSA")   
	sf = sf.set_index("LSA")
	of = of.set_index("LSA")
	pf = pf.set_index("LSA")
	bandf = bandf.set_index("LSA")
	bandexpf = bandexpf.set_index("LSA")
	masterdf = df[masterall]

# 	eff = exp_year_cal_yearly_trends(ef) # for expiry year heatmap year wise
	
# 	bwf = bw_exp_cal_yearly_trends(sff,ef) # hover text for expiry year heatmap year wise
	
	# st.sidebar.title('Navigation')

	#processing "Spectrum_all" excel tab data
	dff = df[spectrumall] #contains information of LSA wise mapping oldoperators with new operators
	dffcopy = dff.copy() #make a copy for "Operator Wise" subfeature under the feature "FreqMap"
	dff = cal_bw_mapped_to_operators_auctionmap(dff)
	dff = coltostr(dff)
	dff = adddummycols(dff,auctionfailyears_dict[Band])
	dff = dff.applymap(lambda x: "NA  " if x=="" else x) # space with NA is delibelitratly added as it gets removed with ","

	#processing pricemaster excel tab data
	pricemaster = df["Master_Price_Sheet"]
	pricemaster.rename(columns = {"FP" : "Auction Price", "DP": "Reserve Price"}, inplace = True)

	#processing & restructuring dataframe spectrum offered vs sold & unsold for hovertext of data3
	offeredvssold = df[spectrumofferedvssold]
	offeredvssold = offeredvssold[(offeredvssold["Band"] == Band) & (offeredvssold["Year"] != 2018)]
	offeredvssold = offeredvssold.drop(columns =["Band"]).reset_index(drop=True)
	offeredspectrum = offeredvssold.pivot(index=["LSA"], columns='Year', values="Offered").fillna("NA")
	offeredspectrum = coltostr(offeredspectrum) #convert columns data type to string
	soldspectrum = offeredvssold.pivot(index=["LSA"], columns='Year', values="Sold").fillna("NA")
	soldspectrum = coltostr(soldspectrum) #convert columns data type to string
	percentsold = offeredvssold.pivot(index=["LSA"], columns='Year', values="%Sold")
	percentsold = percentsold.replace("NA", np.nan)
	percentsold = percentsold*100 #for rationalising the percentage number
	percentsold = percentsold.applymap(lambda x: round(float(x),1))
	percentsold = coltostr(percentsold) #convert columns data type to string
	unsoldspectrum = offeredvssold.pivot(index=["LSA"], columns='Year', values="Unsold").fillna("NA")
	unsoldspectrum = coltostr(unsoldspectrum) #convert columns data type to string
	percentunsold = offeredvssold.pivot(index=["LSA"], columns='Year', values="%Unsold")
	percentunsold = percentunsold.replace("NA", np.nan)
	percentunsold = percentunsold*100 #for rationalising the percentage number
	percentunsold = percentunsold.applymap(lambda x: round(float(x),1))
	percentunsold = coltostr(percentunsold) #convert columns data type to string

	#processing & restructuring dataframe auction price for hovertext of data3
	auctionprice = pricemaster[(pricemaster["Band"] == Band) & (pricemaster["Year"] != 2018)]
	auctionprice = auctionprice.pivot(index=["LSA"], columns='Year', values="Auction Price").fillna("NA")
	auctionprice = auctionprice.loc[:, (auctionprice != 0).any(axis=0)]
	auctionprice = auctionprice.applymap(lambda x: round(x,2))
	auctionprice = coltostr(auctionprice) #convert columns data type to string
	auctionprice = adddummycols(auctionprice,auctionfailyears_dict[Band])
	auctionprice = auctionprice.replace(0,"NA")

	#processing & restructuring dataframe reserve price for hovertext of data3
	reserveprice = pricemaster[(pricemaster["Band"] == Band) & (pricemaster["Year"] != 2018)]
	reserveprice = reserveprice.pivot(index=["LSA"], columns='Year', values="Reserve Price").fillna("NA")
	reserveprice = reserveprice.loc[:, (reserveprice != 0).any(axis=0)]
	reserveprice = reserveprice.applymap(lambda x: round(x,2))
	reserveprice = coltostr(reserveprice) #convert columns data type to string
	reserveprice = reserveprice.replace(0,"NA")

	#mapping the year of auction with channels in the freq maps
	ayear = cal_year_spectrum_acquired(ef,excepf,pf1)

	Feature = st.sidebar.selectbox('Select a Feature', ["Spectrum Map", "Expiry Map", "Auction Map"], 0) #Default Index first

	#Processing For Dimension = "Frequency Band" & Feature 
	if  Feature == "Spectrum Map":
		SubFeature = st.sidebar.selectbox('Select a Sub Feature', ["Frequency Layout", "Operator Holdings", "Operator %Share"],0)
		if SubFeature == "Frequency Layout":
			sf = sff.copy()
			operators = newoperators_dict[Band]
			hf = sf[sf.columns].replace(operators) # dataframe for hovertext
			operatorslist = sorted(list(operators.keys()))
			selected_operators = st.sidebar.multiselect('Filter by Operators', operatorslist)
			if selected_operators==[]:
				sf[sf.columns] = sf[sf.columns].replace(operators) 
				colorscale = colscalefreqlayout(operators, colcodes)
				tickvals = list(operators.values())
				ticktext = list(operators.keys())
			else:
				selected_op_dict ={}
				for op in operators.keys():
					if op not in selected_operators:
						sf.replace(op, np.nan, inplace = True)
				for i, op in enumerate(selected_operators):
					sf.replace(op,i, inplace = True)
					selected_op_dict.update({op : i})
				colorscale = colscalefreqlayout(selected_op_dict, colcodes)
				tickvals = list(selected_op_dict.values())
				ticktext = list(selected_op_dict.keys())	

			hovertext = htext_specmap_freq_layout(hf)
			parttitle ="Spectrum Frequency Layout"
			xdtickangle = -90
			xdtickval = xdtickfreq_dict[Band]
			
			data = [go.Heatmap(
			      z = sf.values,
			      y = sf.index,
			      x = sf.columns,
			      xgap = xgap_dict[Band],
			      ygap = 1,
			      hoverinfo ='text',
			      text = hovertext,
			      colorscale=colorscale,
		# 	      reversescale=True,
			      colorbar=dict(
		# 	      tickcolor ="black",
		# 	      tickwidth =1,
			      tickvals = tickvals,
			      ticktext = ticktext,
			      dtick=1, tickmode="array"),
					    ),
				]
			
			hcolscale=colscale_hbox_spectrum_expiry_maps(operators, colcodes)  #colorscale for hoverbox
			hoverlabel_bgcolor = transform_colscale_for_spec_exp_maps(hcolscale, hf) #shaping the hfcolorscale
			
		if SubFeature == "Operator Holdings":
			selected_operators=[]
			dfff = dffcopy[(dffcopy["Band"]==Band)]
			operatorlist = sorted(list(set(dfff["OperatorNew"])))
			selected_operators = st.sidebar.multiselect('Filter by Operators',operatorlist)
			selected_operators = sorted(selected_operators)
			if len(selected_operators) >0:
				temp = pd.DataFrame()
				for op in selected_operators:
					temp = pd.concat([dfff[dfff["OperatorNew"]==op],temp], axis =0)
				dfff = temp.copy()
			cat_dict = {'Liberalized' : 'L', 'UnLiberalized' : 'U'}
			if len(set(dfff["Cat"])) == 2:
				selected_category = st.sidebar.multiselect('Select a Category', ['Liberalized', 'UnLiberalized'])
				if (len(selected_category) == 0) or (len(selected_category) == 2):
					pass
				else:
					dfff = dfff[(dfff["Cat"] == cat_dict[selected_category[0]])]
			else:
				selected_category=[]
				
			dfff = dfff.groupby(["OperatorNew","Year","Batch No", "Cat"])[list_of_circles_codes].sum()
			dfff = dfff.reset_index().drop(columns = ["Year", "Batch No", "Cat"], axis =1).groupby("OperatorNew").sum().T
			
			if bandtype_dict[Band]=="TDD": #doubling the TDD spectrum for aligning with normal convention 
        			dfff = (dfff*2).round(2)
			
			parttitle ="Operator Holdings"
			xdtickangle = 0
			xdtickval = 1
			
			hovertext,colormatrix = htext_colmatrix_spec_map_op_hold_share(dfff, selected_operators, operatorlist) #processing hovertext and colormatrix for operatorwise in freqband dim
			hoverlabel_bgcolor = colormatrix #colormatrix processed from fuction "hovertext_and_colmatrix" for same above
			
			data = [go.Heatmap(
			      z = dfff.values,
			      y = dfff.index,
			      x = dfff.columns,
			      xgap = 1,
			      ygap = 1,
			      hoverinfo ='text',
			      text = hovertext,
			       colorscale = 'Hot',
			    texttemplate="%{z}", 
			    textfont={"size":10},
			    reversescale=True,)
				]
			
			
		if SubFeature == "Operator %Share":
			selected_operators=[]
			dfff = dffcopy[(dffcopy["Band"]==Band)]
			operatorlist = sorted(list(set(dfff["OperatorNew"])))
			selected_operators = st.sidebar.multiselect('Filter by Operators',operatorlist)
			selected_operators = sorted(selected_operators)
			if len(selected_operators) >0:
				temp = pd.DataFrame()
				for op in selected_operators:
					temp = pd.concat([dfff[dfff["OperatorNew"]==op],temp], axis =0)
				dfff = temp.copy()
			cat_dict = {'Liberalized' : 'L', 'UnLiberalized' : 'U'}
			if len(set(dfff["Cat"])) == 2:
				selected_category = st.sidebar.multiselect('Select a Category', ['Liberalized', 'UnLiberalized'])
				if (len(selected_category) == 0) or (len(selected_category) == 2):
					pass
				else:
					dfff = dfff[(dfff["Cat"] == cat_dict[selected_category[0]])]
			else:
				selected_category=[]
				
			dfff = dfff.groupby(["OperatorNew","Year","Batch No", "Cat"])[list_of_circles_codes].sum()
			dfff = dfff.reset_index().drop(columns = ["Year", "Batch No", "Cat"], axis =1).groupby("OperatorNew").sum().T
			
			if bandtype_dict[Band]=="TDD": #doubling the TDD spectrum for aligning with normal convention 
        			dfff = (dfff*2).round(2)
				
			dfffcopy =dfff.copy()
			dfffcopy["Total"] = dfffcopy.sum(axis=1)

			operators_to_process = list(dfffcopy.columns[:-1]) #new selected operators after category level filtering 
			

			lst =[]
			dfffshare = pd.DataFrame()
			for op in operators_to_process:
				dfffcopy[op+"1"] = dfffcopy[op]/dfffcopy["Total"]
				lst.append(op+"1")

			dfffshare = dfffcopy[lst]
			for col in dfffshare.columns:
				dfffshare.rename(columns = {col:col[:-1]}, inplace = True) #stripping the last digit "1"
			
			dfffshare = round(dfffshare*100,2)
			
			parttitle ="Operator's Spectrum Market Share"
			xdtickangle = 0
			xdtickval = 1
			
			hovertext,colormatrix = htext_colmatrix_spec_map_op_hold_share(dfff, selected_operators, operatorlist) #processing hovertext and colormatrix for operatorwise in freqband dim
			hoverlabel_bgcolor = colormatrix #colormatrix processed from fuction "hovertext_and_colmatrix" for same above

			data = [go.Heatmap(
			      z = dfffshare.values,
			      y = dfffshare.index,
			      x = dfffshare.columns,
			      xgap = 1,
			      ygap = 1,
			      hoverinfo ='text',
			      text = hovertext,
			       colorscale = 'Hot',
			    texttemplate="%{z}", 
			    textfont={"size":10},
			    reversescale=True,)
				]


			

	#Feature ="Expiry Map" linked to Dimension = "Spectrum Band"
	if  Feature == "Expiry Map":
		SubFeature = st.sidebar.selectbox('Select a Sub Feature', ["Frequency Layout", "Yearly Trends"],0)
		if SubFeature == "Frequency Layout":
			sf = sff.copy()
			operators = newoperators_dict[Band]
			hf = sf[sf.columns].replace(operators) # dataframe for hovertext
			operatorslist = sorted(list(operators.keys()))
			operatorstoremove = ["Govt", "Vacant", "Railways"]
			for op in operatorstoremove:
				if op in operatorslist:
					operatorslist.remove(op)
			selected_operators = st.sidebar.multiselect('Filter by Operators', operatorslist)
			if selected_operators==[]:
				expf = ef
			else:
				for op in operators.keys():
					if op not in selected_operators:
						sf.replace(op,0, inplace = True)
					else:
						sf.replace(op,1,inplace = True)

				expf = pd.DataFrame(sf.values*ef.values, columns=ef.columns, index=ef.index)

			hovertext = htext_expmap_freq_layout(hf)
			parttitle ="Spectrum Expiry Layout "+SubFeature
			xdtickangle = -90
			xdtickval = xdtickfreq_dict[Band]

			data = [go.Heatmap(
			      z = expf.values,
			      y = expf.index,
			      x = expf.columns,
			      xgap = xgap_dict[Band],
			      ygap = 1,
			      hoverinfo ='text',
			      text = hovertext,
			      colorscale ='Hot',
			      reversescale=True,
				)
				  ]
			hcolscale=colscale_hbox_spectrum_expiry_maps(operators, colcodes)  #colorscale for hoverbox
			hoverlabel_bgcolor = transform_colscale_for_spec_exp_maps(hcolscale, hf) #shaping the hfcolorscale

		if SubFeature == "Yearly Trends":
			bandexpcalsheetf = bandexpcalsheetf.set_index("LSA") #Loading Dataframe from BandExpCalSheet
			operatorslist = ["All"]+sorted(list(newoperators_dict[Band].keys()))
			selected_operator = st.sidebar.selectbox('Select an Operator', operatorslist)
			if selected_operator == "All":
				eff = exp_year_cal_yearly_trends(ef,selected_operator)
				bwf = bw_exp_cal_yearly_trends(sff,ef)
				hovertext = htext_expmap_yearly_trends_with_all_select(bwf,eff) #hovertext for "All"
			else:
				if selected_operator[-1] in ["R", "U"]: #Last letter of the operator ending with R or U
					regexfilt = '^(?!.*'+selected_operator+').*' #to replace na.npn with text embedded with names of other than the selected operator
					temp = bandexpcalsheetf.replace(regexfilt, np.nan, regex = True)
					temp = temp.replace(selected_operator,'', regex = True)
				else:
					regexfilt = '[0-9.]+'+selected_operator+'U'  #to replace na.npn with text ending RU with names with the selected operator
					temp = bandexpcalsheetf.replace(regexfilt, np.nan, regex = True)
					regexfilt = '^(?!.*'+selected_operator+').*' #to replace na.npn with text embedded with names of other than the selected operator
					temp = temp.replace(regexfilt, np.nan, regex = True)
					temp = temp.replace(selected_operator,'', regex = True)
				
					
				
				for col in temp.columns:
					temp[col] = temp[col].astype(float)
				eff = exp_year_cal_yearly_trends(temp,selected_operator)
				hovertext = htext_expmap_yearly_trends_with_op_select(eff) #hovertext with operator selections
			
			parttitle ="Spectrum Expiry Layout "+SubFeature
			xdtickangle = 0
			xdtickval = dtickauction_dict[Band]
		
		
			#preparing the dataframe of the summary bar chart on top of the heatmap
			summarydf = eff.sum().reset_index()
			summarydf.columns = ["ExpYears", "TotalMHz"]
			summarydf["ExpYears"]= summarydf["ExpYears"].astype(float)
			summarydf["ExpYears"] = summarydf["ExpYears"].sort_values(ascending = True)

			#preparing the summary chart 
			chart = summarychart(summarydf, "ExpYears", "TotalMHz")
			flag = True #for ploting the summary chart
			
			hoverlabel_bgcolor = "#000000" #subdued black

			data = [go.Heatmap(
			  z = eff.values,
			  y = eff.index,
			  x = eff.columns,
			  xgap = 1,
			  ygap = 1,
			  hoverinfo ='text',
			  text = hovertext,
			  colorscale = 'Hot',
			    texttemplate="%{z}", 
			    textfont={"size":10},
			    reversescale=True,
				)]


	#Feature ="Auction Map" linked to Dimension = "Spectrum Band"
	if  Feature == "Auction Map":
		#This dict has been defined for the Feature = Auction Map
		type_dict ={"Auction Price": auctionprice,
			    "Reserve Price": reserveprice, 
			    "Quantum Offered": offeredspectrum, 
			    "Quantum Sold": soldspectrum,
			    "Percent Sold": percentsold,
			    "Quantum Unsold": unsoldspectrum,
			    "Percent Unsold": percentunsold}
		SubFeature = st.sidebar.selectbox('Select a Sub Feature', ["Auction Price","Reserve Price","Quantum Offered", "Quantum Sold", "Percent Sold", "Quantum Unsold", "Percent Unsold"])
		typedf = type_dict[SubFeature].copy()
		parttitle = "Yearly Trend of "+SubFeature
		xdtickangle=0
		xdtickval = dtickauction_dict[Band]
		hovertext = htext_auctionmap(dff)
		
		#preparing the dataframe of the summary bar chart on top of the heatmap
		if SubFeature not in ["Percent Sold", "Percent Unsold"]:
			summarydf = typedf.replace('[a-zA-Z]+\s*',np.nan, regex=True)
			summarydf = summarydf.sum().reset_index()
			summarydf.columns = ["Years", "India Total"]
		#preparing the summary chart 
			chart = summarychart(summarydf, "Years", "India Total")
			flag = True
		
		#setting the data of the heatmap 
		data = [go.Heatmap(
			z = typedf.values,
			y = typedf.index,
			x = typedf.columns,
			xgap = 1,
			ygap = 1,
			hoverinfo ='text',
			text = hovertext,
			colorscale='Hot',
				texttemplate="%{z}", 
				textfont={"size":10},
				reversescale=True,
				),
			]
		hoverlabel_bgcolor = transform_colscale_for_hbox_auction_map(dff,reserveprice,auctionprice)

#Processing For Dimension = "Auction Year"
if selected_dimension == "Auction Years":
	#loading files
	masterall = "MasterAll-TDDValueConventional" #all auction related information
	spectrumofferedvssold = "Spectrum_Offered_vs_Sold"
	masterdf = df[masterall]
	offeredvssolddf = df[spectrumofferedvssold]
	calendaryearlist = sorted(list(set(masterdf["Auction Year"].values)))
	Year = st.sidebar.selectbox('Select a Year',calendaryearlist,7) #Default Index the latest Calendar Year
	df1 = masterdf[masterdf["Auction Year"]==Year]
	df1 = df1.set_index("Circle")
	Feature = st.sidebar.selectbox('Select a Feature',["Band Metric", "Operator Metric"])
	if Feature == "Band Metric":
		subfeature_dict ={"Quantum Offered" : "Sale (MHz)", "Quantum Sold": "Total Sold (MHz)", "Quantum Unsold" : "Total Unsold (MHz)", "Reserve Price" : "RP/MHz" ,  
			       "Auction Price": "Auction Price/MHz", "Total EMD" : "Total EMD"} 
		subfeature_list = ["Reserve Price", "Auction Price", "Auction/Reserve", "Quantum Offered", "Quantum Sold","Percent Sold", "Quantum Unsold", "Percent Unsold", "Total EMD", "Total Outflow"]
		SubFeature = st.sidebar.selectbox('Select a SubFeature', subfeature_list)
		if SubFeature in ["Reserve Price", "Auction Price", "Total EMD", "Quantum Offered", "Quantum Sold", "Quantum Unsold" ]:
			df1 = df1.reset_index()
			df1_temp1 = df1.copy()
			if SubFeature == "Quantum Sold":
				operatorslist = oldoperators_dict[Year]
				selected_operators = st.sidebar.multiselect('Select an Operator', operatorslist)
				if selected_operators == []:
					df1_temp1 = df1_temp1.pivot(index="Circle", columns='Band', values=subfeature_dict[SubFeature])
				else:
					df1_temp1["OperatorTotal"] = df1_temp1[selected_operators].sum(axis=1)
					df1_temp1 = df1_temp1.pivot(index="Circle", columns='Band', values='OperatorTotal')	
			else:
				df1_temp1 = df1_temp1.pivot(index="Circle", columns='Band', values=subfeature_dict[SubFeature])
			df1_temp1.columns = [str(x) for x in sorted(df1_temp1.columns)]
			z = df1_temp1.values.round(1)
			x = df1_temp1.columns
			y = df1_temp1.index
			summarydf = df1_temp1.sum()
		if SubFeature == "Total Outflow":
			df1 = df1.reset_index()
			df1_temp2 = df1.set_index(["Band","Circle"])
			operatorslist = oldoperators_dict[Year]
			selected_operators = st.sidebar.multiselect('Filter by Operators', operatorslist)
			if selected_operators== []:
				df1_temp2["Total Outflow"] = df1_temp2[subfeature_dict["Auction Price"]]*df1_temp2["Total Sold (MHz)"]
			else:
				df1_temp2["Total Outflow"] = df1_temp2[subfeature_dict["Auction Price"]]*df1_temp2[selected_operators].sum(axis=1)
			df1_temp2 = df1_temp2.reset_index()
			df1_temp2 = df1_temp2[["Band", "Circle", "Total Outflow"]]
			df1_temp2 = df1_temp2.pivot(index="Circle", columns='Band', values="Total Outflow")
			df1_temp2.columns = [str(x) for x in sorted(df1_temp2.columns)]
			z = df1_temp2.values.round(0)
			x = df1_temp2.columns
			y = df1_temp2.index
			summarydf = df1_temp2.sum()
		if SubFeature == "Auction/Reserve":
			df1 = df1.reset_index()
			df1_temp3 = df1.set_index(["Band","Circle"])
			df1_temp3["Auction/Reserve"] = np.divide(df1_temp3["Auction Price/MHz"],df1_temp3["RP/MHz"],out=np.full_like(df1_temp3["Auction Price/MHz"], np.nan), where=df1_temp3["RP/MHz"] != 0)
			df1_temp3 = df1_temp3.reset_index()
			df1_temp3 = df1_temp3[["Band", "Circle", "Auction/Reserve"]]
			df1_temp3 = df1_temp3.pivot(index="Circle", columns='Band', values="Auction/Reserve")
			df1_temp3.columns = [str(x) for x in sorted(df1_temp3.columns)]
			z = df1_temp3.values.round(2)
			x = df1_temp3.columns
			y = df1_temp3.index
		if SubFeature == "Percent Unsold":
			df1 = df1.reset_index()
			df1_temp4 = df1.set_index(["Band", "Circle"])
			df1_temp4["Percent Unsold"] = np.divide(df1_temp4["Total Unsold (MHz)"],df1_temp4["Sale (MHz)"],out=np.full_like(df1_temp4["Total Unsold (MHz)"], np.nan), where=df1_temp4["Sale (MHz)"] != 0)*100
			df1_temp4 = df1_temp4.reset_index()
			df1_temp4 = df1_temp4[["Band", "Circle", "Percent Unsold"]]
			df1_temp4 = df1_temp4.pivot(index="Circle", columns = "Band", values ="Percent Unsold")
			df1_temp4.columns = [str(x) for x in sorted(df1_temp4.columns)]
			z = df1_temp4.values.round(1)
			x = df1_temp4.columns
			y = df1_temp4.index
		if SubFeature == "Percent Sold":
			df1 = df1.reset_index()
			df1_temp5 = df1.set_index(["Band", "Circle"])
			df1_temp5["Percent Sold"] = np.divide(df1_temp5["Total Sold (MHz)"],df1_temp5["Sale (MHz)"],out=np.full_like(df1_temp5["Total Sold (MHz)"], np.nan), where=df1_temp5["Sale (MHz)"] != 0)*100
			df1_temp5 = df1_temp5.reset_index()
			df1_temp5 = df1_temp5[["Band", "Circle", "Percent Sold"]]
			df1_temp5 = df1_temp5.pivot(index="Circle", columns = "Band", values ="Percent Sold")
			df1_temp5.columns = [str(x) for x in sorted(df1_temp5.columns)]
			z = df1_temp5.values.round(1)
			x = df1_temp5.columns
			y = df1_temp5.index
			
		#excluding summarydf as it is not needed for these SubFeatures
		if SubFeature not in  ["Auction/Reserve", "Percent Unsold", "Percent Sold"]:
			#preparing the dataframe of the summary bar chart on top of the heatmap
			summarydf = summarydf.round(1)
			summarydf = summarydf.reset_index()
			summarydf.columns = ["Band", "Total"]
			summarydf["Band"] = summarydf["Band"].astype(int)
			summarydf = summarydf.sort_values("Band")

			#preparing the summary chart 
			chart = summarychart(summarydf, 'Band', "Total")
			flag = True
			
		hovertext,colormatrix = htext_colmatrix_auction_year_band_metric(df1) #processing hovertext and colormatrix for bandwise in cal year dim
		hoverlabel_bgcolor = colormatrix #colormatrix processed from fuction "hovertext_and_colmatrix" for same above


	if Feature == "Operator Metric": #for the dimension "Calendar Year"
		df1 = df1.reset_index()
		df2_temp1 = df1.copy()
		selectedbands = st.sidebar.multiselect('Select Bands',bands_auctioned_dict[Year])
		subfeature_list = ["Total Outflow", "Total Purchase"]
		SubFeature = st.sidebar.selectbox('Select a SubFeature', subfeature_list,0)
		
		if SubFeature == "Total Outflow":
			temp1 = pd.DataFrame()
			if selectedbands != []:
				for band in selectedbands:
					temp2= df2_temp1[df2_temp1["Band"]==band]
					temp1 = pd.concat([temp2,temp1], axis =0)
				df2_temp1 = temp1
			else:
				df2_temp1 = df1.copy()
			operators_dim_cy_new=[]
			for op in oldoperators_dict[Year]:
				df2_temp1[op+"1"] = df2_temp1["Auction Price/MHz"]*df2_temp1[op]
				operators_dim_cy_new.append(op+"1")
			columnstoextract = ["Circle", "Band"]+operators_dim_cy_new
			df2_temp1 = df2_temp1[columnstoextract]
			operators_dim_cy_new = [x[:-1] for x in operators_dim_cy_new] # removing the last letter "1" from operator name
			df2_temp1.columns = ["Circle", "Band"]+ operators_dim_cy_new
			df2_temp1.drop("Band", inplace = True, axis =1)
			df2_temp1 = df2_temp1.groupby(["Circle"]).sum().round(0)
			df2_temp1 = df2_temp1.reindex(sorted(df2_temp1.columns), axis=1)
			
			z = df2_temp1.values
			x = df2_temp1.columns
			y = df2_temp1.index
			
			summarydf = df2_temp1.sum(axis=0)
			summarydf = summarydf.reset_index()
			summarydf.columns = ["Operators", SubFeature] 
			summarydf = summarydf.sort_values("Operators", ascending = False)
			#preparing the summary chart 
			chart = summarychart(summarydf, 'Operators', SubFeature)
			flag = True
			
			hovertext,colormatrix = htext_colmatrix_auction_year_operator_metric(df1, selectedbands, SubFeature, df2_temp1) #processing hovertext and colormatrix for operator wise in cal year dim
			hoverlabel_bgcolor = colormatrix #colormatrix processed from fuction "hovertext_and_colmatrix" for same above
		
		if SubFeature == "Total Purchase":
			if selectedbands != []:
				df2_temp2 = df1.copy()
				temp1=pd.DataFrame()
				for band in selectedbands:
					temp2= df2_temp2[df2_temp2["Band"]==band]
					temp1 = pd.concat([temp2,temp1], axis =0)
				df2_temp2 = temp1
			else:
				df2_temp2 = df1.copy()
			columnstoextract = ["Circle", "Band"]+oldoperators_dict[Year]
			df2_temp2 = df2_temp2[columnstoextract]
			df2_temp2.drop("Band", inplace = True, axis =1)
			df2_temp2 = df2_temp2.groupby(["Circle"]).sum().round(2)
			df2_temp2 = df2_temp2.reindex(sorted(df2_temp2.columns), axis=1)
			z = df2_temp2.values
			x = df2_temp2.columns
			y = df2_temp2.index
			
			summarydf = df2_temp2.sum(axis=0)
			summarydf = summarydf.reset_index()
			summarydf.columns = ["Operators", SubFeature] 
			summarydf = summarydf.sort_values("Operators", ascending = False)
			#preparing the summary chart 
			chart = summarychart(summarydf, 'Operators', SubFeature)
			flag = True
			
			hovertext,colormatrix = htext_colmatrix_auction_year_operator_metric(df1, selectedbands, SubFeature, df2_temp2) #processing hovertext and colormatrix for operator wise in cal year dim
			hoverlabel_bgcolor = colormatrix #colormatrix processed from fuction "hovertext_and_colmatrix" for same above
	

	data = [go.Heatmap(
		  z = z,
		  x = x,
		  y = y,
		  xgap = 1,
		  ygap = 1,
		  hoverinfo ='text',
		  text = hovertext,
# 		  hovertemplate = 'LSA: %{y}<extra></extra>',
		  colorscale = 'Hot',
		    texttemplate="%{z}", 
		    textfont={"size":10},
		    reversescale=True,
			)]	



#This is section is to visulize important data related to the telecom industry (may not be directed related to spectrum)

if selected_dimension == "Business Data":


	excel_content = io.BytesIO()

	with open("telecomdata_protected.xlsx", 'rb') as f:
		excel = msoffcrypto.OfficeFile(f)
		excel.load_key(password)
		excel.decrypt(excel_content)

	#loading data from excel file, the last letter "T" stands for telecom
	xlT = pd.ExcelFile(excel_content)
	sheetT = xlT.sheet_names
	dfT = pd.read_excel(excel_content, sheet_name=sheetT)


	Feature = st.sidebar.selectbox('Select a Feature', ["5G BTS Trends", "Subscribers Trends"])

	if Feature== "5G BTS Trends":


		df5gbts = dfT["5GBTS"] #load 5G BTS deployment data from excel file

		df5gbts["Date"] = df5gbts["Date"].dt.date

		df5gbts.drop(columns = "S.No", inplace = True)

		df5gbtsf = pd.pivot(df5gbts, values ="Total", index = "StateCode", columns = "Date")

		df5gbtsf.columns = [str(x) for x in df5gbtsf.columns ] #convet the dates into string 

		lastcolumn = df5gbtsf.columns[-1]


		df5gbtsf = df5gbtsf.sort_values(lastcolumn, ascending = False).head(20) #sort by the last column

		df5gbtsf = round(df5gbtsf/1000,2) #convert the BTS data in thousands (K)

		df5gbtsf = df5gbtsf.iloc[:,-16:] #select on last 16 dates 


		SubFeature = st.sidebar.selectbox('Select a SubFeature', ["Cumulative Values", "Percent of Total", "Incremental Values"])

		if SubFeature == "Cumulative Values":


			hovertext = htext_telecomdata_5gbts(df5gbtsf)

			#setting the data of the heatmap 

			data = [go.Heatmap(
				z = df5gbtsf.values,
				y = df5gbtsf.index,
				x = df5gbtsf.columns,
				xgap = 1,
				ygap = 1,
				hoverinfo ='text',
				text = hovertext,
				colorscale='Hot',
					texttemplate="%{z}", 
					textfont={"size":10},
					reversescale=True,
					),
				]

			summarydf = df5gbtsf.sum(axis=0)
			summarydf = summarydf.reset_index()
			summarydf.columns = ["Dates", SubFeature] 
			summarydf = summarydf.sort_values("Dates", ascending = False)
			#preparing the summary chart 
			chart = summarychart(summarydf, 'Dates', SubFeature)
			flag = True

			hoverlabel_bgcolor = "#000000" #subdued black
			xdtickangle= -45
			xdtickval=1
			title = "Indian 5G Base Stations Roll Out Trends"
			subtitle = "Cumulative BTS growth; Top 20 States/UT; Unit - Thousands; Sorted by the Recent Date"

		if SubFeature == "Percent of Total":


			hovertext = htext_telecomdata_5gbts(df5gbtsf)

			summarydf = df5gbtsf.sum(axis=0)

			df5gbtsfPercent = round((df5gbtsf/summarydf)*100,2)


			#setting the data of the heatmap 

			data = [go.Heatmap(
				z = df5gbtsfPercent.values,
				y = df5gbtsfPercent.index,
				x = df5gbtsfPercent.columns,
				xgap = 1,
				ygap = 1,
				hoverinfo ='text',
				text = hovertext,
				colorscale='Hot',
					texttemplate="%{z}", 
					textfont={"size":10},
					reversescale=True,
					),
				]

			hoverlabel_bgcolor = "#000000" #subdued black
			xdtickangle= -45
			xdtickval=1
			title = "Indian 5G Base Stations Percentage Roll Out Trends"
			subtitle = "Cumulative BTS growth; Top 20 States/UT; Unit - % of Total; Sorted by the Recent Date"
			flag = False #No summary chart to plot


		if SubFeature == "Incremental Values":

			lst =[]
			for row in df5gbtsf.values:

				increments = np.diff(row)
				lst.append(increments)

			df5gbtsincf = pd.DataFrame(lst)

			df5gbtsincf.index = df5gbtsf.index 
			df5gbtsincf.columns = df5gbtsf.columns[1:]


			lastcolumn = df5gbtsincf.columns[-1]


			df5gbtsincf = df5gbtsincf.sort_values(lastcolumn, ascending = False) #sort by the last column


			hovertext = htext_telecomdata_5gbts(df5gbtsf)

			#setting the data of the heatmap 

			data = [go.Heatmap(
				z = df5gbtsincf.values,
				y = df5gbtsincf.index,
				x = df5gbtsincf.columns,
				xgap = 1,
				ygap = 1,
				hoverinfo ='text',
				text = hovertext,
				colorscale='Hot',
					texttemplate="%{z}", 
					textfont={"size":10},
					reversescale=True,
					),
				]

			summarydf = df5gbtsincf.sum(axis=0)
			summarydf = summarydf.reset_index()
			summarydf.columns = ["Dates", SubFeature] 
			summarydf = summarydf.sort_values("Dates", ascending = False)
			#preparing the summary chart 
			chart = summarychart(summarydf, 'Dates', SubFeature)
			flag = True

			hoverlabel_bgcolor = "#000000" #subdued black
			xdtickangle= -45
			xdtickval=1
			title = "Indian 5G Base Stations Roll Out Trends"
			subtitle = "Incremental BTS growth; Top 20 States/UT; Unit - Thousands; Sorted by the Recent Date"


	if Feature== "Subscribers Trends":

		@st.cache_resource
		def loaddata():

			df = dfT["TelecomSubs"] #load 5G BTS deployment data from excel file

			return df

		#function to extract a list of dates from the list using start and end date from the slider
		def get_selected_date_list(listofallcolumns, start_date, end_date):
			    # Find the index of the first selected date
			    index1 = listofallcolumns.index(start_date)

			    # Find the index of the second selected date
			    index2 = listofallcolumns.index(end_date)

			    # Return a new list containing the dates from index1 to index2 (inclusive)
			    return listofallcolumns[index1:index2+1]


		dftelesubs = loaddata()

		dftelesubs = dftelesubs[dftelesubs["Date"]>=datetime(2013,1,31)] #filter the datframe for all dates more than the year 2013

		dftelesubs["Date"] = dftelesubs["Date"].dt.date

		dftelesubs.columns = [str(x) for x in dftelesubs.columns]

		dftelesubs = dftelesubs.replace(',','', regex=True)

		dftelesubs.drop(columns = ["Year","Months"], axis =1, inplace = True)

		selected_category = st.sidebar.multiselect('Select Categories', ["Wireless", "Wireline"])

		if len(selected_category) == 0 or len(selected_category) == 2:

			dftelesubsprocess = dftelesubs.copy()

		if len(selected_category) == 1:

			dftelesubsprocess = dftelesubs[dftelesubs["Category"]==selected_category[0]]
			

		dftelesubsprocess.drop(columns = ["Category"], axis =1, inplace = True)


		#processsing the dataframe for total subs

		dftotal = dftelesubsprocess.copy()


		dftotal = dftotal.melt(id_vars =["Date", "Circle"], value_vars = list(dftotal.columns[2:]))

		dftotal.columns = ["Date", "Circle", "Operator", "Subs"]

		dftotal = dftotal.groupby(["Date","Operator","Circle"]).sum()

		dftotal = dftotal.reset_index()

		dftotal.drop(columns = ["Circle"], axis =1, inplace = True)

		dftotal = dftotal.groupby(["Date","Operator"]).sum()

		dftotal = dftotal.reset_index()


		dftotal = pd.pivot(dftotal, index="Operator", columns = "Date", values = "Subs")

		SubFeature = st.sidebar.selectbox('Select a SubFeature', ["Cumulative Values", "Incremental Values"])

		if SubFeature=="Cumulative Values":

			listofallcolumns = list(dftotal.columns)


			with st.sidebar:

				start_date, end_date = st.select_slider("Select a Range of Dates", 
					options = listofallcolumns, value =(dftotal.columns[-24],dftotal.columns[-1]))


			date_range_list = get_selected_date_list(listofallcolumns, start_date, end_date)


			dftotalfilt = dftotal[date_range_list] #filter the dataframe with the selected dates


			dftotalfilt = dftotalfilt.sort_values(end_date, ascending = False) #filter the data on the first column selected by slider


			dftotalfilt = round(dftotalfilt.loc[~(dftotalfilt ==0).all(axis=1)]/1000000,2) # delete all rows with value zero and convert into millions

			if len(selected_category) ==0:
				selected_category = ["All"]

			subtitle = "Cumulative Values; Selected Category -" +",".join(selected_category)+ "; Unit - Millions; Sorted by the Recent Date"


			if len(date_range_list) >=30:
				texttemplate =""
			else:
				texttemplate = "%{z}"

			#setting the data of the heatmap 

			data = [go.Heatmap(
				z = dftotalfilt.values,
				y = dftotalfilt.index,
				x = dftotalfilt.columns,
				xgap = 1,
				ygap = 1,
				hoverinfo ='text',
				# text = hovertext,
				colorscale='Hot',
					texttemplate=texttemplate, 
					textfont={"size":10},
					reversescale=True,
					),
				]

			# dftotalfilt = (dftotalfilt/1000).round(1)
			# summarydf = dftotalfilt.sum(axis=0)
			# summarydf = summarydf.reset_index()
			# summarydf.columns = ["Dates", Feature] 
			# summarydf = summarydf.sort_values("Dates", ascending = False)
			# #preparing the summary chart 
			# chart = summarychart(summarydf/1000, 'Dates', Feature)
			# flag = True

		if SubFeature=="Incremental Values":


			lst =[]
			for row in dftotal.values:

				increments = np.diff(row)
				lst.append(increments)

			dftotalinc = pd.DataFrame(lst)

			dftotalinc.index = dftotal.index 
			dftotalinc.columns = dftotal.columns[1:]

			listofallcolumns = list(dftotalinc.columns)


			with st.sidebar:

				start_date, end_date = st.select_slider("Select a Range of Dates", 
					options = listofallcolumns, value =(dftotalinc.columns[-24],dftotalinc.columns[-1]))


			date_range_list = get_selected_date_list(listofallcolumns, start_date, end_date)


			dftotalincfilt = dftotalinc[date_range_list] #filter the dataframe with the selected dates


			dftotalincfilt = dftotalincfilt.sort_values(end_date, ascending = False) #filter the data on the first column selected by slider


			dftotalincfilt = round(dftotalincfilt.loc[~(dftotalincfilt ==0).all(axis=1)]/1000000,2) # delete all rows with value zero and convert into millions

			if len(selected_category) ==0:
				selected_category = ["All"]

			subtitle = "Incremental Values; Selected Category -" +",".join(selected_category)+ "; Unit - Millions; Sorted by the Recent Date"

			# hovertext = htext_telecomdata_5gbts(df5gbtsf)

			if len(date_range_list) >=30:
				texttemplate =""
			else:
				texttemplate = "%{z}"

			#setting the data of the heatmap 

			data = [go.Heatmap(
				z = dftotalincfilt.values,
				y = dftotalincfilt.index,
				x = dftotalincfilt.columns,
				xgap = 1,
				ygap = 1,
				hoverinfo ='text',
				# text = hovertext,
				colorscale='Hot',
					texttemplate=texttemplate, 
					textfont={"size":10},
					reversescale=True,
					),
				]

			# summarydf = df5gbtsincf.sum(axis=0)
			# summarydf = summarydf.reset_index()
			# summarydf.columns = ["Dates", SubFeature] 
			# summarydf = summarydf.sort_values("Dates", ascending = False)
			# #preparing the summary chart 
			# chart = summarychart(summarydf, 'Dates', SubFeature)
			# flag = True

			# hoverlabel_bgcolor = "#000000" #subdued black
			# xdtickangle= -45
			# xdtickval=1
			# title = "Indian 5G Base Stations Roll Out Trends"
			# subtitle = "Incremental BTS growth; Top 20 States/UT; Unit - Thousands; Sorted by the Recent Date"




#This section deals with titles and subtitles and hoverlabel color

units_dict = {"Reserve Price" : "Rs Cr/MHz", "Auction Price" : "Rs Cr/MHz", "Quantum Offered": "MHz", 
	      "Quantum Sold" : "MHz", "Quantum Unsold" : "MHz", "Total EMD" : "Rs Cr", "Total Outflow" : "Rs Cr",
	     "Auction/Reserve" : "Ratio", "Percent Unsold" : "% of Total Spectrum", "Percent Sold" : "% of Total Spectrum", "Total Purchase" : "MHz"}

#Plotting the final Heatmap	
fig = go.Figure(data=data)

if selected_dimension == "Spectrum Bands":
	if Feature == "Auction Map":
		fig.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))
		unit = units_dict[SubFeature]
		selected_operators = ["NA"]
		
		subtitle = "Unit - "+unit+"; Selected Operators - "+', '.join(selected_operators)+ " ; Summary Below - Sum of all LSAs"
		
	if (Feature == "Expiry Map") and (SubFeature == "Frequency Layout"):
		fig.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))
		unit = "Ch Size - "+str(channelsize_dict[Band])+" MHz"
		if selected_operators == []:
			selected_operators = ["All"]
		else:
			selected_operators = selected_operators
			
		subtitle = subtitle_freqlayout_dict[Band]+unit+"; Selected Operators - "+', '.join(selected_operators)
	
	if (Feature == "Expiry Map") and (SubFeature == "Yearly Trends"):
		fig.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white'))) #hoverbox color is black
		unit = "MHz"
		if selected_operator == "":
			selected_operator = "All"
		else:
			selected_operator = selected_operator
		subtitle = "Unit - "+unit+"; Selected Operators - "+selected_operator+ "; Summary Below - Sum of all LSAs"
		
	if (Feature == "Spectrum Map") and (SubFeature == "Frequency Layout"):
		fig.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))
		unit = "Ch Size - "+str(channelsize_dict[Band])+" MHz"
		if selected_operators == []:
			selected_operators = ["All"]
		else:
			selected_operators = selected_operators
			
		subtitle = subtitle_freqlayout_dict[Band]+unit+"; Selected Operators - "+', '.join(selected_operators)
			
	if (Feature == "Spectrum Map") and (SubFeature == "Operator Holdings"):
		fig.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))
		if (len(selected_category) == 0) or (len(selected_category) == 2):
			selected_category = "All"
		else:
			selected_category = selected_category[0]
		
		if selected_operators == []:
			selected_operators = ["All"]
		else:
			selected_operators = selected_operators
		
		unit = "MHz"
		subtitle = "Unit - "+unit+"; "+"India Total - Sum of all LSAs "+"; Selected Operators - "+', '.join(selected_operators)+ "; Category - "+ selected_category
	
	
	title = parttitle+" for "+str(Band)+" MHz Band"
	
	
	if (Feature == "Spectrum Map") and (SubFeature == "Operator %Share"):
		fig.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))
		if (len(selected_category) == 0) or (len(selected_category) == 2):
			selected_category = "All"
		else:
			selected_category = selected_category[0]
		
		if len(selected_operators) == 0: #debug
			selected_operators = ["All"]
		else:
			selected_operators = selected_operators
		
		unit = '% of Total'
		subtitle = "Unit - "+unit+ " ; Selected Operators - "+', '.join(selected_operators)+ "; Category - "+ selected_category
	
	
	title = parttitle+" for "+str(Band)+" MHz Band"

if (selected_dimension == "Auction Years") and (Feature == "Band Metric"):
	if (SubFeature =="Total Outflow") or (SubFeature == "Quantum Sold"):
		if selected_operators==[]:
			selected_operators = ["All"]
		else:
			selected_operators = selected_operators
	else:
		selected_operators = ["NA"]
		
	title = "Band Wise Auction Summary for the Year "+str(Year)
	
	if SubFeature in ["Reserve Price", "Auction Price", "Quantum Offered", "Quantum Sold", "Quantum Unsold", "Total EMD", "Total Outflow"]:
		partsubtitle = "; Summary Below - Sum of all LSAs"
	else:
		partsubtitle = ""
	subtitle = SubFeature+"; Unit -"+units_dict[SubFeature]+"; "+ "Selected Operators -" + ', '.join(selected_operators)+ partsubtitle
	fig.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))
	
	xdtickangle =0
	xdtickval =1
	
if (selected_dimension == "Auction Years") and (Feature == "Operator Metric"):
	if (SubFeature =="Total Outflow") or (SubFeature == "Total Purchase"):
		if selectedbands==[]:
			selectedbands = ["All"]
		else:
			selectedbands = selectedbands
	else:
		selectedbands = ["NA"]
	selectedbands = [str(x) for x in selectedbands]	
	title = "Operator Wise Outflow Summary for the Year "+str(Year)
	subtitle = SubFeature + "; Unit -"+units_dict[SubFeature]+"; Selected Bands -" + ', '.join(selectedbands) + "; Summary Below - Sum of all LSAs"
	fig.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))
	xdtickangle =0
	xdtickval =1


if (selected_dimension == "Business Data") and (Feature == "5G BTS Trends"):

	fig.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))
	hoverlabel_bgcolor = "#000000" #subdued black
	xdtickangle= -45
	xdtickval=1
	title = "Indian 5G Base Stations Roll Out Trends"


if (selected_dimension == "Business Data") and (Feature == "Subscribers Trends"):

	# hoverlabel_bgcolor = "#000000" #subdued black
	xdtickangle= -45
	xdtickval=1
	title = "Indian Telecom Subscribers Trends"
	


#updating figure layouts
fig.update_layout(uniformtext_minsize=12, 
		  uniformtext_mode='hide', 
		  xaxis_title=None, 
		  yaxis_title=None, 
		  yaxis_autorange='reversed',
		  font=dict(size=12),
		  template='simple_white',
		  paper_bgcolor=None,
		  height=575, width=1200,
		  margin=dict(t=80, b=50, l=50, r=50, pad=0),
		  yaxis=dict(
        	  tickmode='array'),
		  xaxis = dict(
		  side = 'top',
		  tickmode = 'linear',
		  tickangle=xdtickangle,
		  dtick = xdtickval), 
		)



#Here are only some last minute changes in the plot

#When dimensions as "Telecom Data" then convert x axis into category
if selected_dimension == "Business Data":
	fig.update_layout(xaxis_type='category')
else:
	pass

#removes tic labels if the date_range_list greater than a value
if (selected_dimension == "Business Data") and (Feature == "Subscribers Trends"):
	if len(date_range_list) >= 30:
		fig.update_xaxes(
		    tickmode='array',
		    ticktext=[''] * len(date_range_list),
		    tickvals=list(range(len(date_range_list)))
		)


fig.update_xaxes(fixedrange=True,showline=True,linewidth=1.2,linecolor='black', mirror=True)
fig.update_yaxes(fixedrange=True,showline=True, linewidth=1.2, linecolor='black', mirror=True)

style = "<style>h3 {text-align: left;}</style>"
with st.container():
	#plotting the main chart
	st.markdown(style, unsafe_allow_html=True)
	st.header(title)
	st.markdown(subtitle)
	st.write(fig)
	#plotting the summary chart
	if flag ==True:
		st.altair_chart(chart, use_container_width=False)




