#importing libraries
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
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
import datetime as dt 
import calendar
import time

from PIL import Image

from dateutil import relativedelta

import re

from collections import defaultdict

from dateutil.relativedelta import relativedelta

import io
import msoffcrypto

import pickle
from pathlib import Path
import streamlit_authenticator as stauth

import yaml
from yaml.loader import SafeLoader

from deta import Deta




#Set page layout here
st.set_page_config(layout="wide")

#--------User Authentication Starts-------


# DETA_KEY= st.secrets["deta_auth_tele_app"]

# deta = Deta(DETA_KEY)

# db = deta.Base("users_db")

# def insert_user(username, name, password):

# 	#Returns the users on a successful user creation, othewise raises an error

# 	return db.put({"key" : username, "name": name, "password" : password})

# # insert_user("pparker", "Peter Parker", "abc123")

# def fetch_all_users():
# 	#"Returns a dict of all users"

# 	res = db.fetch()

# 	return res.items

# users = fetch_all_users()

# st.write(pd.DataFrame(users))

# usernames = [user["key"] for user in users]
# names = [user["name"] for user in users]
# hashed_passwords = [user["password"] for user in users]


# authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
#     "telecommapp", "abcdef", 30)


# authenticator = stauth.Authenticate(
# 	users,
# 	'name',
# 	'abcde')


# def convert_dict_to_yaml_text(data):
#     yaml_text = yaml.dump(data, sort_keys=False)
#     return yaml_text

# # Convert dictionary to YAML text
# yaml_text = convert_dict_to_yaml_text(users)

# # Use the YAML text in memory for further processing
# st.write(yaml_text)


# with open('config.yaml') as file:
#     config = yaml.load(file, Loader=SafeLoader)


# authenticator = stauth.Authenticate(
# 	config['credentials'], 
# 	config['cookie']['name'],
# 	config['cookie']['key'],
# 	config['cookie']['expiry_days']
# 	)

# st.write(authenticator)


# name, authentication_status, username = authenticator.login("Login", "main")

# if authentication_status == False:
# 	st.error("Username/password is incorrect")

# if authentication_status == None:
# 	st.warning("Please enter your username and password")


#--------User Authentication Ends-------

#if authentication sucessful then render the app

# if authentication_status: 


#--------Fuctions, Constants, Configurations and Flags-------------


SummaryFlag = False # Code below will toggle to True to show summary chart


#--------hide streamlit style and buttons--------------

hide_st_style = '''
				<style>
				#MainMenu {visibility : hidden;}
				footer {visibility : hidder;}
				header {visibility :hidden;}
				<style>
				'''
st.markdown(hide_st_style, unsafe_allow_html =True)


#--------Functions for loading File Starts---------------------

@st.cache_resource
def loadrstousd():

	df = pd.read_csv("rs_to_usd.csv")

	return df


@st.cache_resource
def loadspectrumfile():

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

	return df

@st.cache_resource
def loadtelecomdatafile():

	password = st.secrets["db_password"]

	excel_content = io.BytesIO()

	with open("telecomdata_protected.xlsx", 'rb') as f:
		excel = msoffcrypto.OfficeFile(f)
		excel.load_key(password)
		excel.decrypt(excel_content)

	#loading data from excel file, the last letter "T" stands for telecom
	xlT = pd.ExcelFile(excel_content)
	sheetT = xlT.sheet_names
	dfT = pd.read_excel(excel_content, sheet_name=sheetT)

	return dfT



@st.cache_resource
def loadauctionbiddata():

	password = st.secrets["db_password"]

	excel_content = io.BytesIO()

	with open("auctionbiddata.xlsx", 'rb') as f:
		excel = msoffcrypto.OfficeFile(f)
		excel.load_key(password)
		excel.decrypt(excel_content)

	xl = pd.ExcelFile(excel_content)
	sheetauctiondata = xl.sheet_names
	df = pd.read_excel(excel_content, sheet_name=sheetauctiondata)

	return df

@st.cache_resource
def loadtraiagr():

	password = st.secrets["db_password"]

	excel_content = io.BytesIO()

	with open("trai_agr.xlsx", 'rb') as f:
		excel = msoffcrypto.OfficeFile(f)
		excel.load_key(password)
		excel.decrypt(excel_content)

	xl = pd.ExcelFile(excel_content)
	sheetauctiondata = xl.sheet_names
	df = pd.read_excel(excel_content, sheet_name=sheetauctiondata)

	return df

#--------Fuctions for loading File Ends--------------------


#--------Setting up the Constants Starts-------------------

  
state_dict = {'AP': 'Andhra Pradesh', 'AS': 'Assam', 'BH': 'Bihar', 'DL': 'Delhi', 'GU': 'Gujarat',
    'HA': 'Haryana','HP': 'Himachal Pradesh','JK': 'Jammu & Kashmir','KA': 'Karnataka',
    'KE': 'Kerala','KO': 'Kolkata','MP': 'Madhya Pradesh','MA': 'Maharashtra','MU': 'Mumbai',
    'NE': 'Northeast','OR': 'Odisha','PU': 'Punjab','RA': 'Rajasthan','TN': 'Tamil Nadu',
    'UPE': 'Uttar Pradesh (East)','UPW': 'Uttar Pradesh (West)','WB': 'West Bengal' }

subtitle_freqlayout_dict = {700:"FDD: Uplink - 703-748 MHz(shown); Downlink - 758-803(notshown); ",
         800:"Uplink - 824-844 MHz(shown); Downlink - 869-889 MHz(not shown); ", 
         900:"Uplink - 890-915 MHz(shown); Downlink - 935-960 MHz(not shown); ", 
         1800:"Uplink - 1710-1785 MHz(shown); Downlink - 1805-1880 MHz(notshown); ", 
         2100:"Uplink - 1919-1979 MHz(shown); Downlink - 2109-2169 MHz(notshown); ",
         2300:"Up & Downlinks - 2300-2400 MHz(shown); ",
         2500:"Up & Downlinks - 2500-2690 MHz(shown); ",
         3500:"Up & Downlinks - 3300-3670 MHz(shown); ",
         26000:"Up & Downlinks - 24250-27500 MHz(shown); "}

#Operators who are the current owners of blocks of spectrum in these bands 
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

#Operators who were the original buyer of spectrum
oldoperators_dict = {2010 : ["Bharti", "QCOM", "Augere", "Vodafone", "Idea", "RJIO", "RCOM", "STel", "Tata", "Aircel", "Tikona"],
		    2012 : ["Bharti", "Vodafone", "Idea", "Telenor", "Videocon"],
		    2013 : ["MTS"],
		    2014 : ["Bharti", "Vodafone", "Idea", "RJIO", "RCOM", "Aircel", "Telenor"],
		    2015 : ["Bharti", "Vodafone", "Idea", "RJIO", "RCOM", "Tata", "Aircel"],
		    2016 : ["Bharti", "Vodafone", "Idea", "RJIO", "RCOM", "Tata", "Aircel"],
		    2021 : ["Bharti", "RJIO", "VodaIdea"],
		    2022 : ["Bharti", "RJIO", "VodaIdea", "Adani"] }

#Spectrum Bands Auctioned in that Calender Year
bands_auctioned_dict = {2010 : [2100, 2300],
	       2012 : [800, 1800],
	       2013 : [800, 900, 1800],
	       2014 : [900, 1800],
	       2015 : [800, 900, 1800, 2100],
	       2016 : [700, 800, 900, 1800, 2100, 2300, 2500],
	       2021 : [700, 800, 900, 1800, 2100, 2300, 2500],
	       2022 : [600, 700, 800, 900, 2100, 2300, 2500, 3500, 26000]}
		    

#if "1" the expiry tab in spectrum_map file is present and if "0" then not present
exptab_dict = {700:1, 800:1, 900:1, 1800:1, 2100:1, 2300:1, 2500:1, 3500:1, 26000:1}

#Setting the channel sizes for respective frequency maps
channelsize_dict = {700:2.5, 800:0.625, 900:0.2, 1800:0.2, 2100:2.5, 2300:2.5, 2500:5, 3500:5, 26000:25}

#scaling the granularity of the layout of the x axis in the heatmap plot for the respective bands
xdtickfreq_dict = {700:1, 800:0.25, 900:0.4, 1800:1, 2100:1, 2300:1, 2500:2, 3500:5, 26000:50}

#used to control the number of ticks on xaxis for chosen feature = AuctionMap
dtickauction_dict = {700:1, 800:1, 900:1, 1800:1, 2100:1, 2300:1, 2500:1, 3500:1, 26000:1}

# used to set the vertical line widths for the heatmap chart 
xgap_dict = {700:1, 800:1, 900:0.5, 1800:0, 2100:1, 2300:1, 2500:1, 3500:1, 26000:1}

#Minor adjustment for tool tip display data for channel frequency on heatmap
#The reason is that the start freq of the spectrum tab is shifted delpberately by few MHz
#This is to align the labels on the xaxis to align properly with the edge of the heatmap
xaxisadj_dict = {700:1, 800:0.25, 900:0, 1800:0, 2100:1, 2300:1, 2500:2, 3500:0, 26000:0}

#Setting the constant to describe the type of band TDD/FDD
bandtype_dict = {700:"FDD", 800:"FDD", 900:"FDD", 1800:"FDD", 2100:"FDD", 2300:"TDD", 2500:"TDD", 3500:"TDD", 26000:"TDD"}

#auctionfailyears when the auction prices for all LSAs were zero and there are no takers 
auctionfailyears_dict = {700:["2016","2021"], 800:["2012"], 900:["2013","2016"], 1800:["2013"], 
        2100:[], 2300:["2022"], 2500:["2021"], 3500:[], 26000:[]}

#auction sucess years are years where at least in one of the LASs there was a winner
auctionsucessyears_dict = {700:[2022], 
        800:[2013, 2015, 2016, 2021, 2022], 
        900:[2014, 2015, 2021, 2022], 
        1800:[2012, 2014, 2015, 2016, 2021, 2022], 
        2100:[2010, 2015, 2016, 2021, 2022], 
        2300:[2010, 2016, 2021, 2022],  #added 2022 an as exception (due to error) need to revist the logic of succes and failure
        2500:[2010, 2016, 2022], 
        3500:[2022], 
        26000:[2022]}

#end of month auction completion dates dictionary for the purpose of evaluting rs-usd rates 

auction_eom_dates_dict = {2010 : datetime(2010,6,30), 2012: datetime(2012,11,30),2013: datetime(2013,3,31), 2014: datetime(2014,2,28),
					2015 : datetime(2015,3,31), 2016 : datetime(2016,10,31), 2021: datetime(2021,3,31), 2022: datetime(2022,8,31)}

#Error dicts defines the window width = difference between the auction closing date and the auction freq assignment dates
#This values is used to map expiry year of a particular freq spot to the operator owning that spot
# errors_dict= {700:0.25, 800:1, 900:1, 1800:1, 2100:1.5, 2300:1.25, 2500:1, 3500:0.1, 26000:0.5}

errors_dict= {700:0.25, 800:1, 900:1, 1800:1, 2100:1.5, 2300:1.25, 2500:1, 3500:1, 26000:10} #debug 2024 (Feb)

list_of_circles_codes = ['AP','AS', 'BH', 'DL', 'GU', 'HA', 'HP', 'JK', 'KA', 'KE', 'KO', 'MA', 'MP',
       	   'MU', 'NE', 'OR', 'PU', 'RA', 'TN', 'UPE', 'UPW', 'WB']


#function to count number of items in a list and outputs the result as dictionary
#Used to extract data table for Spectrum Layout Dimension when it is filtered by Operators      	   
def count_items_in_dataframe(df):
    counts = {}

    for col in df.columns:
        for idx, item in enumerate(df[col]):
            if isinstance(item, (int, float)) and not pd.isnull(item):
                # item_key = str(item)  # Convert float item to string
                item_key = int(item) #adding this item solved the problem
                if item_key not in counts:
                    counts[item_key] = [0] * len(df)
                counts[item_key][idx] += 1

    df_counts = pd.DataFrame.from_dict(counts, orient='columns')
    return df_counts


#function used to prepare the color scale for the freqmap
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

#function used for calculating the expiry year heatmap for the subfeature yearly trends
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

#function used for calculating the quantum of spectrum expiring mapped to LSA and Years 
#This is for feature expiry map and the subfeature yearly trends 
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

#funtion used for processing pricing datframe for hovertext for the feature auction map
#The feature auction map is under the dimension Spectrum Bands
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

#This general function for converting columns of dataframe into string
@st.cache_resource
def coltostr(df):
	lst =[]
	for col in df.columns:
		lst.append(str(col))
	df.columns=lst
	return df

#This functions adds dummy columns to the dataframe for auction failed years
@st.cache_resource
def adddummycols(df,col):
    df[col]="NA  " # space with NA is delibelitratly added.
    cols = sorted(df.columns)
    df =df[cols]
    return df

#This function maps the year in which the spectrum was acquired
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
  
#This fuctions processes the hovertext for the Feature Spectrum Map, and Sub Feature Frequency Layout
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

#This function processes the hovertext for Feature expiry map, and SubFeature Freq Layout
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

#This function is used for processing hovertext for Feature expiry map, and subfeature Yearly Trends with operator selection "All"
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


#processing for hovertext for Fearure expiry map, and SubFeature Yearly Trends along with operator menue
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
	
#This if for processing for hovertext for the Feature Auction Map
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


#processing for hovertext and colormatrix for Spectrum Band, Features- Spectrum Map, SubFeature - Operator Holdings 
@st.cache_resource
def htext_colmatrix_spec_map_op_hold_share(dfff, selected_operators, operatorlist):


	operators_to_process = list(dfff.columns)

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


#processing for hovertext and colormatrix for Dim - Auction Years, Fearure - Band Metric, SubFeatures Reserve Price etc
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

#processing for hovertext and colormatrix for Auction Year, Operator Metric, SubFeatures - Total Outflow, Total Purchase
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


#processing for hovertext for Business Data and 5G BTS Trends
@st.cache_resource
def htext_businessdata_5gbts(df5gbtsf): 

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



#processing for hovertext for Business Data and Subscribers Trends Cumulatitive
@st.cache_resource
def htext_businessdata_telesubscum(dftotalfilt): 

	summarydf = dftotalfilt.sum(axis=0)
	dftotalfiltPercent = round((dftotalfilt/summarydf)*100,2)

	lst =[]
	for row in dftotalfilt.values:

		increments = np.diff(row)
		lst.append(increments)

	dftotalfiltInc = pd.DataFrame(lst)

	dftotalfiltInc.index = dftotalfilt.index 

	dftotalfiltInc.columns = dftotalfilt.columns[1:]

	lastcolumn = dftotalfiltInc.columns[-1]
	dftotalfiltInc = dftotalfiltInc.sort_values(lastcolumn, ascending = False) #sort by the last column

	hovertext=[]

	for yi,yy in enumerate(dftotalfilt.index):
		hovertext.append([])
		for xi,xx in enumerate(dftotalfilt.columns):

			subcum = dftotalfilt.loc[yy,xx]
			subpercent = dftotalfiltPercent.loc[yy,xx]

			try:
				subinc = dftotalfiltInc.loc[yy,xx]
			except:
				subinc = np.nan


			hovertext[-1].append(
					    'Operator: {}\
					    <br>Date: {}\
					    <br>Subs Cum: {} Millions\
					    <br>Subs Inc: {} Millions\
					    <br>Subs Cum: {} % of Total'

				     .format( 
					    yy,
					    xx,
					    subcum,
					    round(subinc,2),
					    subpercent,
					    )
					    )
	return hovertext



#processing for hovertext for Business Data and Subscribers Trends Cumulatitive
@st.cache_resource
def htext_businessdata_telesubsinc(dftotalfiltinc): 
	hovertext=[]

	for yi,yy in enumerate(dftotalfiltinc.index):
		hovertext.append([])
		for xi,xx in enumerate(dftotalfiltinc.columns):

			subinc = dftotalfiltinc.loc[yy,xx]


			hovertext[-1].append(
					    'Operator: {}\
					    <br>Date: {}\
					    <br>Subs Inc: {} Millions'

				     .format( 
					    yy,
					    xx,
					    round(subinc,2),
					    )
					    )
	return hovertext


#processing for hovertext for Business Data and Subscribers Market Share
@st.cache_resource
def htext_businessdata_telesubsms(dftotal,dftotalpercentms): 

	hovertext=[]

	for yi,yy in enumerate(dftotal.index):
		hovertext.append([])
		for xi,xx in enumerate(dftotal.columns):

			subtotal = dftotal.loc[yy,xx]
			subpercentms = dftotalpercentms.loc[yy,xx]


			hovertext[-1].append(
					    'Operator: {}\
					    <br>Circle: {}\
					    <br>Subs Total: {} Millions\
					    <br>Subs Share: {} % of Total'

				     .format( 
					    yy,
					    xx,
					    round(subtotal/1000000,1),
					    subpercentms,
					    )
					    )
	return hovertext


#processing for hovertext for Business Data and License Fees
@st.cache_resource
def htext_businessdata_licensefees(dflfsfbysubfeature, summarydf_for_hovertext): 

	dfabsolute = dflfsfbysubfeature.copy()


	dfpercent = round((dflfsfbysubfeature/summarydf_for_hovertext)*100,2)


	hovertext=[]

	for yi,yy in enumerate(dfabsolute.index):
		hovertext.append([])
		for xi,xx in enumerate(dfabsolute.columns):

			absolute = dfabsolute.loc[yy,xx]
			percent = dfpercent.loc[yy,xx]


			hovertext[-1].append(
					    'Yaxis Label: {}\
					    <br>FY: {}\
					    <br>Abs Value: {} Rs Cr\
					    <br>Percentage: {} % of Total'

				     .format( 
					    yy,
					    xx,
					    absolute,
					    percent,
					    )
					    )
	return hovertext

#---------------Hovertest for BlocksAllocated Starts---------------------

@st.cache_resource
def htext_businessdata_FinancialSPWise(df_finmetric,df_finmetric_prec,df_finmetricINC):

	hovertext = []
	for yi,yy in enumerate(df_finmetric.index):
		hovertext.append([])

		for xi,xx in enumerate(df_finmetric.columns):

			absvalue = df_finmetric.loc[yy,xx]
			percentoftotal = df_finmetric_prec.loc[yy,xx]
			increments = df_finmetricINC.loc[yy,xx]



			hovertext[-1].append(
					    'Bidder: {}\
					    <br>Date: {}\
					    <br>Abs Value : {} Rs K Cr\
					    <br>Perc : {} of Total \
					    <br>Increments : {} Rs K Cr'
				
				     .format( 
					    yy,
					    xx,
					    absvalue,
					    percentoftotal,
					    round(increments,2),
					    )
					    )

	return hovertext


#---------------Hovertest for BlocksAllocated Ends---------------------	

#processing hovertext for auction data 

@st.cache_resource
def htext_colormatrix_auctiondata_2010_3G_BWA_BidsCircleWise(dfbidcirclwise, dftemp, selected_lsa,start_round,end_round,dfprovallcblks_endrd):

	filt_last_round = (dfbidcirclwise["Clk_Round"] == end_round)

	dfbidcirclwiselastrd = dfbidcirclwise[filt_last_round].drop(columns = ["Clk_Round","PWB_Start_ClkRd","Rank_PWB_Start_ClkRd",
		"Possible_Raise_Bid_ClkRd","Bid_Decision","PWB_End_ClkRd"], axis =1).reset_index()

	dfbidcirclwiselastrd = dfbidcirclwiselastrd.pivot(index="Bidder", columns='LSA', values="Rank_PWB_End_ClkRd").sort_index(ascending=False)


	dftempheatperc = dftemp.pivot(index="Bidder", columns='LSA', values="Bid_Decision_Perc")

	dftempheatperc = dftempheatperc.sort_values(selected_lsa, ascending = True)

	dftempheatabs = dftemp.pivot(index="Bidder", columns='LSA', values="Bid_Decision")

	dftempheatabs = dftempheatabs.sort_values(selected_lsa, ascending = True)


	hovertext = []
	dict_col={}
	dict_result={}
	for yi,yy in enumerate(dftempheatabs.index):
		hovertext.append([])
		list_col=[]
		list_result=[]
		for xi,xx in enumerate(dftempheatabs.columns):

			totalbidsagg = dftempheatabs.loc[yy,xx]

			totalbissperc = dftempheatperc.loc[yy,xx]

			totalblksrdend = dfprovallcblks_endrd.loc[yy,xx]

			finalrank = dfbidcirclwiselastrd.loc[yy,xx]
	
		
			if finalrank in [1,2,3,4]:
				result = "WON"
				ccode = '#008000' #(green)
			else:
				result = "LOST"
				ccode = '#FF0000' #(red)

			list_result.append(result)

			list_col.append(ccode)

			hovertext[-1].append(
					    'Bidder: {}\
					    <br>Circle: {}\
					    <br>Agg Bids : {} Nos\
					    <br>Agg Bids: {} % of Total\
					    <br>Prov Result : {}\
					    <br>Prov Rank: {}\
					    <br>Prov BLKs: {}'

				     .format( 
					    yy,
					    state_dict[xx],
					    totalbidsagg,
					    totalbissperc,
					    result,
					    finalrank,
					    round(totalblksrdend,0),
					    )
					    )

		dict_col[yy]=list_col
		dict_result[yy]=list_result

	temp = pd.DataFrame(dict_col).T


	temp.columns = dftempheatabs.columns


	resultdf = pd.DataFrame(dict_result).T

	resultdf.columns = dftempheatabs.columns 
	
	colormatrix = list(temp.values)

	return hovertext, colormatrix, resultdf


#-----------------Hovertext for Provisional Winning Bids Starts----------------------

@st.cache_resource
def htext_colormatrix_auctiondata_2010_3G_BWA_ProvWinningBid(dfrp, dftemp, pwbtype, round_number):


	dftemp = dftemp.sort_index(ascending=True)

	dftemprpmul = round(dftemp/dfrp.values,1)

	hovertext = []
	dict_col={}
	for yi,yy in enumerate(dftemp.index):
		hovertext.append([])
		list_col=[]
		for xi,xx in enumerate(dftemp.columns):

			pwb = dftemp.loc[yy,xx]
			pwbmulofrp = dftemprpmul.loc[yy,xx]


			if str(pwb)  == "nan":
				ccode = '#808080' #(grey)
			else:
				ccode = '#228B22' #(green)

			list_col.append(ccode)

			hovertext[-1].append(
					    'Bidder: {}\
					    <br>Circle: {}\
					    <br>PWB : {} Rs Cr\
					    <br>PWB / Reserve P: {}\
					    <br>PWB Type : {}\
					    <br>Round No: {}'

				     .format( 
					    yy,
					    state_dict[xx],
					    pwb,
					    pwbmulofrp,
					    pwbtype,
					    round_number,
					    )
					    )

		dict_col[yy]=list_col

	temp = pd.DataFrame(dict_col).T


	temp.columns = dftemp.columns
	
	colormatrix = list(temp.values)

	return hovertext, colormatrix

#-----------------Hovertext for Provisional Winning Bids Ends----------------------


#---------------Hovertest for Demand Intensity---------------------

@st.cache_resource
def htext_auctiondata_2010_3G_BWA_DemandIntensity(dfbid,ADPrecOfBlksforSale):

	dfbidaAD = dfbid.pivot(index="LSA", columns='Clock Round', values="Aggregate Demand").sort_index(ascending=True)

	dfbidaED = dfbid.pivot(index="LSA", columns='Clock Round', values="Excess Demand").sort_index(ascending=True)


	hovertext = []
	for yi,yy in enumerate(dfbidaAD.index):
		hovertext.append([])

		for xi,xx in enumerate(dfbidaAD.columns):

			aggdemand = dfbidaAD.loc[yy,xx]
			aggdemperc = ADPrecOfBlksforSale.loc[yy,xx]
			excessdemand = dfbidaED.loc[yy,xx]

			hovertext[-1].append(
					    'Circle: {}\
					    <br>Round No: {}\
					    <br>Agg Demand : {} Slots\
					    <br>Ratio (AD/Total) : {} \
					    <br>Excess Demand : {} Slots'
				

				     .format( 
					    yy,
					    xx,
					    aggdemand,
					    aggdemperc,
					    excessdemand,
					    )
					    )

	return hovertext


#---------------Hovertest for Demand Intensity Ends---------------------


#---------------Hovertest for Bidding Activity Total---------------------

@st.cache_resource
def htext_auctiondata_2010_3G_BWA_BiddingActivity(dfbid, column_name):

	filt = dfbid["Clk_Round"]==1

	dfbidRD1 = dfbid[filt]

	dfbidactivity = dfbid.pivot(index="Bidder", columns='Clk_Round', values=column_name).sort_index(ascending=True)

	dfbidactivityRd1 = dfbidRD1.pivot(index="Bidder", columns='Clk_Round', values="Pts_Start_Round").sort_index(ascending=True)

	dfbidactivityratio = round((dfbidactivity/dfbidactivityRd1.values),2)


	hovertext = []
	for yi,yy in enumerate(dfbidactivity.index):
		hovertext.append([])

		for xi,xx in enumerate(dfbidactivity.columns):

			pointsinplay = dfbidactivity.loc[yy,xx]
			pointsratio = dfbidactivityratio.loc[yy,xx]


			hovertext[-1].append(
					    'Bidder: {}\
					    <br>Round No: {}\
					    <br>Points in Play : {} Nos\
					    <br>Ratio (Actual/Initial) : {}'
				

				     .format( 
					    yy,
					    xx,
					    pointsinplay,
					    pointsratio,
					    )
					    )

	return hovertext


#---------------Hovertest for Bidding Activity Total Ends---------------------



#---------------Hovertest for Points Lost---------------------

@st.cache_resource
def htext_auctiondata_2010_3G_BWA_PointsLost(dfbidactivity, dfbidactivityperc):


	hovertext = []
	for yi,yy in enumerate(dfbidactivity.index):
		hovertext.append([])

		for xi,xx in enumerate(dfbidactivity.columns):

			pointslost = dfbidactivity.loc[yy,xx]
			pointslostperc = dfbidactivityperc.loc[yy,xx]


			hovertext[-1].append(
					    'Bidder: {}\
					    <br>Round No: {}\
					    <br>Points Lost : {} Nos\
					    <br>Points Lost : {} % of Initial'
				
				     .format( 
					    yy,
					    xx,
					    pointslost,
					    pointslostperc,
					    )
					    )

	return hovertext


#---------------Hovertest for Points Lost Ends---------------------


#---------------Hovertest for BlocksAllocated Starts---------------------

@st.cache_resource
def htext_auctiondata_2010_3G_BWA_BlocksAllocated(dftemp):

	dftemp = dftemp.sort_index(ascending=True)

	hovertext = []
	for yi,yy in enumerate(dftemp.index):
		hovertext.append([])

		for xi,xx in enumerate(dftemp.columns):

			blocksalloc = dftemp.loc[yy,xx]
			spectrumMHz = (dftemp.loc[yy,xx])*blocksize


			hovertext[-1].append(
					    'Bidder: {}\
					    <br>Circle: {}\
					    <br>BLKs Allocated : {} Nos\
					    <br>Spectrum : {} MHz'
				
				     .format( 
					    yy,
					    xx,
					    blocksalloc,
					    round(spectrumMHz,2),
					    )
					    )

	return hovertext


#---------------Hovertest for BlocksAllocated Ends---------------------



#---------------Hovertest for LastBidPrice Starts---------------------

@st.cache_resource
def htext_colormatrix_auctiondata_2010_3G_BWA_LastBidPrice(dflastsubbidheat,dflastsubbidratio,dfbid):


	hovertext = []
	dict_col = {}
	for yi,yy in enumerate(dflastsubbidheat.index):
		hovertext.append([])
		list_col=[]
		for xi,xx in enumerate(dflastsubbidheat.columns):

			lastbid = dflastsubbidheat.loc[yy,xx]
			lastbidratiorp = dflastsubbidratio.loc[yy,xx]
			blocksforsale = dfbid.T.loc["Blocks For Sale",xx]

			if lastbid > 0:
				ccode = '#880808' #(red)
			else:
				ccode = '#808080' #(grey)

			list_col.append(ccode)



			hovertext[-1].append(
					    'Bidder: {}\
					    <br>Circle: {}\
					    <br>LastBid : {} RsCr/BLK\
					    <br>LastBidRatio : {} Bid/RP\
					    <br>BLKsForSale : {} Nos'
				
				     .format( 
					    yy,
					    xx,
					    lastbid,
					    round(lastbidratiorp,2),
					    blocksforsale,
					    )
					    )

		dict_col[yy]=list_col

	temp = pd.DataFrame(dict_col).T

	temp.columns = dflastsubbidheat.columns

	colormatrix = list(temp.values)

	return hovertext, colormatrix


#---------------Hovertest for LastBidPrice Ends---------------------


#preparing color scale for hoverbox for Spectrum and Expiry maps
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

#shaping colorscale for driving the color of hoverbox of Spectrum and Expiry maps
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

	
	text = bar.mark_text(size = 10, dx=0, dy=-7, color = 'white').encode(text=ycolumn+':Q')
	
	chart = (bar + text).properties(width=1120, height =150)
	chart = chart.configure_title(fontSize = 20, font ='Arial', anchor = 'middle', color ='black')
	return chart


#**********  Main Program Starts here ***************


# authenticator.logout("Logout", "sidebar") #logging out authentication
# st.sidebar.title(f"Welcome {name}")
# image = Image.open('parag_kar.jpg') #debug
# st.sidebar.image(image) #debug


#set flags extracting chart data in the data tab

chart_data_flag = False #set this to true only if this chart exists.

with st.sidebar:
	selected_dimension = option_menu(
		menu_title = "Select a Menu",
		options = ["Spectrum Bands", "Auction Years", "Business Data", "Auction Data"],
		icons = ["1-circle-fill", "2-circle-fill", "3-circle-fill", "4-circle-fill"],
		menu_icon = "arrow-down-circle-fill",
		default_index =0,
		)

#loading file rupee to USD and finding the exchange rate in the auction eom

auction_eom_list = [x.date() for x in list(auction_eom_dates_dict.values())]

dfrsrate = loadrstousd()

auction_rsrate_dict ={} #the dictionary which stores all the values of the rupee usd rates

dfrsrate["Date"] = pd.to_datetime(dfrsrate["Date"])

dfrsrate = dfrsrate.set_index("Date").asfreq("m")

for index in dfrsrate.index:

	if index.date() in auction_eom_list:

		auction_rsrate_dict[index.year] = dfrsrate.loc[index,:].values[0]

# st.write(auction_rsrate_dict)



if selected_dimension == "Spectrum Bands":

	#selecting a Spectrum band
	Band = st.sidebar.selectbox('Select a Band', list(exptab_dict.keys()), 3) #default index 1800 MHz Band
	
	#setting up tabs for reading data from excel file - spectrum map
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


	#loading spectrum excel file

	df = loadspectrumfile()


	#processing colorcode excel data tab
	colcodes = df["ColorCodes"]
	colcodes=colcodes.set_index("Description")

	#Loading the excel tabs in the file spectrum map into various dataframes
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
	

	#processing "Spectrum_all" excel tab data of the file "spectrum map"
	dff = df[spectrumall] #contains information of LSA wise mapping oldoperators with new operators
	dffcopy = dff.copy() #make a copy for "Operator holdings" subfeature under the feature "Freq Layout"
	dff = cal_bw_mapped_to_operators_auctionmap(dff)
	dff = coltostr(dff)
	dff = adddummycols(dff,auctionfailyears_dict[Band])
	dff = dff.applymap(lambda x: "NA  " if x=="" else x) # space with NA is delibelitratly added as it gets removed with ","

	#processing the pricemaster excel tab data in the file "spectrum map"
	pricemaster = df["Master_Price_Sheet"]
	pricemaster.rename(columns = {"FP" : "Auction Price", "DP": "Reserve Price"}, inplace = True)

	#processing & restructuring dataframe spectrum offered vs sold & unsold for hovertext the data of heatmap
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

	#processing & restructuring dataframe auction price for hovertext of the data of heatmap
	auctionprice = pricemaster[(pricemaster["Band"] == Band) & (pricemaster["Year"] != 2018)]
	auctionprice = auctionprice.pivot(index=["LSA"], columns='Year', values="Auction Price").fillna("NA")
	auctionprice = auctionprice.loc[:, (auctionprice != 0).any(axis=0)]
	auctionprice = auctionprice.applymap(lambda x: round(x,2))
	auctionprice = coltostr(auctionprice) #convert columns data type to string
	auctionprice = adddummycols(auctionprice,auctionfailyears_dict[Band])
	auctionprice = auctionprice.replace(0,"NA")

	#processing & restructuring dataframe reserve price for hovertext of the data of heatmap
	reserveprice = pricemaster[(pricemaster["Band"] == Band) & (pricemaster["Year"] != 2018)]
	reserveprice = reserveprice.pivot(index=["LSA"], columns='Year', values="Reserve Price").fillna("NA")
	reserveprice = reserveprice.loc[:, (reserveprice != 0).any(axis=0)]
	reserveprice = reserveprice.applymap(lambda x: round(x,2))
	reserveprice = coltostr(reserveprice) #convert columns data type to string
	reserveprice = reserveprice.replace(0,"NA")

	#mapping the year of auction with channels in the spectrum maps
	ayear = cal_year_spectrum_acquired(ef,excepf,pf1)

	Feature = st.sidebar.selectbox('Select a Feature', ["Spectrum Map", "Expiry Map", "Auction Map"], 0) #Default Index first

	#Processing For Dimension = "Frequency Band" & Features
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

			#processing for hovertext
			hovertext = htext_specmap_freq_layout(hf)

			#processing for data for the data tab for the SubFeature "Frequency Layout"
			chartdata_df = count_items_in_dataframe(sf)*channelsize_dict[Band]

			chartdata_df.index = sf.index

			def get_key_from_value(dictionary, value):
			    reverse_dict = {v: k for k, v in dictionary.items()}
			    return reverse_dict.get(value)

			if len(selected_operators) ==0:
				for col in chartdata_df.columns:
					operatorname = get_key_from_value(operators,int(col))
					chartdata_df.rename(columns = {col : operatorname}, inplace = True)
			if len(selected_operators) > 0:
				for col in chartdata_df.columns:
					operatorname = get_key_from_value(selected_op_dict,int(float(col)))
					chartdata_df.rename(columns = {col : operatorname}, inplace = True)

			chartdata_df = chartdata_df.T

			chartdata_df["Total"] = chartdata_df.sum(axis=1)

			chartdata_df.sort_values("Total", ascending=False, inplace = True)

			chartdata_df = chartdata_df.T

			chart_data_flag = True #Plot only if this flag is true 

			
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

			fig = go.Figure(data=data)
			
			hcolscale=colscale_hbox_spectrum_expiry_maps(operators, colcodes)  #colorscale for hoverbox
			hoverlabel_bgcolor = transform_colscale_for_spec_exp_maps(hcolscale, hf) #shaping the hfcolorscale

			currency_flag = True #default
			
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

			fig = go.Figure(data=data)

			currency_flag = True # default
			
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

			fig = go.Figure(data=data)

			currency_flag = True #default

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

			fig = go.Figure(data=data)

			currency_flag = True #default

			hcolscale=colscale_hbox_spectrum_expiry_maps(operators, colcodes)  #colorscale for hoverbox
			hoverlabel_bgcolor = transform_colscale_for_spec_exp_maps(hcolscale, hf) #shaping the hfcolorscale

		if SubFeature == "Yearly Trends":
			bandexpcalsheetf = bandexpcalsheetf.set_index("LSA") #Loading Dataframe from BandExpCalSheet
			operatorslist = ["All"]+sorted(list(newoperators_dict[Band].keys()))
			selected_operator = st.sidebar.selectbox('Select an Operator', operatorslist)
			selected_operator = selected_operator.strip()
			if selected_operator == "All":
				eff = exp_year_cal_yearly_trends(ef,selected_operator)
				bwf = bw_exp_cal_yearly_trends(sff,ef)
				hovertext = htext_expmap_yearly_trends_with_all_select(bwf,eff) #hovertext for "All"
			else:
				temp = bandexpcalsheetf.copy()

				for i, row in enumerate(bandexpcalsheetf.values):
					for j, item in enumerate(row):
						try: # debug 2024
							op = item.split(";")[1]
						except: #debug 2024
							pass #debug 2024

						if op != selected_operator:
							temp.iloc[i,j] = np.nan
						else:
							temp.iloc[i,j] = item.split(";")[0]
				
				for col in temp.columns:
					temp[col] = temp[col].astype(float)
				eff = exp_year_cal_yearly_trends(temp,selected_operator)
				hovertext = htext_expmap_yearly_trends_with_op_select(eff) #hovertext with operator selections
		
		
			#preparing the dataframe of the summary bar chart on top of the heatmap
			summarydf = eff.sum().reset_index()
			summarydf.columns = ["ExpYears", "TotalMHz"]
			summarydf["ExpYears"]= summarydf["ExpYears"].astype(float)
			summarydf["ExpYears"] = summarydf["ExpYears"].sort_values(ascending = True)

			#preparing the summary chart 
			chart = summarychart(summarydf, "ExpYears", "TotalMHz")
			SummaryFlag = True #for ploting the summary chart
			

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

			fig = go.Figure(data=data)

			currency_flag = True #default

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

		SubFeature = st.sidebar.selectbox('Select a Sub Feature', ["Auction Price","Reserve Price","Quantum Offered", "Quantum Sold", 
			"Percent Sold", "Quantum Unsold", "Percent Unsold"])

		typedf = type_dict[SubFeature].copy()

		hovertext = htext_auctionmap(dff)

		if SubFeature in ["Auction Price", "Reserve Price"]:
			curr_list=[]
			for col in typedf.columns:
				curr_list.append(auction_rsrate_dict[int(col)])


			radio_currency = st.sidebar.radio('Click Currency', ["Rupees", "US Dollars"])
			if radio_currency == "Rupees":
				z = typedf.values
				currency_flag = True #Rupees
			if radio_currency == "US Dollars":
				lst1=[]
				for line in typedf.values:
					lst2=[]
					for i, val in enumerate(line):
						if str(val).strip() == "NA":
							lst2.append(val)
						else:
							lst2.append(round(val/[curr_list[i]][0]*10,2))
					lst1.append(lst2)
				z = pd.DataFrame(lst1).values
				currency_flag = False #USD

		else:
			z = typedf.values
			currency_flag = True #Default
					
		#preparing the dataframe of the summary bar chart on top of the heatmap
		if SubFeature not in ["Percent Sold", "Percent Unsold"]:
			summarydf = typedf.replace('[a-zA-Z]+\s*',np.nan, regex=True)
			summarydf = summarydf.sum().reset_index()
			summarydf.columns = ["Years", "India Total"]

		if SubFeature in ["Auction Price", "Reserve Price"]:

			if radio_currency == "Rupees":
				summarydf = summarydf
			if radio_currency == "US Dollars":
				summarydf["India Total"] = np.around(summarydf["India Total"].values/curr_list,1)

		#preparing the summary chart 
			chart = summarychart(summarydf, "Years", "India Total")
			SummaryFlag = True
		else:
			pass
		
		#setting the data of the heatmap 
		data = [go.Heatmap(
			z = z,
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
		fig = go.Figure(data=data)

		hoverlabel_bgcolor = transform_colscale_for_hbox_auction_map(dff,reserveprice,auctionprice)


#----------------New Auction Bid Data Code Starts Here------------------
#function used to calculate the total bid values 

def bidvalue(df,dfblocks):

	df = df.replace(np.nan, 0)
	min_values=[]
	for col in df.columns:
		lst =[]
		if sum(list(df[col])) > 0:
			for value in list(df[col]):
				if value != 0:
					lst.append(value)
			min_values.append(min(lst))
		if sum(list(df[col])) == 0:
			min_values.append(np.nan)

	mindf = pd.DataFrame(min_values).T

	mindf.columns = df.columns

	df_final = dfblocks*mindf.values #calculating the total value of bids

	# # Regex pattern to match floating-point numbers
	# pattern = re.compile(r'^[+-]?((?=.*[1-9])\d*\.\d+|0\.\d*[1-9]\d*)$')

	# # Function to replace floating-point numbers with 1
	# replace_func = lambda x: 1 if re.match(pattern, str(x)) else x

	# # Apply the function to each cell in the DataFrame
	# matrix = df.applymap(replace_func)

	# df_final = matrix * mindf.values

	df_final = df_final.sum(axis =1).round(1)

	return df_final


def plotbiddertotal(dftemp,dfblocksalloc_rdend):

	dftemp = round(dftemp,1)
					
	panindiabids = bidvalue(dftemp,dfblocksalloc_rdend).sort_index().reset_index()

	panindiabids.columns =["Bidder","PanIndiaBid"]

	panindiabids = panindiabids.round(0)

	panindiabids = panindiabids.sort_values("Bidder", ascending=False)

	fig = px.bar(panindiabids, y = 'Bidder', x='PanIndiaBid', orientation ='h', height = 625)

	fig.update_layout(xaxis=dict(title='Total Value'), yaxis=dict(title=''))

	fig.update_traces(text=panindiabids['PanIndiaBid'], textposition='auto')

	fig.update_xaxes(tickvals=[])

	fig.update_layout(xaxis=dict(side='top', title_standoff=0, ticklen=0, title_font=dict(size=14)))

	fig.update_layout(xaxis_title_standoff=5)

	fig.update_traces(marker=dict(color='red'))


	return fig


def plotrwototal(sumrows, ydim, xdim):
					

	fig = px.bar(sumrows, y = ydim, x=xdim, orientation ='h', height = 625)

	fig.update_layout(xaxis=dict(title='India Total'), yaxis=dict(title=''))

	fig.update_traces(text=sumrows[xdim], textposition='auto')

	fig.update_xaxes(tickvals=[])

	fig.update_layout(xaxis=dict(side='top', title_standoff=0, ticklen=0, title_font=dict(size=14)))

	fig.update_layout(xaxis_title_standoff=5)

	fig.update_traces(marker=dict(color='red'))


	return fig



def plotlosttotal(df,ydim,xdim):

	fig = px.bar(df, y =ydim, x=xdim, orientation ='h', height = 615)

	fig.update_layout(xaxis=dict(title="Total"), yaxis=dict(title=''))

	fig.update_traces(text=df[xdim], textposition='auto')

	fig.update_xaxes(tickvals=[])

	fig.update_layout(xaxis=dict(side='top', title_standoff=0, ticklen=0, title_font=dict(size=14)))

	fig.update_layout(xaxis_title_standoff=5)

	fig.update_traces(marker=dict(color='red'))

	return fig




if selected_dimension == "Auction Data":

	currency_flag = True #default 

	Feature = st.sidebar.selectbox("Select a Feature", ["2010-Band2100","2010-Band2300", "2012-Band1800","2014-Band1800","2014-Band900",
									"2015-Band800", "2015-Band900","2015-Band1800", "2015-Band2100", "2016-Band800","2016-Band1800",
									"2016-Band2100", "2016-Band2300", "2016-Band2500","2021-Band800","2021-Band900","2021-Band1800",
									"2021-Band2100","2021-Band2300","2022-Band700","2022-Band800","2022-Band900","2022-Band1800",
									"2022-Band2100","2022-Band2500","2022-Band3500","2022-Band26000"])

	
	


	if Feature == "2022-Band26000":

		totalrounds = 40
		mainsheet = "2022_5G_26000"
		mainsheetoriginal = "2022_5G_26000_Original"
		mainoriflag = True
		activitysheet = "2022_4G_5G_Activity"
		demandsheet = "2022_5G_26000_AD"
		titlesubpart = "26000 MHz Auctions (CY-2022)"
		subtitlesubpartbidactivity = "; Combined for All Bands"
		year =2022
		band = 26000
		xdtick =1
		zmin=1
		zmax=5
		zmin_af = 0.5
		zmax_af = 1
		texttempbiddemandactivity = ""
		blocksize = 50
		zmin_blk_sec = 0
		zmax_blk_sec = 4


	if Feature == "2022-Band3500":

		totalrounds = 40
		mainsheet = "2022_5G_3500"
		mainsheetoriginal = "2022_5G_3500_Original"
		mainoriflag = True
		activitysheet = "2022_4G_5G_Activity"
		demandsheet = "2022_5G_3500_AD"
		titlesubpart = "3500 MHz Auctions (CY-2022)"
		subtitlesubpartbidactivity = "; Combined for All Bands"
		year =2022
		band = 3500
		xdtick =1
		zmin=1
		zmax=5
		zmin_af = 0.5
		zmax_af = 1
		texttempbiddemandactivity = ""
		blocksize = 10
		zmin_blk_sec = 0
		zmax_blk_sec = 4


	if Feature == "2022-Band2500":

		totalrounds = 40
		mainsheet = "2022_4G_2500"
		mainsheetoriginal = "2022_4G_2500_Original"
		mainoriflag = True
		activitysheet = "2022_4G_5G_Activity"
		demandsheet = "2022_4G_2500_AD"
		titlesubpart = "2500 MHz Auctions (CY-2022)"
		subtitlesubpartbidactivity = "; Combined for All Bands"
		year =2022
		band = 2500
		xdtick =1
		zmin=1
		zmax=5
		zmin_af = 0.5
		zmax_af = 1
		texttempbiddemandactivity = ""
		blocksize = 10
		zmin_blk_sec = 0
		zmax_blk_sec = 4


	if Feature == "2022-Band2100":

		totalrounds = 40
		mainsheet = "2022_4G_2100"
		mainsheetoriginal = "2022_4G_2100_Original"
		mainoriflag = True
		activitysheet = "2022_4G_5G_Activity"
		demandsheet = "2022_4G_2100_AD"
		titlesubpart = "2100 MHz Auctions (CY-2022)"
		subtitlesubpartbidactivity = "; Combined for All Bands"
		year =2022
		band = 2100
		xdtick =1
		zmin=1
		zmax=5
		zmin_af = 0.5
		zmax_af = 1
		texttempbiddemandactivity = ""
		blocksize = 5
		zmin_blk_sec = 0
		zmax_blk_sec = 4



	if Feature == "2022-Band1800":

		totalrounds = 40
		mainsheet = "2022_4G_1800"
		mainsheetoriginal = "2022_4G_1800_Original"
		mainoriflag = True
		activitysheet = "2022_4G_5G_Activity"
		demandsheet = "2022_4G_1800_AD"
		titlesubpart = "1800 MHz Auctions (CY-2022)"
		subtitlesubpartbidactivity = "; Combined for All Bands"
		year =2022
		band = 1800
		xdtick =1
		zmin=1
		zmax=5
		zmin_af = 0.5
		zmax_af = 1
		texttempbiddemandactivity = "%{z}" #debug 2024
		blocksize = 0.2
		zmin_blk_sec = 0
		zmax_blk_sec = 4


	if Feature == "2022-Band900":

		totalrounds = 40
		mainsheet = "2022_4G_900"
		mainsheetoriginal = "2022_4G_900_Original"
		mainoriflag = True
		activitysheet = "2022_4G_5G_Activity"
		demandsheet = "2022_4G_900_AD"
		titlesubpart = "900 MHz Auctions (CY-2022)"
		subtitlesubpartbidactivity = "; Combined for All Bands"
		year =2022
		band = 900
		xdtick =1
		zmin=1
		zmax=5
		zmin_af = 0.5
		zmax_af = 1
		texttempbiddemandactivity = ""
		blocksize = 0.2
		zmin_blk_sec = 0
		zmax_blk_sec = 4


	if Feature == "2022-Band800":

		totalrounds = 40
		mainsheet = "2022_4G_800"
		mainsheetoriginal = "2022_4G_800_Original"
		mainoriflag = True
		activitysheet = "2022_4G_5G_Activity"
		demandsheet = "2022_4G_800_AD"
		titlesubpart = "800 MHz Auctions (CY-2022)"
		subtitlesubpartbidactivity = "; Combined for All Bands"
		year =2022
		band = 800
		xdtick =1
		zmin=1
		zmax=5
		zmin_af = 0.5
		zmax_af = 1
		texttempbiddemandactivity = ""
		blocksize = 1.25
		zmin_blk_sec = 0
		zmax_blk_sec = 4


	if Feature == "2022-Band700":

		totalrounds = 40
		mainsheet = "2022_5G_700"
		mainsheetoriginal = "2022_5G_700_Original"
		mainoriflag = True
		activitysheet = "2022_4G_5G_Activity"
		demandsheet = "2022_5G_700_AD"
		titlesubpart = "700 MHz Auctions (CY-2022)"
		subtitlesubpartbidactivity = "; Combined for All Bands"
		year =2022
		band = 700
		xdtick =1
		zmin=1
		zmax=5
		zmin_af = 0.5
		zmax_af = 1
		texttempbiddemandactivity = ""
		blocksize = 5
		zmin_blk_sec = 0
		zmax_blk_sec = 4
	
	if Feature == "2021-Band800":

		totalrounds = 6
		mainsheet = "2021_4G_800"
		mainsheetoriginal = "2021_4G_800_Original"
		mainoriflag = True
		activitysheet = "2021_4G_Activity"
		demandsheet = "2021_4G_800_AD"
		titlesubpart = "800 MHz Auctions (CY-2021)"
		subtitlesubpartbidactivity = "; Combined for All Bands"
		year =2021
		band = 800
		xdtick =1
		zmin=1
		zmax=5
		zmin_af = 0.5
		zmax_af = 1
		texttempbiddemandactivity = "%{z}"
		blocksize = 1.25
		zmin_blk_sec = 0
		zmax_blk_sec = 4

	if Feature == "2021-Band900":

		totalrounds = 6
		mainsheet = "2021_4G_900"
		mainsheetoriginal = "2021_4G_900_Original"
		mainoriflag = True
		activitysheet = "2021_4G_Activity"
		demandsheet = "2021_4G_900_AD"
		titlesubpart = "900 MHz Auctions (CY-2021)"
		subtitlesubpartbidactivity = "; Combined for All Bands"
		year =2021
		band = 900
		xdtick =1
		zmin=1
		zmax=5
		zmin_af = 0.5
		zmax_af = 1
		texttempbiddemandactivity = "%{z}"
		blocksize = 0.2
		zmin_blk_sec = 0
		zmax_blk_sec = 4

	if Feature == "2021-Band1800":

		totalrounds = 6
		mainsheet = "2021_4G_1800"
		mainsheetoriginal = "2021_4G_1800_Original"
		mainoriflag = True
		activitysheet = "2021_4G_Activity"
		demandsheet = "2021_4G_1800_AD"
		titlesubpart = "1800 MHz Auctions (CY-2021)"
		subtitlesubpartbidactivity = "; Combined for All Bands"
		year =2021
		band = 1800
		xdtick =1
		zmin=1
		zmax=5
		zmin_af = 0.5
		zmax_af = 1
		texttempbiddemandactivity = "%{z}"
		blocksize = 0.2
		zmin_blk_sec = 0
		zmax_blk_sec = 4

	if Feature == "2021-Band2100":

		totalrounds = 6
		mainsheet = "2021_4G_2100"
		mainsheetoriginal = "2021_4G_2100_Original"
		mainoriflag = True
		activitysheet = "2021_4G_Activity"
		demandsheet = "2021_4G_2100_AD"
		titlesubpart = "2100 MHz Auctions (CY-2021)"
		subtitlesubpartbidactivity = "; Combined for All Bands"
		year =2021
		band = 2100
		xdtick =1
		zmin=1
		zmax=5
		zmin_af = 0.5
		zmax_af = 1
		texttempbiddemandactivity = "%{z}"
		blocksize = 5
		zmin_blk_sec = 0
		zmax_blk_sec = 4

	if Feature == "2021-Band2300":

		totalrounds = 6
		mainsheet = "2021_4G_2300"
		mainsheetoriginal = "2021_4G_2300_Original"
		mainoriflag = True
		activitysheet = "2021_4G_Activity"
		demandsheet = "2021_4G_2300_AD"
		titlesubpart = "2300 MHz Auctions (CY-2021)"
		subtitlesubpartbidactivity = "; Combined for All Bands"
		year =2021
		band = 2300
		xdtick =1
		zmin=1
		zmax=5
		zmin_af = 0.5
		zmax_af = 1
		texttempbiddemandactivity = "%{z}"
		blocksize = 10
		zmin_blk_sec = 0
		zmax_blk_sec = 4




	if Feature == "2016-Band2500":

		totalrounds = 31
		mainsheet = "2016_4G_2500"
		mainsheetoriginal = "2016_4G_2500_Original"
		mainoriflag = True
		activitysheet = "2016_4G_Activity"
		demandsheet = "2016_4G_2500_AD"
		titlesubpart = "2500 MHz Auctions (CY-2016)"
		subtitlesubpartbidactivity = "; Combined for All Bands"
		year =2016
		band = 2500
		xdtick =5
		zmin=1
		zmax=5
		zmin_af = 0.5
		zmax_af = 1
		texttempbiddemandactivity = "%{z}"
		blocksize = 10
		zmin_blk_sec = 0
		zmax_blk_sec = 4


	if Feature == "2016-Band2300":

		totalrounds = 31
		mainsheet = "2016_4G_2300"
		mainsheetoriginal = "2016_4G_2300_Original"
		mainoriflag = True
		activitysheet = "2016_4G_Activity"
		demandsheet = "2016_4G_2300_AD"
		titlesubpart = "2300 MHz Auctions (CY-2016)"
		subtitlesubpartbidactivity = "; Combined for All Bands"
		year =2016
		band = 2300
		xdtick =5
		zmin=1
		zmax=5
		zmin_af = 0.5
		zmax_af = 1
		texttempbiddemandactivity = "%{z}"
		blocksize = 10
		zmin_blk_sec = 0
		zmax_blk_sec = 4

	if Feature == "2016-Band2100":

		totalrounds = 31
		mainsheet = "2016_4G_2100"
		mainsheetoriginal = "2016_4G_2100_Original"
		mainoriflag = True
		activitysheet = "2016_4G_Activity"
		demandsheet = "2016_4G_2100_AD"
		titlesubpart = "2100 MHz Auctions (CY-2016)"
		subtitlesubpartbidactivity = "; Combined for All Bands"
		year =2016
		band = 2100
		xdtick =5
		zmin=1
		zmax=5
		zmin_af = 0.5
		zmax_af = 1
		texttempbiddemandactivity = "%{z}"
		blocksize = 5
		zmin_blk_sec = 0
		zmax_blk_sec = 4
	
	if Feature == "2016-Band1800":

		totalrounds = 31
		mainsheet = "2016_4G_1800"
		mainsheetoriginal = "2016_4G_1800_Original"
		mainoriflag = True
		activitysheet = "2016_4G_Activity"
		demandsheet = "2016_4G_1800_AD"
		titlesubpart = "1800 MHz Auctions (CY-2016)"
		subtitlesubpartbidactivity = "; Combined for All Bands"
		year =2016
		band = 1800
		xdtick =5
		zmin=1
		zmax=5
		zmin_af = 0.5
		zmax_af = 1
		texttempbiddemandactivity = "%{z}"
		blocksize = 0.2
		zmin_blk_sec = 0
		zmax_blk_sec = 4


	if Feature == "2016-Band800":

		totalrounds = 31
		mainsheet = "2016_4G_800"
		mainsheetoriginal = "2016_4G_800_Original"
		mainoriflag = True
		activitysheet = "2016_4G_Activity"
		demandsheet = "2016_4G_800_AD"
		titlesubpart = "800 MHz Auctions (CY-2016)"
		subtitlesubpartbidactivity = "; Combined for All Bands"
		year =2016
		band = 800
		xdtick =5
		zmin=1
		zmax=5
		zmin_af = 0.5
		zmax_af = 1
		texttempbiddemandactivity = "%{z}"
		blocksize = 1.25
		zmin_blk_sec = 0
		zmax_blk_sec = 4


	if Feature == "2015-Band2100":

		totalrounds = 115
		mainsheet = "2015_3G_2100"
		mainsheetoriginal = "2015_3G_2100_Original"
		mainoriflag = True
		activitysheet = "2015_2G_3G_Activity"
		demandsheet = "2015_3G_2100_AD"
		titlesubpart = "2100 MHz Auctions (CY-2015)"
		subtitlesubpartbidactivity = "; Combined for All Bands"
		year =2015
		band = 2100
		xdtick =5
		zmin=1
		zmax=5
		zmin_af = 0.5
		zmax_af = 1
		texttempbiddemandactivity = ""
		blocksize = 5
		zmin_blk_sec = 0
		zmax_blk_sec = 4


	if Feature == "2015-Band1800":

		totalrounds = 115
		mainsheet = "2015_2G_1800"
		mainsheetoriginal = "2015_2G_1800_Original"
		mainoriflag = True
		activitysheet = "2015_2G_3G_Activity"
		demandsheet = "2015_2G_1800_AD"
		titlesubpart = "1800 MHz Auctions (CY-2015)"
		subtitlesubpartbidactivity = "; Combined for All Bands"
		year =2015
		band = 1800
		xdtick =5
		zmin=1
		zmax=5
		zmin_af = 0.5
		zmax_af = 1
		texttempbiddemandactivity = ""
		blocksize = 0.2
		zmin_blk_sec = 0
		zmax_blk_sec = 4


	if Feature == "2015-Band900":

		totalrounds = 115
		mainsheet = "2015_2G_900"
		mainsheetoriginal = "2015_2G_900_Original"
		mainoriflag = True
		activitysheet = "2015_2G_3G_Activity"
		demandsheet = "2015_2G_900_AD"
		titlesubpart = "900 MHz Auctions (CY-2015)"
		subtitlesubpartbidactivity = "; Combined for All Bands"
		year =2015
		band = 900
		xdtick =5
		zmin=1
		zmax=5
		zmin_af = 0.5
		zmax_af = 1
		texttempbiddemandactivity = ""
		blocksize = 0.2
		zmin_blk_sec = 0
		zmax_blk_sec = 4


	if Feature == "2015-Band800":

		totalrounds = 115
		mainsheet = "2015_2G_800"
		mainsheetoriginal = "2015_2G_800_Original"
		mainoriflag = True
		activitysheet = "2015_2G_3G_Activity"
		demandsheet = "2015_2G_800_AD"
		titlesubpart = "800 MHz Auctions (CY-2015)"
		subtitlesubpartbidactivity = "; Combined for All Bands"
		year =2015
		band = 800
		xdtick =5
		zmin=1
		zmax=5
		zmin_af = 0.5
		zmax_af = 1
		texttempbiddemandactivity = ""
		blocksize = 1.25
		zmin_blk_sec = 0
		zmax_blk_sec = 4


	if Feature == "2014-Band1800":

		totalrounds = 68
		mainsheet = "2014_2G_1800"
		mainsheetoriginal = "2014_2G_1800_Original"
		mainoriflag = True
		activitysheet = "2014_2G_Activity"
		demandsheet = "2014_2G_1800_AD"
		titlesubpart = "1800 MHz Auctions (CY-2014)"
		subtitlesubpartbidactivity = "; Combined for both 1800 & 900 MHz Bands"
		year = 2014
		band = 1800
		xdtick =5
		zmin=1
		zmax=5
		zmin_af = 0.5
		zmax_af = 1
		texttempbiddemandactivity = ""
		blocksize = 0.2
		zmin_blk_sec = 0
		zmax_blk_sec = 4

	if Feature == "2014-Band900":

		totalrounds = 68
		mainsheet = "2014_2G_900"
		mainsheetoriginal = "2014_2G_900_Original"
		mainoriflag = True
		activitysheet = "2014_2G_Activity"
		demandsheet = "2014_2G_900_AD"
		titlesubpart = "900 MHz Auctions (CY-2014)"
		subtitlesubpartbidactivity = "; Combined for both 1800 & 900 MHz Bands"
		year = 2014
		band = 900
		xdtick =5
		zmin=1
		zmax=5
		zmin_af = 0.5
		zmax_af = 1
		texttempbiddemandactivity = ""
		blocksize = 1
		zmin_blk_sec = 0
		zmax_blk_sec = 4


	if Feature == "2010-Band2100":

		totalrounds = 183
		mainsheet = "2010_3G"
		mainoriflag = False
		activitysheet = "2010_3G_Activity"
		demandsheet = "2010_3G_AD"
		titlesubpart = "2100 MHz Auctions (CY-2010)"
		subtitlesubpartbidactivity = ""
		year = 2010
		band = 2100
		xdtick =10
		zmin=1 
		zmax=5
		zmin_af = 0.5
		zmax_af = 1
		texttempbiddemandactivity = ""
		blocksize = 5
		zmin_blk_sec = 0
		zmax_blk_sec = 1

	if Feature == "2010-Band2300":

		totalrounds = 117
		mainsheet = "2010_BWA"
		mainoriflag = False
		activitysheet = "2010_BWA_Activity"
		demandsheet = "2010_BWA_AD"
		titlesubpart = "2300 MHz Auctions (CY-2010)"
		subtitlesubpartbidactivity=""
		year = 2010
		band = 2300
		xdtick =10
		zmin=1
		zmax=3
		zmin_af = 0.5
		zmax_af = 1
		texttempbiddemandactivity = ""
		blocksize = 20
		zmin_blk_sec = 0
		zmax_blk_sec = 1

	if Feature == "2012-Band1800":

		totalrounds = 14
		mainsheet = "2012_2G_1800"
		mainoriflag = False
		activitysheet = "2012_2G_1800_Activity"
		demandsheet = "2012_2G_1800_AD"
		titlesubpart = "1800 MHz Auctions (CY-2012)"
		subtitlesubpartbidactivity = ""
		year = 2012
		band = 1800
		xdtick =1
		zmin=1
		zmax=3
		zmin_af = 0.5
		zmax_af = 1
		texttempbiddemandactivity = "%{z}"
		blocksize = 1.25
		zmin_blk_sec = 0
		zmax_blk_sec = 4
	


	if mainoriflag == True:

		#filtering the reserve price for the auction year

		dfprallauctions = loadauctionbiddata()["Reserve_Prices_All_Auctions"]

		filt = (dfprallauctions["Band"]==band) & (dfprallauctions["Auction Year"]==year)

		dfrp = dfprallauctions[filt]

		dfrp = dfrp.drop(columns =["Auction Year","Band"])

		dfrp.columns = ["LSA", "ReservePricePerBLK"]

		dfrp = dfrp.set_index("LSA").sort_index(ascending = True)


		dfbidori = loadauctionbiddata()[mainsheetoriginal].replace('-', np.nan, regex = True)


		dfbidori.columns = ["Clk_Round", "Bidder","LSA","Last_Sub_Bid_Start_CLKRd", "Rank_Start_ClkRd", 
					"Possible_Raise_Bid_ClkRd", "Bid_Decision", "Last_Sub_Bid_End_CLKRd", "Rank_End_ClkRd", 
					"No_of_BLK_Selected", "Prov_Alloc_BLK_Start_ClkRd", "Prov_Alloc_BLK_End_ClkRd", "Prov_Win_Price_End_ClkRd"]

		dfbidori = dfbidori.drop(columns = ["Possible_Raise_Bid_ClkRd"])

		dfbidori = dfbidori.replace("No Bid", 0)
		dfbidori = dfbidori.replace("Bid",1)

		listofbidders = sorted(list(set(dfbidori["Bidder"])))

		listofcircles = sorted(list(set(dfbidori["LSA"])))

		dfbidori = dfbidori.set_index("LSA").sort_index(ascending = False)


	#filtering the reserve price for the auction year

	dfprallauctions = loadauctionbiddata()["Reserve_Prices_All_Auctions"]

	filt = (dfprallauctions["Band"]==band) & (dfprallauctions["Auction Year"]==year)

	dfrp = dfprallauctions[filt]

	dfrp = dfrp.drop(columns =["Auction Year","Band"])

	dfrp.columns = ["LSA", "ReservePricePerBLK"]

	dfrp = dfrp.set_index("LSA").sort_index(ascending = True)

	dfbid = loadauctionbiddata()[mainsheet].replace('-', np.nan, regex = True)

	dfbid.columns = ["Clk_Round", "Bidder","LSA","PWB_Start_ClkRd", "Rank_PWB_Start_ClkRd", 
					"Possible_Raise_Bid_ClkRd", "Bid_Decision", "PWB_End_ClkRd", "Rank_PWB_End_ClkRd", 
					"No_of_BLK_Selected", "Prov_Alloc_BLK_Start_ClkRd", "Prov_Alloc_BLK_End_ClkRd", "Prov_Win_Price_End_ClkRd"]

	dfbid = dfbid.replace("No Bid", 0)
	dfbid = dfbid.replace("Bid",1)

	listofbidders = sorted(list(set(dfbid["Bidder"])))

	listofcircles = sorted(list(set(dfbid["LSA"])))

	dfbid = dfbid.set_index("LSA").sort_index(ascending = False)

	if mainoriflag == True:

		SubFeature = st.sidebar.selectbox("Select a SubFeature", ["BidsCircleWise","RanksCircleWise", "ProvWinningBid", "BlocksSelected",
									  "BlocksAllocated","BiddingActivity", "DemandActivity","LastBidPrice"])

	if mainoriflag == False:

		SubFeature = st.sidebar.selectbox("Select a SubFeature", ["BidsCircleWise","RanksCircleWise", "ProvWinningBid", "BlocksSelected",
									  "BlocksAllocated","BiddingActivity", "DemandActivity"])

	if SubFeature == "BidsCircleWise":

		round_range = st.slider("Select Auction Round Numbers using the Silder below", min_value=1, max_value = totalrounds, value=(1,totalrounds))

		start_round = round_range[0]

		end_round = round_range[1]

		dfbidcirclwise = dfbid.copy()

		dfbidcirclwise_endrd = dfbidcirclwise[dfbidcirclwise["Clk_Round"]==end_round].reset_index()

		dfprovallcblks_endrd = dfbidcirclwise_endrd.pivot(index="Bidder", columns='LSA', values="Prov_Alloc_BLK_End_ClkRd")


		#filter data within the block of selected rounds 

		filt  =(dfbidcirclwise["Clk_Round"] >= start_round) & (dfbidcirclwise["Clk_Round"] <= end_round)

		dfbidcirclwise = dfbidcirclwise[filt]


		dftemp = dfbidcirclwise.drop(columns=["Possible_Raise_Bid_ClkRd", "Rank_PWB_Start_ClkRd", "Rank_PWB_End_ClkRd",
									"PWB_End_ClkRd","Clk_Round", "PWB_Start_ClkRd"], axis=1)

		dftemp = dftemp.groupby(["LSA", "Bidder"]).sum().reset_index()

		summarydf = dftemp.groupby(["LSA"]).sum().reset_index().drop(columns = "Bidder", axis =1)

		summarydf = summarydf.set_index("LSA")

		dftemp = dftemp.set_index("LSA")

		dftemp["Bid_Decision_Perc"] = round((dftemp["Bid_Decision"]/summarydf["Bid_Decision"])*100,1)

		dftemp = dftemp.reset_index()

		# #sort by LSA

		circle_list=[]

		for circle in listofcircles: #this extracts the full name of the circle from the code
			circle_list.append(state_dict[circle])


		# sortbylsa = st.sidebar.selectbox("Select a Circle to Sort", state_dict.values())

		sortbylsa = st.sidebar.selectbox("Select a Circle to Sort", circle_list)

		selected_lsa = [k for k, v in state_dict.items() if v == sortbylsa]

		# dftempheat = dftempheat.sort_values(selected_lsa[0], ascending = True)

		#processing hovertext and colormatrix
		hovertext,colormatrix,resultdf = htext_colormatrix_auctiondata_2010_3G_BWA_BidsCircleWise(dfbidcirclwise, 
											dftemp,selected_lsa[0],start_round,end_round,dfprovallcblks_endrd)
		hoverlabel_bgcolor = colormatrix



		radio_selection = st.sidebar.radio('Click an Option', ["Absolute Values", "Percentage of Total", "Provisional Winners"])

		if radio_selection == "Absolute Values":

			dftempheat = dftemp.pivot(index="Bidder", columns='LSA', values="Bid_Decision")

			dftempheat = dftempheat.sort_values(selected_lsa[0], ascending = True)

			summarydf = dftempheat.sum(axis=0).reset_index()

			summarydf.columns = ["LSA","TotalBids"]

			#preparing the summary chart 
			chart = summarychart(summarydf, 'LSA', "TotalBids")
			SummaryFlag = True

			titlesubpart2 = " - Total Agg Bids (Within Selected Rounds)"


			#--------New Code Starts------------#

			#function to combine text from two dataframe 

			def combine_text(x, y): #sep is seperator
			    if x.notnull().all() and y.notnull().all():
			        return x + '<br>' + y
			    elif x.notnull().all():
			        return x
			    else:
			        return y

			#for rendering text of the final heatmap for Data

			dftempheat = dftempheat.applymap(int)

			df_combined = dftempheat.applymap(str).combine(resultdf.applymap(str), lambda x, y: combine_text(x, y))


			#------New Code Ends----------------#


			data = [go.Heatmap(
				z=dftempheat.values,
		        x=dftempheat.columns,
		        y=dftempheat.index,
				xgap = 1,
				ygap = 1,
				hoverinfo ='text',
				hovertext = hovertext,
				text = df_combined.values,
				colorscale="Hot",
				texttemplate="%{text}",
				textfont={"size":12},
				reversescale=True,
				),
				]

		if radio_selection == "Percentage of Total":

			dftempheatSum = dftemp.pivot(index="Bidder", columns='LSA', values="Bid_Decision") #for summary chrat below

			dftempheat = dftemp.pivot(index="Bidder", columns='LSA', values="Bid_Decision_Perc") #for heatmap

			dftempheat = dftempheat.sort_values(selected_lsa[0], ascending = True)

			summarydf = dftempheatSum.sum(axis=0).reset_index()

			summarydf.columns = ["LSA","TotalBids"]

			#preparing the summary chart 
			chart = summarychart(summarydf, 'LSA', "TotalBids")
			SummaryFlag = True

			titlesubpart2 = " - % of Total Agg Bids (Within Selected Rounds)"


			#--------New Code Starts------------#

			#function to combine text from two dataframe 

			def combine_text(x,sym, y): #sym is for %
			    if x.notnull().all() and y.notnull().all():
			        return x + sym + '<br>' + y
			    elif x.notnull().all():
			        return x
			    else:
			        return y

			#for rendering text of the final heatmap for Data


			df_combined = dftempheat.applymap(str).combine(resultdf.applymap(str), lambda x, y: combine_text(x," %", y))


			#------New Code Ends----------------#

			data = [go.Heatmap(
				z=dftempheat.values,
		        x=dftempheat.columns,
		        y=dftempheat.index,
				xgap = 1,
				ygap = 1,
				hoverinfo ='text',
				hovertext = hovertext,
				text = df_combined.values,
				colorscale="Hot",
					texttemplate="%{text}",
					textfont={"size":12},
					reversescale=True,
					),
				]

		if radio_selection == "Provisional Winners":

			dftempheatSum = dftemp.pivot(index="Bidder", columns='LSA', values="Bid_Decision") #for summary chrat below

			summarydf = dftempheatSum.sum(axis=0).reset_index()

			summarydf.columns = ["LSA","TotalBids"]

			#preparing the summary chart 
			chart = summarychart(summarydf, 'LSA', "TotalBids")
			SummaryFlag = True


			resultdfheat = resultdf.replace("WON",1)

			resultdfheat = resultdfheat.replace("LOST",0)


			titlesubpart2 = " - Provisional Winners (End of Selected Rounds)"

			data = [go.Heatmap(
				z=resultdfheat.values,
		        x=resultdfheat.columns,
		        y=resultdfheat.index,
				xgap = 1,
				ygap = 1,
				hoverinfo ='text',
				text = resultdf.values,
				colorscale="Picnic",
					texttemplate="%{text}",
					textfont={"size":10},
					reversescale=True,
					),
				]
		
		#Ploting the heatmap for all the above three options

		figauc = go.Figure(data=data)

		figauc.update_layout(
		    template="seaborn",
		    xaxis_side= 'top',
		   	height = 650,
		   	yaxis=dict(
	        tickmode='array',
	        showgrid=False,
	        	))

		title = titlesubpart+titlesubpart2
		subtitle = "Source - DoT; Between Round Nos "+str(start_round)+" & "+str(end_round)+ "; Number of Rounds = "+ str(end_round-start_round)

		style = "<style>h3 {text-align: left;}</style>"
		with st.container():
			#plotting the main chart
			st.markdown(style, unsafe_allow_html=True)
			st.header(title)
			st.markdown(subtitle)

		#Drawning a black border around the heatmap chart 
		figauc.update_xaxes(fixedrange=True,showline=True,linewidth=1.2,linecolor='black', mirror=True)
		figauc.update_yaxes(fixedrange=True,showline=True, linewidth=1.2, linecolor='black', mirror=True)

		figauc.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))

		st.plotly_chart(figauc, use_container_width=True)

		#plotting the final summary chart 
		col1,col2,col3 = st.columns([0.2,14,1]) #create collumns of uneven width
		if SummaryFlag ==True:
			# st.altair_chart(chart, use_container_width=True)
			col2.altair_chart(chart, use_container_width=True)

	if SubFeature == "RanksCircleWise":

		plottype = st.sidebar.selectbox("Select a Plot Type", ["RanksInRound", "RanksInRounds"])

		if plottype == "RanksInRound":

			round_number = st.slider("Select Auction Round Numbers using the Silder below", min_value=1, max_value=totalrounds, step=1, value = totalrounds)

			dfbidspec = dfbid.copy()

			filt  =(dfbidspec["Clk_Round"] == round_number) 

			dfbidspec = dfbidspec[filt]

			dftemp = dfbidspec.drop(columns=["PWB_Start_ClkRd","Possible_Raise_Bid_ClkRd", "Rank_PWB_Start_ClkRd","Bid_Decision","PWB_End_ClkRd"], axis=1).reset_index()

			dftemp = dftemp.groupby(["LSA", "Bidder", "Rank_PWB_End_ClkRd"]).sum().reset_index()

			dftempheat = dftemp.pivot(index="Bidder", columns='LSA', values="Rank_PWB_End_ClkRd")

			#sort by LSA 

			circle_list=[]

			for circle in listofcircles: #this extracts the full name of the circle from the code
				circle_list.append(state_dict[circle])

			# sortbylsa = st.sidebar.selectbox("Select a Circle to Sort", state_dict.values())

			sortbylsa = st.sidebar.selectbox("Select a Circle to Sort", circle_list)

			selected_lsa = [k for k, v in state_dict.items() if v == sortbylsa]

			# Custom sorting key function
			def sort_key(value):
			    if pd.isnull(value) or value == 0:
			        return float('inf')  # Assigning a very large value for zeros and np.nan
			    else:
			        return value

			dftempheat = dftempheat.sort_values(selected_lsa[0], key=lambda x: x.map(sort_key), ascending = False)

			dftempheat = dftempheat.replace(0,np.nan)


			data = [go.Heatmap(
				z=dftempheat.values,
		        y= dftempheat.index,
		        x=dftempheat.columns,
				xgap = 1,
				ygap = 1,
				hoverinfo ='text',
				zmin=zmin, zmax=zmax,
				# text = hovertext,
				colorscale='Hot',
				# showscale=False,
					texttemplate="%{z}", 
					textfont={"size":10},
					# reversescale=True,
					)]
				
			figauc = go.Figure(data=data)

			figauc.update_layout(
			    template="seaborn",
			    xaxis_side= 'top',
			   	height = 650,
			   	yaxis=dict(
		        tickmode='array',
		        	))

			title = titlesubpart+" - Bidder's Rank at the End of CLK Round No - "+str(round_number)
			subtitle = "Unit - RankNo; Higher the Rank More Aggressive is the Bidding; Sorted by Circle -"+selected_lsa[0]+" ; Source - DoT"

			style = "<style>h3 {text-align: left;}</style>"
			with st.container():
				#plotting the main chart
				st.markdown(style, unsafe_allow_html=True)
				st.header(title)
				st.markdown(subtitle)


			#Drawning a black border around the heatmap chart 
			figauc.update_xaxes(fixedrange=True,showline=True,linewidth=1.2,linecolor='black', mirror=True)
			figauc.update_yaxes(fixedrange=True,showline=True, linewidth=1.2, linecolor='black', mirror=True)

			figauc.update_layout(
				    xaxis=dict(showgrid=False),
				    yaxis=dict(showgrid=False)
				)

			st.plotly_chart(figauc, use_container_width=True)


		if plottype == "RanksInRounds":

			round_range = st.slider("Select Auction Round Numbers using the Silder below", min_value = 0, max_value = totalrounds, value=(1,totalrounds))

			start_round = round_range[0]

			end_round = round_range[1]

			dfbidsel = dfbid.copy()

			st.write("") #This is very important as this statement triggers section to become active

			filt  =(dfbidsel["Clk_Round"] >= start_round) & (dfbidsel["Clk_Round"] <= end_round)

			dfbidsel = dfbidsel[filt].reset_index()


			listofbidders = sorted(list(set(dfbidsel["Bidder"])))

			lsas = sorted(list(set(dfbidsel["LSA"])))

			lst =[]

			for bidder in listofbidders:

				for lsa in lsas:

					temp = dfbidsel[(dfbidsel["Bidder"]==bidder) & (dfbidsel["LSA"]==lsa)]

					lst.append([1, lsa, bidder, temp[temp["Rank_PWB_End_ClkRd"]==1]["Rank_PWB_End_ClkRd"].count()])

					lst.append([2, lsa, bidder, temp[temp["Rank_PWB_End_ClkRd"]==2]["Rank_PWB_End_ClkRd"].count()])

					lst.append([3, lsa, bidder, temp[temp["Rank_PWB_End_ClkRd"]==3]["Rank_PWB_End_ClkRd"].count()])

					lst.append([4, lsa, bidder, temp[temp["Rank_PWB_End_ClkRd"]==4]["Rank_PWB_End_ClkRd"].count()])

			dfbidsel = pd.DataFrame(lst)

			dfbidsel.columns = ["RankNo","LSA","Bidder", "RankCount"]

			dfbidsel = dfbidsel.set_index("LSA").reset_index()


			dfbidsel["Rank_Bidder"] = dfbidsel[["RankNo", "Bidder"]].apply(lambda x: "-".join(map(str, x)), axis=1)


			dfbidsel = dfbidsel.pivot(index="Rank_Bidder", columns='LSA', values="RankCount")

			dfbidsel = dfbidsel.sort_index(ascending = False)


			circle_list=[]

			for circle in listofcircles: #this extracts the full name of the circle from the code
				circle_list.append(state_dict[circle])

			sortbylsa = st.sidebar.selectbox("Select a Circle to Sort", circle_list)

			selected_lsa = [k for k, v in state_dict.items() if v == sortbylsa]

			dfbidsel = dfbidsel.sort_values(selected_lsa[0], ascending = True)

			dfbidsel = dfbidsel.replace(0, np.nan).dropna(axis =0, how = "all")

			data = [go.Heatmap(
					z=dfbidsel.values,
			        y= dfbidsel.index,
			        x=dfbidsel.columns,
					xgap = 1,
					ygap = 1,
					hoverinfo ='text',
					# text = hovertext,
					colorscale='Hot',
					# showscale=False,
						texttemplate="%{z}", 
						textfont={"size":8},
						reversescale=True,
						)
					]

			figauc = go.Figure(data=data)

			figauc.update_layout(
			    template="seaborn",
			    xaxis_side= 'top',
			   	height = 800,
			   	yaxis=dict(
		        tickmode='array',
		        	))

			title = titlesubpart+" - Bidder's Agg Ranks in the Chosen Window of Rounds"
			subtitle = "Source - DOT; Between Round Nos "+str(start_round)+" & "+str(end_round)+ "; Number of Rounds = "+ str(end_round-start_round)

			style = "<style>h3 {text-align: left;}</style>"
			with st.container():
				#plotting the main chart
				st.markdown(style, unsafe_allow_html=True)
				st.header(title)
				st.markdown(subtitle)

			#Drawning a black border around the heatmap chart 
			figauc.update_xaxes(fixedrange=True,showline=True,linewidth=1.2,linecolor='black', mirror=True)
			figauc.update_yaxes(fixedrange=True,showline=True, linewidth=1.2, linecolor='black', mirror=True)

			figauc.update_layout(
				    xaxis=dict(showgrid=False),
				    yaxis=dict(showgrid=False)
				)

			st.plotly_chart(figauc, use_container_width=True)


	if SubFeature == "ProvWinningBid":

		#------------------New Code Starts---------------------#


		dfbid1 = loadauctionbiddata()[demandsheet].replace('-', np.nan, regex = True)

		dfbid1 = dfbid1.drop(columns =["Clock Round", "Clock Round Price (Rs. Crore)", "Aggregate Demand", "Excess Demand"], axis =1)

		dfbid1 = dfbid1.groupby(["LSA"]).mean().reset_index()

		dfbid1.columns = ["LSA", "BlocksForSale"]


		summarydf = dfbid1.copy()

		#preparing the summary chart total slots up for auctions
		chart = summarychart(summarydf, 'LSA', "BlocksForSale")
		SummaryFlag = True


		#-----------------New Code Ends ----------------------#


		pwbtype = st.sidebar.selectbox("Select a PWB Type", ["Start CLK Round", "End CLK Round"])

		df1strd = dfbid[dfbid["Clk_Round"] == 1].reset_index() #Identifying the 2nd round gives up the reserve price

		dfpwb1strdend = df1strd.pivot(index="Bidder", columns='LSA', values="PWB_End_ClkRd").sort_index(ascending=False)

		# dfrp = dfpwb1strdend.mean()

		# dfrp.columns = ["ReservePrice"]

		dfrp = dfrp.T


		if pwbtype == "Start CLK Round":

			round_number = st.slider("Select Auction Round Numbers using the Silder below", min_value=1, max_value=totalrounds, step=1, value = totalrounds)

			dfbidpwb = dfbid.copy()

			filt  =(dfbidpwb["Clk_Round"] == round_number) 

			dfbidpwb = dfbidpwb[filt]

			dfblocksalloc_rdend = dfbidpwb.reset_index().pivot(index="Bidder", columns='LSA', values="Prov_Alloc_BLK_End_ClkRd")\
														.sort_index(ascending=False).round(0)

			dftemp = dfbidpwb.reset_index().pivot(index="Bidder", columns='LSA', values="PWB_Start_ClkRd").sort_index(ascending=False).round(1)

			chartoption = st.sidebar.radio('Click an Option', ["Absolute Values", "ReservePrice Multiple"])

			if chartoption == "Absolute Values":


				figpanindiabids = plotbiddertotal(dftemp,dfblocksalloc_rdend)


				figpanindiabids.update_yaxes(visible=False, showticklabels=False)

				figpanindiabids.update_layout(height = 615)

				dftemp = dftemp.sort_index(ascending=True)


				hovertext, colormatrix = htext_colormatrix_auctiondata_2010_3G_BWA_ProvWinningBid(dfrp, dftemp, pwbtype, round_number)

				data = [go.Heatmap(
				z=dftemp.values,
		        y= dftemp.index,
		        x=dftemp.columns,
				xgap = 1,
				ygap = 1,
				hoverinfo ='text',
				text = hovertext,
				colorscale='picnic',
				showscale=False,
					texttemplate="%{z}", 
					textfont={"size":12},
					# reversescale=True,
					)]

				figauc = go.Figure(data=data)


				figauc.update_layout(uniformtext_minsize=12, 
				  uniformtext_mode='hide', 
				  xaxis_title=None, 
				  yaxis_title=None, 
				  yaxis_autorange='reversed',
				  font=dict(size=12),
				  template='simple_white',
				  paper_bgcolor=None,
				  height=600, 
				  # width=1200,
				  margin=dict(t=80, b=50, l=0, r=0, pad=0),
				  yaxis=dict(
		        	  tickmode='array'),
				  xaxis = dict(
				  side = 'top',
				  tickmode = 'linear',
				  tickangle=0,
				  dtick = 1), 
				)

				figauc.update_layout(
				    coloraxis=dict(
				        cmin=0,  # Set the minimum value of the color bar
				        # zmax=10  # Set the maximum value of the color bar
				    )
				)
		
			if chartoption == "ReservePrice Multiple":

				dftemp1 = dftemp.copy()

				dftemp = round(dftemp/dfrp.values,1)


				hovertext, colormatrix = htext_colormatrix_auctiondata_2010_3G_BWA_ProvWinningBid(dfrp, dftemp1, pwbtype, round_number)

				dftemp = dftemp.sort_index(ascending=True)

				data = [go.Heatmap(
					z=dftemp.values,
			        y= dftemp.index,
			        x=dftemp.columns,
					xgap = 1,
					ygap = 1,
					hoverinfo ='text',
					text = hovertext,
					colorscale='picnic',
					showscale=True,
						texttemplate="%{z}", 
						textfont={"size":12},
						# reversescale=True,
						)]

				figauc = go.Figure(data=data)

				figauc.update_layout(uniformtext_minsize=12, 
				  uniformtext_mode='hide', 
				  xaxis_title=None, 
				  yaxis_title=None, 
				  yaxis_autorange='reversed',
				  font=dict(size=12),
				  template='simple_white',
				  paper_bgcolor=None,
				  height=600, 
				  # width=1200,
				  margin=dict(t=80, b=50, l=50, r=50, pad=0),
				  yaxis=dict(
		        	  tickmode='array'),
				  xaxis = dict(
				  side = 'top',
				  tickmode = 'linear',
				  tickangle=0,
				  dtick = 1), 
				)

				figauc.update_layout(
				    coloraxis=dict(
				        cmin=0,  # Set the minimum value of the color bar
				        # zmax=10  # Set the maximum value of the color bar
				    )
				)

				
# #-------------New Layout Code for Testing ---------------
# 				figauc.update_layout(  
# 				    plot_bgcolor="#FFFFFF",
# 				    hovermode="x",
# 				    hoverdistance=100, # Distance to show hover label of data point
# 				    spikedistance=1000, # Distance to show spike
# 				    yaxis=dict(
# 				        title="time",
# 				        linecolor="#BCCCDC",
# 				        showspikes=True, # Show spike line for X-axis
# 				        # Format spike
# 				        spikethickness=1,
# 				        spikedash="dot",
# 				        spikecolor="#999999",
# 				        spikemode="across",
# 				    ),
# 				    xaxis=dict(
# 				        title="price",
# 				        linecolor="#BCCCDC"
# 				    )
# 						)
# #-------------New Layout Code for Testing Ends-----------------


			title = titlesubpart+" - PWB/BLK at the Start of Clock Round No - "+str(round_number)
			subtitle = "Unit - Rs Cr; Source - DoT; "+ chartoption+" - May be lower for bidders in same circle who did not agree to the higher round price"

			style = "<style>h3 {text-align: left;}</style>"
			with st.container():
				#plotting the main chart
				st.markdown(style, unsafe_allow_html=True)
				st.header(title)
				st.markdown(subtitle)


			#Drawning a black border around the heatmap chart 
			figauc.update_xaxes(fixedrange=True,showline=True,linewidth=1.2,linecolor='black', mirror=True, range=[-0.5, len(dftemp.columns) -0.5])
			figauc.update_yaxes(fixedrange=True,showline=True, linewidth=1.2, linecolor='black', mirror=True, range=[-0.5, len(dftemp.index) -0.5])

			figauc.update_layout(
				    xaxis=dict(showgrid=False),
				    yaxis=dict(showgrid=False),
				     
							)

			hoverlabel_bgcolor = colormatrix

			figauc.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))

			if chartoption == "Absolute Values":
				col1,col2 = st.columns([8,1]) #create collumns of uneven width
				with col1:
					st.plotly_chart(figauc, use_container_width=True)
				with col2:
					st.markdown("")
					st.plotly_chart(figpanindiabids, use_container_width=True)



			if chartoption =="ReservePrice Multiple":

				st.plotly_chart(figauc, use_container_width=True)

				#plotting the final summary chart 
				col1,col2,col3 = st.columns([0.35, 14,1.1]) #create collumns of uneven width
				if SummaryFlag ==True:
					col2.altair_chart(chart, use_container_width=True)


		if pwbtype == "End CLK Round":

			round_number = st.slider("Select Auction Round Numbers using the Silder below", min_value=1, max_value=totalrounds, step=1, value = totalrounds)

			dfbidpwb = dfbid.copy()

			filt  =(dfbidpwb["Clk_Round"] == round_number) 

			dfbidpwb = dfbidpwb[filt]

			# dftemp = dfbidpwb.drop(columns=["Rank_PWB_End_ClkRd","PWB_Start_ClkRd","Possible_Raise_Bid_ClkRd", "Rank_PWB_Start_ClkRd","Bid_Decision","Clk_Round", "PWB_Start_ClkRd"], axis=1).reset_index()

			# dftemp = dftemp.groupby(["LSA", "Bidder", "PWB_End_ClkRd"]).sum().reset_index()

			dftemp = dfbidpwb.reset_index().pivot(index="Bidder", columns='LSA', values="PWB_End_ClkRd").sort_index(ascending=False).round(1)

			dfblocksalloc_rdend = dfbidpwb.reset_index().pivot(index="Bidder", columns='LSA', values="Prov_Alloc_BLK_End_ClkRd")\
														.sort_index(ascending=False).round(0)

			chartoption = st.sidebar.radio('Click an Option', ["Absolute Values", "ReservePrice Multiple"])

			if chartoption == "Absolute Values":

				figpanindiabids = plotbiddertotal(dftemp,dfblocksalloc_rdend)

				figpanindiabids.update_yaxes(visible=False, showticklabels=False)

				figpanindiabids.update_layout(height = 615)


				hovertext, colormatrix = htext_colormatrix_auctiondata_2010_3G_BWA_ProvWinningBid(dfrp, dftemp, pwbtype, round_number)

				dftemp = dftemp.sort_index(ascending=True)

				data = [go.Heatmap(
				z=dftemp.values,
		        y= dftemp.index,
		        x=dftemp.columns,
				xgap = 1,
				ygap = 1,
				hoverinfo ='text',
				text = hovertext,
				colorscale='picnic',
				showscale=False,
					texttemplate="%{z}", 
					textfont={"size":10},
					# reversescale=True,
					)]

				figauc = go.Figure(data=data)

				figauc.update_layout(uniformtext_minsize=12, 
				  uniformtext_mode='hide', 
				  xaxis_title=None, 
				  yaxis_title=None, 
				  yaxis_autorange='reversed',
				  font=dict(size=12),
				  template='simple_white',
				  paper_bgcolor=None,
				  height=600, 
				  # width=1200,
				  margin=dict(t=80, b=50, l=0, r=0, pad=0),
				  yaxis=dict(
		        	  tickmode='array'),
				  xaxis = dict(
				  side = 'top',
				  tickmode = 'linear',
				  tickangle=0,
				  dtick = 1), 
				)

				figauc.update_layout(
				    coloraxis=dict(
				        cmin=0,  # Set the minimum value of the color bar
				        # zmax=10  # Set the maximum value of the color bar
				    )
				)


			if chartoption == "ReservePrice Multiple":

				dftemp1 = dftemp.copy()

				dftemp = round(dftemp/dfrp.values,1)

				hovertext, colormatrix = htext_colormatrix_auctiondata_2010_3G_BWA_ProvWinningBid(dfrp, dftemp1, pwbtype, round_number)

				dftemp = dftemp.sort_index(ascending=True)

				data = [go.Heatmap(
					z=dftemp.values,
			        y= dftemp.index,
			        x=dftemp.columns,
					xgap = 1,
					ygap = 1,
					hoverinfo ='text',
					text = hovertext,
					colorscale='picnic',
					showscale=True,
						texttemplate="%{z}", 
						textfont={"size":10},
						# reversescale=True,
						)]

				figauc = go.Figure(data=data)

				figauc.update_layout(uniformtext_minsize=12, 
				  uniformtext_mode='hide', 
				  xaxis_title=None, 
				  yaxis_title=None, 
				  yaxis_autorange='reversed',
				  font=dict(size=12),
				  template='simple_white',
				  paper_bgcolor=None,
				  height=600, 
				  # width=1200,
				  margin=dict(t=80, b=50, l=50, r=50, pad=0),
				  yaxis=dict(
		        	  tickmode='array'),
				  xaxis = dict(
				  side = 'top',
				  tickmode = 'linear',
				  tickangle=0,
				  dtick = 1), 
				)

				figauc.update_layout(
				    coloraxis=dict(
				        cmin=0,  # Set the minimum value of the color bar
				        # zmax=10  # Set the maximum value of the color bar
				    )
				)


			title = titlesubpart+" - PWB/BLK at the End of Clock Round No - "+str(round_number)
			subtitle = "Unit - Rs Cr; Source - DoT; "+chartoption+" - May be lower for bidders in same circle who did not agree to the higher round price"

			style = "<style>h3 {text-align: left;}</style>"
			with st.container():
				#plotting the main chart
				st.markdown(style, unsafe_allow_html=True)
				st.header(title)
				st.markdown(subtitle)


			#Drawning a black border around the heatmap chart 
			figauc.update_xaxes(fixedrange=True,showline=True,linewidth=1.2,linecolor='black', mirror=True)
			figauc.update_yaxes(fixedrange=True,showline=True, linewidth=1.2, linecolor='black', mirror=True)

			figauc.update_layout(
				    xaxis=dict(showgrid=False),
				    yaxis=dict(showgrid=False)
				)

			hoverlabel_bgcolor = colormatrix

			figauc.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))

			if chartoption == "Absolute Values":
				col1,col2 = st.columns([8,1]) #create collumns of uneven width
				col1.plotly_chart(figauc, use_container_width=True)
				col2.markdown("")
				col2.plotly_chart(figpanindiabids, use_container_width=True)

			if chartoption == "ReservePrice Multiple":
				st.plotly_chart(figauc, use_container_width=True)

			#plotting the final summary chart 
				col1,col2,col3 = st.columns([0.35, 14,1.1]) #create collumns of uneven width
				if SummaryFlag ==True:
					col2.altair_chart(chart, use_container_width=True)




	if SubFeature == "BlocksSelected":

		round_number = st.slider("Select Auction Round Numbers using the Silder below", min_value=1, max_value=totalrounds, step=1, value = totalrounds)

		dfbidblksec = dfbid.copy()

		filt  =(dfbidblksec["Clk_Round"] == round_number) 

		dfbidblksec = dfbidblksec[filt]

		# dftemp = dfbidblksec.groupby(["LSA", "Bidder", "No_of_BLK_Selected"]).sum().reset_index()

		dftemp = dfbidblksec.reset_index().pivot(index="Bidder", columns='LSA', values="No_of_BLK_Selected").sort_index(ascending=False).round(0)


		sumrows = dftemp.sum(axis=1).reset_index()

		sumrows.columns = ["Bidders", "Total Slots"]

		sumcols = dftemp.sum(axis=0).reset_index()

		sumcols.columns = ["LSA", "Total Slots"]


		figsumcols = summarychart(sumcols, "LSA", "Total Slots")


		data = [go.Heatmap(
				z=dftemp.values,
		        y= dftemp.index,
		        x=dftemp.columns,
				xgap = 1,
				ygap = 1,
				hoverinfo ='text',
				# text = hovertext,
				zmin = zmin_blk_sec, 
				zmax = zmax_blk_sec, 
				colorscale='Hot',
				# showscale=showscale,
					texttemplate="%{z}", 
					textfont={"size":10},
					reversescale=True,
					)]
				

		figauc = go.Figure(data=data)

		figauc.update_layout(
		    template="seaborn",
		    xaxis_side= 'top',
		   	height = 650,
		   	yaxis=dict(
	        tickmode='array',
	        	))

		title = titlesubpart+" - Blocks Selected at Round No -"+str(round_number)
		subtitle = "Unit - Numbers; Block Size = "+ str(blocksize) +" MHz; Source - DoT"

		style = "<style>h3 {text-align: left;}</style>"
		with st.container():
			#plotting the main chart
			st.markdown(style, unsafe_allow_html=True)
			st.header(title)
			st.markdown(subtitle)


		#Drawning a black border around the heatmap chart 
		figauc.update_xaxes(fixedrange=True,showline=True,linewidth=1.2,linecolor='black', mirror=True)
		figauc.update_yaxes(fixedrange=True,showline=True, linewidth=1.2, linecolor='black', mirror=True)

		figauc.update_layout(
			    xaxis=dict(showgrid=False),
			    yaxis=dict(showgrid=False)
			)

		# hoverlabel_bgcolor = "#000000" #subdued black

		# figauc.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))

		
		st.plotly_chart(figauc, use_container_width=True)

		#plotting the column sums of all slots
		col1,col2,col3 = st.columns([0.2,14,1]) #create collumns of uneven width
		col2.altair_chart(figsumcols, use_container_width=True)



	if SubFeature == "BlocksAllocated":

		round_number = st.slider("Select Auction Round Numbers using the Silder below", min_value=1, max_value=totalrounds, step=1, value = totalrounds)

		blkallocoption = st.sidebar.radio('Click an Option', ["Start of Round", "End of Round"])

		if blkallocoption == "Start of Round":

			dfbidblksec = dfbid.copy()

			filt  =(dfbidblksec["Clk_Round"] == round_number) 

			dfbidblksec = dfbidblksec[filt]


			dftemp = dfbidblksec.reset_index().pivot(index="Bidder", columns='LSA', values="Prov_Alloc_BLK_Start_ClkRd").sort_index(ascending=False).round(0)

			sumrows = dftemp.sum(axis=1).reset_index()

			sumrows.columns = ["Bidders", "Total Slots"]

			sumcols = dftemp.sum(axis=0).reset_index()

			sumcols.columns = ["LSA", "Total Slots"]


			figsumcols = summarychart(sumcols, "LSA", "Total Slots")

			figsumrows = plotrwototal(sumrows,"Bidders", "Total Slots")

			figsumrows.update_yaxes(visible=False, showticklabels=False)

			figsumrows.update_layout(height = 615)

			hovertext = htext_auctiondata_2010_3G_BWA_BlocksAllocated(dftemp)

			dftemp = dftemp.sort_index(ascending=True)

			data = [go.Heatmap(
					z=dftemp.values,
			        y= dftemp.index,
			        x=dftemp.columns,
					xgap = 1,
					ygap = 1,
					hoverinfo ='text',
					text = hovertext,
					zmin = zmin_blk_sec, 
					zmax = zmax_blk_sec, 
					colorscale='Hot',
					showscale=False,
						texttemplate="%{z}", 
						textfont={"size":12},
						reversescale=True,
						)]
					

			figauc = go.Figure(data=data)

			figauc.update_layout(uniformtext_minsize=12, 
				  uniformtext_mode='hide', 
				  xaxis_title=None, 
				  yaxis_title=None, 
				  yaxis_autorange='reversed',
				  font=dict(size=12),
				  template='simple_white',
				  paper_bgcolor=None,
				  height=600, 
				  # width=1200,
				  margin=dict(t=80, b=50, l=0, r=0, pad=0),
				  yaxis=dict(
		        	  tickmode='array'),
				  xaxis = dict(
				  side = 'top',
				  tickmode = 'linear',
				  tickangle=0,
				  dtick = 1), 
				)

			figauc.update_layout(
			    coloraxis=dict(
			        cmin=0,  # Set the minimum value of the color bar
			        # zmax=10  # Set the maximum value of the color bar
			    )
			)

			title = titlesubpart+" - Blocks Allocated at the Start of Round No -"+str(round_number)
			subtitle = "Unit - Numbers; Block Size = "+ str(blocksize) +" MHz; Source - DoT"

			style = "<style>h3 {text-align: left;}</style>"
			with st.container():
				#plotting the main chart
				st.markdown(style, unsafe_allow_html=True)
				st.header(title)
				st.markdown(subtitle)


			#Drawning a black border around the heatmap chart 
			figauc.update_xaxes(fixedrange=True,showline=True,linewidth=1.2,linecolor='black', mirror=True)
			figauc.update_yaxes(fixedrange=True,showline=True, linewidth=1.2, linecolor='black', mirror=True)

			figauc.update_layout(
				    xaxis=dict(showgrid=False),
				    yaxis=dict(showgrid=False)
				)

			hoverlabel_bgcolor = "#000000" #subdued black

			figauc.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))

			
			#plotting all charts 
			col1,col2 = st.columns([9,1]) #create collumns of uneven width
			col1.plotly_chart(figauc, use_container_width=True)
			col1.altair_chart(figsumcols, use_container_width=True)
			col2.markdown("")
			col2.plotly_chart(figsumrows, use_container_width=True)


		if blkallocoption == "End of Round":

			dfbidblksec = dfbid.copy()

			filt  =(dfbidblksec["Clk_Round"] == round_number) 

			dfbidblksec = dfbidblksec[filt]

			dftemp = dfbidblksec.reset_index().pivot(index="Bidder", columns='LSA', values="Prov_Alloc_BLK_End_ClkRd").sort_index(ascending=False).round(0)

			sumrows = dftemp.sum(axis=1).reset_index()

			sumrows.columns = ["Bidders", "Total Slots"]

			sumcols = dftemp.sum(axis=0).reset_index()

			sumcols.columns = ["LSA", "Total Slots"]


			figsumrows = plotrwototal(sumrows,"Bidders", "Total Slots")

			figsumrows.update_yaxes(visible=False, showticklabels=False)

			figsumrows.update_layout(height = 615)


			figsumcols = summarychart(sumcols, "LSA", "Total Slots")

			hovertext = htext_auctiondata_2010_3G_BWA_BlocksAllocated(dftemp)

			dftemp = dftemp.sort_index(ascending=True)


			data = [go.Heatmap(
					z=dftemp.values,
			        y= dftemp.index,
			        x=dftemp.columns,
					xgap = 1,
					ygap = 1,
					hoverinfo ='text',
					text = hovertext,
					zmin = zmin_blk_sec, 
					zmax = zmax_blk_sec, 
					colorscale='Hot',
					showscale=False,
						texttemplate="%{z}", 
						textfont={"size":12},
						reversescale=True,
						)]
					

			figauc = go.Figure(data=data)

			figauc.update_layout(uniformtext_minsize=12, 
				  uniformtext_mode='hide', 
				  xaxis_title=None, 
				  yaxis_title=None, 
				  yaxis_autorange='reversed',
				  font=dict(size=12),
				  template='simple_white',
				  paper_bgcolor=None,
				  height=600, 
				  # width=1200,
				  margin=dict(t=80, b=50, l=0, r=0, pad=0),
				  yaxis=dict(
		        	  tickmode='array'),
				  xaxis = dict(
				  side = 'top',
				  tickmode = 'linear',
				  tickangle=0,
				  dtick = 1), 
				)

			figauc.update_layout(
			    coloraxis=dict(
			        cmin=0,  # Set the minimum value of the color bar
			        # zmax=10  # Set the maximum value of the color bar
			    )
			)

			title = titlesubpart+" - Blocks Allocated at the End of Round No -"+str(round_number)
			subtitle = "Unit - Numbers; Block Size = "+ str(blocksize) +" MHz; Source - DoT"

			style = "<style>h3 {text-align: left;}</style>"
			with st.container():
				#plotting the main chart
				st.markdown(style, unsafe_allow_html=True)
				st.header(title)
				st.markdown(subtitle)


			#Drawning a black border around the heatmap chart 
			figauc.update_xaxes(fixedrange=True,showline=True,linewidth=1.2,linecolor='black', mirror=True)
			figauc.update_yaxes(fixedrange=True,showline=True, linewidth=1.2, linecolor='black', mirror=True)

			figauc.update_layout(
				    xaxis=dict(showgrid=False),
				    yaxis=dict(showgrid=False)
				)

			hoverlabel_bgcolor = "#000000" #subdued black

			figauc.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))


			#plotting all charts 
			col1,col2 = st.columns([9,1]) #create collumns of uneven width
			col1.plotly_chart(figauc, use_container_width=True)
			col1.altair_chart(figsumcols, use_container_width=True)
			col2.markdown("")
			col2.plotly_chart(figsumrows, use_container_width=True)


	if SubFeature == "BiddingActivity":

		dfbid = loadauctionbiddata()[activitysheet].replace('-', np.nan, regex = True)

		dfbid.columns = ["Clk_Round", "Bidder", "Pts_Start_Round", "Activity_Factor", "Activity_Requirement",
						"Actual_Activity","Activity_at_PWB","Activity_NewBids","Point_Carry_Forward", "Points_Lost"] 


		dfbidactivity = dfbid.copy()

		optiontype = st.sidebar.radio('Click an Option', ["Total Pts in Play", "Pts in PWB Circles", "Pts in New Circles", "Activity Factor","Points Lost"])

		if optiontype == "Total Pts in Play":

			filt = dfbidactivity["Clk_Round"]==1

			dfbidactivityRd1 = dfbidactivity[filt]

			dfbidactivity = dfbidactivity.pivot(index="Bidder", columns='Clk_Round', values="Actual_Activity").sort_index(ascending=True)

			dfbidactivityRd1 = dfbidactivityRd1.pivot(index="Bidder", columns='Clk_Round', values="Pts_Start_Round").sort_index(ascending=True)

			dfbidactivityratio = round((dfbidactivity/dfbidactivityRd1.values),2)


			hovertext = htext_auctiondata_2010_3G_BWA_BiddingActivity(dfbid, "Actual_Activity")



			data1 = [go.Heatmap(
				z=dfbidactivity.values,
		        y= dfbidactivity.index,
		        x=dfbidactivity.columns,
				xgap = 0.5,
				ygap = 1,
				hoverinfo ='text',
				text = hovertext,
				colorscale='Hot',
				showscale=True,
					texttemplate=texttempbiddemandactivity,
					textfont={"size":8},
					reversescale=True,
					)]

		

			data2 = [go.Heatmap(
					z=dfbidactivityratio.values,
			        y= dfbidactivityratio.index,
			        x=dfbidactivityratio.columns,
					xgap = 0.5,
					ygap = 1,
					hoverinfo ='text',
					text = hovertext,
					colorscale='Hot',
					showscale=True,
						texttemplate=texttempbiddemandactivity, 
						textfont={"size":8},
						reversescale=True,
						)]
						

			figauc1 = go.Figure(data=data1)
			figauc2 = go.Figure(data=data2)

			figauc1.update_layout(uniformtext_minsize=12, 
			  uniformtext_mode='hide', 
			  xaxis_title=None, 
			  yaxis_title=None, 
			  yaxis_autorange='reversed',
			  font=dict(size=12),
			  template='simple_white',
			  paper_bgcolor=None,
			  height=600, 
			  width=1200,
			  margin=dict(t=80, b=50, l=50, r=50, pad=0),
			  yaxis=dict(
	        	  tickmode='array'),
			  xaxis = dict(
			  side = 'top',
			  tickmode = 'linear',
			  tickangle=0,
			  dtick = xdtick), 
			)


			#Drawning a black border around the heatmap chart 
			figauc1.update_xaxes(fixedrange=True,showline=True,linewidth=1.2,linecolor='black', mirror=True)
			figauc1.update_yaxes(fixedrange=True,showline=True, linewidth=1.2, linecolor='black', mirror=True)

			figauc1.update_layout(
				    xaxis=dict(showgrid=False),
				    yaxis=dict(showgrid=False)
				)

			figauc2.update_layout(uniformtext_minsize=12, 
			  uniformtext_mode='hide', 
			  xaxis_title=None, 
			  yaxis_title=None, 
			  yaxis_autorange='reversed',
			  font=dict(size=12),
			  template='simple_white',
			  paper_bgcolor=None,
			  height=600, 
			  width=1200,
			  margin=dict(t=80, b=50, l=50, r=50, pad=0),
			  yaxis=dict(
	        	  tickmode='array'),
			  xaxis = dict(
			  side = 'top',
			  tickmode = 'linear',
			  tickangle=0,
			  dtick = xdtick), 
			)


			#Drawning a black border around the heatmap chart 
			figauc2.update_xaxes(fixedrange=True,showline=True,linewidth=1.2,linecolor='black', mirror=True)
			figauc2.update_yaxes(fixedrange=True,showline=True, linewidth=1.2, linecolor='black', mirror=True)

			figauc2.update_layout(
				    xaxis=dict(showgrid=False),
				    yaxis=dict(showgrid=False)
				)

			title = titlesubpart+" - Points in Play"
			subtitle = "Unit - Nos; Source - DoT; Xaxis - Round Numbers"+subtitlesubpartbidactivity
			style = "<style>h3 {text-align: left;}</style>"
			with st.container():
				#plotting the main chart
				st.markdown(style, unsafe_allow_html=True)
				st.header(title)
				st.markdown(subtitle)


			hoverlabel_bgcolor = "#000000" #subdued black

			figauc1.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))
			figauc2.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))

			tab1, tab2 = st.tabs(["Actual", "Ratio (Actual/Initial)"]) #For showning the absolute and Ratio charts in two differet tabs
			tab1.plotly_chart(figauc1, use_container_width=True)
			tab2.plotly_chart(figauc2, use_container_width=True)



		if optiontype == "Pts in PWB Circles":

			filt = dfbidactivity["Clk_Round"]==1

			dfbidactivityRd1 = dfbidactivity[filt]

			dfbidactivity = dfbidactivity.pivot(index="Bidder", columns='Clk_Round', values="Activity_at_PWB").sort_index(ascending=True)

			# dfbidactivityRd1 = dfbidactivityRd1.pivot(index="Bidder", columns='Clk_Round', values="Actual_Activity").sort_index(ascending=True)

			dfbidactivityRd1 = dfbidactivityRd1.pivot(index="Bidder", columns='Clk_Round', values="Pts_Start_Round").sort_index(ascending=True)

			dfbidactivityratio = round((dfbidactivity/dfbidactivityRd1.values),2)


			hovertext = htext_auctiondata_2010_3G_BWA_BiddingActivity(dfbid, "Activity_at_PWB")


			data1 = [go.Heatmap(
					z=dfbidactivity.values,
			        y= dfbidactivity.index,
			        x=dfbidactivity.columns,
					xgap = 0.5,
					ygap = 1,
					hoverinfo ='text',
					text = hovertext,
					colorscale='Hot',
					showscale=True,
						texttemplate=texttempbiddemandactivity, 
						textfont={"size":8},
						reversescale=True,
						)]

			data2 = [go.Heatmap(
					z=dfbidactivityratio.values,
			        y= dfbidactivityratio.index,
			        x=dfbidactivityratio.columns,
					xgap = 0.5,
					ygap = 1,
					hoverinfo ='text',
					text = hovertext,
					colorscale='Hot',
					showscale=True,
						texttemplate=texttempbiddemandactivity, 
						textfont={"size":8},
						reversescale=True,
						)]
					

			figauc1 = go.Figure(data=data1)
			figauc2 = go.Figure(data=data2)

			figauc1.update_layout(uniformtext_minsize=12, 
			  uniformtext_mode='hide', 
			  xaxis_title=None, 
			  yaxis_title=None, 
			  yaxis_autorange='reversed',
			  font=dict(size=12),
			  template='simple_white',
			  paper_bgcolor=None,
			  height=600, 
			  width=1200,
			  margin=dict(t=80, b=50, l=50, r=50, pad=0),
			  yaxis=dict(
	        	  tickmode='array'),
			  xaxis = dict(
			  side = 'top',
			  tickmode = 'linear',
			  tickangle=0,
			  dtick = xdtick), 
			)


			#Drawning a black border around the heatmap chart 
			figauc1.update_xaxes(fixedrange=True,showline=True,linewidth=1.2,linecolor='black', mirror=True)
			figauc1.update_yaxes(fixedrange=True,showline=True, linewidth=1.2, linecolor='black', mirror=True)

			figauc1.update_layout(
				    xaxis=dict(showgrid=False),
				    yaxis=dict(showgrid=False)
				)

			figauc2.update_layout(uniformtext_minsize=12, 
			  uniformtext_mode='hide', 
			  xaxis_title=None, 
			  yaxis_title=None, 
			  yaxis_autorange='reversed',
			  font=dict(size=12),
			  template='simple_white',
			  paper_bgcolor=None,
			  height=600, 
			  width=1200,
			  margin=dict(t=80, b=50, l=50, r=50, pad=0),
			  yaxis=dict(
	        	  tickmode='array'),
			  xaxis = dict(
			  side = 'top',
			  tickmode = 'linear',
			  tickangle=0,
			  dtick = xdtick), 
			)


			#Drawning a black border around the heatmap chart 
			figauc2.update_xaxes(fixedrange=True,showline=True,linewidth=1.2,linecolor='black', mirror=True)
			figauc2.update_yaxes(fixedrange=True,showline=True, linewidth=1.2, linecolor='black', mirror=True)

			figauc2.update_layout(
				    xaxis=dict(showgrid=False),
				    yaxis=dict(showgrid=False)
				)

			title = titlesubpart+" - Points in Play in LSAs where the Bidder was a PWB"
			subtitle = "Unit - Nos; Source - DoT; Xaxis - Round Numbers"

			style = "<style>h3 {text-align: left;}</style>"
			with st.container():
				#plotting the main chart
				st.markdown(style, unsafe_allow_html=True)
				st.header(title)
				st.markdown(subtitle)

			hoverlabel_bgcolor = "#000000" #subdued black

			figauc1.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))
			figauc2.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))

			tab1, tab2 = st.tabs(["Actual", "Ratio (Actual/Initial)"]) #For showning the absolute and Ratio charts in two differet tabs
			tab1.plotly_chart(figauc1, use_container_width=True)
			tab2.plotly_chart(figauc2, use_container_width=True)


		if optiontype == "Pts in New Circles":


			filt = dfbidactivity["Clk_Round"]==1

			dfbidactivityRd1 = dfbidactivity[filt]

			dfbidactivity = dfbidactivity.pivot(index="Bidder", columns='Clk_Round', values="Activity_NewBids").sort_index(ascending=True)

			# dfbidactivityRd1 = dfbidactivityRd1.pivot(index="Bidder", columns='Clk_Round', values="Actual_Activity").sort_index(ascending=True)

			dfbidactivityRd1 = dfbidactivityRd1.pivot(index="Bidder", columns='Clk_Round', values="Pts_Start_Round").sort_index(ascending=True)

			dfbidactivityratio = round((dfbidactivity/dfbidactivityRd1.values),2)

			hovertext = htext_auctiondata_2010_3G_BWA_BiddingActivity(dfbid, "Activity_NewBids")


			data1 = [go.Heatmap(
					z=dfbidactivity.values,
			        y= dfbidactivity.index,
			        x=dfbidactivity.columns,
					xgap = 0.5,
					ygap = 1,
					hoverinfo ='text',
					text = hovertext,
					colorscale='Hot',
					showscale=True,
						texttemplate=texttempbiddemandactivity, 
						textfont={"size":8},
						reversescale=True,
						)]

			data2 = [go.Heatmap(
					z=dfbidactivityratio.values,
			        y= dfbidactivityratio.index,
			        x=dfbidactivityratio.columns,
					xgap = 0.5,
					ygap = 1,
					hoverinfo ='text',
					text = hovertext,
					colorscale='Hot',
					showscale=True,
						texttemplate=texttempbiddemandactivity, 
						textfont={"size":8},
						reversescale=True,
						)]
					

			figauc1 = go.Figure(data=data1)
			figauc2 = go.Figure(data=data2)

			figauc1.update_layout(uniformtext_minsize=12, 
			  uniformtext_mode='hide', 
			  xaxis_title=None, 
			  yaxis_title=None, 
			  yaxis_autorange='reversed',
			  font=dict(size=12),
			  template='simple_white',
			  paper_bgcolor=None,
			  height=600, 
			  width=1200,
			  margin=dict(t=80, b=50, l=50, r=50, pad=0),
			  yaxis=dict(
	        	  tickmode='array'),
			  xaxis = dict(
			  side = 'top',
			  tickmode = 'linear',
			  tickangle=0,
			  dtick = xdtick), 
			)


			#Drawning a black border around the heatmap chart 
			figauc1.update_xaxes(fixedrange=True,showline=True,linewidth=1.2,linecolor='black', mirror=True)
			figauc1.update_yaxes(fixedrange=True,showline=True, linewidth=1.2, linecolor='black', mirror=True)

			figauc1.update_layout(
				    xaxis=dict(showgrid=False),
				    yaxis=dict(showgrid=False)
				)

			figauc2.update_layout(uniformtext_minsize=12, 
			  uniformtext_mode='hide', 
			  xaxis_title=None, 
			  yaxis_title=None, 
			  yaxis_autorange='reversed',
			  font=dict(size=12),
			  template='simple_white',
			  paper_bgcolor=None,
			  height=600, 
			  width=1200,
			  margin=dict(t=80, b=50, l=50, r=50, pad=0),
			  yaxis=dict(
	        	  tickmode='array'),
			  xaxis = dict(
			  side = 'top',
			  tickmode = 'linear',
			  tickangle=0,
			  dtick = xdtick), 
			)


			#Drawning a black border around the heatmap chart 
			figauc2.update_xaxes(fixedrange=True,showline=True,linewidth=1.2,linecolor='black', mirror=True)
			figauc2.update_yaxes(fixedrange=True,showline=True, linewidth=1.2, linecolor='black', mirror=True)

			figauc2.update_layout(
				    xaxis=dict(showgrid=False),
				    yaxis=dict(showgrid=False)
				)


			title = titlesubpart+" - Points used for making New Bids"
			subtitle = "Unit - Nos; Source - DoT; Xaxis - Round Numbers"

			style = "<style>h3 {text-align: left;}</style>"
			with st.container():
				#plotting the main chart
				st.markdown(style, unsafe_allow_html=True)
				st.header(title)
				st.markdown(subtitle)


			hoverlabel_bgcolor = "#000000" #subdued black

			figauc1.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))
			figauc2.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))

			tab1, tab2 = st.tabs(["Actual", "Ratio (Actual/Initial)"]) #For showning the absolute and Ratio charts in two differet tabs
			tab1.plotly_chart(figauc1, use_container_width=True)
			tab2.plotly_chart(figauc2, use_container_width=True)

		if optiontype == "Activity Factor":


			dfbidactivity = dfbidactivity.pivot(index="Bidder", columns='Clk_Round', values="Activity_Factor").sort_index(ascending=True)

			data = [go.Heatmap(
					z=dfbidactivity.values,
			        y= dfbidactivity.index,
			        x=dfbidactivity.columns,
					xgap = 0.5,
					ygap = 1,
					hoverinfo ='text',
					# text = hovertext,
					colorscale='Hot',
					zmin=zmin_af, zmax=zmax_af,
					showscale=True,
						texttemplate=texttempbiddemandactivity, 
						textfont={"size":8},
						reversescale=True,
						)]
					

			figauc = go.Figure(data=data)

			figauc.update_layout(uniformtext_minsize=12, 
			  uniformtext_mode='hide', 
			  xaxis_title=None, 
			  yaxis_title=None, 
			  yaxis_autorange='reversed',
			  font=dict(size=12),
			  template='simple_white',
			  paper_bgcolor=None,
			  height=600, 
			  width=1200,
			  margin=dict(t=80, b=50, l=50, r=50, pad=0),
			  yaxis=dict(
	        	  tickmode='array'),
			  xaxis = dict(
			  side = 'top',
			  tickmode = 'linear',
			  tickangle=0,
			  dtick = xdtick), 
			)

			title = titlesubpart+" - Activity Factor Announced by the Auctioneer"
			subtitle = "Unit - Nos; Source - DoT; Xaxis - Round Numbers"

			style = "<style>h3 {text-align: left;}</style>"
			with st.container():
				#plotting the main chart
				st.markdown(style, unsafe_allow_html=True)
				st.header(title)
				st.markdown(subtitle)


			#Drawning a black border around the heatmap chart 
			figauc.update_xaxes(fixedrange=True,showline=True,linewidth=1.2,linecolor='black', mirror=True)
			figauc.update_yaxes(fixedrange=True,showline=True, linewidth=1.2, linecolor='black', mirror=True)

			figauc.update_layout(
				    xaxis=dict(showgrid=False),
				    yaxis=dict(showgrid=False)
				)

			# hoverlabel_bgcolor = "#000000" #subdued black

			# figauc.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))

		
			st.plotly_chart(figauc, use_container_width=True)



		if optiontype == "Points Lost":

			filt = dfbidactivity["Clk_Round"]==1 

			dfbidactivityRd1 = dfbidactivity[filt] 

			dfbidactivityRd1 = dfbidactivityRd1.pivot(index="Bidder", columns='Clk_Round', values="Pts_Start_Round").sort_index(ascending=True) 

			dfbidactivity = dfbidactivity.pivot(index="Bidder", columns='Clk_Round', values="Points_Lost").sort_index(ascending=True)

			totalpointslost = dfbidactivity.sum(axis=1).reset_index() 

			totalpointslost.columns = ["Bidder", "Points Lost"]

			totalpointslost = totalpointslost.set_index("Bidder").sort_index(ascending=True)

			totalpointslostperc = round((totalpointslost/dfbidactivityRd1.values)*100,1).sort_index(ascending=False).reset_index()

			dfbidactivityperc = round((dfbidactivity/dfbidactivityRd1.values)*100,1) # % of points lost with respect to the initial awarded

			totalpointslost = totalpointslost.sort_index(ascending=False)

			totalpointslostperc.columns = ["Bidder", "% Pts Lost"]

			totalpointslost = totalpointslost.reset_index()

			figptslostabs = plotlosttotal(totalpointslost, "Bidder", "Points Lost")

			figptslostabs.update_yaxes(visible=False, showticklabels=False)

			figptslostperc = plotlosttotal(totalpointslostperc, "Bidder", "% Pts Lost")

			figptslostperc.update_yaxes(visible=False, showticklabels=False)

			hovertext = htext_auctiondata_2010_3G_BWA_PointsLost(dfbidactivity, dfbidactivityperc)

			data = [go.Heatmap(
					z=dfbidactivity.values,
			        y= dfbidactivity.index,
			        x=dfbidactivity.columns,
					xgap = 0.5,
					ygap = 1,
					hoverinfo ='text',
					text = hovertext,
					colorscale='Jet',
					# zmin=0.5, zmax=1,
					showscale=False,
						texttemplate=texttempbiddemandactivity, 
						textfont={"size":8},
						reversescale=True,
						)]
					

			figauc = go.Figure(data=data)

			figauc.update_layout(uniformtext_minsize=12, 
			  uniformtext_mode='hide', 
			  xaxis_title=None, 
			  yaxis_title=None, 
			  yaxis_autorange='reversed',
			  font=dict(size=12),
			  template='simple_white',
			  paper_bgcolor=None,
			  height=600, 
			  width=1200,
			  margin=dict(t=80, b=50, l=0, r=0, pad=0),
			  yaxis=dict(
	        	  tickmode='array'),
			  xaxis = dict(
			  side = 'top',
			  tickmode = 'linear',
			  tickangle=0,
			  dtick = xdtick), 
			)

			title = titlesubpart+" - Points Lost in Various Rounds During the Auction"
			subtitle = "Unit - Nos; Source - DoT; Xaxis - Round Numbers; Points are lost due to Bidder's inability to confirm to the Activity Factor"

			style = "<style>h3 {text-align: left;}</style>"
			with st.container():
				#plotting the main chart
				st.markdown(style, unsafe_allow_html=True)
				st.header(title)
				st.markdown(subtitle)


			#Drawning a black border around the heatmap chart 
			figauc.update_xaxes(fixedrange=True,showline=True,linewidth=1.2,linecolor='black', mirror=True)
			figauc.update_yaxes(fixedrange=True,showline=True, linewidth=1.2, linecolor='black', mirror=True)

			figauc.update_layout(
				    xaxis=dict(showgrid=False),
				    yaxis=dict(showgrid=False)
				)

			hoverlabel_bgcolor = "#000000" #subdued black

			figauc.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))


			tab1,tab2 = st.tabs(["Pts Lost(Actual)", "Pts Lost(Percentage)"]) 

			with tab1:
				col1,col2 = st.columns([8,1]) #create collumns of uneven width
				with col1:
					st.plotly_chart(figauc, use_container_width=True)
				with col2:
					st.markdown("")
					st.plotly_chart(figptslostabs, use_container_width=True)

			with tab2:
				col1,col2 = st.columns([8,1]) #create collumns of uneven width
				with col1:
					st.plotly_chart(figauc, use_container_width=True)
				with col2:
					st.markdown("")
					st.plotly_chart(figptslostperc, use_container_width=True)



	if SubFeature == "DemandActivity":

		dfbid = loadauctionbiddata()[demandsheet].replace('-', np.nan, regex = True)

		optiontype = st.sidebar.radio('Click an Option', ["Aggregate Demand", "Excess Demand"])

		if optiontype == "Aggregate Demand":

			dfbidaAD = dfbid.pivot(index="LSA", columns='Clock Round', values="Aggregate Demand").sort_index(ascending=True)

			dfbidaBlksSale = dfbid.pivot(index="LSA", columns='Clock Round', values="Blocks For Sale").sort_index(ascending=True)

			ADPrecOfBlksforSale = round((dfbidaAD/dfbidaBlksSale.values),1)


			#summary chart for total blocks for sale on right

			blocksforsale = dfbidaBlksSale.iloc[:,0].sort_index(ascending = False).reset_index()

			blocksforsale.columns = ["LSA", "Blocks"]

			figblkssale = px.bar(blocksforsale, x="Blocks", y="LSA", orientation='h', height = 615) #plotly horizontal bar chart 

			figblkssale.update_layout(xaxis=dict(title='BlocksForSale'), yaxis=dict(title=''))

			figblkssale.update_traces(text=blocksforsale['Blocks'], textposition='auto')

			figblkssale.update_xaxes(tickvals=[])

			figblkssale.update_layout(xaxis=dict(side='top', title_standoff=0, ticklen=0, title_font=dict(size=14)))

			figblkssale.update_layout(xaxis_title_standoff=5)

			figblkssale.update_traces(marker=dict(color='red'))

			figblkssale.update_yaxes(visible=False, showticklabels=False)


			hovertext = htext_auctiondata_2010_3G_BWA_DemandIntensity(dfbid,ADPrecOfBlksforSale)

			data1 = [go.Heatmap(
						z=dfbidaAD.values,
				        y= dfbidaAD.index,
				        x=dfbidaAD.columns,
						xgap = 0.5,
						ygap = 1,
						hoverinfo ='text',
						text = hovertext,
						colorscale='Hot',
						# zmin=0.5, zmax=1,
						showscale=False,
							texttemplate=texttempbiddemandactivity, 
							textfont={"size":12},
							reversescale=True,
							)]

			data2 = [go.Heatmap(
						z=ADPrecOfBlksforSale.values,
				        y= ADPrecOfBlksforSale.index,
				        x=ADPrecOfBlksforSale.columns,
						xgap = 0.5,
						ygap = 1,
						hoverinfo ='text',
						text = hovertext,
						colorscale='Hot',
						# zmin=0.5, zmax=1,
						showscale=False,
							texttemplate=texttempbiddemandactivity, 
							textfont={"size":8},
							reversescale=True,
							)]
						

			figauc1 = go.Figure(data=data1)
			figauc2 = go.Figure(data=data2)

			figauc1.update_layout(uniformtext_minsize=12, 
			  uniformtext_mode='hide', 
			  xaxis_title=None, 
			  yaxis_title=None, 
			  yaxis_autorange='reversed',
			  font=dict(size=12),
			  template='simple_white',
			  paper_bgcolor=None,
			  height=600, 
			  width=1200,
			  margin=dict(t=80, b=50, l=0, r=0, pad=0),
			  yaxis=dict(
	        	  tickmode='array'),
			  xaxis = dict(
			  side = 'top',
			  tickmode = 'linear',
			  tickangle=0,
			  dtick = xdtick), 
			)

			figauc2.update_layout(uniformtext_minsize=12, 
			  uniformtext_mode='hide', 
			  xaxis_title=None, 
			  yaxis_title=None, 
			  yaxis_autorange='reversed',
			  font=dict(size=12),
			  template='simple_white',
			  paper_bgcolor=None,
			  height=600, 
			  width=1200,
			  margin=dict(t=80, b=50, l=0, r=0, pad=0),
			  yaxis=dict(
	        	  tickmode='array'),
			  xaxis = dict(
			  side = 'top',
			  tickmode = 'linear',
			  tickangle=0,
			  dtick = xdtick), 
			)

			title = titlesubpart+" - Aggregrated Demand in Various Rounds"
			subtitle = "Unit - Nos; Source - DoT; Xaxis - Round Numbers"

			style = "<style>h3 {text-align: left;}</style>"
			with st.container():
				#plotting the main chart
				st.markdown(style, unsafe_allow_html=True)
				st.header(title)
				st.markdown(subtitle)


			#Drawning a black border around the heatmap chart 
			figauc1.update_xaxes(fixedrange=True,showline=True,linewidth=1.2,linecolor='black', mirror=True)
			figauc1.update_yaxes(fixedrange=True,showline=True, linewidth=1.2, linecolor='black', mirror=True)

			figauc2.update_xaxes(fixedrange=True,showline=True,linewidth=1.2,linecolor='black', mirror=True)
			figauc2.update_yaxes(fixedrange=True,showline=True, linewidth=1.2, linecolor='black', mirror=True)

			figauc1.update_layout(
				    xaxis=dict(showgrid=False),
				    yaxis=dict(showgrid=False)
				)

			figauc2.update_layout(
				    xaxis=dict(showgrid=False),
				    yaxis=dict(showgrid=False)
				)

			hoverlabel_bgcolor = "#000000" #subdued black

			figauc1.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))

			figauc2.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))


			tab1, tab2 = st.tabs(["Aggregate Demand", "Ratio (AD/BLKsForSale)"]) #For showning the absolute and Ratio charts in two differet tabs
			with tab1:
				col1,col2 = st.columns([8,1]) #create collumns of uneven width
				with col1:
					st.plotly_chart(figauc1, use_container_width=True)
				with col2:
					st.markdown("")
					st.plotly_chart(figblkssale, use_container_width=True)

			with tab2:
				col1,col2 = st.columns([8,1]) #create collumns of uneven width
				with col1:
					st.plotly_chart(figauc2, use_container_width=True)
				with col2:
					st.markdown("")
					st.plotly_chart(figblkssale, use_container_width=True)

		

		if optiontype == "Excess Demand":

			dfbidaED = dfbid.pivot(index="LSA", columns='Clock Round', values="Excess Demand").sort_index(ascending=True)

			dfbidaAD = dfbid.pivot(index="LSA", columns='Clock Round', values="Aggregate Demand").sort_index(ascending=True)

			dfbidaBlksSale = dfbid.pivot(index="LSA", columns='Clock Round', values="Blocks For Sale").sort_index(ascending=True) 

			ADPrecOfBlksforSale = round((dfbidaAD/dfbidaBlksSale.values),1)

			hovertext = htext_auctiondata_2010_3G_BWA_DemandIntensity(dfbid, ADPrecOfBlksforSale)

			data = [go.Heatmap(
						z=dfbidaED.values,
				        y= dfbidaED.index,
				        x=dfbidaED.columns,
						xgap = 0.5,
						ygap = 1,
						hoverinfo ='text',
						text = hovertext,
						colorscale='Hot',
						# zmin=0.5, zmax=1,
						showscale=True,
							texttemplate=texttempbiddemandactivity, 
							textfont={"size":12},
							reversescale=True,
							)]
						

			figauc = go.Figure(data=data)

			figauc.update_layout(uniformtext_minsize=12, 
			  uniformtext_mode='hide', 
			  xaxis_title=None, 
			  yaxis_title=None, 
			  yaxis_autorange='reversed',
			  font=dict(size=12),
			  template='simple_white',
			  paper_bgcolor=None,
			  height=600, 
			  width=1200,
			  margin=dict(t=80, b=50, l=50, r=50, pad=0),
			  yaxis=dict(
	        	  tickmode='array'),
			  xaxis = dict(
			  side = 'top',
			  tickmode = 'linear',
			  tickangle=0,
			  dtick = xdtick), 
			)

			title = titlesubpart+" - Excess Demand in Various Rounds"
			subtitle = "Unit - Nos; Source - DoT; Xaxis - Round Numbers"

			style = "<style>h3 {text-align: left;}</style>"
			with st.container():
				#plotting the main chart
				st.markdown(style, unsafe_allow_html=True)
				st.header(title)
				st.markdown(subtitle)


			#Drawning a black border around the heatmap chart 
			figauc.update_xaxes(fixedrange=True,showline=True,linewidth=1.2,linecolor='black', mirror=True)
			figauc.update_yaxes(fixedrange=True,showline=True, linewidth=1.2, linecolor='black', mirror=True)

			figauc.update_layout(
				    xaxis=dict(showgrid=False),
				    yaxis=dict(showgrid=False)
				)

			hoverlabel_bgcolor = "#000000" #subdued black

			figauc.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))


			st.plotly_chart(figauc, use_container_width=True)



	if SubFeature == "LastBidPrice":

		dfbid = loadauctionbiddata()[demandsheet].replace('-', np.nan, regex = True) #for number of blocks for sale for hovertext

		dfbid = dfbid.drop(columns =["Clock Round", "Clock Round Price (Rs. Crore)","Aggregate Demand", "Excess Demand"], axis =1).drop_duplicates()

		dfbid = dfbid.set_index("LSA")

		round_number = st.slider("Select Auction Round Numbers using the Silder below", min_value=1, max_value=totalrounds, step=1, value = totalrounds)

		dflastsubbid = dfbidori.copy()

		filt1 =(dflastsubbid["Clk_Round"] == round_number)

		filt2 = (dflastsubbid["Clk_Round"] == 2) #bids at start of 2nd round will give us RP when all circles were bid

		dflastsubbidRD2 = dflastsubbid[filt2]

		dflastsubbid = dflastsubbid[filt1]


		roundoption = st.sidebar.radio('Click an Option', ["Start of Round", "End of Round"])

		if roundoption == "Start of Round":

			dflastsubbidheat = dflastsubbid.reset_index().pivot(index="Bidder", columns='LSA', 
							values="Last_Sub_Bid_Start_CLKRd").sort_index(ascending=False).round(2)

			#provisionaally allocated blocks in the start of selected round

			dfBLKsStartRd = dflastsubbid.reset_index().pivot(index="Bidder", columns='LSA', 
			values="Prov_Alloc_BLK_Start_ClkRd").sort_index(ascending=True).round(0)


			#function to combine text from two dataframe 

			def combine_text(sep2, x, y, sep1): #sep is seperator
			    if x.notnull().all() and y.notnull().all():
			        return sep2 + x + '<br>' + sep1 + y
			    elif x.notnull().all():
			        return x
			    else:
			        return y

			#for rendering text of the final heatmap for Data1

			dflastsubbidheat = dflastsubbidheat.round(1)

			df_combined1 = dflastsubbidheat.applymap(str).combine(dfBLKsStartRd.applymap(str), lambda x, y: combine_text('Rs-', x, y, 'BA-'))


			dflastsubbidratio = round((dflastsubbidheat.T/dfrp.values).T,2).sort_index(ascending=True)


			#for rendering text of the final heatmap for Data2

			df_combined2 = dflastsubbidratio.applymap(str).combine(dfBLKsStartRd.applymap(str), lambda x, y: combine_text('Ratio-', x, y, 'BA-'))

			dflastsubbidheat = dflastsubbidheat.sort_index(ascending=True)


			hovertext, colormatrix = htext_colormatrix_auctiondata_2010_3G_BWA_LastBidPrice(dflastsubbidheat,dflastsubbidratio,dfbid)

			data1 = [go.Heatmap(
						z=dflastsubbidheat.values,
				        y= dflastsubbidheat.index,
				        x=dflastsubbidheat.columns,
						xgap = 0.5,
						ygap = 1,
						hoverinfo ='text',
						hovertext= hovertext,
						text = df_combined1.values,
						colorscale='reds',
						# zmin=0.5, zmax=1,
						showscale=True,
							texttemplate="%{text}", 
							textfont={"size":12},
							# reversescale=True,
							)]

			data2 = [go.Heatmap(
						z=dflastsubbidratio.values,
				        y= dflastsubbidratio.index,
				        x=dflastsubbidratio.columns,
						xgap = 0.5,
						ygap = 1,
						hoverinfo ='text',
						hovertext= hovertext,
						text = df_combined2.values,
						colorscale='reds',
						# zmin=0.5, zmax=1,
						showscale=True,
							texttemplate="%{text}", 
							textfont={"size":12},
							# reversescale=True,
							)]			

			figauc1 = go.Figure(data=data1)
			figauc2 = go.Figure(data=data2)


			figauc1.update_layout(uniformtext_minsize=12, 
			  uniformtext_mode='hide', 
			  xaxis_title=None, 
			  yaxis_title=None, 
			  yaxis_autorange='reversed',
			  font=dict(size=12),
			  template='simple_white',
			  paper_bgcolor=None,
			  height=600, 
			  width=1200,
			  margin=dict(t=80, b=50, l=50, r=50, pad=0),
			  yaxis=dict(
	        	  tickmode='array'),
			  xaxis = dict(
			  side = 'top',
			  tickmode = 'linear',
			  tickangle=0,
			  dtick = 1), 
			)

			figauc2.update_layout(uniformtext_minsize=12, 
			  uniformtext_mode='hide', 
			  xaxis_title=None, 
			  yaxis_title=None, 
			  yaxis_autorange='reversed',
			  font=dict(size=12),
			  template='simple_white',
			  paper_bgcolor=None,
			  height=600, 
			  width=1200,
			  margin=dict(t=80, b=50, l=50, r=50, pad=0),
			  yaxis=dict(
	        	  tickmode='array'),
			  xaxis = dict(
			  side = 'top',
			  tickmode = 'linear',
			  tickangle=0,
			  dtick = 1), 
			)

			title = titlesubpart+" - Last Submitted Bid (Start of Round No - "+ str(round_number)+")"
			subtitle = "Unit - Rs Cr (Except Ratio); BlockSize - "+str(blocksize)+" MHz; Source - DoT;"\
			" Winning Price - Min of Bid Value; Text below Bid Value:- (BA)- BLKS Allocated"

			style = "<style>h3 {text-align: left;}</style>"
			with st.container():
				#plotting the main chart
				st.markdown(style, unsafe_allow_html=True)
				st.header(title)
				st.markdown(subtitle)


			#Drawning a black border around the heatmap chart 
			figauc1.update_xaxes(fixedrange=True,showline=True,linewidth=1.2,linecolor='black', mirror=True)
			figauc1.update_yaxes(fixedrange=True,showline=True, linewidth=1.2, linecolor='black', mirror=True)
			figauc1.update_layout(
				    xaxis=dict(showgrid=False),
				    yaxis=dict(showgrid=False)
				)

			#Drawning a black border around the heatmap chart 
			figauc2.update_xaxes(fixedrange=True,showline=True,linewidth=1.2,linecolor='black', mirror=True)
			figauc2.update_yaxes(fixedrange=True,showline=True, linewidth=1.2, linecolor='black', mirror=True)
			figauc2.update_layout(
				    xaxis=dict(showgrid=False),
				    yaxis=dict(showgrid=False)
				)

			hoverlabel_bgcolor = colormatrix

			figauc1.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))

			figauc2.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))


			tab1,tab2 = st.tabs(["Absolute Value", "Ratio (Bid/Reserve)"])  #For showning the absolute and Ratio charts in two differet tabs
			tab1.plotly_chart(figauc1, use_container_width=True)
			tab2.plotly_chart(figauc2, use_container_width=True)

		if roundoption == "End of Round":

			dflastsubbidheat = dflastsubbid.reset_index().pivot(index="Bidder", columns='LSA', 
							values="Last_Sub_Bid_End_CLKRd").sort_index(ascending=False).round(2)

			#provisionaally allocated blocks in the start of selected round

			dfBLKsEndRd = dflastsubbid.reset_index().pivot(index="Bidder", columns='LSA', 
			values="Prov_Alloc_BLK_End_ClkRd").sort_index(ascending=True).round(0)

			dfBLKsSelEndRd = dflastsubbid.reset_index().pivot(index="Bidder", columns='LSA', 
			values="No_of_BLK_Selected").sort_index(ascending=True).round(0)

			#process for win and lost for heatmap color

			# Define a regular expression pattern to match numbers
			pattern = r'\b(?!0)\d+\b'

			# Function to replace numbers with 1, except 0
			def replace_numbers(match):
			    num = int(match.group())
			    if num != 0:
			        return '1'
			    else:
			        return str(num)

			# Apply the regular expression pattern and replacement function to the dataframe
			# mask dataframe has 1 for winners and 0 for losers
			mask = dfBLKsEndRd.applymap(lambda x: re.sub(pattern, replace_numbers, str(x)))

			for col in mask.columns:
				mask[col] = mask[col].astype(int)

			#function to combine text from two dataframe 

			def combine_text(sep2, x, y, sep1): #sep is seperator
			    if x.notnull().all() and y.notnull().all():
			        return sep2 + x + '<br>' + sep1 + y
			    elif x.notnull().all():
			        return x
			    else:
			        return y

			#for rendering text of the final heatmap for Data1

			dflastsubbidheat = dflastsubbidheat.round(1)

			df_combined1 = dflastsubbidheat.applymap(str).combine(dfBLKsSelEndRd.applymap(str), lambda x, y: combine_text('Rs-', x, y,'BS-'))

			df_combined1 = df_combined1.applymap(str).combine(dfBLKsEndRd.applymap(str), lambda x, y: combine_text("", x, y, 'BA-'))


			dflastsubbidratio = round((dflastsubbidheat.T/dfrp.values).T,2).sort_index(ascending=True)

			#for rendering text of the final heatmap for Data2

			df_combined2 = dflastsubbidratio.applymap(str).combine(dfBLKsSelEndRd.applymap(str), lambda x, y: combine_text('Ratio-', x, y, 'BS-'))

			df_combined2 = df_combined2.applymap(str).combine(dfBLKsEndRd.applymap(str), lambda x, y: combine_text("", x, y, 'BA-'))

			dflastsubbidheat = dflastsubbidheat.sort_index(ascending=True)


			#working for the summary bar chart at the side of the heatmap

			dfprovwinbid = dflastsubbid.reset_index().pivot(index="Bidder", columns='LSA', 
							values="Prov_Win_Price_End_ClkRd").sort_index(ascending=False).round(2)

			# Define a regular expression pattern to match numbers
			pattern = r'\b(?!0)\d+\b'

			# Function to replace numbers with 1, except 0
			def replace_numbers(match):
			    num = int(match.group())
			    if num != 0:
			        return '1'
			    else:
			        return str(num)

			# Apply the regular expression pattern and replacement function to the dataframe
			# To calculate the winners who are those who have been assigned blocks
			mask1 = dfBLKsEndRd.applymap(lambda x: re.sub(pattern, replace_numbers, str(x)))

			for col in mask.columns:
				mask1[col] = mask1[col].astype(int)

			dfwithbids = dfprovwinbid*mask.values #final datframe with actual submitted bids


			# To identify those bidders who have submitted bids during the auction
			mask2 = dflastsubbidheat.applymap(lambda x: re.sub(pattern, replace_numbers, str(x).split('.')[0]))

			for col in mask2.columns:
				mask2[col] = mask2[col].astype(int)

			# lst2=[]
			# for index in mask2.index:
			# 	lst1=[]
			# 	for col in mask2.columns:
			# 		mask1val = mask1.loc[index,col]
			# 		mask2val = mask2.loc[index,col]
			# 		if mask1val == mask2val:
			# 			lst1.append(1)
			# 		else:
			# 			lst1.append(2)
			# 	lst2.append(lst1)

			# mask2 = pd.DataFrame(lst2)

			# mask2.index = mask1.index
			# mask2.columns = mask1.columns

			# st.write(mask2)

			# #create a checkbox to filter winners from those who have bid in the auction

			with st.sidebar:

				check = st.checkbox('Filter Winners', value = False)

				if check:
					mask = mask1
				else:
					mask = mask2


			#plotting the barchart for row sums

			figsummry = plotbiddertotal(dfwithbids,dfBLKsEndRd)

			figsummry.update_yaxes(visible=False, showticklabels=False)

			figsummry.update_layout(height=615)

			#plotting the barchart for collumn sums


			df_value = (dfwithbids*dfBLKsEndRd.values).sum(axis=0).reset_index().round(0)

			df_value.columns = ["LSA", "Total"]

			figsumcols = summarychart(df_value, 'LSA', "Total")


			hovertext, colormatrix = htext_colormatrix_auctiondata_2010_3G_BWA_LastBidPrice(dflastsubbidheat,dflastsubbidratio,dfbid)


			data1 = [go.Heatmap(
						z=mask.values,
				        y=mask.index,
				        x=mask.columns,
						xgap = 0.5,
						ygap = 1,
						hoverinfo ='text',
						hovertext = hovertext,
						text = df_combined1.values,
						colorscale='Picnic',
						# zmin=0.5, zmax=1,
						showscale=False,
							texttemplate="%{text}", 
							textfont={"size":12},
							reversescale=True,
							)]

			data2 = [go.Heatmap(
						z=mask.values,
				        y= mask.index,
				        x=mask.columns,
						xgap = 0.5,
						ygap = 1,
						hoverinfo ='text',
						hovertext = hovertext,
						text = df_combined2.values,
						colorscale='Picnic',
						# zmin=0.5, zmax=1,
						showscale=False,
							texttemplate="%{text}", 
							textfont={"size":12},
							reversescale=True,
							)]			

			figauc1 = go.Figure(data=data1)
			figauc2 = go.Figure(data=data2)


			figauc1.update_layout(uniformtext_minsize=12, 
			  uniformtext_mode='hide', 
			  xaxis_title=None, 
			  yaxis_title=None, 
			  yaxis_autorange='reversed',
			  font=dict(size=12),
			  template='simple_white',
			  paper_bgcolor=None,
			  height=600, 
			  width=1200,
			  margin=dict(t=80, b=50, l=0, r=0, pad=0),
			  yaxis=dict(
	        	  tickmode='array'),
			  xaxis = dict(
			  side = 'top',
			  tickmode = 'linear',
			  tickangle=0,
			  dtick = 1), 
			)

			figauc2.update_layout(uniformtext_minsize=12, 
			  uniformtext_mode='hide', 
			  xaxis_title=None, 
			  yaxis_title=None, 
			  yaxis_autorange='reversed',
			  font=dict(size=12),
			  template='simple_white',
			  paper_bgcolor=None,
			  height=600, 
			  width=1200,
			  margin=dict(t=80, b=50, l=0, r=0, pad=0),
			  yaxis=dict(
	        	  tickmode='array'),
			  xaxis = dict(
			  side = 'top',
			  tickmode = 'linear',
			  tickangle=0,
			  dtick = 1), 
			)

			title = titlesubpart+" - Last Submitted Bid (End of Round No - "+ str(round_number)+")"
			subtitle = "Unit - Rs Cr (Except Ratio); BlockSize - "+str(blocksize)+" MHz; Source - DoT;"\
			" Winners - BLUE; Loosers - RED; Text Below Bid Value:- (BS)- Blocks Selected; (BA)- Blocks Allocated"

			style = "<style>h3 {text-align: left;}</style>"
			with st.container():
				#plotting the main chart
				st.markdown(style, unsafe_allow_html=True)
				st.header(title)
				st.markdown(subtitle)


			#Drawning a black border around the heatmap chart 
			figauc1.update_xaxes(fixedrange=True,showline=True,linewidth=1.2,linecolor='black', mirror=True)
			figauc1.update_yaxes(fixedrange=True,showline=True, linewidth=1.2, linecolor='black', mirror=True)
			figauc1.update_layout(
				    xaxis=dict(showgrid=False),
				    yaxis=dict(showgrid=False)
				)

			#Drawning a black border around the heatmap chart 
			figauc2.update_xaxes(fixedrange=True,showline=True,linewidth=1.2,linecolor='black', mirror=True)
			figauc2.update_yaxes(fixedrange=True,showline=True, linewidth=1.2, linecolor='black', mirror=True)
			figauc2.update_layout(
				    xaxis=dict(showgrid=False),
				    yaxis=dict(showgrid=False)
				)

			hoverlabel_bgcolor = colormatrix

			figauc1.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))
			figauc2.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))


			tab1,tab2 = st.tabs(["Absolute Value", "Ratio (Bid/Reserve)"])  #For showning the absolute and Ratio charts in two differet tabs

			with tab1:
				col1,col2 = st.columns([8,1]) #create collumns of uneven width
				with col1:
					st.plotly_chart(figauc1, use_container_width=True)
					st.altair_chart(figsumcols, use_container_width=True)

				with col2:
					st.markdown("")
					st.plotly_chart(figsummry, use_container_width=True)

			with tab2:
				col1,col2 = st.columns([8,1]) #create collumns of uneven width
				with col1:
					st.plotly_chart(figauc2, use_container_width=True)
					st.altair_chart(figsumcols, use_container_width=True)
				with col2:
					st.markdown("")
					st.plotly_chart(figsummry, use_container_width=True)



#---------------New Auction Bid Data Cide Ends Here----------------------



subfeature_dict ={"Quantum Offered" : "Sale (MHz)", "Quantum Sold": "Total Sold (MHz)", "Quantum Unsold" : "Total Unsold (MHz)", 
"Reserve Price" : "RP/MHz" , "Auction Price": "Auction Price/MHz", "Total EMD" : "Total EMD"} 
subfeature_list = ["Reserve Price", "Auction Price", "Auction/Reserve", "Quantum Offered", 
"Quantum Sold","Percent Sold", "Quantum Unsold", "Percent Unsold", "Total EMD", "Total Outflow"]

#Processing For Dimension = "Auction Year"
if selected_dimension == "Auction Years":


	radio_currency = st.sidebar.radio('Click Currency', ["Rupees", "US Dollars"])

	if radio_currency == "Rupees":
		currency_flag = True
	if radio_currency == "US Dollars":
		currency_flag = False


	df = loadspectrumfile()

	#loading data
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

			if currency_flag == False: #USD
				z = np.around(df1_temp1.values/auction_rsrate_dict[Year]*10,2)
				x = df1_temp1.columns
				y = df1_temp1.index
				summarydf = round(df1_temp1.sum()/auction_rsrate_dict[Year]*10,1)
			if currency_flag == True: #Rupee
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

			if currency_flag == False: #USD
				z = np.around(df1_temp2.values/auction_rsrate_dict[Year]*10,2)
				x = df1_temp2.columns
				y = df1_temp2.index
				summarydf = round(df1_temp2.sum()/auction_rsrate_dict[Year]*10,1)
			if currency_flag == True: #Rupee
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
			currency_flag = True #default

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
			currency_flag = True #default

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
			currency_flag = True #default
			
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
			SummaryFlag = True
			
		hovertext,colormatrix = htext_colmatrix_auction_year_band_metric(df1) #processing hovertext and colormatrix for bandwise in cal year dim
		hoverlabel_bgcolor = colormatrix #colormatrix processed from fuction "hovertext_and_colmatrix" for same above


	if Feature == "Operator Metric": #for the dimension "Auction Years"
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
			
			if currency_flag == False: #USD
				z = np.around(df2_temp1.values/auction_rsrate_dict[Year]*10,2)
				x = df2_temp1.columns
				y = df2_temp1.index
				summarydf = round(df2_temp1.sum(axis=0)//auction_rsrate_dict[Year]*10,2)

			if currency_flag == True: #Rupees

				z = df2_temp1.values
				x = df2_temp1.columns
				y = df2_temp1.index
				summarydf = df2_temp1.sum(axis=0)


			summarydf = summarydf.reset_index()
			summarydf.columns = ["Operators", SubFeature] 
			summarydf = summarydf.sort_values("Operators", ascending = False)
			#preparing the summary chart 
			chart = summarychart(summarydf, 'Operators', SubFeature)
			SummaryFlag = True
			
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
			SummaryFlag = True
			
			#processing hovertext and colormatrix for operator wise in cal year dim
			hovertext,colormatrix = htext_colmatrix_auction_year_operator_metric(df1, selectedbands, SubFeature, df2_temp2)
			hoverlabel_bgcolor = colormatrix #colormatrix processed from fuction "hovertext_and_colmatrix" for same above

			currency_flag = True #default

	data = [go.Heatmap(
		  z = z,
		  x = x,
		  y = y,
		  xgap = 1,
		  ygap = 1,
		  hoverinfo ='text',
		  text = hovertext,
		  colorscale = 'Hot',
		    texttemplate="%{z}", 
		    textfont={"size":12},
		    reversescale=True,
			)]

	fig = go.Figure(data=data)


#This is section is to visulize important data related to the telecom industry (may not be directed related to spectrum)

if selected_dimension == "Business Data":

	currency_flag = True #default

	dfT = loadtelecomdatafile()
	
	Feature = st.sidebar.selectbox('Select a Feature', ["5GBTS Trends", "Subscriber Trends", "Subscriber MShare", "License Fees", "TowerBTS Trends",
														"Financial SPWise", "Financial LSAWise", "Subs RuralUrban"])

	if Feature== "5GBTS Trends":


		df5gbts = dfT["5GBTS"] #load 5G BTS deployment data from excel file

		df5gbtsf = pd.pivot(df5gbts, values ="Total", index = "StateCode", columns = "Date")

		df5gbtsf.columns = [str(x) for x in df5gbtsf.columns ] #convet the dates into string 

		lastcolumn = df5gbtsf.columns[-1]


		df5gbtsfall = df5gbtsf.copy()

		df5gbtsf = df5gbtsf.sort_values(lastcolumn, ascending = False).head(20) #sort by the last column

		df5gbtsf = round(df5gbtsf/1000,2) #convert the BTS data in thousands (K)

		df5gbtsf = df5gbtsf.iloc[:,-16:] #select on last 16 dates

		#converting the columns into datetime and then date

		df5gbtsf.columns = pd.to_datetime(df5gbtsf.columns)
		df5gbtsf.columns= [x.date() for x in list(df5gbtsf.columns)]


		SubFeature = st.sidebar.selectbox('Select a SubFeature', ["Cumulative Values", "Percent of Total", "Incremental Values"])

		if SubFeature == "Cumulative Values":


			hovertext = htext_businessdata_5gbts(df5gbtsf)

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
			fig = go.Figure(data=data)

			summarydf = round(df5gbtsfall.sum(axis=0)/1000,2) #debug
			summarydf = summarydf.reset_index()
			summarydf.columns = ["Dates", SubFeature] 
			summarydf = summarydf.sort_values("Dates", ascending = False)
			#preparing the summary chart 
			chart = summarychart(summarydf, 'Dates', SubFeature)
			SummaryFlag = True
			

		if SubFeature == "Percent of Total":


			hovertext = htext_businessdata_5gbts(df5gbtsf)

			summarydf = df5gbtsfall.sum(axis=0).sort_index(ascending=False).head(16) #debug

			df5gbtsf = df5gbtsf.head(16)

			summarydf = summarydf[::-1].reset_index().T

			summarydf.columns = list(summarydf.iloc[0,:])

			summarydf = summarydf.iloc[1:,:]/1000

			summarydf.columns = pd.to_datetime(summarydf.columns)
			summarydf.columns= [x.date() for x in list(summarydf.columns)]



			df5gbtsfPercent = (df5gbtsf/summarydf.values)*100


			# df5gbtsfPercent = round((df5gbtsf/summarydf.T)*100,2)


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
					texttemplate="%{z:.2f}", 
					textfont={"size":10},
					reversescale=True,
					),
				]
			fig = go.Figure(data=data)

			SummaryFlag = False #No summary chart to plot


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


			hovertext = htext_businessdata_5gbts(df5gbtsf)

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

			fig = go.Figure(data=data)

			summarydf = df5gbtsincf.sum(axis=0)
			summarydf = summarydf.reset_index()
			summarydf.columns = ["Dates", SubFeature] 
			summarydf = summarydf.sort_values("Dates", ascending = False)
			#preparing the summary chart 
			chart = summarychart(summarydf, 'Dates', SubFeature)
			SummaryFlag = True



	if Feature== "Subscriber Trends":

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


		list_of_circles = list(state_dict.values())

		selected_circles = st.sidebar.multiselect('Select Circles', list_of_circles) #drop down to select circles


		selected_circle_codes = [k for k, v in state_dict.items() if v in selected_circles]

		if len(selected_circles) > 0:

			temp = pd.DataFrame()
			for circle_code in selected_circle_codes:

				temp = pd.concat([temp, dftotal[dftotal["Circle"]==circle_code]], axis =0)

			dftotal = temp

		else:
			pass


		dftotal = dftotal.groupby(["Date","Operator","Circle"]).sum()

		dftotal = dftotal.reset_index()

		dftotal.drop(columns = ["Circle"], axis =1, inplace = True)

		dftotal = dftotal.groupby(["Date","Operator"]).sum()

		dftotal = dftotal.reset_index()


		dftotal = pd.pivot(dftotal, index="Operator", columns = "Date", values = "Subs")

		SubFeature = st.sidebar.radio('Click an Option', ["Cumulative Values", "Incremental Values"])

		if SubFeature=="Cumulative Values":

			listofallcolumns = list(dftotal.columns)


			# with st.sidebar:

			# 	start_date, end_date = st.select_slider("Select a Range of Dates", 
			# 		options = listofallcolumns, value =(dftotal.columns[-24],dftotal.columns[-1]))

			start_date, end_date = st.select_slider("Select a Range of Dates", 
				options = listofallcolumns, value =(dftotal.columns[-18],dftotal.columns[-1]))


			date_range_list = get_selected_date_list(listofallcolumns, start_date, end_date)


			dftotalfilt = dftotal[date_range_list] #filter the dataframe with the selected dates


			dftotalfilt = dftotalfilt.sort_values(end_date, ascending = False) #filter the data on the first column selected by slider


			dftotalfilt = round(dftotalfilt.loc[~(dftotalfilt ==0).all(axis=1)]/1000000,2) # delete all rows with value zero and convert into millions

			if len(selected_category) ==0:
				selected_category = ["All"]

			if len(selected_circles) == 0:
				selected_circles = ["All"]


			if len(date_range_list) >=24:
				texttemplate =""
			else:
				texttemplate = "%{z}"


			hovertext = htext_businessdata_telesubscum(dftotalfilt)

			#setting the data of the heatmap 

			data = [go.Heatmap(
				z = dftotalfilt.values,
				y = dftotalfilt.index,
				x = dftotalfilt.columns,
				xgap = 1,
				ygap = 1,
				hoverinfo ='text',
				text = hovertext,
				colorscale='reds',
					texttemplate=texttemplate, 
					textfont={"size":10},
					# reversescale=True,
					),
				]

			fig = go.Figure(data=data)

			# dftotalfilt = (dftotalfilt/1000).round(1)
			# summarydf = dftotalfilt.sum(axis=0)
			# summarydf = summarydf.reset_index()
			# summarydf.columns = ["Dates", Feature] 
			# summarydf = summarydf.sort_values("Dates", ascending = False)
			# #preparing the summary chart 
			# chart = summarychart(summarydf/1000, 'Dates', Feature)
			# SummaryFlag = True

		if SubFeature=="Incremental Values":


			lst =[]
			for row in dftotal.values:

				increments = np.diff(row)
				lst.append(increments)

			dftotalinc = pd.DataFrame(lst)

			dftotalinc.index = dftotal.index 
			dftotalinc.columns = dftotal.columns[1:]

			listofallcolumns = list(dftotalinc.columns)


			# with st.sidebar:

			# 	start_date, end_date = st.select_slider("Select a Range of Dates", 
			# 		options = listofallcolumns, value =(dftotalinc.columns[-24],dftotalinc.columns[-1]))


			start_date, end_date = st.select_slider("Select a Range of Dates", 
					options = listofallcolumns, value =(dftotalinc.columns[-18],dftotalinc.columns[-1]))


			date_range_list = get_selected_date_list(listofallcolumns, start_date, end_date)


			dftotalincfilt = dftotalinc[date_range_list] #filter the dataframe with the selected dates


			dftotalincfilt = dftotalincfilt.sort_values(end_date, ascending = False) #filter the data on the first column selected by slider


			dftotalincfilt = round(dftotalincfilt.loc[~(dftotalincfilt ==0).all(axis=1)]/1000000,2) # delete all rows with value zero and convert into millions

			if len(selected_category) ==0:
				selected_category = ["All"]

			if len(selected_circles) == 0:
				selected_circles = ["All"]

			if len(date_range_list) >=24:
				texttemplate =""
			else:
				texttemplate = "%{z}"


			hovertext = htext_businessdata_telesubsinc(dftotalincfilt)

			#setting the data of the heatmap 

			data = [go.Heatmap(
				z = dftotalincfilt.values,
				y = dftotalincfilt.index,
				x = dftotalincfilt.columns,
				xgap = 1,
				ygap = 1,
				hoverinfo ='text',
				text = hovertext,
				colorscale='reds',
					texttemplate=texttemplate, 
					textfont={"size":10},
					reversescale=True,
					),
				]

			fig = go.Figure(data=data)

	if Feature== "Subscriber MShare":

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


		dftotal = dftotal[dftotal["Date"]==sorted(list(set(dftotal["Date"].values)))[-1]] # filtering the dataframe on the latest date

		dftotal.drop(columns = ["Date"], axis =1, inplace = True)

		dftotal = dftotal.groupby(["Circle", "Operator"])["Subs"].sum().reset_index()


		dftotal = pd.pivot(dftotal, values = 'Subs', index='Operator' , columns = 'Circle')

		dftotal= dftotal.loc[~(dftotal == 0).all(axis=1)]

		dftotal["Total"] = dftotal.sum(axis=1)

		dftotal = dftotal.sort_values("Total", ascending = False)

		dftotal.drop(columns = ["Total"], axis =1, inplace = True)

		summarydf = dftotal.sum(axis=0)

		dftotalpercentms = round((dftotal/summarydf)*100,2)


		hovertext = htext_businessdata_telesubsms(dftotal,dftotalpercentms)


		data = [go.Heatmap(
				z = dftotalpercentms.values,
				y = dftotalpercentms.index,
				x = dftotalpercentms.columns,
				xgap = 1,
				ygap = 1,
				hoverinfo ='text',
				text = hovertext,
				colorscale='reds',
					texttemplate="%{z}", 
					textfont={"size":10},
					# reversescale=True,
					),
				]

		fig = go.Figure(data=data)

		summarydf= round(summarydf/1000000,1) # converting the numbers to million
		summarydf = summarydf.reset_index()
		summarydf.columns = ["Circle", "Total Subs"]

		# summarydf = summarydf.sort_values("Dates", ascending = False)

		#preparing the summary chart 
		chart = summarychart(summarydf, 'Circle', "Total Subs")
		SummaryFlag = True


		if len(selected_category) ==0:
			selected_category = ["All"]


	if Feature == "License Fees":

		dflfsf = dfT["LFSF"]

		dfoperatornames = dfT["LFSF_Op_Names_Map"]

		def dataframe_to_dictionary(df):
		    dictionary = {}
		    for index, row in df.iterrows():
		        key = row[0]
		        value = row[1]
		        dictionary[key] = value
		    return dictionary

		operator_dict = dataframe_to_dictionary(dfoperatornames)
		

		dflfsf = dflfsf.replace(r'[^A-Za-z0-9\-()/\s.]','', regex=True)


		dflfsfprocess = dflfsf.copy()

		dflfsfprocess = dflfsfprocess.replace(operator_dict)

		listoflicensetypes = sorted(list(set(dflfsf["LicenseType"])))

		listofFY = sorted(list(set(dflfsf["FY"])), reverse = True)


		selected_category = st.sidebar.multiselect('Select Categories', ["LF", "SF"])

		if (len(selected_category)==0) or (len(selected_category)==2):

			pass
		
		else:

			dflfsfprocess = dflfsf[dflfsf["Category"]==selected_category[0]]

		subfeature_list = ["Operators", "LicenseType"]


		SubFeature = st.sidebar.selectbox('Select a SubFeature', subfeature_list,0)

		if SubFeature == "Operators":

			column_to_drop = "LicenseType"

		if SubFeature == "LicenseType":

			column_to_drop = "Operators"

		#This is for creating a list of 30 important operators 

		if SubFeature == "LicenseType":

			#sort operators by the last FY, and might get revised later

			sorted_df = dflfsfprocess[dflfsfprocess["FY"]=="2022-2023"]

			sorted_df = sorted_df.groupby(["Operators","FY"]).sum().reset_index().sort_values(by='Amount', ascending=False)


			sorted_operators = sorted_df.head(29)["Operators"].tolist() #pick on 29 operators and add BSNL to list later

			sorted_operators = ["BSNL"]+sorted_operators

			#Filtering the viz by the selected operator

			selected_operators = st.sidebar.multiselect('Select Operators', sorted_operators)

			if len(selected_operators)==0:

				selected_operators = sorted_operators

			else:
				pass

			#Filter the dataframe with the list of selected operators

			temp = pd.DataFrame()
			for operator in selected_operators:

				df = dflfsfprocess[dflfsfprocess["Operators"]==operator]

				temp = pd.concat([temp, df], axis =0)

			dflfsfprocess = temp

		else:

			pass


		dflfsfprocess = dflfsfprocess.groupby([SubFeature,'FY']).sum().drop(columns=['Category', column_to_drop], axis =1).reset_index()

		selected_fy_for_sort = st.sidebar.selectbox('Select FY for Sorting', listofFY)


		dflfsfbysubfeature = round(dflfsfprocess.pivot(index =SubFeature, columns ='FY', values ='Amount')
								.sort_values(selected_fy_for_sort, ascending = False)/10000000,0)

		chosen_metric = st.sidebar.radio('Click an Option', ["Absolute", "Percentage"])

		summarydf = dflfsfbysubfeature.sum(axis =0)

		summarydf_for_hovertext = summarydf.copy()

		if chosen_metric=="Absolute":

			df = dflfsfbysubfeature.head(20).copy()

		if chosen_metric=="Percentage":

			df = round(((dflfsfbysubfeature/summarydf).head(20))*100,2)


		#preparing the summary chart 

		summarydf = summarydf.reset_index()
		summarydf.columns = ["FY", "Total Fees"]

		# summarydf = summarydf.sort_values("Dates", ascending = False)

		#preparing the summary chart 
		chart = summarychart(summarydf, 'FY', 'Total Fees')
		SummaryFlag = True

		dflfsfbysubfeature = dflfsfbysubfeature.head(20)

		hovertext = htext_businessdata_licensefees(dflfsfbysubfeature,summarydf_for_hovertext)

		data = [go.Heatmap(
				z = df.values,
				y = df.index,
				x = df.columns,
				xgap = 1,
				ygap = 1,
				hoverinfo ='text',
				text = hovertext,
				colorscale='reds',
					texttemplate="%{z:.2f}", 
					textfont={"size":10},
					# reversescale=True,
					),
				]
		fig = go.Figure(data=data)



	if Feature == "Financial SPWise":


		df = loadtraiagr()


		df_rev = df["TRAI_Financial"]

		df_rev["Date"] = pd.to_datetime(df_rev["Date"]).dt.date

		list_of_dates = sorted(list(set(df_rev["Date"])))[11:]


		df_rev = df_rev.set_index("Date")


		for col in ["GR","APGR", "AGR", "LF", "SF"]:

			df_rev[col] = pd.to_numeric(df_rev[col], errors='coerce') #to convert rougue strings (errors during PDF conversion) into numeric

		df_rev = df_rev.reset_index()

		df_rev.rename(columns = {"index":"Date"}, inplace = True)

		df_rev.drop(columns=["License", "Year", "Month","Circle","Dollar Rate"], inplace = True)

		
		df_temp = df_rev.groupby(["Date","Operator"]).agg({"GR":'sum','APGR':'sum','AGR':'sum','LF':'sum','SF':'sum'})\
					.sort_index(ascending = False).sort_values("GR", ascending = False).round(0)


		start_date, end_date = st.select_slider("Select a Range of Dates", 
			options = list_of_dates, value =(list_of_dates[0],list_of_dates[-1]))


		df_temp = df_temp.reset_index()

		filt = (df_temp["Date"] >= start_date) & (df_temp["Date"] <= end_date) 

		df_temp = df_temp[filt]

		finmetric = st.sidebar.selectbox("Select from Options", ["GrossRevenue", "ApplicableRev", "AdjustedGR", "LicenseFee", "SpectrumFee"])

		fin_dic = {'GrossRevenue':'GR', 'ApplicableRev':'APGR','AdjustedGR':'AGR','LicenseFee':'LF', 'SpectrumFee': 'SF'}


		df_finmetric = (df_temp.pivot(index ="Operator", columns ="Date", values =fin_dic[finmetric])/1000).round(1)

		df_finmetric = df_finmetric.sort_values(df_finmetric.columns[-1], ascending = False)

		df_finmetricINC = (df_finmetric - df_finmetric.shift(1, axis =1)).head(15)

		summarydf_INC = df_finmetricINC.sum(axis=0).round(1)

		summarydf_INC = summarydf_INC.reset_index()

		summarydf_INC.columns = ["Date", "IndiaTotal"]

		summarydf = df_finmetric.sum(axis=0)

		df_finmetric_prec = round((df_finmetric/summarydf.values)*100,1).head(15)
		df_finmetric = df_finmetric.head(15)

		summarydf = summarydf.reset_index()

		summarydf.columns = ["Date", "IndiaTotal"]


		radio_selection = st.sidebar.radio('Click an Option', ["Absolute Values", "Percentage of Total", "Quarterly Increments"])

		if radio_selection == "Absolute Values":
			df_heatmap = df_finmetric.copy()
			summarydf = summarydf.copy()
		if radio_selection == "Percentage of Total":
			df_heatmap = df_finmetric_prec.copy()
			summarydf = summarydf.copy()
		if radio_selection == "Quarterly Increments":
			df_heatmap = df_finmetricINC.iloc[:,1:].copy()
			summarydf = summarydf_INC.iloc[1:,:].copy()



		#preparing the summary chart 
		chart = summarychart(summarydf, "Date", "IndiaTotal")
		SummaryFlag = True #for ploting the summary chart

		hovertext = htext_businessdata_FinancialSPWise(df_finmetric,df_finmetric_prec,df_finmetricINC)

		data = [go.Heatmap(
				z = df_heatmap.values,
				y = df_heatmap.index,
				x = df_heatmap.columns,
				xgap = 1,
				ygap = 1,
				hoverinfo ='text',
				text = hovertext,
				colorscale='Hot',
					texttemplate="%{z}", 
					textfont={"size":8},
					reversescale=True,
					),
				]
		fig = go.Figure(data=data)



#------------------------New Code ----------------------


	if Feature == "Financial LSAWise":


		df = loadtraiagr()


		df_rev = df["TRAI_Financial"]

		df_rev["Date"] = pd.to_datetime(df_rev["Date"]).dt.date

		list_of_dates = sorted(list(set(df_rev["Date"])))[11:]


		df_rev = df_rev.set_index("Date")


		for col in ["GR","APGR", "AGR", "LF", "SF"]:

			df_rev[col] = pd.to_numeric(df_rev[col], errors='coerce') #to convert rougue strings (errors during PDF conversion) into numeric

		df_rev = df_rev.reset_index()

		df_rev.rename(columns = {"index":"Date"}, inplace = True)

		selected_operator = st.sidebar.selectbox("Select from Options", ["RJIO", "Bharti", "Vodafone Idea", "BSNL"])

		df_rev = df_rev[df_rev["Operator"]==selected_operator]

		df_rev.drop(columns=["License", "Year", "Month","Operator","Dollar Rate"], inplace = True)

		
		df_temp = df_rev.groupby(["Date","Circle"]).agg({"GR":'sum','APGR':'sum','AGR':'sum','LF':'sum','SF':'sum'})\
					.sort_index(ascending = False).sort_values("GR", ascending = False).round(0)


		start_date, end_date = st.select_slider("Select a Range of Dates", 
			options = list_of_dates, value =(list_of_dates[0],list_of_dates[-1]))


		df_temp = df_temp.reset_index()

		filt = (df_temp["Date"] >= start_date) & (df_temp["Date"] <= end_date) 

		df_temp = df_temp[filt]

		finmetric = st.sidebar.selectbox("Select from Options", ["GrossRevenue", "ApplicableRev", "AdjustedGR", "LicenseFee", "SpectrumFee"])

		fin_dic = {'GrossRevenue':'GR', 'ApplicableRev':'APGR','AdjustedGR':'AGR','LicenseFee':'LF', 'SpectrumFee': 'SF'}


		df_finmetric = (df_temp.pivot(index ="Circle", columns ="Date", values =fin_dic[finmetric])/1000).round(1)

		df_finmetric = df_finmetric.sort_values(df_finmetric.columns[-1], ascending = False)

		df_finmetricINC = (df_finmetric - df_finmetric.shift(1, axis =1))

		summarydf_INC = df_finmetricINC.sum(axis=0).round(1)

		summarydf_INC = summarydf_INC.reset_index()

		summarydf_INC.columns = ["Date", "IndiaTotal"]

		summarydf = df_finmetric.sum(axis=0)

		df_finmetric_prec = round((df_finmetric/summarydf.values)*100,1)
		df_finmetric = df_finmetric

		summarydf = summarydf.reset_index()

		summarydf.columns = ["Date", "IndiaTotal"]


		radio_selection = st.sidebar.radio('Click an Option', ["Absolute Values", "Percentage of Total", "Quarterly Increments"])

		if radio_selection == "Absolute Values":
			df_heatmap = df_finmetric.copy()
			summarydf = summarydf.copy()
		if radio_selection == "Percentage of Total":
			df_heatmap = df_finmetric_prec.copy()
			summarydf = summarydf.copy()
		if radio_selection == "Quarterly Increments":
			df_heatmap = df_finmetricINC.iloc[:,1:].copy()
			summarydf = summarydf_INC.iloc[1:,:].copy()



		#preparing the summary chart 
		chart = summarychart(summarydf, "Date", "IndiaTotal")
		SummaryFlag = True #for ploting the summary chart



		data = [go.Heatmap(
				z = df_heatmap.values,
				y = df_heatmap.index,
				x = df_heatmap.columns,
				xgap = 1,
				ygap = 1,
				hoverinfo ='text',
				# text = hovertext,
				colorscale='Hot',
					texttemplate="%{z}", 
					textfont={"size":8},
					reversescale=True,
					),
				]
		fig = go.Figure(data=data)


#-------------------- New Code Ends ------------------------


#---------------------New Code Starts----------------------


	if Feature == "Subs RuralUrban":


		dfRU = dfT["TelecomSubsRuralUrban"] #load 5G BTS deployment data from excel file

		dfRU["Date"] = pd.to_datetime(dfRU["Date"]).dt.date

		rural = dfRU[dfRU["Type"]=="Rural"].drop(columns ="Type", axis=1).groupby("Date").sum().drop(columns="Category", axis=1).tail(20).T

		urban = dfRU[dfRU["Type"]=="Urban"].drop(columns ="Type", axis=1).groupby("Date").sum().drop(columns="Category", axis=1).tail(20).T

		rural = round(rural.sort_values(rural.columns[-1], ascending = False)/1000000,1)

		urban = round(urban.sort_values(urban.columns[-1], ascending = False)/1000000,1)

		rural_perc = round(rural/rural.sum(axis=0)*100,1).head(4)

		urban_perc = round(urban/urban.sum(axis=0)*100,1)
		


		data = [go.Heatmap(
				z = rural_perc.values,
				y = rural_perc.index,
				x = rural_perc.columns,
				xgap = 1,
				ygap = 1,
				hoverinfo ='text',
				# text = hovertext,
				colorscale='Hot',
					texttemplate="%{z}", 
					textfont={"size":8},
					reversescale=True,
					),
				]
		fig = go.Figure(data=data)


#-------------------- New Code Ends ------------------------



	if Feature == "TowerBTS Trends":


		dftowersbts = dfT["bts_towers"] #load 5G BTS deployment data from excel file

		# dftowersbts["Date"] = pd.to_datetime(dftowersbts["Date"])

		# dftowersbts['Date'] = dftowersbts['Date'].apply(lambda x: x.strftime('%B-%Y')) 

		dftowersbts = dftowersbts.set_index("Date")

		dftowersbts = dftowersbts.asfreq("m")

		dftowersbts.index = dftowersbts.index.strftime("%Y-%m")


		dftowersbts = dftowersbts.sort_values("Date", ascending=True)
		dftowersbts["Ratio"] = dftowersbts["BTS"] / dftowersbts["Towers"]

		trace1 = go.Scatter(x=dftowersbts.index, y=dftowersbts["Ratio"], name="BTSs/Towers", yaxis="y1", 
							textfont=dict(family="sans serif",size=8,color="DarkBlue"),
							mode = 'lines+markers+text',text=list(round(dftowersbts["Ratio"],1)),
							textposition="bottom center", showlegend = False, line = dict(color ='red'))
		trace2 = go.Scatter(x=dftowersbts.index, y=dftowersbts["BTS"], name="BTS Trends", yaxis="y2", 
							textfont=dict(family="sans serif",size=8,color="DarkBlue"),
							mode = 'lines+markers+text',text=list(round(dftowersbts["BTS"]/100000,1)),
							textposition="bottom center", showlegend = False, line = dict(color = 'green'))
		trace3 = go.Scatter(x=dftowersbts.index, y=dftowersbts["Towers"], name="Tower Trends", yaxis="y3", 
							textfont=dict(family="sans serif",size=8,color="DarkBlue"),
							mode = 'lines+markers+text',text=list(round(dftowersbts["Towers"]/1000,0)),
							textposition="bottom center", showlegend = False, line = dict(color = 'blue'))
		# trace3 = go.Scatter(x=dftowersbts.index, y=dftowersbts["Towers"], name="Tower Trends", yaxis="y3", mode = 'lines+markers',
		# 					showlegend = False, line = dict(color = 'blue'))


		data = [trace1, trace2, trace3] #Data for line chart stacked on top of each other

		figtowerbts = go.Figure(data=data)

		end_date = dt.datetime(2023, 6, 30)  # Use datetime.datetime instead of just datetime
		figtowerbts.update_xaxes(range=[dftowersbts.index[0], end_date], dtick=2)

		figtowerbts.update_layout(
			    # title='Multiple Line Charts',
			    yaxis=dict(
			        title='Ratio - BTS/Towers',
			        range=[2.5, 4],  # Set the range for y-axis 1
			        domain=[0, 0.27]
			    ),
			    yaxis2=dict(
			        title='BTS',
			        range=[1500000, 3000000],  # Set the range for y-axis 2
			        domain=[0.35, 0.62]
			    ),
			    yaxis3=dict(
			        title='Towers',
			        range=[400000, 800000],  # Set the range for y-axis 3
			        domain=[0.69, 1]
			    ),
			    xaxis=dict(
			        title='Date'
			    ),
			    height=900,
			    width=1000,
			)

		# st.plotly_chart(figtowerbts, use_container_width=True) #for towerdata


#----------------End of all Dimensions of Fig Data-----------------------------



#This section deals with titles and subtitles and hoverlabel color for all the heatmap charts

if currency_flag == True: #Rupees

	units_dict = {"Reserve Price" : "Rs Cr/MHz", "Auction Price" : "Rs Cr/MHz", "Quantum Offered": "MHz", 
		          "Quantum Sold" : "MHz", "Quantum Unsold" : "MHz", "Total EMD" : "Rs Cr", "Total Outflow" : "Rs Cr",
		          "Auction/Reserve" : "Ratio", "Percent Unsold" : "% of Total Spectrum", "Percent Sold" : "% of Total Spectrum", 
		          "Total Purchase" : "MHz"}

if currency_flag == False: #USD

		units_dict = {"Reserve Price" : "USD Million/MHz", "Auction Price" : "USD Million/MHz", "Quantum Offered": "MHz", 
			          "Quantum Sold" : "MHz", "Quantum Unsold" : "MHz", "Total EMD" : "USD Million", "Total Outflow" : "USD Million",
			          "Auction/Reserve" : "Ratio", "Percent Unsold" : "% of Total Spectrum", "Percent Sold" : "% of Total Spectrum", 
			          "Total Purchase" : "MHz"}



#---------Dimension = Spectrum Bands Starts -------------------

if (Feature == "Spectrum Map") and (SubFeature == "Frequency Layout"):

	fig.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))

	xdtickangle = -90
	xdtickval = xdtickfreq_dict[Band]

	unit = "Ch Size - "+str(channelsize_dict[Band])+" MHz"

	if selected_operators == []:
		selected_operators = ["All"]
	else:
		selected_operators = selected_operators
		
	subtitle = subtitle_freqlayout_dict[Band]+unit+"; Selected Operators - "+', '.join(selected_operators)+"; Source - DOT"

	title = "Spectrum Frequency Layout for the "+str(Band)+" MHz Band"


if (Feature == "Spectrum Map") and (SubFeature == "Operator Holdings"):

	fig.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))

	xdtickangle = 0
	xdtickval = 1


	if (len(selected_category) == 0) or (len(selected_category) == 2):
		selected_category = "All"
	else:
		selected_category = selected_category[0]
	
	if selected_operators == []:
		selected_operators = ["All"]
	else:
		selected_operators = selected_operators
	
	unit = "MHz"
	subtitle = "Unit - "+unit+"; "+"India Total - Sum of all LSAs "+"; Selected Operators - "+', '.join(selected_operators)+ ";\
	Category - "+ selected_category+"; Source - DOT"

	title = "Operator Holdings for the "+str(Band)+" MHz Band"


if (Feature == "Spectrum Map") and (SubFeature == "Operator %Share"):

	fig.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))

	xdtickangle = 0
	xdtickval = 1

	if (len(selected_category) == 0) or (len(selected_category) == 2):
		selected_category = "All"
	else:
		selected_category = selected_category[0]
	
	if len(selected_operators) == 0: 
		selected_operators = ["All"]
	else:
		selected_operators = selected_operators
	
	unit = '% of Total'
	subtitle = "Unit - "+unit+ " ; Selected Operators - "+', '.join(selected_operators)+ "; Category - "+ selected_category+"; Source - DOT"

	title = "Operator's Spectrum Market Share for the "+str(Band)+" MHz Band"


	
if (Feature == "Expiry Map") and (SubFeature == "Frequency Layout"):

	fig.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))

	xdtickangle = -90
	xdtickval = xdtickfreq_dict[Band]


	unit = "Ch Size - "+str(channelsize_dict[Band])+" MHz"
	if selected_operators == []:
		selected_operators = ["All"]
	else:
		selected_operators = selected_operators
		
	subtitle = subtitle_freqlayout_dict[Band]+unit+"; Selected Operators - "+', '.join(selected_operators)+"; Source - DOT"

	title = "Spectrum Expiry Layout for the "+str(Band)+" MHz Band"


if (Feature == "Expiry Map") and (SubFeature == "Yearly Trends"):

	hoverlabel_bgcolor = "#000000" #subdued black

	fig.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white'))) #hoverbox color is black

	xdtickangle = 0
	xdtickval = dtickauction_dict[Band]

	unit = "MHz"
	if selected_operator == "":
		selected_operator = "All"
	else:
		selected_operator = selected_operator
	subtitle = "Unit - "+unit+"; Selected Operators - "+selected_operator+ "; Summary Below - Sum of all LSAs"+"; Source - DOT"

	title = "Spectrum Expiry Yearly Trends for the "+str(Band)+" MHz Band"


if Feature == "Auction Map":

	fig.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))

	parttitle = "Yearly Trend of "+SubFeature
	xdtickangle=0
	xdtickval = dtickauction_dict[Band]
	unit = units_dict[SubFeature]
	selected_operators = ["NA"]
	
	subtitle = "Unit - "+unit+"; Selected Operators - "+', '.join(selected_operators)+ " ; Summary Below - Sum of all LSAs"+"; Source - DOT"

	title = parttitle+" for the "+str(Band)+" MHz Band"

#---------Dimension = Spectrum Bands Ends -------------------


#---------Dimension = Auction Years Starts ------------------

if (Feature == "Band Metric"):

	fig.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))
	
	xdtickangle =0
	xdtickval =1

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

	subtitle = SubFeature+"; Unit -"+units_dict[SubFeature]+"; "+ "Selected Operators -" + ', '.join(selected_operators)+ partsubtitle+"; Source - DOT"

	
if (Feature == "Operator Metric"):

	fig.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))

	xdtickangle =0
	xdtickval =1

	if (SubFeature =="Total Outflow") or (SubFeature == "Total Purchase"):
		if selectedbands==[]:
			selectedbands = ["All"]
		else:
			selectedbands = selectedbands
	else:
		selectedbands = ["NA"]
	selectedbands = [str(x) for x in selectedbands]	

	title = "Operator Wise Summary for the Year "+str(Year)

	subtitle = SubFeature + "; Unit -"+units_dict[SubFeature]+"; Selected Bands -"+ ', '.join(selectedbands) + \
				"; Summary Below - Sum of all LSAs"+"; Source - DOT"


#---------Dimension = Auction Years Ends ------------------


#---------Dimension = Business Data Starts ----------------


if (Feature == "5GBTS Trends") and (SubFeature == "Cumulative Values"):

	hoverlabel_bgcolor = "#000000" #subdued black

	fig.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))
	xdtickangle= -45
	xdtickval=1
	title = "Indian 5G Base Stations Roll Out Trends"
	subtitle = "Cumulative BTS growth; Top 20 States/UT; Unit - Thousands; Sorted by the Recent Date; Source - DOT"


if (Feature == "5GBTS Trends") and (SubFeature == "Percent of Total"):

	hoverlabel_bgcolor = "#000000" #subdued black

	fig.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))
	xdtickangle= -45
	xdtickval=1
	title = "Indian 5G Base Stations Roll Out Trends"
	subtitle = "Percent of Total; Top 20 States/UT; Unit - %; Sorted by the Recent Date; Source - DOT"

if (Feature == "5GBTS Trends") and (SubFeature == "Incremental Values"):

	hoverlabel_bgcolor = "#000000" #subdued black

	fig.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))
	xdtickangle= -45
	xdtickval=1
	title = "Indian 5G Base Stations Roll Out Trends"
	subtitle = "Incremental Values; Top 20 States/UT; Unit - Thousands; Sorted by the Recent Date; Source - DOT"


if (Feature == "Subscriber Trends"):

	hoverlabel_bgcolor = "#000000" #subdued black

	fig.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))
	xdtickangle= -45
	xdtickval=1

	subtitle = "Cumulative Values; Selected Category -" +",".join(selected_category)+ "; Selected Circles - "+ \
				",".join(selected_circles)+"; Unit - Millions; Sorted by the Recent Date; Source - TRAI"
	title = "Indian Telecom Subscribers Trends"


if (Feature == "Subscriber MShare"):

	hoverlabel_bgcolor = "#000000" #subdued black

	fig.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))
	xdtickangle= 0
	xdtickval=1

	subtitle = "Unit - % of Total; Total in Millions ; Selected Category -"+ ",".join(selected_category)+ " ;Source - TRAI"
	title = "Indian Telecom Operator's Latest Subs Market Share"

if (Feature == "License Fees"):

	if (len(selected_category)==0) or (len(selected_category)==2):

		selected_category=["All"]
	else:
		selected_category = selected_category

	if chosen_metric == "Absolute":
		unit = "Rs Cr"
	if chosen_metric == "Percentage":
		unit = "% of Total"

	# if len(selected_operators) > 0:
	# 	selected_operators = selected_operators


	if SubFeature == "LicenseType":

		if (len(selected_operators)==0) or (len(selected_operators)==30):
			selected_operators = ["NA"]
		else:
			pass
	if SubFeature == "Operators":
		selected_operators = ["NA"]


	hoverlabel_bgcolor = "#000000" #subdued black

	fig.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))
	xdtickangle= 0
	xdtickval=1

	subfeature_dict = {"Operators" : "Operators", "LicenseType" : "License Types"}

	subtitle = "Selected Category - "+",".join(selected_category)+"; "+chosen_metric+\
				"; Unit - "+unit+"; Sorted by - "+selected_fy_for_sort+"; Selected Operators -"+",".join(selected_operators)+"; Source - DOT"
	title = "Indian Telecom Regulatory Fees Trend Top N "+subfeature_dict[SubFeature]

if (Feature == "TowerBTS Trends"):
	xdtickangle =0
	xdtickval = 100
	title = "Indian Telecom Tower and BTS rollout Trends"
	subtitle = ""

if (Feature == "Financial SPWise"):
	xdtickangle =-45
	xdtickval = 2
	title = "Indian Telecom Financial Metric - Top 15 Operators ("+finmetric+")"
	subtitle = "Rs K Cr (Except %); Source - TRAI; ("+radio_selection+")"


if (Feature == "Financial LSAWise"):

	hoverlabel_bgcolor = "#000000" #subdued black

	fig.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white'))) #hoverbox color is black

	xdtickangle =-45
	xdtickval = 2
	title = "Indian Telecom Financial Metric ("+finmetric+")"
	subtitle = "Rs K Cr (Except %); Source - TRAI; ("+radio_selection+")"

if (Feature == "Subs RuralUrban"):

	# hoverlabel_bgcolor = "#000000" #subdued black

	# fig.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white'))) #hoverbox color is black

	xdtickangle =-45
	xdtickval = 2
	title = "Indian Telecom Financial Metric"
	subtitle = ""


	
#---------Dimension = Business Data Ends ----------------




if selected_dimension in ["Spectrum Bands", "Auction Years", "Business Data"]:

	#layout for heatmaps 

	fig.update_layout(uniformtext_minsize=12, 
			  uniformtext_mode='hide', 
			  xaxis_title=None, 
			  yaxis_title=None, 
			  yaxis_autorange='reversed',
			  font=dict(size=12),
			  template='simple_white',
			  paper_bgcolor=None,
			  height=600, 
			  width=1200,
			  margin=dict(t=80, b=50, l=50, r=50, pad=0),
			  yaxis=dict(
	        	  tickmode='array'),
			  xaxis = dict(
			  side = 'top',
			  tickmode = 'linear',
			  tickangle=xdtickangle,
			  dtick = xdtickval), 
			)

		#Drawning a black border around the heatmap chart 
	fig.update_xaxes(fixedrange=True,showline=True,linewidth=1.2,linecolor='black', mirror=True)
	fig.update_yaxes(fixedrange=True,showline=True, linewidth=1.2, linecolor='black', mirror=True)


	#Some last minute exceptions and changes in the plot

	#converts x axis into category
	if selected_dimension == "Business Data":
		fig.update_layout(xaxis_type='category')
	else:
		pass

	#removes tic labels if the date_range_list greater than a value
	#This is done to prevent cluttering of xaxis labels when a large range is selected
	if (selected_dimension == "Business Data") and (Feature == "Subscriber Trends"):
		# fig.data[0].update(zmin=110, zmax=450) #setting the max and min value of the colorscale
		if len(date_range_list) >= 30:
			fig.update_xaxes(
			    tickmode='array',
			    ticktext=[''] * len(date_range_list),
			    tickvals=list(range(len(date_range_list)))
			)

	#encircle the heatmaps with a rectangular box made up of black lines
	#Except for features which are not heatmaps


	#Final plotting of various charts on the output page
	style = "<style>h3 {text-align: left;}</style>"
	with st.container():
		#plotting the main chart
		st.markdown(style, unsafe_allow_html=True)
		st.header(title)
		st.markdown(subtitle)

		if chart_data_flag==True:
			tab1, tab2 = st.tabs(["📈 Chart", "🗃 Data"]) #for listing the summary chart for freq layout
			tab1.plotly_chart(fig, use_container_width=True)
			tab2.table(chartdata_df)
		else:
			st.plotly_chart(fig, use_container_width=True) # for heatmaps


		#preparing the container layout for the dimension business data
		if (selected_dimension=="Business Data") and (Feature == "License Fees") and (SubFeature=="Operators"):
			col1val =4.5
		if (selected_dimension=="Business Data") and (Feature == "License Fees") and (SubFeature=="LicenseType"):
			col1val =1
		if (selected_dimension=="Business Data") and (Feature == "Financial SPWise"):
			col1val =0.7
		if (selected_dimension=="Business Data") and (Feature == "Financial LSAWise"):
			col1val =0.7
		else:
			col1val = 0.2


		#plotting the final summary chart 

		col1,col2,col3 = st.columns([col1val,14,1.1]) #create collumns of uneven width
		if SummaryFlag ==True:
			# st.altair_chart(chart, use_container_width=True)
			col2.altair_chart(chart, use_container_width=True)


#--------The expander is used to add note for the user on reading the color codes for every chart -------

	expander = st.expander("Click Here - To Learn About the Color Codes", expanded = False)

	with expander:
		if (Feature == "Spectrum Map") and (SubFeature=="Frequency Layout"):
			st.info("Heatmap and Hoverbox's Background Color - Maps to the Specific Operator")

		if (Feature == "Expiry Map") and (SubFeature=="Frequency Layout"):
			st.info("Heatmap's Color Intensity - Directly Proportional to length of the expiry period in years")
			st.info("Hoverbox's Background Color - Directly Maps to the Specific Operator of the 'Spectrum Map' Layout")

		if (Feature == "Auction Map"):
			st.info("Heatmap's Color Intensity - Directly Proportional to the Value of the Cell")
			st.info("Hoverbox's Background Color = BLACK (Failed/No Auction)")
			st.info("Hoverbox's Background Color = GREEN (Auction Price = Reserve Price)")
			st.info("Hoverbox's Background Color = RED (Auction Price > Reserve Price)")

		if (Feature == "Operator Metric"):
			st.info("Heatmap's Color Intensity - Directly proportional to value on Color Bar on the left")
			st.info("Hoverbox's Background Color = GREEN (Purchase Made)")
			st.info("Hoverbox's Background Color = GREY (No Purchase Made)")


		if (Feature == "Band Metric"):
			st.info("Heatmap's Color Intensity - Directly Proportional to the Value of the Cell")
			st.info("Hoverbox's Background Color = GREY (No Auction)")
			st.info("Hoverbox's Background Color = BLACK (Failed Auction)")
			st.info("Hoverbox's Background Color = GREEN (Auction Price = Reserve Price)")
			st.info("Hoverbox's Background Color = RED (Auction Price > Reserve Price)")


