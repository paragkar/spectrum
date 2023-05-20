#importing libraries
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from collections import OrderedDict
from plotly.subplots import make_subplots
import plotly
import pandas as pd
import plotly.figure_factory as ff
import streamlit as st

state_dict = {'AP': 'Andhra Pradesh', 'AS': 'Assam', 'BH': 'Bihar', 'DL': 'Delhi', 'GU': 'Gujarat',
    'HA': 'Haryana','HP': 'Himachal Pradesh','JK': 'Jammu & Kashmir','KA': 'Karnataka',
    'KE': 'Kerala','KO': 'Kolkata','MP': 'Madhya Pradesh','MA': 'Maharashtra','MU': 'Mumbai',
    'NE': 'Northeast','OR': 'Odisha','PU': 'Punjab','RA': 'Rajasthan','TN': 'Tamil Nadu',
    'UPE': 'Uttar Pradesh (East)','UPW': 'Uttar Pradesh (West)','WB': 'West Bengal' }

#preparing color scale for freqmap
def colscalefreqmap(operators, colcodes):
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
def forexpyearheatmap(ef):
	lst1 =[]
	for i, line1 in enumerate(ef.values):
		explst = list(set(line1))
		l1 = [[ef.index[i],round(list(line1).count(x)*ChannelSize[Band],2), round(x,2)] for x in explst]
		lst1.append(l1)

	lst2 =[]
	for i, val in enumerate(lst1):
		for item in val:
			lst2.append(item)
	df = pd.DataFrame(lst2)
	df.columns = ["LSA", "Spectrum", "ExpYrs"]
	df = df.pivot_table(index='LSA', columns='ExpYrs', values=['Spectrum'], aggfunc='first')
	df.columns = df.columns.droplevel(0)
	df.columns = [str(x) for x in df.columns]
	df = df.iloc[:,1:]
	df = df.fillna(0)
	return df

#function for calculating quantum of spectrum expiring mapped to LSA and Years for expiry map yearwise
def BWExpiring(sff,ef):
	
	lst=[]
	for j, index in enumerate(ef.index):
		for i, col in enumerate(ef.columns):
			l= [index, sff.iloc[j,i],ef.iloc[j,i]]
			lst.append(l)	
	df = pd.DataFrame(lst)
	df.columns = ["LSA","Operators", "ExpYear"]
	df = df.groupby(["ExpYear"])[["LSA","Operators"]].value_counts()*ChannelSize[Band]
	df = df.reset_index()
	df.columns =["ExpYear","LSA", "Operators","BW"]
	return df

#funtion to process pricing datframe for hovertext for auction map
def processdff(dff):
    dff = dff.replace(0,np.nan).fillna(0)
    dff = dff.applymap(lambda x: round(x,2) if type(x)!=str else x)
    dff = dff[(dff["Band"]==Band) & (dff["Cat"]=="L") & (dff["OperatorOld"] != "Free") & (dff["Year"] >= 2010)]
    dff = dff.drop(['OperatorNew', 'Band','Cat'], axis = 1)
    for col in dff.columns[3:]:
        dff[col]=dff[col].astype(float)
    dff = dff.groupby(["OperatorOld", "Year"]).sum()
    dff = dff.drop(['Batch No',], axis = 1) 
    if BandType[Band]=="TDD": #doubling the TDD spectrum for aligning with normal convention 
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
    dff.columns = ["LSA"]+auctionsucessyears[Band]
    dff = dff.set_index("LSA")
    return dff

#convert columns of dataframe into string
def coltostr(df):
	lst =[]
	for col in df.columns:
		lst.append(str(col))
	df.columns=lst
	return df

#add dummy columns for auction failed years
def adddummycols(df,col):
    df[col]="NA  " # space with NA is delibelitratly added.
    cols = sorted(df.columns)
    df =df[cols]
    return df

#function to calculate the year in which the spectrum was acquired
def auctioncalyear(ef,excepf,pf1):
	lst=[]
	for col in ef.columns:
		for i, (efval,excepfval) in enumerate(zip(ef[col].values, excepf[col].values)):
			for j, pf1val in enumerate(pf1.values):
				if excepfval == 0:
					error = abs(efval-pf1val[6]) #orignal
				else:
					error = 0
				if (ef.index[i] == pf1val[0]) and error <= errors[Band]:
					lst.append([ef.index[i],col-xaxisadj[Band],pf1val[1],pf1val[2], pf1val[3], pf1val[4], error]) 
				
	df_final = pd.DataFrame(lst)
	df_final.columns = ["LSA", "StartFreq", "TP", "RP", "AP", "Year", "Error"]
	df_final["Year"] = df_final["Year"].astype(int)
	ayear = df_final.pivot_table(index=["LSA"], columns='StartFreq', values="Year", aggfunc='first').fillna("NA")
	return ayear
    
#defining all dictionaries here with data linked to a specific band
title_map = {700:"(FDD Up/Down - 703-748 / 758-803 MHz)",
         800:"(FDD UP/DN : 824-844 / 869-889 MHz)", 
         900:"(FDD UP/DN : 890-915 / 935-960 MHz)", 
         1800:"(FDD UP/DN : 1710-1785 / 1805-1880 MHz)", 
         2100:"(FDD UP/DN : 1919-1979 / 2109- 2169 MHz)",
         2300:"(TDD UP/DN : 2300-2400 MHz)",
         2500:"(TDD UP/DN : 2500-2690 MHz)",
         3500:"(TDD UP/DN : 3300-3670 MHz)",
         26000:"(TDD UP/DN : 24250-27500 MHz)"}

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
ExpTab = {700:1, 800:1, 900:1, 1800:1, 2100:1, 2300:1, 2500:1, 3500:1, 26000:1}

#Setting the channel sizes for respective frequency maps
ChannelSize = {700:2.5, 800:0.625, 900:0.2, 1800:0.2, 2100:2.5, 2300:2.5, 2500:5, 3500:5, 26000:25}

# scale of the x axis plots
dtickfreq = {700:1, 800:0.25, 900:0.4, 1800:1, 2100:1, 2300:1, 2500:2, 3500:5, 26000:50}

# used to control the number of ticks on xaxis for chosen feature = AuctionMap
dtickauction = {700:1, 800:1, 900:1, 1800:1, 2100:1, 2300:1, 2500:1, 3500:1, 26000:1}

# vertical line widths
xgap = {700:1, 800:1, 900:0.5, 1800:0, 2100:1, 2300:1, 2500:1, 3500:1, 26000:1}

# adjustment need for tool tip display data for channel frequency
xaxisadj = {700:1, 800:0.25, 900:0, 1800:0, 2100:1, 2300:1, 2500:2, 3500:0, 26000:0}

#describing the type of band TDD/FDD
BandType = {700:"FDD", 800:"FDD", 900:"FDD", 1800:"FDD", 2100:"FDD", 2300:"TDD", 2500:"TDD", 3500:"TDD", 26000:"TDD"}

#auctionfailyears when all auction prices are zero and there are no takers 
auctionfailyears = {700:["2016","2021"], 800:["2012"], 900:["2013","2016"], 1800:["2013"], 
        2100:[], 2300:["2022"], 2500:["2021"], 3500:[], 26000:[]}

#auction sucess years are years where at least in one circle there was a winner
auctionsucessyears = {700:[2022], 
        800:[2013, 2015, 2016, 2021, 2022], 
        900:[2014, 2015, 2021, 2022], 
        1800:[2012, 2014, 2015, 2016, 2021, 2022], 
        2100:[2010, 2015, 2016, 2021, 2022], 
        2300:[2010, 2016, 2021], 
        2500:[2010, 2016, 2022], 
        3500:[2022], 
        26000:[2022]}

#Error is added to auction closing date so that freq assignment dates fall within the window.
#This helps to identify which expiry year is linked to which operators
errors= {700:0.25, 800:1, 900:1, 1800:1, 2100:1.5, 2300:1.25, 2500:1, 3500:0.1, 26000:0.5}

st.set_page_config(layout="wide")

file = "https://paragkar.com/wp-content/uploads/2023/05/spectrum_map.xlsx"
#loading data from excel file
xl = pd.ExcelFile(file)
sheet = xl.sheet_names
df = pd.read_excel(file, sheet_name=sheet)

#choose a dimension
Dimension = st.sidebar.selectbox('Select a Dimension', options = ["Frequency Band", "Calendar Year"])

if Dimension == "Calendar Year":
	masterall = "MasterAll-TDDValueConventional" #all auction related information
	spectrumofferedvssold = "Spectrum_Offered_vs_Sold"
	masterdimdf = df[masterall]
	offeredvssolddimdf = df[spectrumofferedvssold]
	calendaryearlist = sorted(list(set(masterdf["Auction Year"].values)))
	YearDim = st.sidebar.selectbox('Select a Year', options = calendaryearlist)
	dfdim1 = masterdimdf[masterdf["Auction Year"]==YearDim]
	dfdim2 = offeredvssolddimdf[offeredvssold["Year"]==YearDim]
	feature_dict ={"Spectrum Offered" : "Offered", "Spectrum Sold": "Sold", "Spectrum Unsold" : "Unsold", "Reserve Price" : "RP/MHz" ,  
		       "Auction Price": "Auction Price/MHz", "Block Size": "Block Size", "Total EMD" : "Total EMD"} 
	feature_list = ["Reserve Price",  "Auction Price", "Spectrum Offered", "Spectrum Sold", "Spectrum Unsold", "Block Size", "Total EMD"]
	Feature = st.sidebar.selectbox('Select a Feature', options = feature_list)
	z = dfdim1[feature_dict[Feature]].round(2)
	x = dfdim1["Band"].astype(str)
	y = dfdim1["Circle"]
	
	data = [go.Heatmap(
		  z = z,
		  x = x,
		  y = y,
		  xgap = 1,
		  ygap = 1,
		  hoverinfo ='text',
# 		  text = hovertext,
		  colorscale = 'Hot',
		    texttemplate="%{z}", 
		    textfont={"size":10},
		    reversescale=True,
			)]

if Dimension == "Frequency Band":
	#Selecting a Freq Band
	Band = st.sidebar.selectbox('Select a Band', options = list(ExpTab.keys()))
	Feature = st.sidebar.selectbox('Select a Feature', options = ["FreqMap", "ExpiryMap", "AuctionMap"])
	
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
	if ExpTab[Band]==1:
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

	eff = forexpyearheatmap(ef) # for expiry year heatmap year wise
	bwf = BWExpiring(sff,ef) # hover text for expiry year heatmap year wise

	# st.sidebar.title('Navigation')

	#processing "Spectrum_all" excel tab data
	dff = df[spectrumall] #contains information of LSA wise mapping oldoperators with new operators
	dff = processdff(dff)
	dff = coltostr(dff)
	dff = adddummycols(dff,auctionfailyears[Band])
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
	unsoldspectrum = offeredvssold.pivot(index=["LSA"], columns='Year', values="Unsold").fillna("NA")
	unsoldspectrum = coltostr(unsoldspectrum) #convert columns data type to string

	#processing & restructuring dataframe auction price for hovertext of data3
	auctionprice = pricemaster[(pricemaster["Band"] == Band) & (pricemaster["Year"] != 2018)]
	auctionprice = auctionprice.pivot(index=["LSA"], columns='Year', values="Auction Price").fillna("NA")
	auctionprice = auctionprice.loc[:, (auctionprice != 0).any(axis=0)]
	auctionprice = auctionprice.applymap(lambda x: round(x,2))
	auctionprice = coltostr(auctionprice) #convert columns data type to string
	auctionprice = adddummycols(auctionprice,auctionfailyears[Band])
	auctionprice = auctionprice.replace(0,"NA")

	#processing & restructuring dataframe reserve price for hovertext of data3
	reserveprice = pricemaster[(pricemaster["Band"] == Band) & (pricemaster["Year"] != 2018)]
	reserveprice = reserveprice.pivot(index=["LSA"], columns='Year', values="Reserve Price").fillna("NA")
	reserveprice = reserveprice.loc[:, (reserveprice != 0).any(axis=0)]
	reserveprice = reserveprice.applymap(lambda x: round(x,2))
	reserveprice = coltostr(reserveprice) #convert columns data type to string
	reserveprice = reserveprice.replace(0,"NA")
	
	#mapping the year of auction with channels in the freq maps
	ayear = auctioncalyear(ef,excepf,pf1)

#processing for hovertext for freq map
def hovertext1(sf,sff,bandf,ExpTab,ChannelSize,xaxisadj):  
	hovertext = []
	for yi, yy in enumerate(sf.index):
		hovertext.append([])
		for xi, xx in enumerate(sf.columns):
			operatornew = sff.values[yi][xi]
			bandwidth = bandf.values[yi][xi]
			hovertext[-1].append(
					    'StartFreq: {} MHz\
					     <br>Channel Size : {} MHz\
					     <br>Circle : {}\
				             <br>Operator: {}\
					     <br>Total BW: {} MHz'

				     .format(
					    round(xx-xaxisadj[Band],2),
					    ChannelSize[Band],
					    state_dict.get(yy),
					    operatornew,
					    bandwidth,
					    )
					    )
	return hovertext


#processing for hovertext for expiry map freq wise
def hovertext21(sf,sff,ef,bandf,bandexpf,ExpTab,ChannelSize,xaxisadj,ayear):
	hovertext = []
	for yi, yy in enumerate(sf.index):
		hovertext.append([])
		for xi, xx in enumerate(sf.columns):
			if ExpTab[Band]==1: #1 means that the expiry table in the excel sheet has been set and working 
				expiry = round(ef.values[yi][xi],2)
			else:
				expiry = "NA"
			try:
			    auction_year = round(ayear.loc[yy,round(xx-xaxisadj[Band],3)])
			except:
			    auction_year ="NA"
			operatornew = sff.values[yi][xi]
			bandwidthexpiring = bandexpf.values[yi][xi]
			bandwidth = bandf.values[yi][xi]
			hovertext[-1].append(
					    'StartFreq: {} MHz\
					     <br>Channel Size : {} MHz\
					     <br>Circle : {}\
				             <br>Operator: {}\
					     <br>Expiring BW: {} of {} MHz\
					     <br>Expiring In: {} Years\
					     <br>Acquired In: {}'

				     .format(
					    round(xx-xaxisadj[Band],2),
					    ChannelSize[Band],
					    state_dict.get(yy),
					    operatornew,
					    bandwidthexpiring,
					    bandwidth,
					    expiry,
					    auction_year,
					    )
					    )
	return hovertext

#processing for hovertext for expiry map year wise

def hovertext22(bwf,eff): 
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
					    round(xx,2), 
					    opwiseexpMHz,
					    )
					    )
	return hovertext
	
#processing for hovertext for Auction Map
def hovertext3(dff,reserveprice,auctionprice,offeredspectrum,soldspectrum,unsoldspectrum):  
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
					     <br / >RP/AP: Rs {} / {} Cr/MHz\
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

#preparing color scale for hoverbox for freq and exp maps
def hcolscalefreqexp(operators, colcodes):
    scale = [round(x/(len(operators)-1),2) for x in range(len(operators))]
    colors =[]
    for k, v  in operators.items():
        colors.append(colcodes.loc[k,:].values[0])
    colorscale=[]
    for i in range(len(scale)):
        colorscale.append([scale[i],colors[i]])
    return colorscale

#shaping colorscale for driving the color of hoverbox of freq and exp maps
def hcolmatrixfreqexp(colorscale, sf):
	hlabel_bgcolor = [[x[1] for x in colorscale if x[0] == round(value/(len(colorscale) - 1),2)] 
			      for row in sf.values for value in row]
	hlabel_bgcolor = list(np.array(hlabel_bgcolor).reshape(22,int(len(hlabel_bgcolor)/22)))
	return hlabel_bgcolor

#preparing and shaping the colors for hoverbox for auction map
def hovermatrixauction(dff,reserveprice, auctionprice): 
	lst =[]
	for yi, yy in enumerate(dff.index):
		reserveprice = reserveprice.replace("NA\s*", np.nan, regex = True)
		auctionprice = auctionprice.replace("NA\s*", np.nan, regex = True)
		delta = auctionprice-reserveprice
		delta = delta.replace(np.nan, "NA")
		for xi, xx in enumerate(dff.columns):
			delval = delta.values[yi][xi]
			if delval =="NA":
				ccode = '#000000' #auction failed 
			elif delval == 0:
				ccode = '#00FF00' #auction price = reserve price
			else:
				ccode = '#FF0000' #auction price > reserve price
			lst.append([yy,xx,ccode])
			temp = pd.DataFrame(lst)
			temp.columns = ["LSA", "Year", "Color"]
			colormatrix = temp.pivot(index='LSA', columns='Year', values="Color")
			colormatrix = list(colormatrix.values)
	return colormatrix

#**********Main Program Starts here***************

#Feature ="Frequency Map" linked to Dimension = "Frequency"
if  (Dimension == "Frequency Band") and (Feature == "FreqMap"):
	sf = sff.copy()
	operators = operators[Band]
	hf = sf[sf.columns].replace(operators) # dataframe for hovertext
	operatorslist = sorted(list(operators.keys()))
	selected_operators = st.sidebar.multiselect('Select Operators', options = operatorslist)
	if selected_operators==[]:
		sf[sf.columns] = sf[sf.columns].replace(operators) 
		colorscale = colscalefreqmap(operators, colcodes)
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
		colorscale = colscalefreqmap(selected_op_dict, colcodes)
		tickvals = list(selected_op_dict.values())
		ticktext = list(selected_op_dict.keys())	
		
	hovertext = hovertext1(hf,sff,bandf, ExpTab,ChannelSize,xaxisadj)
	subtitle ="Frequency Map"
	tickangle = -90
	dtickval = dtickfreq[Band]
	
	data = [go.Heatmap(
	      z = sf.values,
	      y = sf.index,
	      x = sf.columns,
	      xgap = xgap[Band],
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
	hcolscale=hcolscalefreqexp(operators, colcodes)  #colorscale for hoverbox
	hoverlabel_bgcolor = hcolmatrixfreqexp(hcolscale, hf) #shaping the hfcolorscale

#Feature ="Expiry Map" linked to Dimension = "Frequency"
if  (Dimension == "Frequency Band") and (Feature == "ExpiryMap"):
	SubFeature = st.sidebar.selectbox('Select a Sub Feature', options = ["Freq Wise", "Year Wise"])
	if SubFeature == "Freq Wise":
		sf = sff.copy()
		operators = operators[Band]
		hf = sf[sf.columns].replace(operators) # dataframe for hovertext
		operatorslist = sorted(list(operators.keys()))
		operatorstoremove = ["Govt", "Vacant", "Railways"]
		for op in operatorstoremove:
			if op in operatorslist:
				operatorslist.remove(op)
		selected_operators = st.sidebar.multiselect('Select Operators', options = operatorslist)
		if selected_operators==[]:
			expf = ef
		else:
			for op in operators.keys():
				if op not in selected_operators:
					sf.replace(op,0, inplace = True)
				else:
					sf.replace(op,1,inplace = True)

			expf = pd.DataFrame(sf.values*ef.values, columns=ef.columns, index=ef.index)

		hovertext = hovertext21(hf,sff,ef,bandf, bandexpf, ExpTab,ChannelSize,xaxisadj,ayear)
		subtitle ="Expiry Map "+SubFeature
		tickangle = -90
		dtickval = dtickfreq[Band]

		data = [go.Heatmap(
		      z = expf.values,
		      y = expf.index,
		      x = expf.columns,
		      xgap = xgap[Band],
		      ygap = 1,
		      hoverinfo ='text',
		      text = hovertext,
		      colorscale ='Hot',
		      reversescale=True,
			)
			  ]
		hcolscale=hcolscalefreqexp(operators, colcodes)  #colorscale for hoverbox
		hoverlabel_bgcolor = hcolmatrixfreqexp(hcolscale, hf) #shaping the hfcolorscale
		
	if SubFeature == "Year Wise":
		hovertext = hovertext22(bwf,eff)
		subtitle ="Expiry Map "+SubFeature
		tickangle = 0
		dtickval = dtickauction[Band]

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
		hoverlabel_bgcolor = "#002855" #subdued black


#Feature ="Auction Map" linked to Dimension = "Frequency"
if  (Dimension == "Frequency Band") and (Feature == "AuctionMap"):
	pricemaster = pricemaster[(pricemaster["Band"]==Band) & (pricemaster["Year"] != 2018)]
	pricemaster["Year"] = sorted([str(x) for x in pricemaster["Year"].values])
	#This dict has been defined for the Feature = Auction Map
	type_dict ={"Auction Price": auctionprice,
		    "Reserve Price": reserveprice, 
		    "Quantum Offered": offeredspectrum, 
		    "Quantum Sold": soldspectrum, 
		    "Quantum Unsold": unsoldspectrum}
	Type = st.sidebar.selectbox('Select Price Type', options = ["Auction Price","Reserve Price","Quantum Offered", "Quantum Sold", "Quantum Unsold"])
	typedf = type_dict[Type].copy()
	subtitle = Type
	tickangle=0
	dtickval = dtickauction[Band]
	hovertext = hovertext3(dff,reserveprice,auctionprice,offeredspectrum,soldspectrum,unsoldspectrum)
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
	hoverlabel_bgcolor = hovermatrixauction(dff,reserveprice,auctionprice)

	
units_dict = {"Reserve Price" : "Rs Cr/MHz", "Auction Price" : "Rs Cr/MHz", "Quantum Offered": "MHz", 
	      "Quantum Sold" : "MHz", "Quantum Unsold" : "MHz"}

#Plotting the final Heatmap	
fig = go.Figure(data=data)

if Dimension == "Frequency Band":
	if Feature == "AuctionMap":
		unit = units_dict[Type]
	if (Feature == "ExpiryMap") and (SubFeature == "Freq Wise"):
		unit = "Ch Size - "+str(ChannelSize[Band])+" MHz"
	if (Feature == "ExpiryMap") and (SubFeature == "Year Wise"):
		unit = "No of Years"
	if Feature == "FreqMap":
		unit = "Ch Size - "+str(ChannelSize[Band])+" MHz"
	if BandType[Band] == "FDD":
		title_x =0.09
	if BandType[Band] == "TDD":
		title_x = 0.10
	fig.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))
	title = "<b>"+"Spectrum "+subtitle+" for "+str(Band)+" MHz "+title_map[Band]+" ("+unit+")"+"<b>"

if Dimension == "Calendar Year":
	title = "<b>"+str(YearDim)+" - Band Wise Trend of "+Feature+" ("+units_dict[Feature]+")"+"<b>"
	title_x =0.28
	tickangle =0
	dtickval =1

#updating figure layouts
fig.update_layout(uniformtext_minsize=12, 
		  uniformtext_mode='hide', 
		  xaxis_title=None, 
		  yaxis_title=None, 
		  yaxis_autorange='reversed',
		  font=dict(size=12),
		  template='simple_white',
		  paper_bgcolor=None,
		  height=600, width=1200,
		  title=title,
		  margin=dict(t=80, b=50, l=50, r=50, pad=0),
		  title_x=title_x, title_y=0.99,
		  title_font=dict(size=22),
		  yaxis=dict(
        	  tickmode='array'),
		  xaxis = dict(
		  side = 'top',
		  tickmode = 'linear',
		  tickangle=tickangle,
		  dtick = dtickval),
		)

fig.update_xaxes(fixedrange=True,showline=True,linewidth=1.2,linecolor='black', mirror=True)
fig.update_yaxes(fixedrange=True,showline=True, linewidth=1.2, linecolor='black', mirror=True)

st.write(fig)



