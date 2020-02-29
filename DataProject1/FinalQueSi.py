# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:02:57 2019

@author: Salim
"""

#load basiclibraries
import os
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype #For definition of custom categorical data types (ordinal if necesary)
import matplotlib.pyplot as plt
import seaborn as sns  # For hi level, Pandas oriented, graphics
import scipy.stats as stats  # For statistical inference 


# Get working directory
os.getcwd()


# Change working directory

os.chdir('C:\\Users\MASCIAVE\Dropbox\Hyper')
os.getcwd()


#change the format document. Delete the first 6 row. f8th problem solve
ine =pd.read_csv ("3989.csv", sep='\t', decimal='.',skiprows=6, index_col=None)

#delete row and column
ine=ine.drop([53,54,55], axis=0)
ine=ine.drop(['Unnamed: 356'], axis=1)


#change city names
ine=ine.replace({'Unnamed: 0':"Provincias"})
ine=ine.replace({'03 Alicante/Alacant':"Alicante/Alacant"})
ine=ine.replace({'02 Albacete':"Albacete"})
ine=ine.replace({'04 Almería':"Almería"})
ine=ine.replace({'01 Araba/Álava':"Araba/Álava"})
ine=ine.replace({'33 Asturias':"Asturias"})
ine=ine.replace({'05 Ávila':"Ávila"})
ine=ine.replace({'06 Badajoz':"Badajoz"})
ine=ine.replace({'07 Balears, Illes':"Balears (Illes)"})
ine=ine.replace({'08 Barcelona':"Barcelona"})
ine=ine.replace({'48 Bizkaia':"Bizkaia"})
ine=ine.replace({'09 Burgos':"Burgos"})
ine=ine.replace({'10 Cáceres':"Cáceres"})
ine=ine.replace({'11 Cádiz':"Cádiz"})
ine=ine.replace({'39 Cantabria':"Cantabria"})
ine=ine.replace({'12 Castellón/Castelló':"Castellón/Castelló"})
ine=ine.replace({'13 Ciudad Real':"Ciudad Real"})
ine=ine.replace({'14 Córdoba':"Córdoba"})
ine=ine.replace({'15 Coruña, A':"Coruña (A)"})
ine=ine.replace({'16 Cuenca':"Cuenca"})
ine=ine.replace({'20 Gipuzkoa':"Gipuzkoa"})
ine=ine.replace({'17 Girona':"Girona"})
ine=ine.replace({'18 Granada':"Granada"})
ine=ine.replace({'19 Guadalajara':"Guadalajara"})
ine=ine.replace({'21 Huelva':"Huelva"})
ine=ine.replace({'22 Huesca':"Huesca"})
ine=ine.replace({'23 Jaén':"Jaén"})
ine=ine.replace({'24 León':"León"})
ine=ine.replace({'25 Lleida':"Lleida"})
ine=ine.replace({'27 Lugo':"Lugo"})
ine=ine.replace({'28 Madrid':"Madrid"})
ine=ine.replace({'29 Málaga':"Málaga"})
ine=ine.replace({'30 Murcia':"Murcia"})
ine=ine.replace({'31 Navarra':"Navarra"})
ine=ine.replace({'32 Ourense':"Ourense"})
ine=ine.replace({'34 Palencia':"Palencia"})
ine=ine.replace({'35 Palmas, Las':"Palmas (Las)"})
ine=ine.replace({'36 Pontevedra':"Pontevedra"})
ine=ine.replace({'26 Rioja, La':"Rioja (La)"})
ine=ine.replace({'37 Salamanca':"Salamanca"})
ine=ine.replace({'38 Santa Cruz de Tenerife':"Tenerife"})
ine=ine.replace({'40 Segovia':"Segovia"})
ine=ine.replace({'41 Sevilla':"Sevilla"})
ine=ine.replace({'42 Soria':"Soria"})
ine=ine.replace({'43 Tarragona':"Tarragona"})
ine=ine.replace({'44 Teruel':"Teruel"})
ine=ine.replace({'45 Toledo':"Toledo"})
ine=ine.replace({'46 Valencia/València':"Valencia/València"})
ine=ine.replace({'47 Valladolid':"Valladolid"})
ine=ine.replace({'49 Zamora':"Zamora"})
ine=ine.replace({'50 Zaragoza':"Zaragoza"})
ine=ine.replace({'51 Ceuta':"Ceuta"})
ine=ine.replace({'52 Melilla':"Melilla"})


#delete soria because in DGT there is not this row
ine=ine.drop([43], axis=0)


#control data
ine.dtypes
ine.columns
ine.shape
ine.dtypes
ine.head()
ine.tail()


#change empty los vacios por 0
ine.replace('', 0, inplace=True)


#change nan los vacios por 0
ine.replace('nan', 0, inplace=True)
ine.replace('..',0.0, inplace=True)
ine=ine.dropna()


#change 5 columns to float. format problems
ine["2017T4.1"]= ine["2017T4.1"].apply(pd.to_numeric, errors='coerce')
ine["2017T3.1"]= ine["2017T3.1"].apply(pd.to_numeric, errors='coerce')
ine["2015T4.1"]= ine["2015T4.1"].apply(pd.to_numeric, errors='coerce')
ine["2014T2.1"]= ine["2014T2.1"].apply(pd.to_numeric, errors='coerce')
ine["2014T1.1"]= ine["2014T1.1"].apply(pd.to_numeric, errors='coerce')


#Average per year (T1,T2,T3,T4 per year per group 16-19, 20-24, 25-54 )
ine['total_2018_19'] = ine[["2018T4.1", "2018T3.1", "2018T2.1","2018T1.1"]].sum(axis=1)/4
ine['total_2017_19'] = ine[["2017T4.1", "2017T3.1", "2017T2.1","2017T1.1"]].sum(axis=1)/4
ine['total_2016_19'] = ine[["2016T4.1", "2016T3.1", "2016T2.1","2016T1.1"]].sum(axis=1)/4
ine['total_2015_19'] = ine[["2015T4.1", "2015T3.1", "2015T2.1","2015T1.1"]].sum(axis=1)/4
ine['total_2014_19'] = ine[["2014T4.1", "2014T3.1", "2014T2.1","2014T1.1"]].sum(axis=1)/4

ine['total_2018_24'] = ine[["2018T4.2", "2018T3.2", "2018T2.2","2018T1.2"]].sum(axis=1)/4
ine['total_2017_24'] = ine[["2017T4.2", "2017T3.2", "2017T2.2","2017T1.2"]].sum(axis=1)/4
ine['total_2016_24'] = ine[["2016T4.2", "2016T3.2", "2016T2.2","2016T1.2"]].sum(axis=1)/4
ine['total_2015_24'] = ine[["2015T4.2", "2015T3.2", "2015T2.2","2015T1.2"]].sum(axis=1)/4
ine['total_2014_24'] = ine[["2014T4.2", "2014T3.2", "2014T2.2","2014T1.2"]].sum(axis=1)/4

ine['total_2018_55'] = ine[["2018T4.3", "2018T3.3", "2018T2.3","2018T1.3"]].sum(axis=1)/4
ine['total_2017_55'] = ine[["2017T4.3", "2017T3.3", "2017T2.3","2017T1.3"]].sum(axis=1)/4
ine['total_2016_55'] = ine[["2016T4.3", "2016T3.3", "2016T2.3","2016T1.3"]].sum(axis=1)/4
ine['total_2015_55'] = ine[["2015T4.3", "2015T3.3", "2015T2.3","2015T1.3"]].sum(axis=1)/4
ine['total_2014_55'] = ine[["2014T4.3", "2014T3.3", "2014T2.3","2014T1.3"]].sum(axis=1)/4

#total per group and years
ine['total_P2018'] = ine[['total_2018_19', 'total_2018_24', 'total_2018_55']].sum(axis=1)
ine['total_P2017'] = ine[['total_2017_19', 'total_2017_24', 'total_2017_55']].sum(axis=1)
ine['total_P2016'] = ine[['total_2016_19', 'total_2016_24', 'total_2016_55']].sum(axis=1)
ine['total_P2015'] = ine[['total_2015_19', 'total_2015_24', 'total_2015_55']].sum(axis=1)
ine['total_P2014'] = ine[['total_2014_19', 'total_2014_24', 'total_2014_55']].sum(axis=1)




#show only total columns
col = list(ine.columns)

#Filtro 1: Columnas totals
filtro1 = [col for col in ine if col.find('Unnamed: 0')>=0 or col.find('total_P2018')>=0 or col.find('total_P2017')>=0 or col.find('total_P2016')>=0 or col.find('total_P2015')>=0 or col.find('total_P2014')>=0]


#Selection
ine_top15= ine[filtro1]

#Drop Total nacional 
ine_top15=ine_top15.drop([0], axis=0)


#TOP 15 
ine_top15=ine_top15.nlargest(15,'total_P2018')


#detale datasets we don't need
del(col)
del(filtro1)


#remplace the name of the city columms
ine_top15.rename(columns={'Unnamed: 0':'PROVINCIAS'}, inplace=True)


#grafic % pob activa sobre tota pob activa. top 15
f=ine_top15.loc[:,['PROVINCIAS','total_P2018']]
n=ine.loc[0,'total_P2018']


#hacer porcentajes
f['m']=ine_top15['total_P2018']*100/n
plt.bar(f.PROVINCIAS, f['m'],edgecolor='blue')
plt.xticks(f.PROVINCIAS,rotation='vertical')
plt.ylabel('Miles de Personas')
plt.xlabel(r"$\it{"+'Source'+ "}$"+': INE_Actvos por grupo de edad y provincia')
plt.title('Figura 1. Top 15 Personas por Provincias')
props = dict(boxstyle='round', facecolor='white', lw=0.5)



#desviation per year and per 2014 to 2018
ine_totals_top15=ine_top15
ine_totals_top15['delta14_15'] = ine_top15["total_P2015"]- ine_top15["total_P2014"]
ine_totals_top15['delta15_16'] = ine_top15["total_P2016"]- ine_top15["total_P2015"]
ine_totals_top15['delta16_17'] = ine_top15["total_P2017"]- ine_top15["total_P2016"]
ine_totals_top15['delta17_18'] = ine_top15["total_P2018"]- ine_top15["total_P2017"]
ine_totals_top15['delta14_18'] = ine_top15["total_P2018"]- ine_top15["total_P2014"]

# Division y multiplicar por 100 para llevar a %
ine_totals_top15['deltaP14_15'] = (ine_totals_top15["delta14_15"] / ine_totals_top15["total_P2014"])*100
ine_totals_top15['deltaP15_16'] = (ine_totals_top15["delta15_16"] / ine_totals_top15["total_P2015"])*100
ine_totals_top15['deltaP16_17'] = (ine_totals_top15["delta16_17"] / ine_totals_top15["total_P2016"])*100
ine_totals_top15['deltaP17_18'] = (ine_totals_top15["delta17_18"] / ine_totals_top15["total_P2017"])*100
ine_totals_top15['deltaP14_18'] = (ine_totals_top15["delta14_18"] / ine_totals_top15["total_P2014"])*100
########################################################################################################

########################################################################################################
inetop10=ine_totals_top15.nlargest(10,'deltaP14_18')
inetop10_1= inetop10.loc[[8,40,30,9,36],:]
inetop10_2= inetop10.loc[[2,47,42,18,10],:]

#new table
Top10_D14_15_1=inetop10_1.loc[:,['PROVINCIAS','delta14_15']]
Top10_D15_16_1=inetop10_1.loc[:,['PROVINCIAS','delta15_16']]
Top10_D16_17_1=inetop10_1.loc[:,['PROVINCIAS','delta16_17']]
Top10_D17_18_1=inetop10_1.loc[:,['PROVINCIAS','delta17_18']]

Top10_D14_15_2=inetop10_2.loc[:,['PROVINCIAS','delta14_15']]
Top10_D15_16_2=inetop10_2.loc[:,['PROVINCIAS','delta15_16']]
Top10_D16_17_2=inetop10_2.loc[:,['PROVINCIAS','delta16_17']]
Top10_D17_18_2=inetop10_2.loc[:,['PROVINCIAS','delta17_18']]

#new column ano
Top10_D14_15_1['DIFERENCIAL']=2015
Top10_D15_16_1['DIFERENCIAL']=2016
Top10_D16_17_1['DIFERENCIAL']=2017
Top10_D17_18_1['DIFERENCIAL']=2018

Top10_D14_15_2['DIFERENCIAL']=2015
Top10_D15_16_2['DIFERENCIAL']=2016
Top10_D16_17_2['DIFERENCIAL']=2017
Top10_D17_18_2['DIFERENCIAL']=2018

Top10_D14_15_1.rename(columns={'delta14_15':'POBLACION ACTIVA'}, inplace=True)
Top10_D15_16_1.rename(columns={'delta15_16':'POBLACION ACTIVA'}, inplace=True)
Top10_D16_17_1.rename(columns={'delta16_17':'POBLACION ACTIVA'}, inplace=True)
Top10_D17_18_1.rename(columns={'delta17_18':'POBLACION ACTIVA'}, inplace=True)

Top10_D14_15_2.rename(columns={'delta14_15':'POBLACION ACTIVA'}, inplace=True)
Top10_D15_16_2.rename(columns={'delta15_16':'POBLACION ACTIVA'}, inplace=True)
Top10_D16_17_2.rename(columns={'delta16_17':'POBLACION ACTIVA'}, inplace=True)
Top10_D17_18_2.rename(columns={'delta17_18':'POBLACION ACTIVA'}, inplace=True)

#fusion filas
Top10D_14_15_1=Top10_D14_15_1.append(Top10_D15_16_1, ignore_index=True)
Top10D_14_16_1=Top10D_14_15_1.append(Top10_D16_17_1, ignore_index=True)
Top10D_14_17_1=Top10D_14_16_1.append(Top10_D17_18_1, ignore_index=True)

Top10D_14_15_2=Top10_D14_15_2.append(Top10_D15_16_2, ignore_index=True)
Top10D_14_16_2=Top10D_14_15_2.append(Top10_D16_17_2, ignore_index=True)
Top10D_14_17_2=Top10D_14_16_2.append(Top10_D17_18_2, ignore_index=True)

#graphy

d1=sns.lmplot('DIFERENCIAL','POBLACION ACTIVA',data=Top10D_14_17_1,hue='PROVINCIAS',col='PROVINCIAS',height=5, aspect=.5 )
d2=sns.lmplot('DIFERENCIAL','POBLACION ACTIVA',data=Top10D_14_17_2,hue='PROVINCIAS',col='PROVINCIAS',height=5, aspect=.5 )

d1.savefig('graficoVariacion1.png')
d2.savefig('graficoVariacion2.png')


#deleted

del(n)
del(f)
del(Top10D_14_15_1)
del(Top10D_14_16_1)
del(Top10D_14_17_1)
del(Top10_D14_15_1)
del(Top10_D15_16_1)
del(Top10_D16_17_1)
del(Top10_D17_18_1)
del(Top10D_14_15_2)
del(Top10D_14_16_2)
del(Top10D_14_17_2)
del(Top10_D14_15_2)
del(Top10_D15_16_2)
del(Top10_D16_17_2)
del(Top10_D17_18_2)
del(ine)
del(ine_top15)
del(ine_totals_top15)
del(props)



#cargar el archivo como excel y tomar solo la pestana que queremos
dgtotal2018= pd.ExcelFile('matriculaciones_2018_anuario.xlsx')
dgtotal2018.sheet_names
dgt2018= dgtotal2018.parse('V_1_1_2_MATRI', sep='\t', decimal='.',skiprows=2, index_col=None )

dgtotal2017= pd.ExcelFile('matriculaciones_2017_anuario.xlsx')
dgtotal2017.sheet_names
dgt2017= dgtotal2017.parse('V_1_1_2_MATRI', sep='\t', decimal='.',skiprows=2, index_col=None )

dgtotal2016= pd.ExcelFile('matriculaciones_2016_anuario.xlsx')
dgtotal2016.sheet_names
dgt2016= dgtotal2016.parse('V_1_1_2_MATRI', sep='\t', decimal='.',skiprows=2, index_col=None )

dgtotal2015= pd.ExcelFile('matriculaciones_2015_anuario.xlsx')
dgtotal2015.sheet_names
dgt2015= dgtotal2015.parse('V_1_1_2_MATRI', sep='\t', decimal='.',skiprows=2, index_col=None )

dgtotal2014= pd.ExcelFile('matriculaciones_2014_anuario.xlsx')
dgtotal2014.sheet_names
dgt2014= dgtotal2014.parse('V_1_1_2_MATRI', sep='\t', decimal='.',skiprows=2, index_col=None )


#Proceso para hacer un nuevo DataSet con las columnas que queremos Repetir por ano


#CORRER ESTE PRIMER
#2018 correr cada ano por separado
#aqui tenemos un dataset con la suma de total turismo (coches y motos) 
dgt2018['total_2018'] = dgt2018[["Total\nTurismos", "Total\nMoto"]].sum(axis=1)
col = list(dgt2018.columns)
#Filtro 1: Columnas que tienen una A
filtro1 = [col for col in dgt2018 if col.find('PROVINCIAS')>=0 or col.find('total_2018')>=0]
#Seleccionar
dgt_2018 = dgt2018[filtro1]

#2017 correr cada ano por separado
dgt2017['total_2017'] = dgt2017[["Total\nTurismos", "Total\nMoto"]].sum(axis=1)
col = list(dgt2017.columns)
#Filtro 1: Columnas que tienen una A
filtro1 = [col for col in dgt2017 if col.find('PROVINCIAS')>=0 or col.find('total_2017')>=0]
#Seleccionar
dgt_2017 = dgt2017[filtro1]


#2016 correr cada ano por separado
dgt2016['total_2016'] = dgt2016[["Total\nTurismos", "Total\nMoto"]].sum(axis=1)
col = list(dgt2016.columns)
#Filtro 1: Columnas que tienen una A
filtro1 = [col for col in dgt2016 if col.find('PROVINCIAS')>=0 or col.find('total_2016')>=0]
#Seleccionar
dgt_2016 = dgt2016[filtro1]


#2015 correr cada ano por separado
dgt2015['total_2015'] = dgt2015[["Total\nTurismos", "Total\nMoto"]].sum(axis=1)
col = list(dgt2015.columns)
#Filtro 1: Columnas que tienen una A
filtro1 = [col for col in dgt2015 if col.find('PROVINCIAS')>=0 or col.find('total_2015')>=0]
#Seleccionar
dgt_2015 = dgt2015[filtro1]


#2014 correr cada ano por separado
dgt2014['total_2014'] = dgt2014[["Total\nTurismos", "Total\nMoto"]].sum(axis=1)
col = list(dgt2014.columns)
#Filtro 1: Columnas que tienen una A
filtro1 = [col for col in dgt2014 if col.find('PROVINCIAS')>=0 or col.find('total_2014')>=0]
#Seleccionar
dgt_2014 = dgt2014[filtro1]

##################################################################33

#limpieza variables extras
del(col)
del(filtro1)


#MERGE DE TODOS LOS AÑOS
dgt_14_15 =pd.merge(dgt_2014, dgt_2015, on="PROVINCIAS")
dgt_14_16 =pd.merge(dgt_14_15, dgt_2016, on="PROVINCIAS")
dgt_14_16=dgt_14_16.replace({'Santa Cruz de\nTenerife':"Tenerife"})
dgt_2017=dgt_2017.replace({'Santa Cruz de Tenerife':"Tenerife"})
dgt_14_17 =pd.merge(dgt_14_16, dgt_2017, on="PROVINCIAS")
dgt_2018=dgt_2018.replace({'Santa Cruz de\nTenerife':"Tenerife"})
dgt_14_18 =pd.merge(dgt_14_17, dgt_2018, on="PROVINCIAS")


#MERGE DEL TOP10 INE CON EL DATASET COMPLETO DE MATRICULACION PARA SACAR SOLO TOPTEN
IneDgt_Top10=pd.merge(inetop10, dgt_14_18, on='PROVINCIAS', how='inner')


#restamos las Columnas 

IneDgt_Top10['deltaM14_15'] = IneDgt_Top10["total_2015"]- IneDgt_Top10["total_2014"]
IneDgt_Top10['deltaM15_16'] = IneDgt_Top10["total_2016"]- IneDgt_Top10["total_2015"]
IneDgt_Top10['deltaM16_17'] = IneDgt_Top10["total_2017"]- IneDgt_Top10["total_2016"]
IneDgt_Top10['deltaM17_18'] = IneDgt_Top10["total_2018"]- IneDgt_Top10["total_2017"]
IneDgt_Top10['deltaM14_18'] = IneDgt_Top10["total_2018"]- IneDgt_Top10["total_2014"]

# Division y multiplicar por 100 para llevar a %
IneDgt_Top10['deltaMP14_15'] = (IneDgt_Top10["deltaM14_15"] / IneDgt_Top10["total_2014"])*100
IneDgt_Top10['deltaMP15_16'] = (IneDgt_Top10["deltaM15_16"] / IneDgt_Top10["total_2015"])*100
IneDgt_Top10['deltaMP16_17'] = (IneDgt_Top10["deltaM16_17"] / IneDgt_Top10["total_2016"])*100
IneDgt_Top10['deltaMP17_18'] = (IneDgt_Top10["deltaM17_18"] / IneDgt_Top10["total_2017"])*100
IneDgt_Top10['deltaMP14_18'] = (IneDgt_Top10["deltaM14_18"] / IneDgt_Top10["total_2014"])*100


# HACER CODIDGO PARA SOLO SACAR EL TOP 5
IneDgt_Top5=IneDgt_Top10.nsmallest(5,'deltaMP14_18')


#new table
Top5_D14_15=IneDgt_Top5.loc[:,['PROVINCIAS','deltaM14_15']]
Top5_D15_16=IneDgt_Top5.loc[:,['PROVINCIAS','deltaM15_16']]
Top5_D16_17=IneDgt_Top5.loc[:,['PROVINCIAS','deltaM16_17']]
Top5_D17_18=IneDgt_Top5.loc[:,['PROVINCIAS','deltaM17_18']]

#new column ano
Top5_D14_15['DIFERENCIAL']=2015
Top5_D15_16['DIFERENCIAL']=2016
Top5_D16_17['DIFERENCIAL']=2017
Top5_D17_18['DIFERENCIAL']=2018

Top5_D14_15.rename(columns={'deltaM14_15':'MATRICULACION'}, inplace=True)
Top5_D15_16.rename(columns={'deltaM15_16':'MATRICULACION'}, inplace=True)
Top5_D16_17.rename(columns={'deltaM16_17':'MATRICULACION'}, inplace=True)
Top5_D17_18.rename(columns={'deltaM17_18':'MATRICULACION'}, inplace=True)

#fusion filas
Top5D_14_15=Top5_D14_15.append(Top5_D15_16, ignore_index=True)
Top5D_14_16=Top5D_14_15.append(Top5_D16_17, ignore_index=True)
Top5D_14_17=Top5D_14_16.append(Top5_D17_18, ignore_index=True)



#graphy
d=sns.lmplot('DIFERENCIAL','MATRICULACION',data=Top5D_14_17,hue='PROVINCIAS',col='PROVINCIAS',height=5, aspect=.5 )



#limpieza variables extras
del(Top5D_14_15)
del(Top5D_14_16)
del(Top5D_14_17)
del(Top5_D14_15)
del(Top5_D15_16)
del(Top5_D16_17)
del(Top5_D17_18)
del(dgt2014)
del(dgt2015)
del(dgt2016)
del(dgt2017)
del(dgt2018)
del(dgt_14_15)
del(dgt_14_16)
del(dgt_14_17)
del(dgt_14_18)
del(dgt_2014)
del(dgt_2015)
del(dgt_2016)
del(dgt_2017)
del(dgt_2018)


#A CORUNA
#read
clima_coruna = pd.read_csv ("valoresclimatologicos_a-coruna.csv", sep='","', decimal='.',engine='python',skiprows=2,index_col=None)

#clean and rename
clima_coruna= clima_coruna.rename(columns={'TM': 'temperatura'})
clima_coruna= clima_coruna.rename(columns={'DH': 'helada'})
clima_coruna= clima_coruna.rename(columns={'DN': 'nieve'})
clima_coruna= clima_coruna.rename(columns={'DR': 'lluvia'})
clima_coruna= clima_coruna.rename(columns={'DT': 'tormenta'})
##insertamos una columna con la ciudad
clima_coruna['PROVINCIA']="Coruña (A)"



#Bilbao read_csv(StringIO(data),sep=",",engine='c',skipinitialspace=True)
clima_bilbao = pd.read_csv ("valoresclimatologicos_bilbao-aeropuerto.csv",sep='","',decimal='.',engine='python',skiprows=2,index_col=None)
clima_bilbao= clima_bilbao.rename(columns={'TM': 'temperatura'})
clima_bilbao= clima_bilbao.rename(columns={'DR': 'lluvia'})
clima_bilbao= clima_bilbao.rename(columns={'DT': 'tormenta'})
clima_bilbao= clima_bilbao.rename(columns={'DH': 'helada'})
clima_bilbao= clima_bilbao.rename(columns={'DN': 'nieve'})

##insertamos una columna con la ciudad
clima_bilbao['PROVINCIA']="Bizkaia"


#Barcelona
clima_barcelona = pd.read_csv ("valoresclimatologicos_barcelona-fabra.csv", sep='","', decimal='.',engine='python',skiprows=2,index_col=None)
clima_barcelona= clima_barcelona.rename(columns={'TM': 'temperatura'})
clima_barcelona= clima_barcelona.rename(columns={'DR': 'lluvia'})
clima_barcelona= clima_barcelona.rename(columns={'DT': 'tormenta'})
clima_barcelona= clima_barcelona.rename(columns={'DH': 'helada'})
clima_barcelona= clima_barcelona.rename(columns={'DN': 'nieve'})

##insertamos una columna con la ciudad
clima_barcelona['PROVINCIA']="Barcelona"


#Madrid
clima_madrid = pd.read_csv ("valoresclimatologicos_madrid-retiro.csv", sep='","', decimal='.',engine='python',skiprows=2,index_col=None)
clima_madrid= clima_madrid.rename(columns={'TM': 'temperatura'})
clima_madrid= clima_madrid.rename(columns={'DR': 'lluvia'})
clima_madrid= clima_madrid.rename(columns={'DT': 'tormenta'})
clima_madrid= clima_madrid.rename(columns={'DH': 'helada'})
clima_madrid= clima_madrid.rename(columns={'DN': 'nieve'})

##insertamos una columna con la ciudad
clima_madrid['PROVINCIA']="Madrid"


#Sevilla
clima_sevilla = pd.read_csv ("valoresclimatologicos_sevilla-aeropuerto.csv", sep='","', decimal='.',engine='python',skiprows=2,index_col=None)
clima_sevilla = clima_sevilla.rename(columns={'TM': 'temperatura'})
clima_sevilla = clima_sevilla.rename(columns={'DR': 'lluvia'})
clima_sevilla = clima_sevilla.rename(columns={'DT': 'tormenta'})
clima_sevilla = clima_sevilla.rename(columns={'DH': 'helada'})
clima_sevilla = clima_sevilla.rename(columns={'DN': 'nieve'})


##insertamos una columna con la ciudad
clima_sevilla['PROVINCIA']="Sevilla"


#Fusionamos los ficheros
Clima_coruna_bilbao=clima_coruna.append(clima_bilbao, ignore_index=True)
Clima_coruna_bilbao_barcelona=Clima_coruna_bilbao.append(clima_barcelona, ignore_index=True)
Clima_coruna_bilbao_barcelona_madrid=Clima_coruna_bilbao_barcelona.append(clima_madrid, ignore_index=True)
Clima_coruna_bilbao_barcelona_madrid_sevilla=Clima_coruna_bilbao_barcelona_madrid.append(clima_sevilla, ignore_index=True)
Clima_total=Clima_coruna_bilbao_barcelona_madrid_sevilla

#Limpiamos los ficheros que sobran
del(Clima_coruna_bilbao)
del(Clima_coruna_bilbao_barcelona)
del(Clima_coruna_bilbao_barcelona_madrid)
del(Clima_coruna_bilbao_barcelona_madrid_sevilla)



#limpiamos los meses
Clima_total= Clima_total.rename(columns={'"Mes': 'Mes'})
Clima_total= Clima_total.rename(columns={'I",': 'I'})
Clima_total=Clima_total.replace({'"Enero':'Enero'})
Clima_total=Clima_total.replace({'"Febrero':'Febrero'})
Clima_total=Clima_total.replace({'"Marzo':'Marzo'})
Clima_total=Clima_total.replace({'"Abril':'Abril'})
Clima_total=Clima_total.replace({'"Mayo':'Mayo'})
Clima_total=Clima_total.replace({'"Junio':'Junio'})
Clima_total=Clima_total.replace({'"Julio':'Julio'})
Clima_total=Clima_total.replace({'"Agosto':'Agosto'})
Clima_total=Clima_total.replace({'"Septiembre':'Septiembre'})
Clima_total=Clima_total.replace({'"Octubre':'Octubre'})
Clima_total=Clima_total.replace({'"Noviembre':'Noviembre'})
Clima_total=Clima_total.replace({'"Diciembre':'Diciembre'})
Clima_total=Clima_total.replace({'"Año':'Año'})



#Drop columnas no utilizadas
Clima_total= Clima_total.drop(['T','Tm','R','H','DF','DD','I'],axis=1)

#Sumamos las columnas del mal tiempo (en días) 
Clima_total['dias'] = Clima_total[["lluvia", "nieve","tormenta","helada"]].sum(axis=1)
#Días de mal tiempo por ciudad por año (sacar la gráfica : ayuda Sergio?)
Mal_tiempo=Clima_total[Clima_total.Mes=="Año"]
Clima_total.dtypes

col = list(Clima_total.columns)
#Filtro 1: Columnas que tienen una A
filtro1 = [col for col in Clima_total  if  col.find('PROVINCIA')>=0 or col.find('dias')>=0]
#Eliminar
Clima_total_drop = Clima_total.drop(columns=filtro1)
#Seleccionar
clima_total1 = Clima_total[filtro1]


clima_total1 = clima_total1[clima_total1.dias > 50]
clima_total1.dtypes


#hacer porcentajes y grafico
clima_total1['m']=clima_total1['dias']*100/360
plt.bar(clima_total1.PROVINCIA, clima_total1['m'],edgecolor='blue')
plt.xticks(clima_total1.PROVINCIA,rotation='vertical')
plt.ylabel('Dias Climatologicos Adversos')
plt.xlabel(r"$\it{"+'Source'+ "}$"+': AMET cllima por Provincia')
plt.title('Figura 4. Top 5 Dias Climatologicos Adversos por Provincia')
props = dict(boxstyle='round', facecolor='white', lw=0.5)


#Guardamos el top 3 (menos días de mal tiempo)
ClimaD_top3=Mal_tiempo.nsmallest(3,'dias')

#Creamos subsetting para crear la gráfica de la temperatura media
T_barcelona = Clima_total[Clima_total['PROVINCIA'] == 'Barcelona']
T_barcelona=T_barcelona.iloc[:, 0:2] # Primeras cinco columnas
T_bilbao = Clima_total[Clima_total['PROVINCIA'] == 'Bizkaia']
T_bilbao=T_bilbao.iloc[:, 0:2] # Primeras cinco columnas
T_coruna = Clima_total[Clima_total['PROVINCIA'] == 'Coruña (A)']
T_coruna=T_coruna.iloc[:, 0:2] 
T_madrid = Clima_total[Clima_total['PROVINCIA'] == 'Madrid']
T_madrid=T_madrid.iloc[:, 0:2] 
T_sevilla=  Clima_total[Clima_total['PROVINCIA'] == 'Sevilla']
T_sevilla=T_sevilla.iloc[:, 0:2] 

plt.plot(T_barcelona.Mes,T_barcelona.temperatura, "b", label="Barcelona")
plt.plot(T_madrid.Mes,T_madrid.temperatura, "g", label="Madrid")
plt.plot(T_sevilla.Mes,T_sevilla.temperatura, "c", label="Sevilla")
plt.xticks(rotation=90)
plt.legend(loc="best")
plt.ylabel('Temperatura media')
plt.title('Figura 5. Top 3 Temperatura media por provincia')
plt.xlabel(r"$\it{"+'Source'+ "}$"+': AMET cllima por Provincia')
plt.show()




del(T_barcelona)
del(T_bilbao)
del(T_coruna)
del(T_madrid)
del(T_sevilla)
del(col)
del(filtro1)
del(clima_madrid)
del(clima_sevilla)
del(clima_coruna)
del(clima_bilbao)
del(Clima_total_drop)
del(clima_barcelona)
del(clima_total1)
del(Mal_tiempo)
del(props)

ClimaD_top3.to_csv('ClimaD_top3.csv')
ClimaD_top3_1= pd.read_csv ('ClimaD_top3.csv', sep=',', decimal='.', skiprows=1, index_col=None)
col1=list(ClimaD_top3_1.columns)
ClimaD_top3_1.set_index(col1[1], inplace=True)
ClimaD_top3_1.to_csv('ClimaD_top3_1.csv', sep=',')

Clima_total.to_csv('Clima_total.csv')
Clima_total_1= pd.read_csv ('Clima_total.csv', sep=',', decimal='.',skiprows=1, index_col=None)
col2=list(Clima_total_1.columns)
Clima_total_1.set_index(col2[1], inplace=True)
Clima_total_1.set_index('0', inplace=True)
Clima_total_1.to_csv('Clima_total_1.csv', sep=',')

IneDgt_Top10.to_csv('IneDgt_Top10.csv')
IneDgt_Top10_1= pd.read_csv ('IneDgt_Top10.csv', sep=',', decimal='.',skiprows=1, index_col=None)
col3=list(IneDgt_Top10_1.columns)
IneDgt_Top10_1.set_index(col3[1], inplace=True)
IneDgt_Top10_1.to_csv('IneDgt_Top10_1.csv', sep=',')

IneDgt_Top5.to_csv('IneDgt_Top5.csv')
IneDgt_Top5_1= pd.read_csv ('IneDgt_Top5.csv', sep=',', decimal='.',skiprows=1, index_col=None)
col4=list(IneDgt_Top5_1.columns)
IneDgt_Top5_1.set_index(col4[1], inplace=True)
IneDgt_Top5_1.to_csv('IneDgt_Top5_1.csv', sep=',')

inetop10.to_csv('inetop10.csv')
inetop10_1= pd.read_csv ('inetop10.csv', sep=',', decimal='.',skiprows=1, index_col=None)
col5=list(inetop10_1.columns)
inetop10_1.set_index(col5[1], inplace=True)
inetop10_1.to_csv('IneDgt_Top5_1.csv', sep=',')







