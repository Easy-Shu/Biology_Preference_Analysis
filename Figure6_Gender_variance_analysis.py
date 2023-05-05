# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 12:30:18 2019

@author: Peter_Zhang
"""
import pandas as pd
import numpy as np
from plotnine import *
#from plotnine.data import *
import matplotlib.pyplot as plt 
from functools import reduce
import scipy.stats as stats
from scipy.cluster import hierarchy  #用于进行层次聚类，话层次聚类图的工具包
from pandas.api.types import CategoricalDtype

def trianle_down(a,b,fill):
    x=np.array([0,1,1])+a
    y=np.array([0,0,1])+b
    group=str(a)+"_down_"+str(b)
    df_trianledown=pd.DataFrame(dict(x=x,y=y,group=group,fill=fill))
    return df_trianledown

def trianle_up(a,b,fill):
    x=np.array([0,0,1])+a
    y=np.array([0,1,1])+b
    group=str(a)+"_up_"+str(b)
    df_trianledown=pd.DataFrame(dict(x=x,y=y,group=group,fill=fill))
    return df_trianledown

def trianle(df_xyz,type="up"):
    df_new=pd.DataFrame(columns=['x','y','group','fill'])
    if(type=="up"):
        df_temp=df_xyz.apply(lambda x: trianle_up(x['x'],x['y'],x['z']),axis=1)
        for x in df_temp.values:
            df_new=df_new.append(x)
    if(type=="down"):
        df_temp=df_xyz.apply(lambda x: trianle_down(x['x'],x['y'],x['z']),axis=1)
        for x in df_temp.values:
            df_new=df_new.append(x)
    return df_new

df_xyz=pd.DataFrame(dict(x=[1,2,3],y=[1,2,3],z=[1,2,3]))

df=trianle(df_xyz,type="up").append(trianle(df_xyz,type="down"))

df['fill']=df['fill'].astype(float)

plot=(ggplot() +
geom_polygon(df,aes(x='x', y='y',group='group',fill='fill'),color='k')+
scale_fill_gradient2(low="#00A08A",mid="white",high="#FF0000",midpoint = np.mean(df.fill)))  
print(plot)  
    


file = open('Anamial_Feature_Analysis.csv')
mydata=pd.read_csv(file,encoding = "utf-8")
file.close()


features=mydata.columns.values.tolist()[3:]

df_mean=mydata.groupby(['Animal'],as_index=True).mean()[features]

fig=plt.figure(figsize=(10, 10), dpi= 80)
Z = hierarchy.linkage(df_mean.values, method ='ward',metric='euclidean')
dend_left=hierarchy.dendrogram(Z,labels = df_mean.index,orientation='left')
plt.savefig('Figure_6_Gender_variance_analysis_left.pdf')

fig=plt.figure(figsize=(10, 10), dpi= 80)
Z = hierarchy.linkage(df_mean.values.T, method ='ward',metric='euclidean')
dend_top=hierarchy.dendrogram(Z,labels = df_mean.columns,orientation='top')
plt.savefig('Figure_6_Gender_variance_analysis_top.pdf')

features=dend_top['ivl']#mydata.columns.values.tolist()[3:]
samples=dend_left['ivl']#mydata.Animal.unique()
N_feature=len(features)
N_Sample=len(samples)


df_gender=mydata.groupby(['Animal','gender'],as_index=False).mean()

df_melt=pd.melt(df_gender,id_vars=['Animal','gender','ID'])
cat_features = CategoricalDtype(categories=features, ordered=True)
df_melt['variable']=df_melt['variable'].astype(cat_features)
cat_samples= CategoricalDtype(categories=samples, ordered=True)
df_melt['Animal']=df_melt['Animal'].astype(cat_samples)


df_male=df_melt[df_melt.gender=='男'][['Animal','variable','value']]
df_male['x']=df_male['Animal'].values.codes
df_male['y']=df_male['variable'].values.codes
df_male['z']=df_male['value']

df_female=df_melt[df_melt.gender=='女'][['Animal','variable','value']]
df_female['x']=df_female['Animal'].values.codes
df_female['y']=df_female['variable'].values.codes
df_female['z']=df_female['value']

df_display=trianle(df_male,type="up").append(trianle(df_female,type="down"))


df_display['fill']=df_display['fill'].astype(float)
df_display['x']=df_display['x'].astype(float)
df_display['y']=df_display['y'].astype(float)


T_value_Fabric, P_value_Fabric=np.full([N_Sample,N_feature],np.NAN),np.full([N_Sample,N_feature],np.NAN)

for i in range(0,N_Sample):
    for j in range(0,N_feature):
        x=np.array(mydata[(mydata['Animal']==samples[i]) & (mydata['gender']=='男')][features[j]])
        y=np.array(mydata[(mydata['Animal']==samples[i]) & (mydata['gender']=='女')][features[j]])
        T_value_Fabric[i,j], P_value_Fabric[i,j]=stats.ttest_ind(x, y, equal_var=True)#stats.f_oneway(x,y)#   
       
P_value_Fabric=pd.DataFrame(P_value_Fabric,columns=features)
P_value_Fabric['Animal']=samples#P_value_Fabric.index.values+1
df_Pvalue=pd.melt(P_value_Fabric,id_vars='Animal')
df_Pvalue['variable']=df_Pvalue['variable'].astype(cat_features)
df_Pvalue['Animal']=df_Pvalue['Animal'].astype(cat_samples)

df_Pvalue['x']=df_Pvalue['Animal'].values.codes+0.5
df_Pvalue['y']=df_Pvalue['variable'].values.codes+0.5

df_Pvalue['lable']=['**' if x<=0.01 else '*' if x<=0.05 else '' for x in df_Pvalue['value'] ]

base_plot=(ggplot() 
+geom_polygon(df_display,aes(x='x', y='y',group='group',fill='fill'),color=None)
+geom_tile(df_Pvalue,aes(x='x', y='y'),fill='none',color='k',size=0.25)
+geom_text(df_Pvalue,aes(x='x', y='y',label='lable'),size=15,ha='center',va='center')
+scale_fill_gradient2(name='mean\nvalue',low="#3069A5",mid="white",high="#FF0000",midpoint = np.mean(df_display.fill))
+scale_y_continuous(breaks=np.arange(0,N_feature,1)+0.5,labels=features,expand=[0,0])
+scale_x_continuous(breaks=np.arange(0,N_Sample,1)+0.5,labels=samples,expand=[0,0])
+xlab('Animals')
+ylab('Features')
#+geom_text(size=3,colour="white")
+coord_equal()
+theme_matplotlib()
+theme(
    #text=element_text(size=15,face="plain",color="black"),
    axis_title=element_text(size=14,face="plain",color="black"),
    axis_text_y = element_text(size=13,face="plain",color="black"),
    axis_text_x = element_text(size=13,face="plain",color="black",angle = 90),
    legend_title = element_text(size=13,face="plain",color="black"),
    legend_text= element_text(size=14,face="plain",color="black"),
    #plot_margin=0,
    #legend_position='none',
    #legend_position = (0,0),
    figure_size = (10, 10),
    dpi = 50
))
print(base_plot)  
base_plot.save('Figure_6_Gender_variance_analysis.pdf')


# #-----------------------------------------------
# import seaborn as sns
# df_mean=mydata.groupby(['Animal'],as_index=True).mean()[features]

# #Z = hierarchy.linkage(df_mean, method ='ward',metric='euclidean')
# #dend=hierarchy.dendrogram(Z,labels = df_mean.index)

# h=sns.clustermap(df_mean, center=np.mean(df_mean.values), cmap="RdYlBu_r",
#                linewidths=.15,linecolor='k', figsize=(8, 8))
# #sns.savefig('clustering.pdf')