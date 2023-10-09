# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 23:38:57 2023

@author: abppa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df1=pd.read_excel('ap_data_attributes_max_r2_site1.xlsx')
df1.drop(columns=['Unnamed: 0'],axis=1,inplace=True)
df2=df1.copy()
df1.columns
df1.info()
sns.pairplot(x_vars=['NOx', 'CO', 'T2M', 'T2MDEW', 'RH2M', 'PRECTOTCORR', 'PS', 'WS10M',
       'SR'], y_vars=['pm10','sqrtpm10'],data=df2)

c=df2.corr()

sns.heatmap(df2.corr())
import statsmodels.api as sm
x=df2[[ 'NOx', 'CO','T2M', 'T2MDEW',
        'RH2M', 'PRECTOTCORR', 'PS',
       'WS10M','SR']]
d=x.describe()
d=round(d,2)
y=df1[['pm10']]
y2=df1[['sqrtpm10']]
x_train_sm=sm.add_constant(x)
lr=sm.OLS(y,x_train_sm).fit()
lr2=sm.OLS(y2,x_train_sm).fit()
k=lr.summary()
k2=lr2.summary()
k
k2
with open('summary_site1_mlr.txt','w') as fh:
    fh.write(lr.summary().as_text())
with open('summary_site1_mlr_sqrt.txt','w') as fh:
        fh.write(lr2.summary().as_text())

filename='correct_model_variable.xlsx'
filename1='correlatrion.xlsx'
filename2='ols_summary.doc'
filename4='ols_summary_sqrt.doc'
filename3='data_description.xlsx'
x.to_excel(filename)
c.to_excel(filename1)
k2.to_word(filename4)
d.to_excel(filename3)
