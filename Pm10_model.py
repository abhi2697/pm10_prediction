# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 18:58:42 2023

@author: abppa
"""
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.formula.api import ols
from sklearn.ensemble import RandomForestRegressor

sns.set(rc={'figure.figsize':((11.7,8.27))})
df1=pd.read_excel('ap_data.xlsx',na_values=["None"])
df2=df1.copy()
df2.info()
df2.isnull().sum()

df2['BP'].fillna(value=df2['BP'].mode()[0],inplace=True)
df2['AT'].fillna(value=df2['AT'].mode()[0],inplace=True)
df2['SR'].fillna(value=df2['SR'].mode()[0],inplace=True)
df2['pm10'].fillna(value=df2['pm10'].mode()[0],inplace=True)
df2['SO2'].fillna(value=df2['SO2'].mode()[0],inplace=True)
df2['NO'].fillna(value=df2['NO'].mode()[0],inplace=True)
df2['SR']=df2['SR'].astype(float)

df2=df2.drop(columns=['YEAR','MO','DY','date  of cpcb'],axis=1)
df2.columns
df2['sqrtpm10']=np.sqrt(df2['pm10'])

x1=df2[[ 'NOx', 'CO','T2M', 'T2MDEW',
        'RH2M', 'PRECTOTCORR', 'PS',
       'WS10M','SR']]
 
y1=df2['pm10']
y2=df2['sqrtpm10']
x1.info()
des1=x1.describe()
x_train,x_test,y_train,y_test=train_test_split(x1,y1,test_size=0.2,random_state=3)
lgr=LinearRegression(fit_intercept=True)
model_mlr_pm10=lgr.fit(x_train,y_train)
print(model_mlr_pm10.score(x_train,y_train))
mlr_prediction_pm10=lgr.predict(x_test)
mlr_mse=mean_squared_error(y_test, mlr_prediction_pm10)
mlr_rmse_pm10=np.sqrt(mlr_mse)
print(mlr_rmse_pm10)
mlr_r2_train_pm10=model_mlr_pm10.score(x_train,y_train)
mlr_r2_test=model_mlr_pm10.score(x_test,y_test)
mlr_residual_pm10=y_test-mlr_prediction_pm10
mlr_MAR_pm10=mlr_residual_pm10.mean()
sns.regplot(x=mlr_prediction_pm10,y=mlr_residual_pm10,scatter=True,fit_reg=True)
print(mlr_r2_train_pm10,mlr_r2_test)
x_train,x_test,y2_train,y2_test=train_test_split(x1,y2,test_size=0.2,random_state=3)
lgr=LinearRegression(fit_intercept=True)
model_mlr_sqrtpm10=lgr.fit(x_train,y2_train)
print(model_mlr_sqrtpm10.score(x_train,y2_train))

mlr_prediction_sqrtpm10=lgr.predict(x_test)
mlr_mse=mean_squared_error(y2_test, mlr_prediction_sqrtpm10)
mlr_rmse_sqrtpm10=np.sqrt(mlr_mse)
print(mlr_rmse_sqrtpm10)
mlr_r2_train_sqrtpm10=model_mlr_sqrtpm10.score(x_train,y2_train)
mlr_r2_test=model_mlr_sqrtpm10.score(x_test,y2_test)
mlr_residual=y2_test-mlr_prediction_sqrtpm10
mlr_MAR_sqrtpm10=mlr_residual.mean()
sns.regplot(x=mlr_prediction_sqrtpm10,y=mlr_residual,scatter=True,fit_reg=True)
print(mlr_r2_train_sqrtpm10,mlr_r2_test)
rf=RandomForestRegressor(n_estimators=100,max_depth=100,max_features='auto',min_samples_leaf=4,min_samples_split=10,random_state=4)
model_rf_pm10=rf.fit(x_train,y_train)
print(model_rf_pm10.score(x_train,y_train))
print(model_rf_pm10.score(x_test,y_test))
rf_r2score_pm10=model_rf_pm10.score(x_train,y_train)
rf_prediction_pm10=rf.predict(x_test)
rf_mse=mean_squared_error(y_test,rf_prediction_pm10)
rf_residual_pm10=y_test-rf_prediction_pm10
rf_MAR_pm10=rf_residual_pm10.mean()
rf_rmse_pm10=np.sqrt(rf_mse)
print(rf_rmse_pm10)
rf=RandomForestRegressor(n_estimators=100,max_depth=100,max_features='auto',min_samples_leaf=4,min_samples_split=10,random_state=4)
model_rf=rf.fit(x_train,y2_train)
print(model_rf.score(x_train,y2_train))
print(model_rf.score(x_test,y2_test))
rf_r2score_sqrtpm10=model_rf.score(x_train,y2_train)
rf_prediction=rf.predict(x_test)
rf_residual=y2_test-rf_prediction
rf_MAR_sqrtpm10=rf_residual.mean()
rf_mse=mean_squared_error(y2_test,rf_prediction)
rf_rmse_sqrtpm10=np.sqrt(rf_mse)
print(rf_rmse_sqrtpm10)
result_mlr_pm10=[mlr_MAR_pm10,mlr_r2_train_pm10,mlr_rmse_pm10]
result_mlr_sqrtpm10=[mlr_MAR_sqrtpm10,mlr_r2_train_sqrtpm10,mlr_rmse_sqrtpm10]
result_rf_pm10=[rf_MAR_pm10,rf_r2score_pm10,rf_rmse_pm10]
result_rf_sqrtpm10=[rf_MAR_sqrtpm10,rf_r2score_sqrtpm10,rf_rmse_sqrtpm10]
result_site_1=[result_mlr_pm10,result_mlr_sqrtpm10,result_rf_pm10,result_rf_sqrtpm10]
result_mlr_pm10=pd.Series(result_mlr_pm10)
result_mlr_sqrtpm10=pd.Series(result_mlr_sqrtpm10)
result_rf_pm10=pd.Series(result_rf_pm10)
result_rf_sqrtpm10=pd.Series(result_rf_sqrtpm10)
file_name='ap_data_attributes_max_r2_site1.xlsx'
file_name1='result_site1.xlsx'
result_site_1.to_word(file_name1)
file_name_y1='outputpm10.xlsx'
y1.to_excel(file_name_y1)
file_name_y2='outputsqrtpm10.xlsx'
y2.to_excel(file_name_y2)

