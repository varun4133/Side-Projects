#importing modules

import tensorflow as tf

tf.compat.v1.logging.set_verbosity
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


import pandas as pd
import numpy as np


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse

import seaborn as sns
import matplotlib.pyplot as plt





#setting random seed
tf.random.set_seed(2)




#Reading in Files

train_file='data.csv'
validation_file='data_validation.csv'
test_file='data_test.csv'

df_train=pd.read_csv(train_file,header=0,index_col=0)

df_validate=pd.read_csv(validation_file,header=0,index_col=0)
df_test=pd.read_csv(test_file,header=0,index_col=0)


#Getting data into more plottable form by melting

df_train_melted=df_train.melt(id_vars=['rpm','MFR'],var_name='Target',value_name='Target Value')
df_validate_melted=df_validate.melt(id_vars=['rpm','MFR'],var_name='Target',value_name='Target Value')

#Plotting Data
fig=plt.figure()
ax=sns.relplot(x='MFR',y='Target Value',data=df_train_melted,col='Target',hue='rpm',palette='tab10',facet_kws=dict(sharey=False))

fig=plt.figure()

ax=sns.relplot(x='MFR',y='Target Value',data=df_validate_melted,col='Target',hue='rpm',kind='scatter',palette='tab10',facet_kws=dict(sharey=False))



#Setting Feature Df

X_train=df_train[['rpm','MFR']]
X_validate=df_validate[['rpm','MFR']]
X_test=df_test[['rpm','MFR']]

#Setting Target Df
y_train=df_train[['Kv','eff_tt','eff_ts','ptr','psr']]
y_validate=df_validate[['Kv','eff_tt','eff_ts','ptr','psr']]
y_test=df_test[['Kv','eff_tt','eff_ts','ptr','psr']]





#Scaling

# paramgrid={}
# paramlist=[]

# def Param_Transform(df):
#         rpm_val=df['rpm'].iloc[0]
#         mean=df['MFR'].mean()
#         std=df['MFR'].std()
#         df['MFR_new']=(df['MFR']-mean)/std
#         paramgrid[rpm_val]=(mean,std)
#         paramlist.append(rpm_val)
#         return df
     
# def transform(series,mean,std):
#     series=(series-mean)/std
#     return series

# def interpolate(left_series,right_series,left_rpm,right_rpm,rpm):
#     new_series=(((right_series-left_series)/(right_rpm-left_rpm))*(rpm-left_rpm))+left_series
#     return new_series

# def Test_Transform(df):
#     rpm_val=df['rpm'].iloc[0]
#     if paramgrid.get(rpm_val):
#         mean,std=paramgrid[rpm_val]
#         df['MFR_new']=(df['MFR']-mean)/std
#     else:
#         counter_list=rpm_val-paramlist
#         left_list=counter_list[counter_list>0]
#         left_rpm_val=rpm_val-min(left_list)
#         mean,std=paramgrid[left_rpm_val]
#         left_series=transform(df['MFR'],mean,std)
        
#         right_list=counter_list[counter_list<0]
#         right_rpm_val=rpm_val+min(abs(right_list))
#         mean,std=paramgrid[right_rpm_val]
#         right_series=transform(df['MFR'],mean,std)
#         df['MFR_new']=interpolate(left_series,right_series,left_rpm_val,right_rpm_val,rpm_val)
        
#     return df


# X_train=X_train.groupby('rpm').apply(Param_Transform)
# X_validate=X_validate.groupby('rpm').apply(Test_Transform)
# X_test=X_test.groupby('rpm').apply(Test_Transform)


ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_validate=ss.transform(X_validate)
X_test=ss.transform(X_test)





ss2=StandardScaler()
y_train=ss2.fit_transform(y_train)
y_validate=ss2.transform(y_validate)






#Creating Model + Compiling
model=Sequential()
firstlayer=Dense(5,activation='elu',input_shape=(2,),kernel_initializer='he_uniform')
secondlayer=Dense(7    ,activation='elu',kernel_initializer='he_uniform')


outputlayer=Dense(5)


model.add(firstlayer)
   
model.add(secondlayer)

model.add(outputlayer)

optimizer=Adam(learning_rate=0.07)
model.compile(optimizer=optimizer,loss='mean_squared_error')


#Training model + plotting training history

history=model.fit(X_train,y_train,validation_data=(X_validate,y_validate),epochs=1500,shuffle=True,batch_size=41)  
training_loss=history.history['loss']
val_loss=history.history['loss']
epochs=np.arange(1,len(val_loss)+1)
fig=plt.figure()
sns.lineplot(epochs,training_loss)
sns.lineplot(epochs,val_loss)
plt.legend(labels=['Training Loss','Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')



#Transforming true target values into more useable dataframe
y_validate_true=pd.DataFrame(ss2.inverse_transform(y_validate)).melt(var_name='label',value_name='True')
y_test_true=pd.DataFrame(y_test).melt(var_name='label',value_name='True')

#Creating dataframe of predicted targets +transforming
y_validate_pred=pd.DataFrame(model.predict(X_validate))
y_test_pred=pd.DataFrame(model.predict(X_test))

y_validate_pred=pd.DataFrame(ss2.inverse_transform(y_validate_pred))
y_test_pred=pd.DataFrame(ss2.inverse_transform(y_test_pred))

y_validate_pred=y_validate_pred.melt(var_name='label',value_name='Prediction')

y_test_pred=y_test_pred.melt(var_name='label',value_name='Prediction')


#Merging true dfs and predicted dfs
y_validate_true=y_validate_true.iloc[:,1]
y_test_true=y_test_true.iloc[:,1]

y_validate_df=pd.merge(y_validate_true,y_validate_pred,left_index=True,right_index=True)
y_test_df=pd.merge(y_test_true,y_test_pred,left_index=True,right_index=True)


#Calculating rms error
def error_calc(df):
    error=mse(df['True'],df['Prediction'])
    return pd.DataFrame([error])     

validate_mse=y_validate_df.groupby('label',as_index=False).apply(error_calc).reset_index(drop=True)
test_mse=y_test_df.groupby('label',as_index=False).apply(error_calc).reset_index(drop=True)

overall_validate_mse=validate_mse.mean().values[0]
overall_test_mse=test_mse.mean().values[0]

