

import pandas as pd
import numpy as np
from pandas import read_csv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
#from mpl_toolkits.basemap import Basemap
import plotly.plotly as py
py.sign_in(username='roudgarian', api_key='o4hQBxVInVrbRO5I0Cji')
from plotly.graph_objs import *
import warnings
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from IPython.display import Image
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error





warnings.filterwarnings("ignore") 



training_df = pd.read_csv("trainingData.csv")
validation_df = pd.read_csv("validationData.csv")



df_tr = training_df
df_val = validation_df
df = training_df


X_train = df.iloc[:,:520]
for i in range(-30, 1):
    X_train=X_train[X_train !=i]
X_train = X_train[X_train.std()[X_train.std()!=0].index]
X_train = (X_train.replace(to_replace=100,value=np.nan))
X_train = X_train.stack(dropna=False)


sns.distplot(X_train.dropna(),kde = False, color='blue',axlabel="RSSI in 3 Buildings",bins=15)



df_B1=df.loc[df['BUILDINGID'] == 0]
df_B2=df.loc[df['BUILDINGID'] == 1]
df_B3=df.loc[df['BUILDINGID'] == 2]


# In[9]:


X_train_B1 = df_B1.iloc[:,:520]
for i in range(-30, 1):
    X_train_B1=X_train_B1[X_train_B1 !=i]
X_train_B1 = X_train_B1[X_train_B1.std()[X_train_B1.std()!=0].index]
X_train_B1 = (X_train_B1.replace(to_replace=100,value=np.nan))
X_train_B1 = X_train_B1.stack(dropna=False)


# In[10]:


sns.distplot(X_train_B1.dropna(),kde = False,color='blue',axlabel="RSSI in Buildings No.1",bins=15)


# In[11]:


X_train_B2 = df_B2.iloc[:,:520]
for i in range(-30, 1):
    X_train_B2=X_train_B2[X_train_B2 !=i]
X_train_B2 = X_train_B2[X_train_B2.std()[X_train_B2.std()!=0].index]
X_train_B2 = (X_train_B2.replace(to_replace=100,value=np.nan))
X_train_B2 = X_train_B2.stack(dropna=False)


# In[12]:


sns.distplot(X_train_B2.dropna(),kde = False,color='blue',axlabel="RSSI in Buildings No.2",bins=15)


# In[13]:


X_train_B3 = df_B3.iloc[:,:520]
for i in range(-30, 1):
    X_train_B1=X_train_B3[X_train_B3 !=i]
X_train_B3 = X_train_B3[X_train_B3.std()[X_train_B3.std()!=0].index]
X_train_B3 = (X_train_B3.replace(to_replace=100,value=np.nan))
X_train_B3 = X_train_B3.stack(dropna=False)


# In[14]:


sns.distplot(X_train_B3.dropna(),kde = False,color='blue',axlabel="RSSI in Buildings No.3",bins=15)


# In[5]:


df0_tr=df_tr.iloc[:,520:529]
df1_tr=df_tr.iloc[:,0:520]


# In[6]:


df0_val=df_val.iloc[:,520:529]
df1_val=df_val.iloc[:,0:520]


# In[7]:


df1_tr = df1_tr[df1_tr.std()[df1_tr.std()!=0].index]#35 attributes will remove by this
df1_tr = df1_tr.mask(df1_tr ==100 , -105)


# In[8]:


df1_val = df1_val[df1_val.std()[df1_val.std()!=0].index]#35 attributes will remove by this
df1_val = df1_val.mask(df1_val ==100 , -105)



# In[9]:


df1 = pd.concat([df1_tr, df0_tr], axis=1)
df2 = pd.concat([df1_val, df0_val], axis=1)


# In[10]:


df1 = df1[df1.USERID != 6]# 18957 rows remain after we remove USERID 6
df2 = df2[df2.USERID != 6]


# In[11]:


df1 = df1.drop_duplicates( keep='last') #Remove all repetetive rows (18323 remain)
df2 = df2.drop_duplicates( keep='last')


# In[12]:


df1 = df1[df1.loc[:,'WAP001': 'WAP519'].std(axis=1)!=0] #18250 rows remain after we remove row Standard.Devision=0
df2 = df2[df2.loc[:,'WAP001': 'WAP520'].std(axis=1)!=0]



# In[13]:


for i in range(-30, 0):
    df1=df1[df1 !=i]


# In[14]:


for i in range(-30, 0):
    df2=df2[df2 !=i]


# In[15]:


df1=df1.fillna(-150)
df2=df2.fillna(-150)


# In[16]:


df = df1[df1.columns & df2.columns] # Matching the column names of two data-frames 


# In[19]:


df_bil.to_csv('DF.csv')



# In[18]:


df_bil=df.drop(['SPACEID','RELATIVEPOSITION','USERID','PHONEID','TIMESTAMP','LONGITUDE','LATITUDE','FLOOR','BUILDINGID'], axis=1)


# In[20]:


X =df_bil
y = df.BUILDINGID


# In[21]:


df_bil.head()


# In[260]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=None)


# In[261]:


RF_b = RandomForestClassifier(n_estimators=100)
model_rf_b=RF_b.fit(X_train, y_train)
y_pred_rf_b = RF_b.predict(X_test)


# In[368]:


print(metrics.accuracy_score(y_test,y_pred_rf_b))


# In[263]:


scores_rf_b = cross_val_score(model_rf_b, X, y, cv=10,scoring='neg_mean_squared_error')
predictions_rf_b=cross_val_predict(model_rf_b,X,y,cv=10)


# In[369]:


mse_scores_rf_b = -scores_rf_b
accuracy_rf_b=metrics.r2_score(y,predictions_rf_b)
rmse_rf_b = np.sqrt(mse_scores_rf_b)
cohen_score_rf_b = cohen_kappa_score(y, predictions_rf_b)
print("Random Forest ACC:  ",accuracy_rf_b)
print("Random Forest RMSE: ", rmse_rf_b.mean())
print("Random Forest Kappa:",cohen_score_rf_b)
#Random Forest ACC:   1.0
#Random Forest RMSE:  0.006194932847291901
#Random Forest Kappa: 1.0


# In[269]:


np.savetxt("BUILDINGID.csv", predictions_rf_b, header="BUILDINGID")


# In[22]:


df3=read_csv('DF.csv')


# In[23]:


X1 =df3
y1 = df.FLOOR


# In[25]:


X1.head()


# In[26]:


X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.25, random_state=None)


# In[37]:


knn_f = KNeighborsClassifier(n_neighbors=1)
model_rf_f=knn_f.fit(X1_train, y1_train)
y_pred_rf_f = knn_f.predict(X1_test)
#print(metrics.accuracy_score(y_test, y_pred))


# In[38]:


print(metrics.r2_score(y1_test,y_pred_rf_f))


# In[44]:


scores_kkn_f = cross_val_score(model_rf_f, X1, y1, cv=10,scoring='neg_mean_squared_error')
predictions_knn_f=cross_val_predict(model_rf_f,X1,y1,cv=10)


# In[45]:


mse_scores_knn_f = -scores_kkn_f
accuracy_knn_f=metrics.r2_score(y1,predictions_knn_f)
rmse_knn_f = np.sqrt(mse_scores_knn_f)
cohen_score_knn_f = cohen_kappa_score(y1_test, y_pred_rf_f)
print("Random Forest ACC:  ",accuracy_knn_f)
print("Random Forest RMSE: ",rmse_knn_f.mean())
print("Random Forest Kappa:",cohen_score_knn_f)
#Random Forest ACC:   0.9932826875794417
#Random Forest RMSE:  0.049645133608777356
#Random Forest Kappa: 1.0


# In[47]:


np.savetxt("FLOOR.csv", predictions_knn_f, header="FLOOR")



# In[67]:


X2 =df4
y2 = df.LONGITUDE


# In[69]:


X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.25, random_state=None)


# In[76]:


knn_lo= KNeighborsRegressor(n_neighbors=8)
model_knn_lo=knn_lo.fit(X2_train, y2_train)
y_pred_knn_lo = knn_lo.predict(X2_test)


# In[77]:


print(metrics.r2_score(y2_test,y_pred_knn_lo))
#0.9999998447778784


# In[78]:


scores_knn_lo = cross_val_score(model_knn_lo, X2, y2, cv=10,scoring='neg_mean_squared_error')
predictions_knn_lo=cross_val_predict(model_knn_lo,X2,y2,cv=10)


# In[75]:


mse_scores_knn_lo = -scores_knn_lo
accuracy_knn_lo=metrics.r2_score(y2,predictions_knn_lo,multioutput='raw_values')
rmse_knn_lo = np.sqrt(mse_scores_knn_lo)
print("Random Forest ACC:  ",accuracy_knn_lo)
print("Random Forest RMSE: ", rmse_knn_lo.mean())
#Random Forest ACC:   0.9826746789262286
#Random Forest RMSE:  11.990642750798951


# In[ ]:


np.savetxt("LONGITUDE.csv", predictions_rf_lo, header="LONGITUDE", comments="")


# In[294]:


df5=read_csv('DF.csv')


# In[295]:


df5.head()


# In[296]:


X3 =df5
y3 = df.LATITUDE


# In[297]:


X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.25, random_state=None)


# In[298]:


RF_la= RandomForestRegressor(n_estimators=100)
model_rf_la=RF_la.fit(X3_train, y3_train)
y_pred_rf_la = RF_la.predict(X3_test)
print(metrics.r2_score(y3_test,y_pred_rf_la))
#0.9999983216904653


# In[299]:


scores_rf_la = cross_val_score(model_rf_la, X3, y3, cv=10,scoring='neg_mean_squared_error')
predictions_rf_la=cross_val_predict(model_rf_la,X3,y3,cv=10)


# In[301]:


mse_scores_rf_la = -scores_rf_la
accuracy_rf_la=metrics.r2_score(y3,predictions_rf_la)
rmse_rf_la = np.sqrt(mse_scores_rf_la)

print("Random Forest ACC:  ",accuracy_rf_la)
print("Random Forest RMSE: ", rmse_rf_la.mean())
#Random Forest ACC:   0.9790335034905598
#Random Forest RMSE:  8.625631508805103


# In[319]:


np.savetxt("LATITUDE.csv", predictions_rf_la, header="LATITUDE", comments="")


# In[445]:


df_val=df2[['WAP001','WAP008','WAP009','WAP010','WAP011','WAP012','WAP013','WAP014','WAP015','WAP016','WAP017','WAP018','WAP019','WAP020','WAP021','WAP022','WAP023','WAP024','WAP025','WAP026','WAP027','WAP028','WAP029','WAP030','WAP031','WAP032','WAP033','WAP034','WAP035','WAP036','WAP037','WAP038','WAP039','WAP040','WAP041','WAP042','WAP043','WAP044','WAP045','WAP046','WAP047','WAP048','WAP049','WAP050','WAP051','WAP052','WAP053','WAP054','WAP055','WAP056','WAP057','WAP058','WAP059','WAP060','WAP061','WAP062','WAP063','WAP064','WAP065','WAP066','WAP067','WAP068','WAP069','WAP070','WAP071','WAP072','WAP073','WAP074','WAP075','WAP076','WAP077','WAP078','WAP080','WAP081','WAP082','WAP083','WAP084','WAP085','WAP086','WAP087','WAP088','WAP089','WAP090','WAP091','WAP096','WAP097','WAP098','WAP099','WAP100','WAP101','WAP102','WAP103','WAP104','WAP105','WAP106','WAP107','WAP108','WAP109','WAP110','WAP111','WAP112','WAP113','WAP114','WAP115','WAP116','WAP117','WAP118','WAP119','WAP120','WAP121','WAP122','WAP123','WAP124','WAP125','WAP126','WAP127','WAP128','WAP129','WAP130','WAP131','WAP132','WAP134','WAP135','WAP136','WAP137','WAP138','WAP139','WAP140','WAP141','WAP142','WAP143','WAP144','WAP145','WAP146','WAP147','WAP148','WAP149','WAP150','WAP151','WAP153','WAP154','WAP155','WAP156','WAP161','WAP162','WAP164','WAP165','WAP166','WAP167','WAP168','WAP169','WAP170','WAP171','WAP172','WAP173','WAP174','WAP175','WAP176','WAP177','WAP178','WAP179','WAP180','WAP181','WAP182','WAP183','WAP184','WAP185','WAP186','WAP187','WAP188','WAP189','WAP190','WAP191','WAP192','WAP195','WAP196','WAP201','WAP202','WAP203','WAP204','WAP207','WAP216','WAP222','WAP223','WAP224','WAP225','WAP229','WAP232','WAP233','WAP234','WAP236','WAP237','WAP248','WAP249','WAP253','WAP255','WAP256','WAP257','WAP258','WAP259','WAP260','WAP261','WAP262','WAP263','WAP264','WAP265','WAP266','WAP267','WAP268','WAP269','WAP270','WAP271','WAP272','WAP273','WAP274','WAP275','WAP276','WAP277','WAP278','WAP279','WAP280','WAP281','WAP282','WAP283','WAP284','WAP285','WAP286','WAP287','WAP288','WAP289','WAP290','WAP292','WAP294','WAP295','WAP297','WAP299','WAP300','WAP305','WAP308','WAP309','WAP310','WAP311','WAP312','WAP313','WAP314','WAP315','WAP316','WAP317','WAP318','WAP319','WAP320','WAP321','WAP322','WAP323','WAP324','WAP325','WAP326','WAP327','WAP328','WAP329','WAP330','WAP331','WAP332','WAP334','WAP335','WAP336','WAP337','WAP338','WAP340','WAP341','WAP342','WAP343','WAP344','WAP345','WAP346','WAP348','WAP350','WAP351','WAP352','WAP354','WAP355','WAP356','WAP358','WAP359','WAP362','WAP364','WAP418','WAP422','WAP426','WAP434','WAP443','WAP449','WAP452','WAP456','WAP475','WAP478','WAP481','WAP483','WAP484','WAP486','WAP489','WAP492','WAP493','WAP494','WAP495','WAP496','WAP498','WAP499','WAP500','WAP501','WAP502','WAP508','BUILDINGID','FLOOR','LONGITUDE','LATITUDE']]
df_val_org=df2[['WAP001','WAP008','WAP009','WAP010','WAP011','WAP012','WAP013','WAP014','WAP015','WAP016','WAP017','WAP018','WAP019','WAP020','WAP021','WAP022','WAP023','WAP024','WAP025','WAP026','WAP027','WAP028','WAP029','WAP030','WAP031','WAP032','WAP033','WAP034','WAP035','WAP036','WAP037','WAP038','WAP039','WAP040','WAP041','WAP042','WAP043','WAP044','WAP045','WAP046','WAP047','WAP048','WAP049','WAP050','WAP051','WAP052','WAP053','WAP054','WAP055','WAP056','WAP057','WAP058','WAP059','WAP060','WAP061','WAP062','WAP063','WAP064','WAP065','WAP066','WAP067','WAP068','WAP069','WAP070','WAP071','WAP072','WAP073','WAP074','WAP075','WAP076','WAP077','WAP078','WAP080','WAP081','WAP082','WAP083','WAP084','WAP085','WAP086','WAP087','WAP088','WAP089','WAP090','WAP091','WAP096','WAP097','WAP098','WAP099','WAP100','WAP101','WAP102','WAP103','WAP104','WAP105','WAP106','WAP107','WAP108','WAP109','WAP110','WAP111','WAP112','WAP113','WAP114','WAP115','WAP116','WAP117','WAP118','WAP119','WAP120','WAP121','WAP122','WAP123','WAP124','WAP125','WAP126','WAP127','WAP128','WAP129','WAP130','WAP131','WAP132','WAP134','WAP135','WAP136','WAP137','WAP138','WAP139','WAP140','WAP141','WAP142','WAP143','WAP144','WAP145','WAP146','WAP147','WAP148','WAP149','WAP150','WAP151','WAP153','WAP154','WAP155','WAP156','WAP161','WAP162','WAP164','WAP165','WAP166','WAP167','WAP168','WAP169','WAP170','WAP171','WAP172','WAP173','WAP174','WAP175','WAP176','WAP177','WAP178','WAP179','WAP180','WAP181','WAP182','WAP183','WAP184','WAP185','WAP186','WAP187','WAP188','WAP189','WAP190','WAP191','WAP192','WAP195','WAP196','WAP201','WAP202','WAP203','WAP204','WAP207','WAP216','WAP222','WAP223','WAP224','WAP225','WAP229','WAP232','WAP233','WAP234','WAP236','WAP237','WAP248','WAP249','WAP253','WAP255','WAP256','WAP257','WAP258','WAP259','WAP260','WAP261','WAP262','WAP263','WAP264','WAP265','WAP266','WAP267','WAP268','WAP269','WAP270','WAP271','WAP272','WAP273','WAP274','WAP275','WAP276','WAP277','WAP278','WAP279','WAP280','WAP281','WAP282','WAP283','WAP284','WAP285','WAP286','WAP287','WAP288','WAP289','WAP290','WAP292','WAP294','WAP295','WAP297','WAP299','WAP300','WAP305','WAP308','WAP309','WAP310','WAP311','WAP312','WAP313','WAP314','WAP315','WAP316','WAP317','WAP318','WAP319','WAP320','WAP321','WAP322','WAP323','WAP324','WAP325','WAP326','WAP327','WAP328','WAP329','WAP330','WAP331','WAP332','WAP334','WAP335','WAP336','WAP337','WAP338','WAP340','WAP341','WAP342','WAP343','WAP344','WAP345','WAP346','WAP348','WAP350','WAP351','WAP352','WAP354','WAP355','WAP356','WAP358','WAP359','WAP362','WAP364','WAP418','WAP422','WAP426','WAP434','WAP443','WAP449','WAP452','WAP456','WAP475','WAP478','WAP481','WAP483','WAP484','WAP486','WAP489','WAP492','WAP493','WAP494','WAP495','WAP496','WAP498','WAP499','WAP500','WAP501','WAP502','WAP508','BUILDINGID','FLOOR','LONGITUDE','LATITUDE']]


# In[446]:


df_val_org.to_csv('DF_VAL_O.csv')


# In[322]:


y_val_b=df_val.BUILDINGID


# In[323]:


df_val_b=df_val.drop(['FLOOR','LONGITUDE','BUILDINGID','LATITUDE'], axis=1)


# In[324]:


X_val=df_val_b


# In[167]:


#df_val.to_csv('df_val.csv')


# In[327]:


X_val_train, X_val_test, y_val_train, y_val_test = train_test_split(X_val, y_val, test_size=0.25, random_state=None)


# In[328]:


y_pred_rf_b_val = RF_b.predict(X_val_test)
print(metrics.r2_score(y_val_test,y_pred_rf_b_val))


# In[329]:


scores_rf_b_val = cross_val_score(model_rf_b, X_val, y_val, cv=10,scoring='neg_mean_squared_error')
predictions_rf_b_val=cross_val_predict(model_rf_b,X_val,y_val,cv=10)


# In[330]:


mse_scores_rf_b_val = -scores_rf_b_val
accuracy_rf_b_val=metrics.r2_score(y_val,predictions_rf_b_val)
rmse_rf_b_val = np.sqrt(mse_scores_rf_b_val)
cohen_score_rf_b_val = cohen_kappa_score(y_val, predictions_rf_b_val)
print("Random Forest ACC:  ",accuracy_rf_b_val)
print("Random Forest RMSE: ", rmse_rf_b_val.mean())
print("Random Forest Kappa:",cohen_score_rf_b_val)
#Random Forest ACC:   0.9891797131796158
#Random Forest RMSE:  0.022941324947832735
#Random Forest Kappa: 0.9886123403895481


# In[332]:


np.savetxt("BUILDINGID_val.csv", predictions_rf_b_val, header="BUILDINGID")


# In[333]:


df_val_1=read_csv('DF_VAL.csv')


# In[334]:


y_val_f=df_val.FLOOR


# In[335]:


df_val_f=df_val_1
X_val_f=df_val_f


# In[336]:


X_val_f.head()


# In[337]:


X_val_f_train, X_val_f_test, y_val_f_train, y_val_f_test = train_test_split(X_val_f, y_val_f, test_size=0.25, random_state=None)


# In[338]:


y_pred_rf_f_val = RF_f.predict(X_val_f_test)
print(metrics.r2_score(y_val_f_test,y_pred_rf_f_val))


# In[339]:


scores_rf_f_val = cross_val_score(model_rf_f, X_val_f, y_val_f, cv=10,scoring='neg_mean_squared_error')
predictions_rf_f_val=cross_val_predict(model_rf_f,X_val_f,y_val_f,cv=10)


# In[340]:


mse_scores_rf_f_val = -scores_rf_f_val
accuracy_rf_f_val=metrics.r2_score(y_val_f,predictions_rf_f_val)
rmse_rf_f_val = np.sqrt(mse_scores_rf_f_val)
cohen_score_rf_f_val = cohen_kappa_score(y_val_f, predictions_rf_f_val)
print("Random Forest ACC:  ",accuracy_rf_f_val)
print("Random Forest RMSE: ", rmse_rf_f_val.mean())
print("Random Forest Kappa:",cohen_score_rf_f_val)
#Random Forest ACC:   0.8079100153346228
#Random Forest RMSE:  0.42147148051722805
#Random Forest Kappa: 0.7913126356603528


# In[341]:


np.savetxt("FLOOR_val.csv", predictions_rf_f_val, header="FLOOR")


# In[343]:


df_val_lo=read_csv('DF_VAL.csv')
y_val_lo=df_val.LONGITUDE


# In[344]:


#df_val_lo=df_val_lo.drop(['LONGITUDE','LATITUDE'], axis=1)
X_val_lo=df_val_lo


# In[346]:


X_val_lo_train, X_val_lo_test, y_val_lo_train, y_val_lo_test = train_test_split(X_val_lo, y_val_lo, test_size=0.25, random_state=None)


# In[361]:


y_pred_rf_lo_val = RF_lo.predict(X_val_lo_test)
print(metrics.r2_score(y_val_lo_test,y_pred_rf_lo_val))


# In[350]:


scores_rf_lo_val = cross_val_score(model_rf_lo, X_val_lo, y_val_lo, cv=10,scoring='neg_mean_squared_error')
predictions_rf_lo_val=cross_val_predict(model_rf_lo,X_val_lo,y_val_lo,cv=10)


# In[354]:


mse_scores_rf_lo_val = -scores_rf_lo_val
accuracy_rf_lo_val=metrics.r2_score(y_val_lo,predictions_rf_lo_val)
rmse_rf_lo_val = np.sqrt(mse_scores_rf_lo_val)
print("Random Forest ACC:  ",accuracy_rf_lo_val)
print("Random Forest RMSE: ", rmse_rf_lo_val.mean())


# In[356]:


np.savetxt("LONGITUDE_val.csv", predictions_rf_lo_val, header="LONGITUDE", comments="")


# In[357]:


df_val_la=read_csv('DF_VAL.csv')
y_val_la=df_val.LATITUDE


# In[358]:


X_val_la=df_val_la


# In[359]:


X_val_la_train, X_val_la_test, y_val_la_train, y_val_la_test = train_test_split(X_val_la, y_val_la, test_size=0.25, random_state=None)


# In[360]:


y_pred_rf_la_val = RF_la.predict(X_val_la_test)
print(metrics.r2_score(y_val_la_test,y_pred_rf_la_val))


# In[362]:


scores_rf_la_val = cross_val_score(model_rf_la, X_val_la, y_val_la, cv=10,scoring='neg_mean_squared_error')
predictions_rf_la_val=cross_val_predict(model_rf_la,X_val_la,y_val_la,cv=10)


# In[363]:


mse_scores_rf_la_val = -scores_rf_la_val
accuracy_rf_la_val=metrics.r2_score(y_val_la,predictions_rf_la_val)
rmse_rf_la_val = np.sqrt(mse_scores_rf_la_val)
print("Random Forest ACC:  ",accuracy_rf_la_val)
print("Random Forest RMSE: ", rmse_rf_la_val.mean())


# In[414]:


np.savetxt("LATITUDE_val.csv", predictions_rf_la_val, header="LATITUDE", comments="")


# In[467]:


DF_VAL=read_csv('DF_VAL.csv')


# In[468]:


trace1 = {
  "x":list(df_val_org['LONGITUDE']), 
  "y":list(df_val_org['LATITUDE']), 
  "z":list(df_val_org['FLOOR']), 
  "marker": {
    "color": "red", 
    "colorscale": "Viridis", 
    "opacity": 1, 
    "size": 4
  }, 
  "mode": "markers",
    "name":"Real Data",
  "type": "scatter3d"
}


# In[469]:


trace2 = {
  "x":list(DF_VAL['LONGITUDE']), 
  "y":list(DF_VAL['LATITUDE']), 
  "z":list(DF_VAL['FLOOR']), 
  "marker": {
    "color": "blue", 
    "colorscale": "Viridis", 
    "opacity": 1, 
    "size": 4
  }, 
  "mode": "markers",
    "name":"Predicted Data",
  "type": "scatter3d"
}


# In[470]:


data = Data([trace1,trace2])
layout = {
  "scene": {
    "xaxis": {"title": "Longitude"}, 
    "yaxis": {"title": "Atitude"}, 
    "zaxis": {"title": "Floor"}
  }, 
  "title": "3D plot Wifi Location"
}


# In[471]:


fig = Figure(data=data, layout=layout)
py.plot(fig)





