import seaborn as sns
import pandas as pd
import numpy as np
import operator, statistics
from statistics import mean
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from msda.msda import *
from msda.anamoly import *

#train data
df = pd.read_csv('mem2.csv')
df.shape
df=df.drop(['S.no.','free','_db_name','name','available_percent','commit_limit','committed_as','dc','deployment','dirty','env','high_free','high_total','host','huge_page_size','huge_pages_total','huge_pages_free','low_free','low_total','mapped','page_tables','region','role','slab','swap_cached','swap_free','used_percent','vmalloc_chunk','vmalloc_total','vmalloc_used','wired','write_back','write_back_tmp','zone','total'], axis =1 )
df["time"] = pd.to_datetime(df["time"])

df1=df


#test data
df1 = pd.read_csv('mem2.csv')
df1.shape
dep_column = df1.loc[:,'host']
dep = dep_column.values           #stores host
#print(dep[0])
df1=df1.drop(['S.no.','free','_db_name','name','available_percent','commit_limit','committed_as','dc','deployment','dirty','env','high_free','high_total','host','huge_page_size','huge_pages_total','huge_pages_free','low_free','low_total','mapped','page_tables','region','role','slab','swap_cached','swap_free','used_percent','vmalloc_chunk','vmalloc_total','vmalloc_used','wired','write_back','write_back_tmp','zone','total'], axis =1 )
df1["time"] = pd.to_datetime(df1["time"])
#print(df1)

#data pre-processing
anamoly_data, anamoly_df = Anamoly.read_data(data=df, column_index_to_drop=1, timestamp_column_index=0)
anamoly_data, anamoly_df = Anamoly.read_data(data=df, column_index_to_drop=0, timestamp_column_index=0)
#anamoly_data.columns
anamoly_data1, anamoly_df1 = Anamoly.read_data(data=df1, column_index_to_drop=1, timestamp_column_index=0)
anamoly_data1, anamoly_df1 = Anamoly.read_data(data=df1, column_index_to_drop=0, timestamp_column_index=0)


#data mapping
X,Y,timesteps,X_data = Anamoly.data_pre_processing(df=anamoly_df, LOOKBACK_SIZE=30)
X1,Y1,timesteps1,X_data1 = Anamoly.data_pre_processing(df=anamoly_df1, LOOKBACK_SIZE=30)


#Model and training loss
MODEL_SELECTED, LOOKBACK_SIZE, KERNEL_SIZE = Anamoly.set_config(MODEL_SELECTED='deepcnn', LOOKBACK_SIZE=30, KERNEL_SIZE=2)
loss, train_data, model = Anamoly.compute(X, Y,X1,Y1,LOOKBACK_SIZE=30, num_of_numerical_features=7, MODEL_SELECTED=MODEL_SELECTED, KERNEL_SIZE=KERNEL_SIZE, epocs=30)


#to store loss index, where loss is greater than the threshold defined by us
it = np.nditer(loss, flags=['f_index'])
anamoly_index=[]
for x in it:
	if x > 1.2 :
		anamoly_index.append(it.index)
		#print("%d <%d>" % (x, it.index), end=' ')

#print uniques servers having potential issues
print("Servers to look for")
servers_list=[]
for y in anamoly_index:
	servers_list.append(dep[y+LOOKBACK_SIZE])
print(np.unique(servers_list))


#plot 
loss_df = Anamoly.find_anamoly(loss=loss, T=timesteps1)
#print(loss_df)
plt.show(block=True)
plt.figure(figsize=(20,10))
sns.set_style("darkgrid")
ax = sns.distplot(loss_df["loss"], bins=100, label="Frequency")
ax.set_title("Frequency Distribution | Kernel Density Estimation")
ax.set(xlabel='Anomaly Confidence Score', ylabel='Frequency (sample)')
plt.axvline(1.80, color="k", linestyle="--")
plt.legend()
plt.show()

loss_df = loss_df.iloc[: , 0:]
print(loss_df)
loss_df["loss"].plot()
plt.show()






#plt.figure(figsize=(20,10))
#ax = sns.lineplot(x="timestamp", y="loss", data=loss_df, color='g', label="Anomaly Score")
#ax.set_title("Anomaly Confidence Score vs Timestamp")
#ax.set(ylabel="Anomaly Confidence Score", xlabel="Timestamp")
#plt.legend()






#for row in loss_df:
 #   for cell in row:
  #      print(cell, end=' ')






#loss_df = loss_df.loc[:,~loss_df.columns.duplicated()]
#Anamoly.plot_anamoly_results(loss_df=loss_df)

#plt.show()
#plt.show(block=True)   #added extra
#Anamoly.plot_anamoly_results(loss_df=loss_df)
#plt.figure(figsize=(20,10))
#sns.set_style("darkgrid")
#ax = sns.distplot(loss_df["loss"], bins=100, label="Frequency")
#ax.set_title("Frequency Distribution | Kernel Density Estimation")
#ax.set(xlabel='Anomaly Confidence Score', ylabel='Frequency (sample)')
#plt.axvline(1.80, color="k", linestyle="--")
#plt.legend()
#plt.show()
