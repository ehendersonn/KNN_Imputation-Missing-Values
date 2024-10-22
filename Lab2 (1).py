#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.impute import KNNImputer


# In[15]:


#Loading data
flight_data = pd.read_csv("/Users/emilyhenderson/Downloads/FlightsDataLab2.csv")


# In[16]:


#1-B: Detecting missing values (n_MV means number of missing values)
print('Number of missing values:')
for col in flight_data.columns:
    n_MV = sum(flight_data[col].isna())
    print('{}:{}'.format(col,n_MV))


# In[17]:


#1-B1: t-test to detect meaningful relationship between distance & occurrence of missing values in delay_time
flight_data=flight_data.assign(MVDelay_Time = flight_data['Delay_Time'].isna())
print(ttest_ind(flight_data['Distance'][flight_data['MVDelay_Time']==True], flight_data['Distance'][flight_data['MVDelay_Time']==False]))


# In[22]:


#1-B1: Because the p-value of the above ttest is greater than 0.05, there is no meaningful relationship between Distance and the occurrence
# of missing values in Delay_Time. There is no difference in Distance values between the missing-values and non-missing-values groups


# In[24]:


#1-B2: Creating dummy variables for Destination variable
flight_data=flight_data.assign(JFK=flight_data['Destination'].isin(['JFK']))
flight_data['JFK']=flight_data['JFK'].map({True:1,False:0})
flight_data=flight_data.assign(LGA=flight_data['Destination'].isin(['LGA']))
flight_data['LGA']=flight_data['LGA'].map({True:1,False:0})
flight_data=flight_data.assign(EWR=flight_data['Destination'].isin(['EWR']))
flight_data['EWR']=flight_data['EWR'].map({True:1,False:0})


# In[30]:


#1-B2: ttest to detect meaningful relationship between destinations and the occurrence of missing values in delay_time
print(ttest_ind(flight_data['JFK'][flight_data['MVDelay_Time']==True], flight_data['JFK'][flight_data['MVDelay_Time']==False]).pvalue)
print(ttest_ind(flight_data['LGA'][flight_data['MVDelay_Time']==True], flight_data['LGA'][flight_data['MVDelay_Time']==False]).pvalue)
print(ttest_ind(flight_data['EWR'][flight_data['MVDelay_Time']==True], flight_data['EWR'][flight_data['MVDelay_Time']==False]).pvalue)


# In[32]:


#1-B2: There is no meaningful relationship between destinations and the occurence of missing values in delay_time, as the p-values of each
#destination are greater than 0.05. 


# In[34]:


#1-B3: Creating dummy variable for weekends
flight_data=flight_data.assign(Weekend=flight_data['Day_of_Week'].isin(['Sat','Sun']))
flight_data['Weekend']=flight_data['Weekend'].map({True:1,False:0})


# In[38]:


#1-B3: ttest to detect meaningful relationship between weekend and occurrence of missing values in delay_time
ttest_ind(flight_data['Weekend'][flight_data['MVDelay_Time']==True], flight_data['Weekend'][flight_data['MVDelay_Time']==False])


# In[40]:


#1-B3: There is no meaningful relationship between whether it was a weekend and the occurrence of missing values in delay_time, as the 
#p-value of the above t-test is not lower than 0.05. 


# In[56]:


#1-B4: Creating a dummy variable for whether it was January 1 or not
#Convert date variable first
flight_data = flight_data.assign(Holiday=flight_data['Date'].isin(['1/1/2004']))
flight_data['Holiday']=flight_data['Holiday'].map({True:1, False:0})


# In[60]:


#1-B4: ttest to detect meaningful relationship between Jan 1 holiday and occurrence of missing values in delay_time
ttest_ind(flight_data['Holiday'][flight_data['MVDelay_Time']==True], flight_data['Holiday'][flight_data['MVDelay_Time']==False])


# In[62]:


#1-B4: There is a meaningful relationship between whether or not it was January 1st and the occurrence of missing values in delay_time, as the 
#p-value of that ttest is a great deal smaller than 0.05. 


# In[64]:


#1-B5: Diagnosing missing values with group mean of the January 1st analysis
print(flight_data['Holiday'].groupby(flight_data['MVDelay_Time']).mean())


# In[66]:


#1-B5: The group mean with the occurrence of missing values is much higher than the group mean without missing values, which means that
#most of the occurrences of missing values are on the January 1st holiday. Because January 1st is a holiday, the missing values may be due
#to the airports being too busy and data not being entered, or the system may be going down and not recording delays because of the amount
#of traffic and flights due to the holiday. Without knowing whether this data is entered by employees or automatically recorded by the system,
#it's hard to say for sure what the issue is, but it may also be due to a shortage of employees to record this data due to it being a holiday.


# In[70]:


#1-C: Detecting a meaningful relationship between origin and the occurrence of missing values in weather
flight_data=flight_data.assign(MV_Weather = flight_data['Weather'].isna())


# In[72]:


#1-C: Creating dummy variables for origin
flight_data=flight_data.assign(BWI=flight_data['Origin'].isin(['BWI']))
flight_data['BWI']=flight_data['BWI'].map({True:1,False:0})
flight_data=flight_data.assign(DCA=flight_data['Origin'].isin(['DCA']))
flight_data['DCA']=flight_data['DCA'].map({True:1,False:0})
flight_data=flight_data.assign(IAD=flight_data['Origin'].isin(['IAD']))
flight_data['IAD']=flight_data['IAD'].map({True:1,False:0})


# In[74]:


#1-C: ttest to detect meaningful relationship between origins and missing values in weather
print(ttest_ind(flight_data['BWI'][flight_data['MV_Weather']==True], flight_data['BWI'][flight_data['MV_Weather']==False]).pvalue)
print(ttest_ind(flight_data['DCA'][flight_data['MV_Weather']==True], flight_data['DCA'][flight_data['MV_Weather']==False]).pvalue)
print(ttest_ind(flight_data['IAD'][flight_data['MV_Weather']==True], flight_data['IAD'][flight_data['MV_Weather']==False]).pvalue)


# In[76]:


#1-C: There is a meaningful relationship between the BWI and DCA origins and missing values in weather, as the p-values for those
#ttests were lower than 0.05. 


# In[80]:


#2-A: Finding the correlation between flight_status and weather and flight_status and delay_time
#Creating a dummy variable for flight_status first
flight_data=flight_data.assign(ontime=flight_data['Flight_Status'].isin(['ontime']))
flight_data['ontime']=flight_data['ontime'].map({True:1,False:0})
flight_data=flight_data.assign(delayed=flight_data['Flight_Status'].isin(['delayed']))
flight_data['delayed']=flight_data['delayed'].map({True:1,False:0})


# In[88]:


#2-A: Correlation between flight_status and weather
corr_weather_delayed = flight_data['delayed'].corr(flight_data['Weather'])
corr_weather_ontime = flight_data['ontime'].corr(flight_data['Weather'])
print(corr_weather_delayed, corr_weather_ontime)


# In[90]:


#There is a strong relationship between weather and flight status. When the weather is good, flights are more likely to be on time and 
#when the weather is bad, flights are more likely to be delayed. 


# In[94]:


#2-A: Correlation between flight_status and delay_time
corr_dt_ontime = flight_data['ontime'].corr(flight_data['Delay_Time'])
corr_dt_delayed = flight_data['delayed'].corr(flight_data['Delay_Time'])
print(corr_dt_ontime, corr_dt_delayed)


# In[ ]:


#There is a strong relationship between flight_status and delay_time. This means that when a flight's status is ontime, the delay time is 
#less and when a flight's status is delayed, the delay time will be greater. 


# In[96]:


#2-B: Calculating means
ontime_mean = flight_data[flight_data['ontime'] == 1]['Delay_Time'].mean()
delayed_mean = flight_data[flight_data['delayed'] == 1]['Delay_Time'].mean()


# In[116]:


#2-B: Filling missing values function
def fill_MV(row):
    if pd.isna(row['Delay_Time']):
        if row['ontime'] == 1:
            return ontime_mean
        elif row['delayed'] == 1:
            return delayed_mean
    return row['Delay_Time']

flight_data['Delay_Time'] = flight_data.apply(fill_MV, axis=1)
flight_data.head(5)


# In[84]:


#2-C: Fill missing values in weather using KNN imputation
imputer = KNNImputer(n_neighbors=2)
flight_data['Weather_i'] = imputer.fit_transform(flight_data[['Weather']])


# In[110]:


#3-A: Finding outliers in Delay_Time using boxplot
plt.boxplot(flight_data['Delay_Time'].dropna(),vert=False)
plt.title("Delay_Time")
plt.show()


# In[112]:


#3-A: Using quartiles
Q1 = flight_data['Delay_Time'].quantile(0.25)
Q3 = flight_data['Delay_Time'].quantile(0.75)
IQR = Q3-Q1

delay_outliers = (flight_data['Delay_Time'] > (Q3+1.5*IQR)) | (flight_data['Delay_Time'] < (Q1-1.5*IQR))
flight_data[delay_outliers]


# In[ ]:


#3-B: The outlier that says "ontime" in the Flight_Status column must be a data error, because the delay_time for that flight is so large, 
#it couldn't possibly be ontime, so it must be a data error. 

