
# coding: utf-8

# In[1]:


wget https://www.dropbox.com/s/xtms9g31aicstvv/train.csv?dl=1
    


# In[2]:


plot(arange(5))


# In[3]:


import pandas as pd


# In[4]:


import numpy as npimport matplotlib as plt%matplotlib inlinedf = pd.read_csv("/home/kunal/Downloads/Loan_Prediction/train.csv") #Reading the dataset in a dataframe using Pandas


# In[5]:


df.describe()


# In[6]:


df['Property_Area'].value_counts()


# In[7]:


df.boxplot(column='ApplicantIncome')


# In[8]:


df.boxplot(column='ApplicantIncome', by = 'Education')


# In[9]:


df['LoanAmount'].hist(bins=50)


# In[10]:


temp1 = df['Credit_History'].value_counts(ascending=True)


# In[11]:


temp2 = df.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())print ('Frequency Table for Credit History:') print (temp1)print ('\nProbility of getting loan for each Credit History class:')print (temp2)


# In[12]:


import matplotlib.pyplot as pltfig = plt.figure(figsize=(8,4))


# In[13]:


ax1 = fig.add_subplot(121)ax1.set_xlabel('Credit_History')ax1.set_ylabel('Count of Applicants')ax1.set_title("Applicants by Credit_History")temp1.plot(kind='bar')ax2 = fig.add_subplot(122)temp2.plot(kind = 'bar')ax2.set_xlabel('Credit_History')ax2.set_ylabel('Probability of getting loan')ax2.set_title("Probability of getting loan by credit history")


# In[14]:


temp3 = pd.crosstab(df['Credit_History'], df['Loan_Status'])temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


# In[15]:


df.apply(lambda x: sum(x.isnull()),axis=0) 


# In[16]:


df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)


# In[17]:


df['Self_Employed'].fillna('No',inplace=True)


# In[18]:


table = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)# Define function to return value of this pivot_tabledef fage(x): return table.loc[x['Self_Employed'],x['Education']]# Replace missing valuesdf['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)


# In[19]:


df['LoanAmount_log'] = np.log(df['LoanAmount'])df['LoanAmount_log'].hist(bins=20)


# In[20]:


df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']df['TotalIncome_log'] = np.log(df['TotalIncome'])


# In[21]:


df['LoanAmount_log'].hist(bins=20) 


# In[ ]:




