#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Library yang dibutuhkan
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder
import copy


# In[2]:


# Pengecekan Struktur Data set
import os
df = pd.read_csv(r'C:\Users\PPL2\Documents\Private\Model\Pacman Data Science\Project\Project III/insurance.csv')
df.head(10)


# In[3]:


# 1.ANALISA DESCRIPTIVE STATISTIC
"""Analisa Descriptive Statistic melingkupi 5 Analisa """


# In[4]:


# 1.Rata-rata Umur pada Data
df.describe().T


# In[5]:


# 2.Rata-rata BMI dari yang merokok
df['bmi'].groupby(df['smoker']).mean()


# In[6]:


# 3.Variansi Tagihan dari Perokok dan Non Perokok
df['charges'].groupby(df['smoker']).var()


# In[7]:


# 4.Rata-Rata Umur dari Laki-Laki dan Perempuan yang Merokok
df.groupby(['smoker', 'sex']).mean('age')


# In[8]:


# 5.Rata-rata tagihan kesehatan perokok dan non-perokok
df.groupby(['smoker']).mean('charges')


# In[9]:


# 2.ANALISA VARIABEL DISKRIT
"""Analisa Variabel Diskrit melingkupi 5 Analisa """


# In[10]:


# 1.Gender yang memiliki tagihan paling tinggi
df['charges'].groupby(df['sex']).max()


# In[11]:


# 2.Proporsi Data di Tiap Region
df['smoker'].groupby(df['region']).describe()


# In[12]:


df['charges'].groupby(df['region']).describe()


# In[13]:


# 3.Proporsi Perokok dan Non Perokok
df.groupby(['smoker']).agg('count')


# In[14]:


# 4.Peluang seseorang adalah perempuan diketahui dia adalah perokok
smokers = df['smoker'].where(df['smoker']=='yes').value_counts()
smokers
female_smoke = df['smoker'].where(df['sex']=='female').value_counts()
p_female_smoker = female_smoke[1] / smokers.sum()
print("Peluang jenis kelamin perempuan sebagai perokok adalah {}".format(np.round(p_female_smoker, 2)))


# In[15]:


# 5.Peluang seseorang adalah laki-laki diketahui dia adalah perokok
smokers = df['smoker'].where(df['smoker']=='yes').value_counts()
smokers
male_smoke = df['smoker'].where(df['sex']=='male').value_counts()
p_male_smoker = male_smoke[1] / smokers.sum()
print("Peluang jenis kelamin laki-laki sebagai perokok adalah {}".format(np.round(p_male_smoker, 2)))


# In[16]:


# 3.ANALISA VARIABEL KONTINU
"""Analisa Variabel Kontinu melingkupi 2 Analisa """


# In[17]:


# 1. Peluang Seseorang dengan BMI di atas 25 Mendapatkan Tagihan Kesehatan di atas 16.7 K
n_smoker_bmi25 = len(df[(df["smoker"]=="yes") & (df["bmi"]>25)])
n_smoker_bmi25_ch167 = len(df[(df["smoker"]=="yes") & (df["bmi"]>25) & (df["charges"]>16700)])
print("n(smoker n BMI>25)=",n_smoker_bmi25)
print("n(smoker n BMI>25 n charges>16.7k)=",n_smoker_bmi25_ch167)
p_smoker_bmi25_ch167 = n_smoker_bmi25_ch167 / n_smoker_bmi25
print("P(charges>16.7k | smoker n BMI>25)=",p_smoker_bmi25_ch167)


# In[18]:


# 2.Peluang Seseorang Perokok dengan BMI di atas 25 Mendapatkan Tagihan Kesehatan di atas 16.7 K
n_nonsmoker_bmi25 = len(df[(df["smoker"]=="no") & (df["bmi"]>25)])
n_nonsmoker_bmi25_ch167 = len(df[(df["smoker"]=="no") & (df["bmi"]>25) & (df["charges"]>16700)])
print("n(non-smoker n BMI>25)=",n_nonsmoker_bmi25)
print("n(non-smoker n BMI>25 n charges>16.7k)=",n_nonsmoker_bmi25_ch167)
p_nonsmoker_bmi25_ch167 = n_nonsmoker_bmi25_ch167 / n_nonsmoker_bmi25
print("P(charges>16.7k | non-smoker n BMI>25)=",p_nonsmoker_bmi25_ch167)


# In[19]:


# 4.ANALISA KORELASI VARIABEL
"""Analisa Variabel Diskrit dilakukan dengan mengecek korelasi antara varibel yang ada """


# In[20]:


#Tes Korelasi Antar Variabel
cov_df = df.cov()
corr_df = df.corr()

print('covariance between quantitative variables')
print(cov_df,'\n')
print('correlation coefficient between quantitative variables')
print(corr_df)

f, ax = plt.subplots(figsize=(5, 4))
plt.title("Heat map for correlation coefficient between quantitative variables")
sns.heatmap(corr_df, mask=np.zeros_like(corr_df, dtype=np.bool), cmap=sns.diverging_palette(240,10,as_cmap=True),
            square=True, annot = True, ax=ax)


# In[21]:


df_encoded = copy.deepcopy(df)
df_encoded.loc[:,['sex', 'smoker', 'region']] = df_encoded.loc[:,['sex', 'smoker', 'region']].apply(LabelEncoder().fit_transform) 

sns.pairplot(df_encoded)  #pairplot
plt.show()


# In[22]:


# 5.PENGUJIAN HIPOTESIS
"""Pengujian 3 hipotesis tentang karakter populasi dari data """


# In[23]:


# 1.Chi_square test to check if smoking habits are different for different genders
Ho = "Gender has no effect on smoking habits"   # Stating the Null Hypothesis
Ha = "Gender has an effect on smoking habits"   # Stating the Alternate Hypothesis

crosstab = pd.crosstab(df['sex'],df['smoker'])  # Contingency table of sex and smoker attributes

chi, p_value, dof, expected =  stats.chi2_contingency(crosstab)

if p_value < 0.05:  # Setting our significance level at 5%
    print(f'{Ha} as the p_value ({p_value.round(3)}) < 0.05')
else:
    print(f'{Ho} as the p_value ({p_value.round(3)}) > 0.05')
crosstab


# In[24]:


# 2.T-test to check dependency of bmi on gender
Ho = "Gender has no effect on bmi"   # Stating the Null Hypothesis
Ha = "Gender has an effect on bmi"   # Stating the Alternate Hypothesis

x = np.array(df[df.sex == 'male'].bmi)  # Selecting bmi values corresponding to males as an array
y = np.array(df[df.sex == 'female'].bmi) # Selecting bmi values corresponding to females as an array

t, p_value  = stats.ttest_ind(x,y, axis = 0)  #Performing an Independent t-test

if p_value < 0.05:  # Setting our significance level at 5%
    print(f'{Ha} as the p_value ({p_value.round()}) < 0.05')
else:
    print(f'{Ho} as the p_value ({p_value.round(3)}) > 0.05')


# In[25]:


# 3.T-test to check dependency of smoking on charges
Ho = "Charges of smoker and non-smoker are same"   # Stating the Null Hypothesis
Ha = "Charges of smoker and non-smoker are not the same"   # Stating the Alternate Hypothesis

x = np.array(df[df.smoker == 'yes'].charges)  # Selecting charges corresponding to smokers as an array
y = np.array(df[df.smoker == 'no'].charges) # Selecting charges corresponding to non-smokers as an array

t, p_value  = stats.ttest_ind(x,y, axis = 0)  #Performing an Independent t-test

if p_value < 0.05:  # Setting our significance level at 5%
    print(f'{Ha} as the p_value ({p_value}) < 0.05')
else:
    print(f'{Ho} as the p_value ({p_value}) > 0.05')


# In[ ]:




