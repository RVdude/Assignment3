#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
# dataset : # https://data.worldbank.org/indicator/NY.GDP.PCAP.CD


# In[4]:


def data(path):
    """
    Function that reads World Bank data from a CSV file.
    Returns:
    - data: Original dataframe.
    - transposed data: Transposed dataframe.
    """
    orginal_data = pd.read_csv(path, skiprows=4)
    transposed_data = pd.read_csv(path, skiprows=4).set_index(['Country Name']).T
    return orginal_data, transposed_data


# In[5]:


path = 'API_NY.GDP.PCAP.CD_DS2_en_csv_v2_6298251.csv'

orginal_data, transposed_data = data(path)


# In[6]:


orginal_data.head()


# In[7]:


transposed_data.head()


# ## Clustering

# In[10]:


def select_yearly_data(df, start_year=2001, end_year=2020):
    """
    Function that selects yearly data for the specified range of years .
    Returns:
    - DataFrame: Subset of the input DataFrame with specified range of years data.
    """
    years_list = list(map(str, range(start_year, end_year + 1)))
    selected_columns = ['Country Name', 'Indicator Name']
    for year in years_list:
        selected_columns.append(year)
    data = df[selected_columns]
    return data


# In[25]:


data = select_yearly_data(orginal_data)


# In[26]:


data = data.dropna()


# In[27]:


data.head()


# In[28]:


data.describe()


# In[29]:


subset = data[["Country Name", "2020"]].copy()
subset.head()


# In[30]:


def calculate_percentage_change(data, subset, start_year=2001, end_year=2020):
    """
    Function that calculates the percentage change between two specified years.
    Returns:
    - DataFrame: Original DataFrame with an additional 'Change' column representing the percentage change.
    """
    percentage_change = 100.0 * (data[f"{end_year}"] - data[f"{start_year}"]) / data[f"{start_year}"]
    subset = subset.assign(Change=percentage_change)
    return subset

subset = calculate_percentage_change(data, subset)


# In[31]:


subset.head()


# In[32]:


subset.isnull().sum()


# In[33]:


subset = subset.dropna()


# In[34]:


subset.describe()


# In[37]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.scatterplot(data=subset, x="2020", y="Change", label="Countries", color='black')
plt.title("GDP per capita in 2020 vs. Percentage Change (2001 to 2020)")
plt.xlabel("GDP per capita in 2020")
plt.ylabel("Percentage Change (2001 to 2020)")
plt.show()


# In[38]:


def one_silhouette(xy, n):
    """Calculates silhouette score for n clusters"""
    kmeans = KMeans(n_clusters=n, n_init=20)
    kmeans.fit(xy)
    labels = kmeans.labels_
    score = skmet.silhouette_score(xy, labels)
    return score


# In[43]:


import sklearn.preprocessing as pp
scaler = pp.RobustScaler()
x_norm = scaler.fit(subset[["2020", "Change"]]).transform(subset[["2020", "Change"]])


# In[44]:


from sklearn.cluster import KMeans
import sklearn.metrics as skmet

silhouette_scores = []

for i in range(2, 12):
    score = one_silhouette(x_norm, i)
    silhouette_scores.append(score)
    print(f"The silhouette score for {i: 3d} is {score: 7.4f}")


# In[50]:


kmeans = KMeans(n_clusters=3, n_init=10)
kmeans.fit(x_norm)
labels = kmeans.labels_
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
xkmeans, ykmeans = centroids[:, 0], centroids[:, 1]


# In[56]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x=subset["2020"], y=subset["Change"], hue=labels, palette="Set1", s=10, marker="o")
sns.scatterplot(x=xkmeans, y=ykmeans, color="black", s=50, label='Centroids')
plt.title("Clusters of GDP per capita in 2020 vs. Percentage Change (2001 to 2020)")
plt.xlabel("GDP per capita in 2020")
plt.ylabel("Percentage Change (2001 to 2020)")
plt.show()


# ## Curve Fitting

# In[57]:


import errors


# In[61]:


uk_data = transposed_data[['United Kingdom']].loc['2001':'2020'].reset_index().rename(columns={'index': 'Years'}).rename(columns={'United Kingdom': 'GDP per capita'})
uk_data = uk_data.apply(pd.to_numeric, errors='coerce')


# In[62]:


uk_data.describe()


# In[63]:


plt.figure(figsize=(8, 5))
sns.lineplot(data=uk_data, x='Years', y='GDP per capita')
plt.xlabel('Year')
plt.ylabel('GDP per capita')
plt.title('GDP per capita of United Kingdom')
plt.show()


# In[64]:


def poly(x, a, b, c, d, e):
    """ Calulates polynominal"""
    x = x - 2001
    f = a + b*x + c*x**2 + d*x**3 + e*x**4
    return f


# In[66]:


import scipy.optimize as opt
import numpy as np

param, covar = opt.curve_fit(poly, uk_data["Years"], uk_data["GDP per capita"])
sigma = np.sqrt(np.diag(covar))
year = np.arange(2001, 2031)
forecast = poly(year, *param)
sigma = errors.error_prop(year, poly, param, covar)
low = forecast - sigma
up = forecast + sigma
uk_data["fit"] = poly(uk_data["Years"], *param)
plt.figure(figsize=(8, 5))
plt.plot(uk_data["Years"], uk_data["GDP per capita"], label="GDP per capita")
plt.plot(year, forecast, label="forecast")
plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.title("GDP per capita Prediction of United Kingdom")
plt.xlabel("Year")
plt.ylabel("GDP per capita")
plt.legend()
plt.show()


# In[ ]:




