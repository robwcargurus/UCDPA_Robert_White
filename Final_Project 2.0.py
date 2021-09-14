import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import os
working_directory = os.getcwd()
print(working_directory)

data_folder = '/Users/rwhite/PycharmProjects/final_project/'
dataset_names = ['bmw', 'merc', 'hyundi', 'ford', 'vauxhall', 'vw', 'audi','skoda', 'toyota']

df = pd.DataFrame()
for dataset_name in dataset_names:
    dataset = pd.read_csv(data_folder+dataset_name + '.csv')
    if(dataset_name == 'hyundi'):
        dataset.rename(columns={"tax(£)": "tax"}, inplace=True)
    dataset['manufacturer'] = dataset_name
    df = pd.concat([df, dataset], ignore_index=True)

    print(df.info)
    print(df.describe())
    print(df.isnull().any())
    print(df['model'].unique())

##Removing vehicle that is 2060
df.loc[df.year > 2020, 'year'] = 2017

corr_=df.corr()
sns.heatmap(corr_,annot=True)
plt.show()

##Identifying & Removing Duplicates
duplicates = df[df.duplicated(keep=False)]
print(duplicates.head())
df.drop_duplicates(ignore_index=True, inplace=True)
print(df.describe())
##Number of rows has changed from 99,187 to 97,712, proving that duplicates have been removed

## Checking for missing values
missing_values = df.isnull().sum()
print(missing_values)
## fill the missing values with 0
df1 = df.fillna(0)
df1.isnull().sum()
print(df1)
##Proving the blank values have been filled
missing_values = df1.isnull().sum()
print(missing_values)

## We can see the number of vehicles by multiple views
plt.figure(figsize=(15,10))
sns.countplot(df1['manufacturer'])
plt.show()

sns.countplot(df1['fuelType'])
plt.title('Fuel Type')
plt.show()

sns.countplot(df1['transmission'])
plt.title('Transmission Type')
plt.show()

## Data Subsetting/ Grouping
df2 = df
model_buys= df2.groupby('model')['model'].count()
model_buys = pd.DataFrame(model_buys)
model_buys.columns = ['Buys']
model_buys.sort_values(by=['Buys'], inplace=True, ascending=False)
model_buys = model_buys.head(10)
print(model_buys.head(20))
model_buys.plot.bar()
plt.show()

sns.lineplot(df2['year'],df2['price'], hue =df2["manufacturer"])
plt.show()

## Strip the data to see only Merc, BMW, Audi
audi = df[df['manufacturer'] == "audi"]
bmw = df[df['manufacturer'] == "bmw"]
merc = df[df['manufacturer'] == "merc"]
merc.loc[merc.year < 1980, 'year'] = 2017

sns.lineplot(audi['year'],audi['price'], hue = audi["manufacturer"], palette =['blue'], linewidth=2.5)
sns.lineplot(bmw['year'],bmw['price'],hue = bmw["manufacturer"], palette =['red'], linewidth=2.5)
sns.lineplot(merc['year'],merc['price'],hue = merc["manufacturer"], palette =['green'], linewidth=2.5)
plt.show()
## 2000 to 2005 models: BMW are the most expensive
## 2005 to 2010 models: Audi are the most expensive
## 2010 to 2015 models: Merc are the most expensive

df_pair = df
df_pair.loc[df_pair.year < 1980, 'year'] = 2017
sns.set_theme(style="ticks")
sns.pairplot(df_pair, hue="fuelType")
plt.show()

premium = [audi,bmw,merc]
premium_cars = pd.concat(premium)
print(premium_cars)

hyundi = df[df['manufacturer'] == "hyundi"]
ford = df[df['manufacturer'] == "ford"]
vauxhall = df[df['manufacturer'] == "vauxhall"]
vw = df[df['manufacturer'] == "vw"]
toyota = df[df['manufacturer'] == "toyota"]
skoda = df[df['manufacturer'] == "skoda"]

volume = [hyundi,ford,vauxhall,vw,skoda,toyota]
volume_cars = pd.concat(volume)
print(volume_cars)

sns.lineplot(premium_cars['year'],premium_cars['price'], palette=['blue'])
sns.lineplot(volume_cars['year'],volume_cars['price'], palette=['red'])
plt.show()

sns.lineplot(premium_cars['year'],premium_cars['mpg'], palette=['blue'])
sns.lineplot(volume_cars['year'],volume_cars['mpg'], palette=['red'])
plt.show()

sns.lineplot(premium_cars['year'],premium_cars['tax'], palette=['blue'])
sns.lineplot(volume_cars['year'],volume_cars['tax'], palette=['red'])
plt.show()

sns.lineplot(premium_cars['year'],premium_cars['engineSize'], palette=['blue'])
sns.lineplot(volume_cars['year'],volume_cars['engineSize'], palette=['red'])
plt.show()

sns.lineplot(premium_cars['year'],premium_cars['engineSize'], palette=['blue'])
sns.lineplot(volume_cars['year'],volume_cars['engineSize'], palette=['red'])
plt.show()

sns.lineplot(premium_cars['year'],premium_cars["mpg"],hue = premium_cars["fuelType"]).set_title('Premium')
plt.show()

sns.lineplot(volume_cars['year'],volume_cars["mpg"],hue = volume_cars["fuelType"]).set_title('Volume')
plt.show()


