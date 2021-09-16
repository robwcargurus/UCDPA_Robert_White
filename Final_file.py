import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import os
working_directory = os.getcwd()
print(working_directory)

# preparing the path for uploading files
data_folder = '/Users/rwhite/PycharmProjects/final_project/UCDPA_Robert_White/'
dataset_names = ['bmw', 'merc', 'hyundai', 'ford', 'vauxhall', 'vw', 'audi', 'skoda', 'toyota']

# Uploading the files, adjusting the naming in the tax column and merging the data together
df = pd.DataFrame()
for dataset_name in dataset_names:
    dataset = pd.read_csv(data_folder+dataset_name + '.csv')
    if (dataset_name == 'hyundai'):
        dataset.rename(columns={"tax(£)": "tax"}, inplace=True)
    dataset['manufacturer'] = dataset_name
    df = pd.concat([df, dataset], ignore_index=True)

    # viewing the data and understanding it
    print(df.info)
    print(df.describe())
    print(df.isnull().any())
    print(df.shape)

    # Removing vehicle that is 2060 and Model from 1970
    df.loc[df.year > 2020, 'year'] = 2017
    df.loc[df.year < 1980, 'year'] = 2017

    # Identifying & Removing duplicates
    duplicates = df[df.duplicated(keep=False)]
    print(duplicates.head())
    df.drop_duplicates(ignore_index=True, inplace=True)
    print(df.describe())
    # Number of rows has changed from 99,187 to 97,712, proving that duplicates have been removed

    # Checking for missing values
    missing_values = df.isnull().sum()
    print(missing_values)

    # fill the missing values with mean averages
    cols = ['tax', 'mpg', 'engineSize']
    for col in cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        nan_count = df[col].isnull().sum()
        rand_values = np.random.randint(mean_val - std_val, mean_val + std_val, size=nan_count)

        col_copy = df[col].copy()
        col_copy[np.isnan(col_copy)] = rand_values
        df[col] = col_copy
print(df.isnull().sum())

# Visualising the data to get an understanding of it
corr_ = df.corr()
sns.heatmap(corr_, annot=True)
plt.savefig("Heatmap.png")
plt.show()

# Understanding the data and Grouping and Sorting
manufacturer_data = df
manufacturer_data1 = manufacturer_data.groupby('manufacturer')['manufacturer'].count()
manufacturer_data1 = pd.DataFrame(manufacturer_data1)
manufacturer_data1.columns = ['Buys']
manufacturer_data1.sort_values(by=['Buys'], inplace=True, ascending=False)
manufacturer_data1 = manufacturer_data1.head(10)
print(manufacturer_data1.head(20))
manufacturer_data1.plot.bar()
plt.xticks(rotation=30)
plt.savefig("Top_Makes.png")
plt.show()

model_data = df
model_data1 = model_data.groupby('model')['model'].count()
model_data1 = pd.DataFrame(model_data1)
model_data1.columns = ['Buys']
model_data1.sort_values(by=['Buys'], inplace=True, ascending=False)
model_data1 = model_data1.head(10)
print(model_data1.head(20))
model_data1.plot.bar()
plt.xticks(rotation=30)
plt.savefig("Top_Models.png")
plt.show()


year_data = df
year_data1 = year_data.groupby('year')['model'].count()
year_data1 = pd.DataFrame(year_data1)
year_data1.columns = ['Buys']
year_data1.sort_values(by=['Buys'], inplace=True, ascending=False)
year_data1 = year_data1.head(10)
print(year_data1.head(20))
year_data1.plot.bar()
plt.xticks(rotation=30)
plt.savefig("Top_Year.png")
plt.show()

# Visualising the data
# 1 Relplot
sns.set_theme(style="white")
df1 = df

sns.relplot(df1['manufacturer'], df1['mpg'], hue=df1['fuelType'], size=df1["engineSize"],
            sizes=(40, 400), alpha=.5, palette="muted",
            height=6)
plt.savefig("relplot_fueltypes.png")
plt.show()

# 2 Line chart
sns.lineplot(df1['year'], df1["tax"], hue=df1["fuelType"]).set_title('Tax cost by Fuel Type by Year')
plt.savefig("Tax_Fueltype.png")
plt.show()

sns.lineplot(df1['year'], df1["engineSize"], hue=df1["fuelType"]).set_title('Engine size by Fuel Type by Year')
plt.savefig("Fuel_Enginesize.png")
plt.show()

sns.lineplot(df1['year'], df1["mpg"], hue=df1["fuelType"]).set_title('MPG by Fuel Type by Year')
plt.savefig("MPG_fueltype.png")
plt.show()
# 3 Pairplot by Fueltype
df_pair = df
df_pair.loc[df_pair.year < 1980, 'year'] = 2017
sns.set_theme(style="ticks")
sns.pairplot(df_pair, hue="fuelType")
plt.savefig("pairplot_fueltype.png")
plt.show()

# 4 Subplotting and Grouping

# Creating two group of vehicles: Premium & Volume
audi = df[df['manufacturer'] == "audi"]
bmw = df[df['manufacturer'] == "bmw"]
merc = df[df['manufacturer'] == "merc"]
merc.loc[merc.year < 1980, 'year'] = 2017
hyundi = df[df['manufacturer'] == "hyundai"]
ford = df[df['manufacturer'] == "ford"]
vauxhall = df[df['manufacturer'] == "vauxhall"]
vw = df[df['manufacturer'] == "vw"]
toyota = df[df['manufacturer'] == "toyota"]
skoda = df[df['manufacturer'] == "skoda"]

premium = [audi, bmw, merc]
premium_cars = pd.concat(premium)
premium_cars['brandType'] = 'Premium'
print(premium_cars)
print(premium_cars.describe())

volume = [hyundi, ford, vauxhall, vw, skoda, toyota]
volume_cars = pd.concat(volume)
volume_cars['brandType'] = 'Volume'
print(volume_cars)
print(volume_cars.describe())

cars = [premium_cars, volume_cars]
total_cars = pd.concat(cars)

sns.lineplot(total_cars['year'], total_cars['price'],
             hue=total_cars['brandType']).set_title('Price of Premium Cars vs Volume Cars')
plt.savefig("premvsvol_pricing.png")
plt.show()

# Boxplot
sns.boxplot(total_cars['fuelType'], total_cars['mpg'], hue=total_cars['brandType'])
plt.savefig("premvsvol_fuelmpg.png")
plt.show()

# Working out total Fuel used in litres
total_fuel_used = total_cars["mileage"] / total_cars["mpg"]
total_cars["gallonsFuel"] = total_fuel_used

total_fuel_litres = total_cars["gallonsFuel"] * 3.785
total_fuel_litres1 = total_fuel_litres / 1000
total_cars["litresFuelthou"] = total_fuel_litres1
print(total_cars)


# Top 10 manufacturers by litres of fuel used vs total number of cars
litre_model_data = total_cars
litre_model_data1 = litre_model_data.groupby('manufacturer')['litresFuelthou'].sum()
litre_model_data1 = pd.DataFrame(litre_model_data1)
litre_model_data1.columns = ['Fuel per Thousand Litres']
litre_model_data1.sort_values(by=['Fuel per Thousand Litres'], inplace=True, ascending=False)
litre_model_data1 = litre_model_data1.head(10)
print(litre_model_data1.head(20))
litre_model_data1.plot.bar()
plt.xticks(rotation=30)
plt.savefig("fuellitres_manu.png")
plt.show()

# Fuel types by fuel used
litre_fuel_data = total_cars
litre_fuel_data1 = litre_fuel_data.groupby('fuelType')['litresFuelthou'].sum()
litre_fuel_data1 = pd.DataFrame(litre_fuel_data1)
litre_fuel_data1.columns = ['Fuel per Thousand Litres']
litre_fuel_data1.sort_values(by=['Fuel per Thousand Litres'], inplace=True, ascending=False)
litre_fuel_data1 = litre_fuel_data1.head(10)
print(litre_fuel_data1.head(20))
litre_fuel_data1.plot.bar()
plt.xticks(rotation=30)
plt.savefig("fuellitres_fueltype.png")
plt.show()

total_mileage = total_cars.groupby('fuelType')['mileage'].sum()
total_mileage1 = pd.DataFrame(total_mileage)
total_mileage1.columns = ['Total Mileage']
total_mileage1.sort_values(by=['Total Mileage'], inplace=True, ascending=False)
total_mileage1.plot.bar()
plt.xticks(rotation=30)
plt.savefig("mileage_fueltype.png")
plt.show()
print(total_mileage1)