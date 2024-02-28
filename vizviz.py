# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 16:34:15 2024

@author: lenovo
"""

import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import pygwalker as pyg
import seaborn as sns

df = pd.read_csv(r"C:\Users\lenovo\Desktop\heart_2022_with_nans.csv")
msno.matrix(df)

vars_quant = df.select_dtypes(include='number')

vars_qualt = df.select_dtypes(exclude='number')
df.isna().sum()

def fill_missing_mode(series):
    if series.dtype == 'O':  # 'O' represents object data type (non-numeric)
        return series.fillna(series.mode().iloc[0])
    else:
        return series

# Apply the function to each column in vars_qualt
df = df.apply(fill_missing_mode, axis=0)

for col in vars_quant.columns:
    df[col] = df.groupby('State')[col].transform(lambda x: x.fillna(x.mean()))
    
df.isna().sum()

# Remove multiple elements from the categorical_columns list
vars_qualt = [col for col in vars_qualt if col not in ['State', 'GeneralHealth', 'LastCheckupTime', 'TetanusLast10Tdap', 'RemovedTeeth']]

df.drop_duplicates(inplace=True)
df[df.duplicated()]

for col in df.describe().columns:
    sns.set_style('ticks')
    plt.figure(figsize=(16, 2))
    sns.boxplot(data=df, x=col)
    plt.show()


#Function for extracting outliers in column of dataframe
def get_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3-Q1

    lower_limit = Q1 - (IQR * 1.5)
    upper_limit = Q3 + (IQR * 1.5)

    outliers = df[(df[column] < lower_limit) | (df[column] > upper_limit)]

    return outliers, lower_limit, upper_limit


sleep_hours_outliers, lower_sleep, upper_sleep = get_outliers(df, 'SleepHours')
sleep_hours_outliers

print(f"Lower Limit:{lower_sleep})\nUpper Limit:{upper_sleep})")


# Dropping records with sleep less than 3 hours
df = df.drop(df[df['SleepHours'] < 3].index)
df.reset_index(drop=True, inplace=True)
df.shape


# Dropping record with sleep greater than 16 hourss
df = df.drop(df[df['SleepHours'] > 16].index)
df.reset_index(drop=True, inplace=True)
df.shape


height_outliers, lower_height, upper_height = get_outliers(df, 'HeightInMeters')
height_outliers

print(f"Lower Limit:{lower_height})\nUpper Limit:{upper_height})")

# Dropping records with height less than 1.3 meters
df = df.drop(df[df['HeightInMeters'] < 1.3].index)
df.reset_index(drop=True, inplace=True)
df.shape


# Dropping records with height greater than 2.1 meters
df = df.drop(df[df['HeightInMeters'] > 2.1].index)
df.reset_index(drop=True, inplace=True)
df.shape

weight_outliers, lower_weight, upper_weight = get_outliers(df, 'WeightInKilograms')
weight_outliers

print(f"Lower Limit:{lower_weight})\nUpper Limit:{upper_weight})")

# Dropping records with weight less than 40 kg
df = df.drop(df[df['WeightInKilograms'] < 40].index)
df.reset_index(drop=True, inplace=True)
df.shape

# Dropping records with weight greater than 200 kg
df = df.drop(df[df['WeightInKilograms'] > 200].index)
df.reset_index(drop=True, inplace=True)
df.shape


# Sleep Hours, BMI, MentalHealthDays are useless here
# Weight and heights had some kind of difference but in genreal the physical stuff is the best one
# Assuming your DataFrame is named df

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# Box plot
sns.boxplot(x="HadHeartAttack", y="PhysicalHealthDays", data=df, palette="Set3")

# Add labels and title
plt.xlabel("Had Heart Attack")
plt.ylabel("PhysicalHealthDays")
plt.title("Distribution of PhysicalHealthDays by Heart Attack Status")

# Show the plot
plt.show()

sns.scatterplot(x='BMI', y='WeightInKilograms', hue='HadHeartAttack', data=df)


state_coordinates = {
    'Alabama': (32.806671, -86.791130),
    'Alaska': (61.016042, -149.737070),
    'Arizona': (33.729759, -111.431221),
    'Arkansas': (34.969704, -92.373123),
    'California': (36.778259, -119.417931),
    'Colorado': (39.550051, -105.782067),
    'Connecticut': (41.603221, -73.087749),
    'Delaware': (39.739071, -75.539787),
    'District of Columbia': (38.895110, -77.036366),
    'Florida': (27.994402, -81.760254),
    'Georgia': (33.040619, -83.643074),
    'Hawaii': (20.796021, -156.331925),
    'Idaho': (44.068203, -114.742043),
    'Illinois': (40.633125, -89.398528),
    'Indiana': (40.551217, -85.602364),
    'Iowa': (41.878003, -93.097702),
    'Kansas': (39.011902, -98.484246),
    'Kentucky': (37.839333, -84.270020),
    'Louisiana': (31.244823, -92.145024),
    'Maine': (45.253783, -69.445469),
    'Maryland': (39.045753, -76.641273),
    'Massachusetts': (42.407211, -71.382439),
    'Michigan': (44.314844, -85.602364),
    'Minnesota': (46.729553, -94.685900),
    'Mississippi': (32.354668, -89.398528),
    'Missouri': (37.964253, -91.831833),
    'Montana': (46.879682, -110.362566),
  'Nebraska': (41.492537, -99.901813),
  'Nevada': (38.802610, -116.419389),
  'New Hampshire': (43.193852, -71.572395),
  'New Jersey': (40.058324, -74.405661),
  'New Mexico': (34.972730, -105.032363),
  'New York': (40.712776, -74.005974),
  'North Carolina': (35.759573, -79.019300),
  'North Dakota': (47.551493, -101.002012),
  'Ohio': (40.417287, -82.907123),
  'Oklahoma': (35.007752, -97.092877),
  'Oregon': (43.804133, -120.554201),
  'Pennsylvania': (41.203322, -77.194525),
  'Rhode Island': (41.580095, -71.477429),
  'South Carolina': (33.836082, -81.163727),
  'South Dakota': (43.969515, -99.901813),
  'Tennessee': (35.517491, -86.580447),
  'Texas': (31.968599, -99.901813),
  'Utah': (39.320980, -111.093731),
  'Vermont': (44.558803, -72.577841),
  'Virginia': (37.431573, -78.656894),
  'Washington': (47.751074, -120.740139),
  'West Virginia': (38.597626, -80.454903),
  'Wisconsin': (43.784439, -88.787868),
  'Wyoming': (43.075970, -107.290283),
    'Guam': (13.444304, 144.793731),
    'Puerto Rico': (18.220833, -66.590149),
    'Virgin Islands': (18.335765, -64.896335)
}


heart_attack_counts = df[df['HadHeartAttack'] == 'Yes']['State'].value_counts().to_dict()

heart_attack_counts

import folium
import pandas as pd

# Assuming you have state_coordinates and heart_attack_counts defined

# Create a DataFrame with state, latitude, longitude, and heart attacks data
df_map = pd.DataFrame(list(heart_attack_counts.items()), columns=['State', 'HeartAttacks'])
df_map['Latitude'] = df_map['State'].map(lambda state: state_coordinates[state][0])
df_map['Longitude'] = df_map['State'].map(lambda state: state_coordinates[state][1])

# Create a folium map centered at the average latitude and longitude
average_lat = sum(lat for lat, _ in state_coordinates.values()) / len(state_coordinates)
average_lon = sum(lon for _, lon in state_coordinates.values()) / len(state_coordinates)
m = folium.Map(location=[average_lat, average_lon], zoom_start=4)

# Create a choropleth map with GeoJsonTooltip
folium.Choropleth(
	geo_data='us-states.json',  # Path to the GeoJSON file containing state boundaries
	name='choropleth',
	data=df_map,
	columns=['State', 'HeartAttacks'],
	key_on='feature.properties.name',
	fill_color='YlOrRd',
	fill_opacity=0.7,
	line_opacity=0.2,
	legend_name='Heart Attacks',
	highlight=True,  # Enable highlighting
).add_to(m)

# Add GeoJsonTooltip to display state names
folium.GeoJson(
    'us-states.json',
    name='geojson',
    style_function=lambda feature: {
        'fillColor': 'transparent',
        'color': 'transparent',
    },
    highlight_function=lambda x: {'weight': 3, 'color': 'black'},
    tooltip=folium.features.GeoJsonTooltip(fields=['name'], aliases=['State'], labels=True, sticky=True)
).add_to(m)

# Display the map
m


# Group the data by height and weight, and calculate the count of heart attacks for each group
grouped = df.groupby(['HeightInMeters', 'WeightInKilograms'])['HadHeartAttack'].count().reset_index()

# Sort the results in descending order
sorted_grouped = grouped.sort_values('HadHeartAttack', ascending=False)

# Get the range with the highest count
height_range = sorted_grouped['HeightInMeters'].iloc[0]
weight_range = sorted_grouped['WeightInKilograms'].iloc[0]

# Print the result
print(f"People within the height range {height_range} and weight range {weight_range} suffer the most from heart attacks.")



# 8. Point Plot with Error Bars
plt.figure(figsize=(10, 8))
sns.pointplot(x='AgeCategory', y='BMI', hue='HadHeartAttack', data=df, capsize=.1, errwidth=1, palette="Set2")
plt.show()

# 1. Bar Chart comparing HadHeartAttack with other categorical variables
categorical_vars = ['GeneralHealth', 'SmokerStatus', 'HIVTesting', 'FluVaxLast12', 'RemovedTeeth']

for cat_var in categorical_vars:
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x=cat_var, hue='HadHeartAttack')
    plt.title(f'Count of {cat_var} by HadHeartAttack')
    plt.xticks(rotation=45)
    plt.show()
    
############
import streamlit as st
import folium
import pandas as pd

# Load data and create DataFrame df_map

# Assuming you have state_coordinates and heart_attack_counts defined
state_coordinates = {'New York': (40.7128, -74.0060), 'California': (36.7783, -119.4179)}  # Example coordinates
heart_attack_counts = {'New York': 100, 'California': 150}  # Example data

# Create a DataFrame with state, latitude, longitude, and heart attacks data
df_map = pd.DataFrame(list(heart_attack_counts.items()), columns=['State', 'HeartAttacks'])
df_map['Latitude'] = df_map['State'].map(lambda state: state_coordinates[state][0])
df_map['Longitude'] = df_map['State'].map(lambda state: state_coordinates[state][1])

# Create a folium map centered at the average latitude and longitude
average_lat = sum(lat for lat, _ in state_coordinates.values()) / len(state_coordinates)
average_lon = sum(lon for _, lon in state_coordinates.values()) / len(state_coordinates)
m = folium.Map(location=[average_lat, average_lon], zoom_start=4)

# Create a choropleth map with GeoJsonTooltip
folium.Choropleth(
    geo_data='us-states.json',  # Path to the GeoJSON file containing state boundaries
    name='choropleth',
    data=df_map,
    columns=['State', 'HeartAttacks'],
    key_on='feature.properties.name',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Heart Attacks',
    highlight=True,  # Enable highlighting
).add_to(m)

# Add GeoJsonTooltip to display state names
folium.GeoJson(
    'us-states.json',
    name='geojson',
    style_function=lambda feature: {
        'fillColor': 'transparent',
        'color': 'transparent',
    },
    highlight_function=lambda x: {'weight': 3, 'color': 'black'},
    tooltip=folium.features.GeoJsonTooltip(fields=['name'], aliases=['State'], labels=True, sticky=True)
).add_to(m)

# Display the map
st.write(m._repr_html_(), unsafe_allow_html=True)


