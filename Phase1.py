#%%
################################################

## Importing the libraries

#################################################

import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import numpy as np
from tabulate import tabulate
from prettytable import PrettyTable
from scipy.stats import skew, kurtosis


#%%

################################################

#Read Datatset and explore features

#################################################

#read dataset and show the shape
airline_df =pd.read_csv("Airline_Passenger_Satisfaction.csv")
airline_df.shape
#show the first 5 rows of the dataset
airline_df.head()
data_info = airline_df.info()
# Extract column names and data types from the info() output
column_info = [(col, dtype) for col, dtype in zip(airline_df.columns, airline_df.dtypes)]
# Print column names, data types
print("Features in the data:")
for col, dtype in column_info:
    print(f"- {col}: {dtype}")
print("""Description of Features in the data:
- Gender: Gender of the passengers (Female, Male)
- Customer Type: The customer type (Loyal customer, disloyal customer)
- Age: The actual age of the passengers
- Type of Travel: Purpose of the flight of the passengers (Personal Travel, Business Travel)
- Class: Travel class in the plane of the passengers (Business, Eco, Eco Plus)
- Flight distance: The flight distance of this journey
- Inflight wifi service: Satisfaction level of the inflight wifi service (0:Not Applicable;1-5)
- Departure/Arrival time convenient: Satisfaction level of Departure/Arrival time convenient
- Ease of Online booking: Satisfaction level of online booking
- Gate location: Satisfaction level of Gate location
- Food and drink: Satisfaction level of Food and drink
- Online boarding: Satisfaction level of online boarding
- Seat comfort: Satisfaction level of Seat comfort
- Inflight entertainment: Satisfaction level of inflight entertainment
- On-board service: Satisfaction level of On-board service
- Leg room service: Satisfaction level of Leg room service
- Baggage handling: Satisfaction level of baggage handling
- Check-in service: Satisfaction level of Check-in service
- Inflight service: Satisfaction level of inflight service
- Cleanliness: Satisfaction level of Cleanliness
- Departure Delay in Minutes: Minutes delayed when departure
- Arrival Delay in Minutes: Minutes delayed when Arrival
- Satisfaction: Airline satisfaction level(Satisfaction, neutral or dissatisfaction)""")


# %%
################################################

#Data Cleaning

#################################################

#check null values
airline_df.isnull().sum()

######## Handle Missing Values ########

# Imputing missing value with mean
airline_df['Arrival Delay in Minutes'] = airline_df['Arrival Delay in Minutes'].fillna(airline_df['Arrival Delay in Minutes'].mean())
airline_df.isnull().sum()

# Check the list of categorical variables
cat_variables = airline_df.select_dtypes(include=['object']).columns
# Count the number of NaN values in each categorical variable
nan_counts = airline_df[cat_variables].isnull().sum()
# Print the NaN counts for each categorical variable
print("NaN counts in categorical variables:")
print(nan_counts)

#######drop unnecessary columns #########
airline_df = airline_df.drop('Unnamed: 0', axis=1)
airline_df = airline_df.drop('id', axis=1)
airline_df.info()



# %%
################################################

#Univariate Analysis

#################################################
#Tables
###### Summary Statistics ########

# Separating numeric and categorical columns
numeric_cols = airline_df.select_dtypes(include=['int64', 'float64'])
categorical_cols = airline_df.select_dtypes(include=['object'])

# Calculate numeric statistics
numeric_stats = numeric_cols.describe()
numeric_stats.loc['median'] = numeric_cols.median()
numeric_stats.loc['mode'] = numeric_cols.mode().iloc[0]

# Calculate categorical statistics (mode and count)
categorical_stats = categorical_cols.describe(include=['object'])
categorical_stats.loc['mode'] = categorical_cols.mode().iloc[0]

# numeric statistics 
numeric_table = PrettyTable()
numeric_table.title = "Numeric Statistics of Dataset"
numeric_table.field_names = ["Statistic"] + list(numeric_stats.columns)


for index, row in numeric_stats.iterrows():
    formatted_row = [f"{value:.2f}" if isinstance(value, float) else value for value in row]
    numeric_table.add_row([index] + formatted_row)

# categorical statistics
categorical_table = PrettyTable()
categorical_table.title = "Categorical Statistics of Dataset"
categorical_table.field_names = ["Statistic"] + list(categorical_stats.columns)
for index, row in categorical_stats.iterrows():
    categorical_table.add_row([index] + row.tolist())

# Print the tables
print(numeric_table)
print("\n")
print(categorical_table)


#%%
##Skewness & kurtosis

# Calculate skewness and kurtosis for numerical variables
numerical_columns = airline_df.select_dtypes(include=['int64', 'float64']).columns

# Create a PrettyTable object
table = PrettyTable()
table.field_names = ["Variable", "Skewness", "Kurtosis"]
table.title = "Skewness and Kurtosis Analysis"

for column in numerical_columns:
    column_skewness = skew(airline_df[column].dropna())
    column_kurtosis = kurtosis(airline_df[column].dropna(), fisher=False)
    table.add_row([column, f"{column_skewness:.2f}", f"{column_kurtosis:.2f}"])

print(table)




# %% 
#Visualization
# Define title, xlabel, and ylabel settings
title_font = {'fontname': 'serif', 'color': 'blue', 'fontsize': 16}
label_font = {'fontname': 'serif', 'color': 'darkred', 'fontsize': 14}

class_counts = airline_df['Class'].value_counts()

# Plotting the pie chart
plt.figure(figsize=(5, 5))
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Passenger Class', **title_font)
plt.legend()
plt.tight_layout()
plt.show()

#%%

########Histograms for numerical data########

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
fig.suptitle('Key Trends in Passenger Experience and Flight Operations', **title_font)
variables = {
    'Age': ('Age Distribution', 0, 0, 'skyblue'),
    'Flight Distance': ('Flight Distance Distribution', 0, 1, 'lightgreen'),
    'Inflight wifi service': ('Inflight Wifi Service Rating', 1, 0, 'salmon'),
    'Departure Delay in Minutes': ('Departure Delay Distribution', 1, 1, 'gold')
}
for var, (title, i, j, color) in variables.items():
    airline_df[var].plot(kind='hist', bins=30, ax=axes[i, j], color=color, title=title)
    axes[i, j].set_title(title, fontdict=title_font)
    axes[i, j].set_xlabel(var, fontdict=label_font)
    axes[i, j].set_ylabel('Frequency', fontdict=label_font)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()



#%%
#Box plot for outliers in delay

# Create a figure and axes for subplot
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12))

# Plotting the box plot for Departure Delay in Minutes
axes[0].boxplot(airline_df['Departure Delay in Minutes'].dropna(), vert=False, patch_artist=True, boxprops=dict(facecolor='brown'))
axes[0].set_title('Departure Delay Spread & Outliers', **title_font)
axes[0].set_xlabel('Delay (Minutes)', **label_font)

# Plotting the box plot for Arrival Delay in Minutes
axes[1].boxplot(airline_df['Arrival Delay in Minutes'].dropna(), vert=False, patch_artist=True, boxprops=dict(facecolor='black'))
axes[1].set_title('Arrival Delay Spread & Outliers', **title_font)
axes[1].set_xlabel('Delay (Minutes)', **label_font)

plt.tight_layout()
plt.show()


#%%


#pre-flight service categories
# Data for pre-flight services to create a bar chart
pre_flight_services = [
    'Departure/Arrival time convenient', 
    'Ease of Online booking', 
    'Gate location', 
    'Checkin service'
]

# Extracting these columns from the dataset
pre_flight_data = airline_df[pre_flight_services]

# Creating a  bar chart for each service rating
plt.figure(figsize=(14, 8))
for i, column in enumerate(pre_flight_services):
    plt.subplot(2, 2, i+1)
    sns.countplot(x=column, data=pre_flight_data, palette='coolwarm')
    plt.title(f'Count of Ratings: {column}', **title_font)
    plt.xlabel('Rating',**label_font)
    plt.ylabel('Count',**label_font)
    plt.tight_layout()

plt.show()











#%%
###### Frequency Count plots #######

# Define the categorical variables 
categorical_vars = ['Gender', 'Customer Type', 'Type of Travel', 'Class']

# Create a figure to plot
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 15))
axes = axes.flatten()  # Flatten the axes array for easy iteration

# Define assorted colors
assorted_colors = ["#FF5733", "#33FFC7", "#336CFF", "#FF33E8"]

# Loop through the list of categorical variables and create count plots
for i, var in enumerate(categorical_vars):
    sns.countplot(x=var, data=airline_df, ax=axes[i], palette=assorted_colors)
    axes[i].set_title(f'Frequency of {var}', **title_font)
    axes[i].set_xlabel(var, **label_font)
    axes[i].set_ylabel('Count', **label_font)
    axes[i].tick_params(labelsize=10)
    
    # Add legend
    handles, labels = axes[i].get_legend_handles_labels()
    axes[i].legend(handles, labels, loc='upper right')


# Add a title for the entire plot
fig.suptitle('Frequency of Categorical Variables', **title_font)

# Adjust layout to prevent overlap
plt.tight_layout()

# Adjust the gap between the overall title and the subplot titles
plt.subplots_adjust(top=0.9)

# Display the plot
plt.show()

#%%
##### Pie chart #####

# What is the general level of satisfaction among passengers within the dataset?


satisfaction_counts = airline_df['satisfaction'].value_counts()
labels = satisfaction_counts.index
sizes = satisfaction_counts.values

# Define colors for the pie slices
colors = ['lightcoral', 'paleturquoise']

# Create the pie chart
fig, ax = plt.subplots(figsize=(8, 6))
patches, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=120, pctdistance=0.85)

# Adjust the legend
legend_labels = [f"{label} ({size})" for label, size in zip(labels, sizes)]
ax.legend(patches, legend_labels, loc='upper left', bbox_to_anchor=(1, 1, 0.5, 0.5))

# Add a title
plt.title('Passenger Satisfaction Levels', **title_font)

plt.tight_layout()
# Display the plot
plt.show()



#%%

##### Stack Bar plot#####


# Create a crosstab to calculate the percentage of satisfaction by class
class_satisfaction = pd.crosstab(airline_df['Class'], airline_df['satisfaction'], normalize='index') * 100

# Plot a stacked bar chart
class_satisfaction.plot(kind='bar', stacked=True, figsize=(10, 6), color=['#d62728', '#2ca02c'])

# Adding labels and title
plt.title('Percentage of Passenger Satisfaction by Class', **title_font)
plt.xlabel('Class', **label_font)
plt.ylabel('Percentage of Satisfaction', **label_font)
plt.xticks(rotation=0)  # Keep the class names horizontal
plt.legend(title='Satisfaction', loc='upper left', bbox_to_anchor=(1,1), title_fontsize='13', fontsize='11')

# Show the plot
plt.tight_layout()
plt.show()


#%%
#mapping the satisfaction column for analysis
# Create a dictionary to map the values
satisfaction_map = {'satisfied': 1, 'neutral or dissatisfied': 0}
# Use map function to map values
airline_df['Satisfaction_Coded'] = airline_df['satisfaction'].map(satisfaction_map)

# %%

#Breakdown of passengers across different age groups, and how does age influence their level of satisfaction

# Define age bins and labels
bins = [0, 13, 18, 26, 41, 66, 100]
labels = ['0-12', '13-17', '18-25', '26-40', '41-65', '66+']
labels_legend = ['0-12 Children', '13-17 Teenagers', '18-25 Young Adults', '26-40 Early-Middle-Aged',
                 '41-65 Late-Middle-Aged', '66+ Seniors']
airline_df['Age Group'] = pd.cut(airline_df['Age'], bins=bins, labels=labels, right=False)

# Calculate age distribution
age_data = airline_df['Age Group'].value_counts().sort_index()

# Calculate the satisfaction rate for each Age Group
airline_df_new = airline_df.groupby('Age Group')['Satisfaction_Coded'].mean().reset_index()

# Rename the columns
airline_df_new.columns = ['Age Group', 'Satisfaction Rate']

# Create a subplot with 2 rows and 1 column
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 20))

# Plot the pie chart for age distribution
# colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0','#ffb3e6']
colors = ['#1f77b4', '#aec7e8', '#6baed6', '#3182bd', '#08519c', '#084594']  
ax1.pie(age_data, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
ax1.set_title('Passenger Age Distribution')
ax1.legend(labels_legend, loc='upper left', bbox_to_anchor=(1, 1))


# Plot the bar plot for satisfaction rate by age group
# Define fading colors based on age
fade_colors = ['#1f77b4', '#4477aa', '#66aabb', '#88bbcc', '#aaddcc', '#ccffcc']

for i, (age_group, satisfaction_rate) in enumerate(zip(airline_df_new['Age Group'], airline_df_new['Satisfaction Rate'])):
    ax2.bar(age_group, satisfaction_rate, color=fade_colors[i], label=age_group)
    ax2.text(age_group, satisfaction_rate, f'{satisfaction_rate:.1%}', ha='center', va='bottom')

ax2.set_title('Satisfaction Rate by Age Group', **title_font)
ax2.set_xlabel('Age Group',**label_font)
ax2.set_ylabel('Satisfaction Rate (%)', **label_font)

# Show plot
plt.tight_layout()
plt.show()





#%%
#################################################

#Bivariate/Multivariate Analysis

#################################################

######## Heat Map - Correlation analysis ########

# Select only numeric columns from the dataset
numeric_df = airline_df.select_dtypes(include=['number'])

# Calculate the correlation matrix
corr = numeric_df.corr()

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 20))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(150, 275, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=None, center=0,
            square=True, annot=True, linewidths=.5, cbar_kws={"shrink": .9})

# Add a title
plt.title("Correlation Matrix", **title_font)
plt.show()



#%%
##### Pivot Table #####

numeric_column_for_counting = 'Age' 

# Pivot tables for visualization
pivot_tables = {
    'Class': pd.pivot_table(airline_df, values=numeric_column_for_counting, index='Class', columns='satisfaction', aggfunc='count'),
    'Customer Type': pd.pivot_table(airline_df, values=numeric_column_for_counting, index='Customer Type', columns='satisfaction', aggfunc='count'),
    'Type of Travel': pd.pivot_table(airline_df, values=numeric_column_for_counting, index='Type of Travel', columns='satisfaction', aggfunc='count'),
    'Gender': pd.pivot_table(airline_df, values=numeric_column_for_counting, index='Gender', columns='satisfaction', aggfunc='count')
}

# Plotting the pivot tables
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
axes = axes.flatten()

for i, (title, pivot_table) in enumerate(pivot_tables.items()):
    sns.heatmap(pivot_table, annot=True, fmt="d", cmap="BuPu", ax=axes[i])
    axes[i].set_title(f'{title} vs Satisfaction', fontsize=14)

plt.suptitle('Passenger Satisfaction Multidimensional Analysis' , **title_font)
plt.tight_layout()
plt.show()



#%% 

#####Corss Tabulation#####
# Perform cross-tabulation between 'Customer Type' and 'satisfaction'
cross_customer_satisfaction = pd.crosstab(index=airline_df['Customer Type'], columns=airline_df['satisfaction'])

# Convert the cross-tabulation to a DataFrame for plotting
cross_customer_satisfaction_df = cross_customer_satisfaction.reset_index()

# Melt the DataFrame for seaborn barplot
cross_customer_satisfaction_melted = cross_customer_satisfaction_df.melt(id_vars='Customer Type', value_vars=cross_customer_satisfaction.columns)

# Create the bar plot
plt.figure(figsize=(10, 6))
sns.barplot(data=cross_customer_satisfaction_melted, x='Customer Type', y='value', hue='satisfaction')

# Add labels and title for presentation
plt.title('Customer Satisfaction by Customer Type', fontsize=16)
plt.xlabel('Customer Type', fontsize=12)
plt.ylabel('Number of Passengers', fontsize=12)
plt.legend(title='Satisfaction')

# Show the plot
plt.tight_layout()
plt.show()


# %%

#####Comparitive Plots#############



#Delay Impact Analysis

# Setting up the plot with adjusted labels and legends
plt.figure(figsize=(16, 8))

# Scatter plot for Departure Delays vs Satisfaction
plt.subplot(1, 2, 1)
sns.regplot(x=airline_df['Departure Delay in Minutes'], y=airline_df['Satisfaction_Coded'], scatter_kws={'alpha':0.1}, line_kws={'color': 'red'})
plt.title('Departure Delays vs. Passenger Satisfaction', **title_font)
plt.xlabel('Departure Delay in Minutes', **label_font)
plt.ylabel('Satisfaction Level', **label_font)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(['Trend Line', 'Data Points'],loc='lower left')

# Scatter plot for Arrival Delays vs Satisfaction
plt.subplot(1, 2, 2)
sns.regplot(x=airline_df['Arrival Delay in Minutes'], y=airline_df['Satisfaction_Coded'], scatter_kws={'alpha':0.1}, line_kws={'color': 'red'})
plt.title('Arrival Delays vs. Passenger Satisfaction', **title_font)
plt.xlabel('Arrival Delay in Minutes', **label_font)
plt.ylabel('Satisfaction Level', **label_font)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(['Trend Line', 'Data Points'], loc='lower left')

# Show the plots
plt.tight_layout()
plt.show()


#%%


# Violin plot - boarding service

ratings_columns = ['Ease of Online booking', 'Online boarding', 'On-board service',
                   'Baggage handling', 'Checkin service']

# Melt the DataFrame to get it in long-form for the ratings
melted_df = airline_df.melt(id_vars=['Customer Type'], value_vars=ratings_columns, 
                            var_name='Service', value_name='Rating')

# Now create the violin plots
plt.figure(figsize=(14, 8))
sns.violinplot(x='Service', y='Rating', hue='Customer Type', data=melted_df, split=True, palette="muted")

# Customize the plot for presentation
plt.title('Boarding Service Ratings by Customer Type', **title_font)
plt.xlabel('Service', **label_font)
plt.ylabel('Rating', **label_font)
# plt.xticks(rotation=45)

# Place the legend outside the plot
plt.legend(title='Customer Type', loc='center left', bbox_to_anchor=(1, 0.5))

# Adjust the layout to make space for the legend
plt.tight_layout(rect=[0, 0, 0.75, 1])

plt.show()


#%%


#area chart

# Define distance categories
bins = [0, 1000, 3000, 7000]  # Example thresholds for short, medium, and long-haul flights
labels = ['Short-haul', 'Medium-haul', 'Long-haul']
airline_df['Distance Category'] = pd.cut(airline_df['Flight Distance'], bins=bins, labels=labels, right=False)

# Aggregate the data by 'Distance Category'
category_counts = airline_df['Distance Category'].value_counts().reindex(labels)

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))
category_counts.plot(kind='area', ax=ax, color='skyblue', alpha=0.5)  # You can choose any color you like
plt.title('Volume of Flights by Distance Category')
plt.xlabel('Flight Distance Category')
plt.ylabel('Number of Flights')
plt.xticks(range(len(labels)), labels)  # Ensure x-ticks match the labels
plt.grid(True)
plt.show()


# %%


# Define gender categories and labels
gender_labels = ['Female','Male']
gender_labels_legend = ['Male', 'Female']

# Calculate gender distribution
gender_data = airline_df['Gender'].value_counts().sort_index()

# Calculate the satisfaction rate for each Gender Group
gender_df_new = airline_df.groupby('Gender')['Satisfaction_Coded'].mean().reset_index()

# Rename the columns
gender_df_new.columns = ['Gender', 'Satisfaction Rate']

# Create a subplot with 2 rows and 1 column
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# Plot the pie chart for gender distribution
gender_colors = ['deepskyblue', 'orchid']
ax1.pie(gender_data, labels=gender_labels, autopct='%1.1f%%', startangle=90, colors=gender_colors)
ax1.set_title('Passenger Gender Distribution')
ax1.legend(gender_labels_legend, loc='upper left', bbox_to_anchor=(1, 1))


for i, (gender_group, satisfaction_rate) in enumerate(zip(gender_df_new['Gender'], gender_df_new['Satisfaction Rate'])):
    ax2.bar(i, satisfaction_rate, color=gender_colors[i], label=gender_group)
    ax2.text(i, satisfaction_rate, f'{satisfaction_rate:.1%}', ha='center', va='bottom')

ax2.set_title('Satisfaction Rate by Gender Group')
ax2.set_xlabel('Gender Group')
ax2.set_ylabel('Satisfaction Rate (%)')

# Set x-axis tick labels
ax2.set_xticks(range(len(gender_labels)))
ax2.set_xticklabels(gender_labels)

# Show plot
plt.tight_layout()
plt.show()


# %%


#%%
#Distribution of each gender by satisfaction according to the mean of Flight Distance


print("Mean Flight Distance by Gender and Satisfaction\n")
airline_df.pivot_table(index='satisfaction',columns='Gender',values='Flight Distance',aggfunc='mean').style.background_gradient(cmap='BuGn').format("{:.2f}")


# %%
#Delay Impact Analysis

# Setting up the plot with adjusted labels and legends
plt.figure(figsize=(16, 8))

# Scatter plot for Departure Delays vs Satisfaction
plt.subplot(1, 2, 1)
sns.regplot(x=airline_df['Departure Delay in Minutes'], y=airline_df['Satisfaction_Coded'], scatter_kws={'alpha':0.1}, line_kws={'color': 'red'})
plt.title('Departure Delays vs. Passenger Satisfaction', fontsize=16)
plt.xlabel('Departure Delay in Minutes', fontsize=14)
plt.ylabel('Satisfaction Level', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(['Trend Line', 'Data Points'])

# Scatter plot for Arrival Delays vs Satisfaction
plt.subplot(1, 2, 2)
sns.regplot(x=airline_df['Arrival Delay in Minutes'], y=airline_df['Satisfaction_Coded'], scatter_kws={'alpha':0.1}, line_kws={'color': 'red'})
plt.title('Arrival Delays vs. Passenger Satisfaction', fontsize=16)
plt.xlabel('Arrival Delay in Minutes', fontsize=14)
plt.ylabel('Satisfaction Level', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(['Trend Line', 'Data Points'])

# Show the plots
plt.tight_layout()
plt.show()



# %%


# Group data by 'Age' and calculate the mean for selected service aspects

grouped_data = airline_df.groupby('Age')[['Seat comfort', 'Inflight entertainment', 'Cleanliness']].mean()

# Setting up the plot again
plt.figure(figsize=(14, 7))

# Plotting multiple lines
for column in grouped_data.columns:
    sns.lineplot(data=grouped_data, x=grouped_data.index, y=column, label=column)

# Customizing the plot
plt.title('Average Ratings for Various Services by Age', fontsize=16)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Average Rating', fontsize=14)
plt.grid(True)
plt.legend(title='Service Type')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Show plot
plt.show()


# %%
#histogram plot

# Setting the style
sns.set(style="whitegrid")

# Creating the histogram with KDE plot
plt.figure(figsize=(10, 6))
sns.histplot(airline_df['Flight Distance'], kde=True, color='skyblue', binwidth=250, binrange=(0, 7000))

# Adding customizations
plt.title('Distribution of Flight Distances', fontsize=16)
plt.xlabel('Flight Distance (miles)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.axvline(airline_df['Flight Distance'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {airline_df["Flight Distance"].mean():.2f} miles')
plt.legend()

# Display the plot
plt.show()

# %%
