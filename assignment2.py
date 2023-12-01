# %%
import json
from collections import defaultdict
from sklearn import linear_model
from sklearn.metrics import mean_squared_error 
import numpy as np
import random
import gzip
import dateutil.parser
import math
import seaborn as sns
import matplotlib.pyplot as plt
import calendar


# %%
def parseData(fname):
    for l in open(fname):
        yield eval(l)

def getDictOfCategoryAndCounts(data, cat):
    d = defaultdict(set)
    for i in data:
        entry = i[cat]
        if entry not in d:
            d[entry] = 1
        else:
            d[entry] += 1
    return d

def getDictOfTimeCategoryAndCounts(data, time):
    d = defaultdict(set)
    for i in data:
        entry = i['review/timeStruct'][time]
        if entry not in d:
            d[entry] = 1
        else:
            d[entry] += 1
    return d

def getProfilesAndBeerDicts(data):
    profilesPerBeer = defaultdict(set)
    beersPerProfile = defaultdict(set)
    beerIdToBeerName = {}

    for d in data:
        profile,beerId = d['user/profileName'], d['beer/beerId']
        beersPerProfile[profile].add(beerId)
        profilesPerBeer[beerId].add(profile)
        if beerId not in beerIdToBeerName:
            beerIdToBeerName[beerId] = d['beer/name']

    return profilesPerBeer, beersPerProfile, beerIdToBeerName

# %%
data = list(parseData("beer_50000.json"))

# %%
print(data[0])

# %% [markdown]
# # Get some data to plot

# %%
beer_overall_ratings = getDictOfCategoryAndCounts(data, 'review/overall')
print(beer_overall_ratings)
print(len(beer_overall_ratings))

categories = beer_overall_ratings.keys()
values = beer_overall_ratings.values()

# Setting the style (optional, for aesthetics)
sns.set_style('whitegrid')

# Creating the bar chart using Seaborn
plt.figure(figsize=(8, 6))
sns.barplot(x=categories, y=values, palette='viridis')

# Adding titles and labels
plt.title('Distrubution of Overall Beer Ratings')
plt.xlabel('Overall Rating')
plt.ylabel('Count')

# Displaying the chart
plt.tight_layout()  # Adjust layout to prevent cutting off labels
plt.show()

# %%
beer_types = getDictOfCategoryAndCounts(data, 'beer/style')
print(beer_types)
print(len(beer_types))

categories = beer_types.keys()
values = beer_types.values()

# Create a horizontal bar plot
plt.figure(figsize=(10, 12))  # Adjust figure size as needed
sns.barplot(y=categories, x=values, palette='viridis')

# Adding titles and labels
plt.title('Distribution of Beer Styles')
plt.xlabel('Count')
plt.ylabel('Beer Styles')

# Rotate x-axis labels for better visibility
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

# %%
taste_reviews = getDictOfCategoryAndCounts(data, 'review/taste')
taste_reviews[0.0] = 0

print(taste_reviews)
print(len(taste_reviews))

categories = taste_reviews.keys()
values = taste_reviews.values()

# Setting the style (optional, for aesthetics)
sns.set_style('whitegrid')

# Creating the bar chart using Seaborn
plt.figure(figsize=(8, 6))
sns.barplot(x=categories, y=values, palette='viridis')

# Adding titles and labels
plt.title('Distrubution of Taste Rating')
plt.xlabel('Taste Rating')
plt.ylabel('Count')

# Displaying the chart
plt.tight_layout()  # Adjust layout to prevent cutting off labels
plt.show()

# %%
# Sample data for two sets of values
categories = beer_overall_ratings.keys()
values_set1 = beer_overall_ratings.values()
values_set2 = taste_reviews.values()

# Create bar plots for both sets of values
plt.figure(figsize=(8, 6))  # Set figure size

# Plotting the first set of values (blue bars)
sns.barplot(x=categories, y=values_set1, color='blue', alpha=.7, label='Overall Rating')

# Plotting the second set of values (orange bars, overlaying on the first)
sns.barplot(x=categories, y=values_set2, color='red', alpha=.7, label='Taste Rating')

# Adding legend, title, and labels
plt.legend()  # Show legend based on the labels provided in the plots
plt.title('Taste Rating v. Overall Rating')
plt.xlabel('Ratings')
plt.ylabel('Counts')

plt.tight_layout()
plt.show()

# %%
years = getDictOfTimeCategoryAndCounts(data, 'year')
print(years)
print(len(years))

categories = years.keys()
values = years.values()

# Setting the style (optional, for aesthetics)
sns.set_style('whitegrid')

# Creating the bar chart using Seaborn
plt.figure(figsize=(8, 6))
sns.barplot(x=categories, y=values, palette='viridis')

# Adding titles and labels
plt.title('Distrubution of Years when Review Written')
plt.xlabel('Year Review Written')
plt.ylabel('Count')

# Displaying the chart
plt.tight_layout()  # Adjust layout to prevent cutting off labels
plt.show()

# %%
months = getDictOfTimeCategoryAndCounts(data, 'mon')
print(months)
print(len(months))

sorted_months = dict(sorted(months.items()))  # Sorting by keys
print(sorted_months)


categories = [calendar.month_name[x] for x in sorted_months.keys()]
values = sorted_months.values()

# Setting the style (optional, for aesthetics)
sns.set_style('whitegrid')

# Creating the bar chart using Seaborn
plt.figure(figsize=(8, 6))
sns.barplot(x=categories, y=values, palette='viridis')

# Adding titles and labels
plt.title('Distrubution of Months when Review Written')
plt.xlabel('Day Review Written')
plt.ylabel('Count')

# Displaying the chart
plt.xticks(rotation=45)
plt.tight_layout()  # Adjust layout to prevent cutting off labels
plt.show()

# %%
days = getDictOfTimeCategoryAndCounts(data, 'wday')
print(days)
print(len(days))

sorted_days = dict(sorted(days.items()))  # Sorting by keys
print(sorted_days)

categories = [calendar.day_name[x] for x in sorted_days.keys()]
values = sorted_days.values()

# Setting the style (optional, for aesthetics)
sns.set_style('whitegrid')

# Creating the bar chart using Seaborn
plt.figure(figsize=(8, 6))
sns.barplot(x=categories, y=values, palette='viridis')

# Adding titles and labels
plt.title('Distrubution of Weekdays when Review Written')
plt.xlabel('Day Review Written')
plt.ylabel('Count')

# Displaying the chart
plt.xticks(rotation=45)
plt.tight_layout()  # Adjust layout to prevent cutting off labels
plt.show()

# %%
# profilesPerBeer maps each beer ID to a set of user profiles, e.g. {beer1 -> {userA, userB, userC}}
# beersPerProfile maps each user profile to a set of beer IDs, e.g. {userA -> {beer1, beer2, beer3}}
# beerIdToBeerNmae maps the beer ID to the name of the beer
profilesPerBeer, beersPerProfile, beerIdToBeerName = getProfilesAndBeerDicts(data)
print("Amount of users:", len(beersPerProfile))
print("Amount of beers:", len(profilesPerBeer))




