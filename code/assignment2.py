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
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


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

def getMostAndLeastPopularBeers(data, beerIdToBeerName, atLeastXReviews):
    # {'47986': [1.5], '48213': [3.0], '48215': [3.0], '47969': [3.0], '64883': [4.0], '52159': [
    # Calculate average ratings for each beer
    beer_ratings = defaultdict(list)

    for d in data:
        beer_ratings[d["beer/beerId"]].append(d["review/overall"])

    elem_to_del = []
    for d in beer_ratings:
        if len(beer_ratings[d]) < atLeastXReviews:
            elem_to_del.append(d)

    for d in elem_to_del:
        beer_ratings.pop(d)

    print("[getMostAndLeastPopularBeers] Length of beer ratings dict:", len(beer_ratings))

    beer_average_ratings = {
        beer_id: sum(ratings) / len(ratings) for beer_id, ratings in beer_ratings.items()
    }

    # Find the most and least popular beers by average rating
    most_popular_beer = max(beer_average_ratings, key=beer_average_ratings.get)
    least_popular_beer = min(beer_average_ratings, key=beer_average_ratings.get)

    most_popular_beer_str = beerIdToBeerName[most_popular_beer]
    least_popular_beer_str = beerIdToBeerName[least_popular_beer]
    print("The most popular beer is: '", most_popular_beer_str, "' with an average rating of", beer_average_ratings[most_popular_beer])
    print(len(beer_ratings[most_popular_beer]), "people reviewed this beer")

    print("")

    print("The least popular beer is: '", least_popular_beer_str, "' with an average rating of", beer_average_ratings[least_popular_beer])
    print(len(beer_ratings[least_popular_beer]), "people reviewed this beer")

    # print(beer_ratings)

# %%
data = list(parseData("../data/beer_50000.json"))

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

weighted_sum = 0
total_count = 0
for rating in beer_overall_ratings:
    weighted_sum += rating * beer_overall_ratings[rating]
    total_count += beer_overall_ratings[rating]

print(weighted_sum/total_count)

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
plt.xlabel('Rating')
plt.ylabel('Count')

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



# %%
getMostAndLeastPopularBeers(data, beerIdToBeerName, 10)


# %% [markdown]
# # Basic Sentiment Analysis

# %%
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# if sentiment_score > 0:
#     sentiment = "positive"
# elif sentiment_score < 0:
#     sentiment = "negative"
# else:
#     sentiment = "neutral"

review_text = []
overall_reviews = []

for d in data:
    review_text.append(d['review/text'])
    overall_reviews.append(d['review/overall'])


# %%
accuracy = 0

for i in range(len(data)):
    review = review_text[i]
    overall_rating = overall_reviews[i]
    sentiment_score = sia.polarity_scores(review)['compound']

    # anything 3 or above is seen as a positive recommendation
    if sentiment_score >= 0 and overall_rating >= 3:
        accuracy += 1
    elif sentiment_score < 0 and overall_rating < 3:
        accuracy += 1

print(accuracy/len(data))


