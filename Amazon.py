##############################################################
# Rating Product & Sorting Reviews and Keywords in Amazon
##############################################################

##############################################################
# 1. Business Problem
##############################################################

# Accurate product ratings and review sorting are crucial challenges in e-commerce. This project aims to enhance
# customer satisfaction and boost product visibility by implementing precise rating systems and effective review
# sorting mechanisms. By building trust and fostering a healthy trading environment, it benefits both customers and
# sellers. Additionally, providing users with a quick overview and easy review scanning improves the overall shopping
# experience. (4915 rows, 12 columns)

# Variables
# reviewerID: User ID
# asin: Product ID
# reviewerName: Username
# helpful: Helpful rating score [helpful_yes,total_vote]
# reviewText: Review text
# overall: Product rating
# summary: Review summary
# unixReviewTime: Review time (in Unix format)
# reviewTime: Review time (raw format)
# day_diff: Number of days since the review was posted
# helpful_yes: Number of times the review was found helpful
# total_vote: Total number of votes received for the review

###############################################################
# 2. Data Preparation
###############################################################

# Importing libraries
##############################################
import pandas as pd
import math
import scipy.stats as st
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 200)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv('Datasets/amazon_review.csv')

# Data understanding
##############################################
def check_df(dataframe, head=5):
    print('################# Columns ################# ')
    print(dataframe.columns)
    print('################# Types  ################# ')
    print(dataframe.dtypes)
    print('##################  Head ################# ')
    print(dataframe.head(head))
    print('#################  Shape ################# ')
    print(dataframe.shape)
    print('#################  NA ################# ')
    print(dataframe.isnull().sum())
    print('#################  Quantiles ################# ')
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99]).T)
    print('')

check_df(df)

# Remove rows with missing reviewer ID and reviewText
df = df.dropna(subset=['reviewerName', 'reviewText'])

#####################################################
# 1. Calculating Rating of Product
#####################################################
# Average rating
##############################################
df['overall'].value_counts()
df['overall'].mean()

# Weighted Average Rating by Date
##############################################
def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["day_diff"] <= 30, "overall"].mean() * w1 / 100 + \
        dataframe.loc[(dataframe["day_diff"] > 30) & (dataframe["day_diff"] <= 90), "overall"].mean() * w2 / 100 + \
        dataframe.loc[(dataframe["day_diff"] > 90) & (dataframe["day_diff"] <= 180), "overall"].mean() * w3 / 100 + \
        dataframe.loc[(dataframe["day_diff"] > 180), "overall"].mean() * w4 / 100

time_based_weighted_average(df)

#####################################################
# 2. Sorting Reviews
#####################################################

# Up-Down Diff Score = (up ratings) − (down ratings)
#####################################################
df['helpful_no'] = df['total_vote'] - df['helpful_yes']

def score_pos_neg_diff(pos, neg):
    df['score_pos_neg_diff'] = df[pos] - df[neg]
    return df

score_pos_neg_diff('helpful_yes', 'helpful_no').head(5)

# Score = Average rating = (up ratings) / (all ratings)
########################################################
def score_average_rating(pos, neg):
    df['score_average_rating'] = df[pos] / (df[pos] + df[neg])
    df['score_average_rating'] = df['score_average_rating'].fillna(0)  # for ZeroDivisionError
    return df

score_average_rating('helpful_yes', 'helpful_no').head(5)

# Wilson lower bound
########################################################
def wilson_lower_bound(pos, neg, confidence=0.95):
    n = pos + neg
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * pos / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

# Sort the DataFrame by the Wilson Lower Bound in descending order
###################################################################
top20_reviews = df.sort_values("wilson_lower_bound", ascending=False).head(20)
print(top20_reviews)

# What is Wilson Lower Bound? : Lower bound of Wilson score confidence interval for a Bernoulli parameter provides
# a way to sort a product based on positive and negative ratings. The idea here is to treat the existing set of
# user ratings as a statistical sampling of a hypothetical set of user ratings from all users and then use this score.
# In other words, what user community would think about upvoting a product with 95% confidence given that we have an
# existing rating for this product with a sample (subset from the whole community) user ratings.
# There exists many other approaches to calculate confidence interval for binomial distribution like Wald interval,
# Normal approximate interval, Agresti–Coull and many more, but Wilson confidence interval proves to be more robust and accurate.
# You can check this paper for more reference (Interval Estimation for a Binomial Proportion)
# [http://www-stat.wharton.upenn.edu/~tcai/paper/Binomial-StatSci.pdf].
# However, Bayesian Approximation makes more sense to use over Wilson Score for K-scale rating to take into effect where
# new product without rating and a product with only low ratings gets a zero rating.

############################################################################
#Function for Bayesian Approximation Rating
############################################################################
def calculate_bayesian_rating_products(rating_counts,confidence_level=0.95):
    if sum(rating_counts)==0:
        return 0
    # Calculate the expected expected value of the rating distribution
    num_scores=len(rating_counts)
    z=st.norm.ppf(1-(1-confidence_level)/2)
    total_ratings=sum(rating_counts)
    expected_value=0.0
    expected_value_squared=0.0
    for score,count in enumerate(rating_counts):
        probability=(count+1)/(total_ratings+num_scores)
        expected_value += (score + 1) * probability
        expected_value_squared += (score + 1) * (score + 1) * probability
    # Calculate the variance of the rating distribution
    variance=(expected_value_squared-expected_value **2)/(total_ratings+num_scores+1)
    # Calculate the Bayesian avg score
    bayesian_average=expected_value-z*math.sqrt(variance)
    return bayesian_average