#!/usr/bin/env python
# coding: utf-8

# 

# In[2]:


### EDA Pkgs
import pandas as pd


# In[3]:


# Data Viz Pkg
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


# Hide warnings
import warnings
warnings.filterwarnings('ignore')


# In[5]:


# Load Dataset
df = pd.read_csv("C:/Users/vksta/Downloads/tie/covid19_tweets.csv")


# In[6]:


# Preview
df.head()


# #### Task
# + Text
#     - Text Preprocessing
#     - Sentiment Analysis
#     - Keyword Extraction
#     - Entity Extraction

# In[7]:


# Check Columns
df.columns


# In[8]:


# Datatype
df.dtypes


# In[9]:


# Source/ Value Count/Distribution of the Sources
df['source'].unique()


# In[10]:


# Source/ Value Count/Distribution of the Sources
df['source'].value_counts()


# In[11]:


# Plot the top value_counts
df['source'].value_counts().nlargest(30)


# In[12]:


# Plot the top value_counts
df['source'].value_counts().nlargest(30).plot(kind='bar')


# In[13]:


# Plot the top value_counts
plt.figure(figsize=(20,10))
df['source'].value_counts().nlargest(30).plot(kind='bar')
plt.xticks(rotation=45)
plt.show()


# #### Text Analysis of tweet

# In[14]:


#!pip install neattext


# In[ ]:


# Load Text Cleaning Package
#import neattext.functions as nfx
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download NLTK data (if not already downloaded)
nltk.download('stopwords')
nltk.download('punkt')
# Example text
text = ""

# Tokenize the text
tokens = word_tokenize(text)

# Remove punctuation and lowercase
cleaned_tokens = [word.lower() for word in tokens if word.isalpha()]

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in cleaned_tokens if word not in stop_words]

# Join the cleaned tokens into a sentence if needed
cleaned_text = ' '.join(filtered_tokens)

print(cleaned_text)


# In[ ]:


# Methods/Attrib
#dir(nfx)


# In[ ]:


df['text'].iloc[2]


# ### Noise
# + remove mentions/userhandles
# + remove hashtags
# + urls
# + emojis
# + special char

# In[ ]:


df.head()


# In[ ]:


#df['text'].apply(nfx.extract_hashtags)


# In[ ]:


#df['extracted_hashtags'] = df['text'].apply(nfx.extract_hashtags)


# In[ ]:


#df[['extracted_hashtags','hashtags']]


# In[ ]:


# Cleaning Text
#df['clean_tweet'] = df['text'].apply(nfx.remove_hashtags)


# In[ ]:


df[['text','text']]


# In[ ]:


#df['clean_tweet'] = df['clean_tweet'].apply(lambda x: nfx.remove_userhandles(x))


# In[ ]:


df[['text','text']]


# In[ ]:


df['text'].iloc[10]


# In[ ]:


# Cleaning Text: Multiple WhiteSpaces
#df['clean_tweet'] = df['text'].apply(nfx.remove_multiple_spaces)


# In[ ]:


df['text'].iloc[10]


# In[ ]:


# Cleaning Text : Remove urls
#df['clean_tweet'] = df['clean_tweet'].apply(nfx.remove_urls)


# In[ ]:


# Cleaning Text: Punctuations
#df['clean_tweet'] = df['clean_tweet'].apply(nfx.remove_puncts)


# In[ ]:


#df[['text','clean_tweet']]


# ### Sentiment Analysis

# In[ ]:


from textblob import TextBlob


# In[ ]:


def get_sentiment(text):
    blob = TextBlob(text)
    sentiment_polarity = blob.sentiment.polarity
    sentiment_subjectivity = blob.sentiment.subjectivity
    if sentiment_polarity > 0:
        sentiment_label = 'Positive'
    elif sentiment_polarity < 0:
        sentiment_label = 'Negative'
    else:
        sentiment_label = 'Neutral'
    result = {'polarity':sentiment_polarity,
              'subjectivity':sentiment_subjectivity,
              'sentiment':sentiment_label}
    return result


# In[ ]:


# Text
ex1 = df['text'].iloc[0]


# In[ ]:


get_sentiment(ex1)


# In[ ]:


print(df['text'].dtypes)
df['text'] = df['text'].astype(str)

import pandas as pd

# Load your DataFrame, assuming df contains your data
# ...

# Convert 'text' column to strings
df['text'] = df['text'].astype(str)

# Handle missing values (if needed)
df['text'].fillna('', inplace=True)  # Replace missing values with an empty string

# Now you can perform text processing on the 'text' column

df['sentiment_results'] = df['text'].apply(get_sentiment)




# In[ ]:


df['sentiment_results']


# In[ ]:


df['sentiment_results'].iloc[0]


# In[ ]:


pd.json_normalize(df['sentiment_results'].iloc[0])


# In[ ]:


df = df.join(pd.json_normalize(df['sentiment_results']))


# In[ ]:


df.head()


# In[ ]:


df['sentiment'].value_counts()


# In[ ]:


df['sentiment'].value_counts().plot(kind='bar')


# In[ ]:


# Plot with seaborn
#sns.countplot(df['sentiment'])
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df is your DataFrame
sns.set(style="darkgrid")
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='sentiment')
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()





# In[ ]:


### Keyword Extraction
#+ For Positive and Negative Sentiment
#+ General


# In[ ]:


positive_tweet = df[df['sentiment'] == 'Positive']['text']


# In[ ]:


neutral_tweet = df[df['sentiment'] == 'Neutral']['text']
negative_tweet = df[df['sentiment'] == 'Negative']['text']


# In[ ]:


positive_tweet


# In[ ]:


# Remove Stopwords and Convert to Tokens
#positive_tweet_list = positive_tweet.apply(.remove_stopwords).tolist()

from nltk.corpus import stopwords

text = "Remove stopwords from this positive_tweet_list"
stop_words = set(stopwords.words("english"))
filtered_text = "positive_tweet".join(word for word in text.split() if word.lower() not in stop_words)


# In[ ]:


#negative_tweet_list = negative_tweet.apply(nfx.remove_stopwords).tolist()
#neutral_tweet_list = neutral_tweet.apply(nfx.remove_stopwords).tolist()

from nltk.corpus import stopwords

text = "Remove stopwords from this negative_tweet_list"
stop_words = set(stopwords.words("english"))
filtered_text = "negative_tweet".join(word for word in text.split() if word.lower() not in stop_words)

from nltk.corpus import stopwords

text = "Remove stopwords from thisneutral_tweet_list"
stop_words = set(stopwords.words("english"))
filtered_text = "neutral_tweet".join(word for word in text.split() if word.lower() not in stop_words)


# In[ ]:


positive_tweet


# In[ ]:


# Tokenization
for line in positive_tweet:
#     print(line)
    for token in line.split():
        print(token)


# In[ ]:


pos_tokens = [token for line in positive_tweet  for token in line.split()]


# In[ ]:


neg_tokens = [token for line in negative_tweet  for token in line.split()]
neut_tokens = [token for line in neutral_tweet  for token in line.split()]


# In[ ]:


pos_tokens


# In[ ]:


# Get Most Commonest Keywords
from collections import Counter


# In[ ]:


def get_tokens(docx,num=30):
    word_tokens = Counter(docx)
    most_common = word_tokens.most_common(num)
    result = dict(most_common)
    return result


# In[ ]:


get_tokens(pos_tokens)


# In[ ]:


most_common_pos_words = get_tokens(pos_tokens)
most_common_neg_words = get_tokens(neg_tokens)
most_common_neut_words = get_tokens(neut_tokens)


# In[ ]:


# Plot with seaborn
neg_df = pd.DataFrame(most_common_neg_words.items(),columns=['words','scores'])


# In[ ]:


neg_df


# In[ ]:


plt.figure(figsize=(20,10))
sns.barplot(x='words',y='scores',data=neg_df)
plt.xticks(rotation=45)
plt.show()


# In[ ]:


# Plot with seaborn
pos_df = pd.DataFrame(most_common_pos_words.items(),columns=['words','scores'])
plt.figure(figsize=(20,10))
sns.barplot(x='words',y='scores',data=pos_df)
plt.xticks(rotation=45)
plt.show()


# In[ ]:


# Plot with seaborn
neut_df = pd.DataFrame(most_common_neut_words.items(),columns=['words','scores'])
plt.figure(figsize=(20,10))
sns.barplot(x='words',y='scores',data=neut_df)
plt.xticks(rotation=45)
plt.show()


# In[ ]:


### Word Cloud
from wordcloud import WordCloud


# In[ ]:


def plot_wordcloud(docx):
    plt.figure(figsize=(20,10))
    mywordcloud = WordCloud().generate(docx)
    plt.imshow(mywordcloud,interpolation='bilinear')
    plt.axis('off')
    plt.show()


# In[ ]:


pos_docx = ' '.join(pos_tokens)
neg_docx = ' '.join(neg_tokens)
neu_docx = ' '.join(neut_tokens)


# In[ ]: 

import pandas as pd

# Assuming df1 and df2 are your DataFrames
df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'sentiment': ['Positive', 'Negative', 'Neutral']})
df2 = pd.DataFrame({'C': [7, 8, 9], 'D': [10, 11, 12], 'sentiment': ['Positive', 'Negative', 'Neutral']})

# Specify a suffix for overlapping columns
suffixes = ('_df1', '_df2')

# Merge the DataFrames with suffixes
merged_df = df1.merge(df2, left_index=True, right_index=True, suffixes=suffixes)

print(merged_df)



plot_wordcloud(pos_docx)


# In[ ]:


plot_wordcloud(neg_docx)


# In[ ]:


plot_wordcloud(neu_docx)


# In[ ]:


plot_wordcloud(pos_docx)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




