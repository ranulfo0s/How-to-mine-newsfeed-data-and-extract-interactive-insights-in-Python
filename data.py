import pandas as pd
pd.options.display.max_columns = 200
pd.options.mode.chained_assignment = None

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
from string import punctuation
import matplotlib.pyplot as plt

from collections import Counter
import re
import numpy as np

from tqdm import tqdm_notebook
tqdm_notebook().pandas()

data = pd.read_csv('./data/news.csv')
print(data.shape)
data.head(3)
data.category.value_counts(normalize=True).plot(kind='bar', grid=True, figsize=(16, 9))
plt.show()
data = data.drop_duplicates('description')
data = data[~data['description'].isnull()]
print(data.shape)
# data = data[(data.description.map(len) > 140) & (data.description.map(len) <= 300)]
data = data[(data.description.map(len) > 100)]
data.reset_index(inplace=True, drop=True)

print(data.shape)
# (13416, 9)
data.description.map(len).hist(figsize=(15, 5), bins=100)
# data = data.sample(25000, random_state=42)
data.reset_index(inplace=True, drop=True)

data.head(2)
plt.show()

#########################

stop_words = []

f = open('./data/stopwords.txt', 'r')
for l in f.readlines():
    stop_words.append(l.replace('\n', ''))

additional_stop_words = ['t', 'will']
stop_words += additional_stop_words

print(len(stop_words))

def _removeNonAscii(s):
    return "".join(i for i in s if ord(i)<128)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = text.replace('(ap)', '')
    text = re.sub(r"\'s", " is ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)
    text = re.sub('[^a-zA-Z ?!]+', '', text)
    text = _removeNonAscii(text)
    text = text.strip()
    return text

def tokenizer(text):
    text = clean_text(text)
    tokens = [word_tokenize(sent) for sent in sent_tokenize(text)]
    tokens = list(reduce(lambda x,y: x+y, tokens))
    tokens = list(filter(lambda token: token not in (stop_words + list(punctuation)) , tokens))
    return tokens

data['source'] = data['source'].map(lambda d: unicode(d.decode('utf-8')))
data['category'] = data['category'].map(lambda d: unicode(d.decode('utf-8')))
data['description'] = data['description'].map(lambda d: unicode(d.decode('utf-8')))
data['tokens'] = data['description'].progress_map(lambda d: tokenizer(d))

for descripition, tokens, category, source in zip(data['description'], data['tokens'], data['category'], data['source']):
    gotJapan = False
    gotHouse = True
    for token in tokens:
        # print(token)
        # print("+++++++++")
        if token == 'japan':
            gotJapan = True
        if token == 'house':
            gotHouse = True
    if gotJapan and gotHouse:
        print('description:', descripition)
        print('tokens:', tokens)
        print('Category:',  category)
        print('Source:', source)
        print()

# def keywords(category):
#     tokens = data[data['category'] == category]['tokens']
#     alltokens = []
#     for token_list in tokens:
#         alltokens += token_list
#     counter = Counter(alltokens)
#     return counter.most_common(10)
#
# for category in set(data['category']):
#     print('category :', category)
#     print('top 10 keywords:', keywords(category))
#     print('---')