from itertools import islice
from collections import defaultdict
import numpy as np
import numpy
import json
import nltk
import string
from PIL import Image
from nltk.stem.porter import *
from sklearn import linear_model
from langdetect import detect
punctuation = set(string.punctuation)
stemmer = PorterStemmer()

print "Reading data..."
parsedData = [json.loads(s) for s in open("data.json")]
print "done"

########################################################################
######## Reading Dataset and Storing in relevant Data Structures #######
########################################################################

users      = defaultdict(int)       ## Count of reviews usr has given
businesses = defaultdict(int)       ## Count of reviews business got
usr_rating = defaultdict(float)     ## Sum of usr ratings
bus_rating = defaultdict(float)     ## Sum of bus ratings
avg_usr_rating = defaultdict(float) ## Average usr rating
avg_bus_rating = defaultdict(float) ## Average bus rating
usr_list      = []
business_list = []
review1       = []
review3       = []
review5       = []

for d in parsedData:
    bus = d[u'business_id'].encode('utf-8')
    u   = d[u'user_id'].encode('utf-8')
    users[u]        += 1
    businesses[bus] += 1
    usr_rating[u]   += d[u'stars']
    bus_rating[bus] += d[u'stars']
    usr_list.append(u)
    business_list.append(bus)
    if d[u'stars'] == 1:
        review1.append(d[u'text'].encode('utf-8'))
    if d[u'stars'] == 3:
        review3.append(d[u'text'].encode('utf-8'))
    if d[u'stars'] == 5:
        review5.append(d[u'text'].encode('utf-8'))
    
for u in usr_rating:
    avg_usr_rating[u] = usr_rating[u]/users[u]
for b in bus_rating:
    avg_bus_rating[b] = bus_rating[b]/businesses[b]
    

users_sort = defaultdict(int)
businesses_sort = defaultdict(int)

users_sort = sorted(users.iteritems(), key=lambda (k,v): (v,k), reverse = True)
businesses_sort = sorted(businesses.iteritems(), key=lambda (k,v): (v,k), reverse = True)

avg_usr_rating = sorted(avg_usr_rating.iteritems(), key=lambda (k,v): (v,k), reverse = True)
avg_bus_rating = sorted(avg_bus_rating.iteritems(), key=lambda (k,v): (v,k), reverse = True)

print "User with max reviews = ",users_sort[:1]
print "Business with max reviews = ", businesses_sort[:1]
print "No. of unique users = ", len(users) 
print "No. of unique businesses = ", len(businesses)
print "User with max average rating = ", avg_usr_rating[:1]
print "Business with max average rating = ", avg_bus_rating[:1]


wordCount = []
i = 0
for d in parsedData:
    r = ''.join([c for c in d[u'text'].encode('utf-8').lower() if not c in punctuation])
    wordCount.append(len(r.split()))
    i+=1
index = wordCount.index(max(wordCount))
print "Index = ", index
print "User who gave longest review = ", usr_list[index]
print "Max Review length = ", max(wordCount)

index = wordCount.index(min(wordCount))
print "Index = ", index
print "User who gave Shortest review = ", usr_list[index]
print "Max Review length = ", min(wordCount)

index = wordCount.index(4)
print "Index = ", index
print "User who gave Shortest review = ", usr_list[index]
print "Max Review length = ", min(wordCount)


######################################################
############## Generating Wordclouds #################
######################################################
import nltk
nltk.download('popular')
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

###### Unigram ######
rating1=[d['text'] for d in parsedData if d['stars'] in [1]]
rating3=[d['text'] for d in parsedData if d['stars'] in [3]]
rating5=[d['text'] for d in parsedData if d['stars'] in [5]]
def unigramCount(rating):
    wordCountUnigram = defaultdict(int)
    for d in rating:
        r = ''.join([c for c in d.lower() if not c in punctuation])
        for w in r.split():
            wordCountUnigram[w] += 1
    countsUnigram = [(wordCountUnigram[w], w) for w in wordCountUnigram]
    countsUnigram.sort()
    countsUnigram.reverse()
    wordsUnigram = [x[1] for x in countsUnigram[:1000]]
    return wordsUnigram

def makeWordCloud(rating):
    maskpic=numpy.array(Image.open("mask1.jpg"))
    wordcloud = WordCloud(background_color="white",relative_scaling = 1.0,max_words=128,stopwords = STOPWORDS,mask=maskpic).generate(' '.join(rating))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

makeWordCloud(unigramCount(rating1))
makeWordCloud(unigramCount(rating3))
makeWordCloud(unigramCount(rating5))


###### Bigram ######
def bigramCount(rating):
    wordCountBigram = defaultdict(int)
    for d in rating:
        r = ''.join([c.encode('ascii', 'ignore') for c in d.lower() if c.encode('ascii', 'ignore') not in punctuation])
        r = ' '.join([e for e in r.split() if e not in STOPWORDS])
        for w in ngrams(r.split(),2):
            wordCountBigram[w] += 1
    countsBigram = [(wordCountBigram[w], w) for w in wordCountBigram]
    countsBigram.sort()
    countsBigram.reverse()
    wordsBigram = [x[1] for x in countsBigram[:1000]]
    return countsBigram

def makeWordCloud(rating):
    WC_max_words = 100
    word_dict = {}
    for i in range(len(rating)):
        word_dict['_'.join(rating[i][1])] = rating[i][0]
    maskpic=numpy.array(Image.open("11.jpg"))
    wordcloud = WordCloud(width=900,height=500,max_words=128,background_color="white",mask=maskpic,relative_scaling=.4,normalize_plurals=False).generate_from_frequencies(word_dict)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.figure()
    plt.show()

makeWordCloud(bigramCount(rating1))
makeWordCloud(bigramCount(rating3))
makeWordCloud(bigramCount(rating5))

###### Trigram ######
rating1=[d['text'] for d in parsedData if d['stars'] in [1]]
rating3=[d['text'] for d in parsedData if d['stars'] in [3]]
rating5=[d['text'] for d in parsedData if d['stars'] in [5]]
def unigramCount(rating):
    wordCountUnigram = defaultdict(int)
    for d in rating:
        r = ''.join([c for c in d.lower() if not c in punctuation])
        for w in r.split():
            wordCountUnigram[w] += 1
    countsUnigram = [(wordCountUnigram[w], w) for w in wordCountUnigram]
    countsUnigram.sort()
    countsUnigram.reverse()
    wordsUnigram = [x[1] for x in countsUnigram[:1000]]
    return wordsUnigram

def makeWordCloud(rating):
    maskpic=numpy.array(Image.open("mask1.jpg"))
    wordcloud = WordCloud(background_color="white",relative_scaling = 1.0,max_words=128,stopwords = STOPWORDS,mask=maskpic).generate(' '.join(rating))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

makeWordCloud(unigramCount(rating1))
makeWordCloud(unigramCount(rating3))
makeWordCloud(unigramCount(rating5))
