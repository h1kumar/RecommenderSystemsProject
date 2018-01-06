import random
from random import shuffle
from nltk import ngrams
from random import shuffle
from nltk import ngrams
from collections import defaultdict
import json
import string
import nltk
from sklearn import linear_model,ensemble
from sklearn.ensemble import RandomForestRegressor
from wordcloud import WordCloud, STOPWORDS

nltk.download('universal_tagset')
punctuation = set(string.punctuation)

#########################################################################
##### Reading Data and splitting it into Train/Validation/Test Set ######
#########################################################################
print "Reading data..."
parsedData = [json.loads(s) for s in open("data.json")]
print "done"

#shuffle data
random.Random(4).shuffle(parsedData)

trainData=parsedData[:60000]
trainData_reviews=[d['text'] for d in trainData]

validationData=parsedData[60000:80000]
validationData_reviews=[d['text'] for d in validationData]

testData=parsedData[80000:]
testData_reviews=[d['text'] for d in testData]


##############################################################
#### Feature Vector Extraction using N-Grams #################
##############################################################
def ngramCount(n,data):
    wordCountNgram = defaultdict(int)
    for d in data:
        r = ''.join([c for c in d.lower() if not c in punctuation])
        r = ' '.join([e for e in r.split() if e not in STOPWORDS])
        if n==1:
            for w in r.split():
            #w = stemmer.stem(w) # with stemming
                wordCountNgram[w] += 1
        else:
            for w in ngrams(r.split(),2):
                #w = stemmer.stem(w) # with stemming
                wordCountNgram[w] += 1
    countsNgram = [(wordCountNgram[w], w) for w in wordCountNgram]
    countsNgram.sort()
    countsNgram.reverse()
    wordsNgram = [x[1] for x in countsNgram[:1000]]
    return wordsNgram

def feature(datum,wordsNgram,n):
  wordIdNgram = dict(zip(wordsNgram, range(len(wordsNgram))))
  feat = [0]*len(wordsNgram)
  r = ''.join([c for c in datum['text'].lower() if not c in punctuation])
  r = ' '.join([e for e in r.split() if e not in STOPWORDS])

  if n==1:
      for w in r.split():
        if w in wordsNgram:
          feat[wordIdNgram[w]] += 1
  else:
      for w in ngrams(r.split(),n):
        if w in wordsNgram:
          feat[wordIdNgram[w]] += 1
  feat.append(1) #offset
  return feat

def adjectives(data):
    wordCountadj = defaultdict(int)
    for d in data:
        r = ''.join([c for c in d.lower() if not c in punctuation])
        r = ' '.join([e for e in r.split() if e not in STOPWORDS])
        tags = nltk.pos_tag(r.split(),tagset='universal')
        for t in tags:
            if t[1] == 'ADJ':
                wordCountadj[t[0]] += 1
    countsadj = [(wordCountadj[w], w) for w in wordCountadj]
    countsadj.sort()
    countsadj.reverse()
    wordsadj = [x[1] for x in countsadj[:1000]]
    print len(countsadj)
    return wordsadj

######################
####### Unigram ######
######################
wordsNgram=ngramCount(1,trainData_reviews)
X_train = [feature(d,wordsNgram,1) for d in trainData]
y_train = [d['stars'] for d in trainData]
regr = RandomForestRegressor(max_depth=30, random_state=0,n_estimators=50)
regr.fit(X_train, y_train)

## Unigram Validation ##
X_validation = [feature(d,wordsNgram,1) for d in validationData]
predictions = regr.predict(X_validation)
correct=0
actualRatings=[d['stars'] for d in validationData]
for (p,a) in zip(predictions,actualRatings):
    correct += (p-a)**2
MSE =correct/float(len(predictions))
print "MSE validation= ",MSE

## Unigram TEST ##
X_test = [feature(d,wordsNgram,1) for d in testData]
y_test = [d['stars'] for d in testData]
predictions = regr.predict(X_test)
correct=0
actualRatings=[d['stars'] for d in testData]
for (p,a) in zip(predictions,actualRatings):
    correct += (p-a)**2
MSE =correct/float(len(predictions))
print "MSE test= ",MSE


######################
####### Bigram #######
######################
wordsNgram=ngramCount(2,trainData_reviews)
X_train = [feature(d,wordsNgram,2) for d in trainData]
y_train = [d['stars'] for d in trainData]
regr = RandomForestRegressor(max_depth=30, random_state=0,n_estimators=50)
regr.fit(X_train, y_train)

### BIGRAM Validation ###
X_validation = [feature(d,wordsNgram,2) for d in validationData]
predictions = regr.predict(X_validation)
correct=0
actualRatings=[d['stars'] for d in validationData]
for (p,a) in zip(predictions,actualRatings):
    correct += (p-a)**2
MSE =correct/float(len(predictions))
print "MSE validation= ",MSE

### BIGRAM TEST ###
X_test = [feature(d,wordsNgram,2) for d in testData]
predictions = regr.predict(X_test)
correct=0
actualRatings=[d['stars'] for d in testData]
for (p,a) in zip(predictions,actualRatings):
    correct += (p-a)**2
MSE =correct/float(len(predictions))
print "MSE test= ",MSE


##################################
######## Unigram + BIGRAM ########
##################################
def ngramCount(n,data):
    wordCountNgram = defaultdict(int)
    for d in data:
        r = ''.join([c for c in d.lower() if not c in punctuation])
        r = ' '.join([e for e in r.split() if e not in STOPWORDS])
        if n==1:
            for w in r.split():
            #w = stemmer.stem(w) # with stemming
                wordCountNgram[w] += 1
        else:
            for w in ngrams(r.split(),2):
                #w = stemmer.stem(w) # with stemming
                wordCountNgram[w] += 1
    countsNgram = [(wordCountNgram[w], w) for w in wordCountNgram]
    countsNgram.sort()
    countsNgram.reverse()
    wordsNgram = [x[1] for x in countsNgram[:1000]]
    return countsNgram,wordsNgram

def feature(datum,wordsNgram):
  wordIdNgram = dict(zip(wordsNgram, range(len(wordsNgram))))
  feat = [0]*len(wordsNgram)
  r = ''.join([c for c in datum['text'].lower() if not c in punctuation])
  r = ' '.join([e for e in r.split() if e not in STOPWORDS])

  for w in r.split():
        if w in wordsNgram:
          feat[wordIdNgram[w]] += 1
        
  for w in ngrams(r.split(),2):
        if w in wordsNgram:
          feat[wordIdNgram[w]] += 1
  feat.append(1) #offset
  return feat
  
countsUnigram,wordsUnigram=ngramCount(1,trainData_reviews)
countsBigram,wordsBigram=ngramCount(2,trainData_reviews)

countsNgram=countsUnigram+countsBigram
countsNgram.sort()
countsNgram.reverse()
wordsNgram = [x[1] for x in countsNgram[:1000]]

###### UNI + BI TRAIN ########
X_train = [feature(d,wordsNgram) for d in trainData]
y_train = [d['stars'] for d in trainData]
regr = RandomForestRegressor(max_depth=30, random_state=0,n_estimators=50)
regr.fit(X_train, y_train)

###### UNI + BI VALID ########
X_validation = [feature(d,wordsNgram) for d in validationData]
predictions = regr.predict(X_validation)
correct=0
actualRatings=[d['stars'] for d in validationData]
for (p,a) in zip(predictions,actualRatings):
    correct += (p-a)**2
MSE =correct/float(len(predictions))
print "MSE validation= ",MSE

###### UNI + BI TEST ########
X_test = [feature(d,wordsNgram) for d in testData]
predictions = regr.predict(X_test)
correct=0
actualRatings=[d['stars'] for d in testData]
for (p,a) in zip(predictions,actualRatings):
    correct += (p-a)**2
MSE =correct/float(len(predictions))
print "MSE test= ",MSE

############### END OF N-GRAMS ###################################
##################################################################

##################################
########## TFIDF #################
##################################


########### UNIGRAM TF-IDF ################
from random import shuffle
from nltk import ngrams
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
punctuation = set(string.punctuation)
from collections import Counter
import numpy as np 


def ngramCount(n,data):
    wordCountNgram = defaultdict(int)
    for d in data:
        r = ''.join([c for c in d.lower() if not c in punctuation])
        r = ' '.join([e for e in r.split() if e not in STOPWORDS])
        if n==1:
            for w in r.split():
            #w = stemmer.stem(w) # with stemming
                wordCountNgram[w] += 1
        else:
            for w in ngrams(r.split(),2):
                #w = stemmer.stem(w) # with stemming
                wordCountNgram[w] += 1
    countsNgram = [(wordCountNgram[w], w) for w in wordCountNgram]
    countsNgram.sort()
    countsNgram.reverse()
    wordsNgram = [x[1] for x in countsNgram[:1000]]
    return countsNgram,wordsNgram


df=defaultdict(int)
countsUnigram,wordsUnigram=ngramCount(1,trainData_reviews)

for d in trainData:
      r = ''.join([c for c in d['text'].lower() if not c in punctuation])
      r = ' '.join([c for c in r.split() if not c in STOPWORDS])
      for w in wordsUnigram:    
        if w in r.split():
            df[w]+=1
            
def feature(datum):
  rwords = ''.join([c for c in datum['text'].lower() if not c in punctuation])
  rwords = ' '.join([c for c in rwords.split() if not c in STOPWORDS])

  tf=Counter(rwords.split())
  feat=[]

  for w in wordsUnigram:
    feat.append(np.log10(len(trainData)/float(df[w]))*tf[w])
    
  feat.append(1) #offset
  return feat

ngramCount
X_train = [feature(d) for d in trainData]
y_train = [d['stars'] for d in trainData]
regr = RandomForestRegressor(max_depth=30, random_state=0,n_estimators=50)
regr.fit(X_train, y_train)

########### UNIGRAM TF-IDF VALIDATION ################
X_validation = [feature(d) for d in validationData]
predictions = regr.predict(X_validation)
correct=0
actualRatings=[d['stars'] for d in validationData]
for (p,a) in zip(predictions,actualRatings):
    correct += (p-a)**2
MSE =correct/float(len(predictions))
print "MSE validation= ",MSE

###### UNIGRAM TF-IDF TEST ########
X_test = [feature(d) for d in testData]
predictions = regr.predict(X_test)
correct=0
actualRatings=[d['stars'] for d in testData]
for (p,a) in zip(predictions,actualRatings):
    correct += (p-a)**2
MSE =correct/float(len(predictions))
print "MSE test= ",MSE


###########################################
########### BIGRAM TF-IDF #################
###########################################
df=defaultdict(int)
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
punctuation = set(string.punctuation)

def ngramCount(n,data):
    wordCountNgram = defaultdict(int)
    for d in data:
        r = ''.join([c for c in d.lower() if not c in punctuation])
        r = ' '.join([e for e in r.split() if e not in STOPWORDS])
        if n==1:
            for w in r.split():
            #w = stemmer.stem(w) # with stemming
                wordCountNgram[w] += 1
        else:
            for w in ngrams(r.split(),2):
                #w = stemmer.stem(w) # with stemming
                wordCountNgram[w] += 1
    countsNgram = [(wordCountNgram[w], w) for w in wordCountNgram]
    countsNgram.sort()
    countsNgram.reverse()
    wordsNgram = [x[1] for x in countsNgram[:1000]]
    return countsNgram,wordsNgram

countsBigram,wordsBigram=ngramCount(2,trainData_reviews)

for d in trainData:
      r = ''.join([c for c in d['text'].lower() if not c in punctuation])
      r = ' '.join([c for c in r.split() if not c in STOPWORDS])
      for w in wordsBigram:    
        if w in ngrams(r.split(),2):
            df[w]+=1

def feature(datum):
  rwords = ''.join([c for c in datum['text'].lower() if not c in punctuation])
  rwords = ' '.join([c for c in rwords.lower() if not c in STOPWORDS])

  tf=Counter(ngrams(r.split(),2))
  feat=[]

  for w in wordsBigram:
    feat.append(np.log10(len(trainData)/float(df[w]))*tf[w])
    
  feat.append(1) #offset
  return feat


X_train = [feature(d) for d in trainData]
y_train = [d['stars'] for d in trainData]
regr = RandomForestRegressor(max_depth=30, random_state=0,n_estimators=50)
regr.fit(X_train, y_train)

### BIGRAM TF-IDF VALIDATION #####
X_validation = [feature(d) for d in validationData]
predictions = regr.predict(X_validation)
correct=0
actualRatings=[d['stars'] for d in validationData]
for (p,a) in zip(predictions,actualRatings):
    correct += (p-a)**2
MSE =correct/float(len(predictions))
print "MSE validation= ",MSE

###### BIGRAM TF-IDF TEST ########
X_test = [feature(d) for d in testData]
predictions = regr.predict(X_test)
correct=0
actualRatings=[d['stars'] for d in testData]
for (p,a) in zip(predictions,actualRatings):
    correct += (p-a)**2
MSE =correct/float(len(predictions))
print "MSE test= ",MSE



###########################################
####### UNIGRAM + BIGRAM TF-IDF ###########
###########################################
from collections import Counter
import numpy
def ngramCount(n,data):
    wordCountNgram = defaultdict(int)
    for d in data:
        r = ''.join([c for c in d.lower() if not c in punctuation])
        r = ' '.join([e for e in r.split() if e not in STOPWORDS])
        if n==1:
            for w in r.split():
            #w = stemmer.stem(w) # with stemming
                wordCountNgram[w] += 1
        else:
            for w in ngrams(r.split(),2):
                #w = stemmer.stem(w) # with stemming
                wordCountNgram[w] += 1
    countsNgram = [(wordCountNgram[w], w) for w in wordCountNgram]
    countsNgram.sort()
    countsNgram.reverse()
    wordsNgram = [x[1] for x in countsNgram[:1000]]
    return countsNgram,wordsNgram


df=defaultdict(int)
countsUnigram,wordsUnigram=ngramCount(1,trainData_reviews)
countsBigram,wordsBigram=ngramCount(2,trainData_reviews)

countsBoth=countsUnigram+countsBigram
countsBoth.sort()
countsBoth.reverse()
countsBoth=countsBoth
wordsBoth= [x[1] for x in countsBoth[:1000]]

for d in trainData:
      r = ''.join([c for c in d['text'].lower() if not c in punctuation])
      r = ' '.join([c for c in r.split() if not c in STOPWORDS])
      for w in wordsBoth:    
        if w in r.split():
            df[w]+=1

for d in trainData:
      r = ''.join([c for c in d['text'].lower() if not c in punctuation])
      r = ' '.join([c for c in r.split() if not c in STOPWORDS])
      for w in wordsBoth:    
        if w in ngrams(r.split(),2):
            df[w]+=1
            
 
def feature(datum):
  rwords = ''.join([c for c in datum['text'].lower() if not c in punctuation])
  rwords = ' '.join([c for c in rwords.lower() if not c in STOPWORDS])

  tf_unigram=Counter(rwords.split())
  tf_bigram=Counter(ngrams(rwords.split(),2))
  tf=tf_unigram+tf_bigram
  feat=[]

  for w in wordsBoth:
    feat.append(numpy.log10(len(trainData)/float(df[w]))*tf[w])
    
  feat.append(1) #offset
  return feat

X_train = [feature(d) for d in trainData]
y_train = [d['stars'] for d in trainData]
regr = RandomForestRegressor(max_depth=30, random_state=0,n_estimators=50)
regr.fit(X_train, y_train)

###### UNI+BI TF-IDF VALIDATION ###########
X_validation = [feature(d) for d in validationData]
predictions = regr.predict(X_validation)
correct=0
actualRatings=[d['stars'] for d in validationData]
for (p,a) in zip(predictions,actualRatings):
    correct += (p-a)**2
MSE =correct/float(len(predictions))
print "MSE validation= ",MSE

###### UNI+BI TF-IDF TEST ########
X_test = [feature(d) for d in testData]
predictions = regr.predict(X_test)
correct=0
actualRatings=[d['stars'] for d in testData]
for (p,a) in zip(predictions,actualRatings):
    correct += (p-a)**2
MSE =correct/float(len(predictions))
print "MSE test= ",MSE