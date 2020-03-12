#!/usr/bin/env python
# coding: utf-8

# ### Load the data

# In[1]:


import numpy as np
import pandas as pd

# Menampilkan tweet
pd.options.display.max_colwidth = 100

data_train = pd.read_csv('D:\\Rosebay\\Data Mining\\Sentimen Analisis\\train.csv', encoding = 'ISO-8859-1')


# In[2]:


data_train.head()


# In[3]:


data_train.info()


# ### Visualize the tweets

# In[4]:


rand_data = np.random.randint(1,len(data_train),50).tolist()
data_train['SentimentText'][rand_data]


# In[5]:


# Finding emoticon
import re
twits = data_train.SentimentText.str.cat()
emot = set(re.findall(r" ([xX:;][-']?.) ", twits))
hit_emot = []
for emo in emot:
    hit_emot.append((twits.count(emo), emo))
sorted(hit_emot, reverse = True)


# In[6]:


emot_seneng = r" ([xX;:]-?[dD]|:-?[\]|[;:][pP]) "
emot_sedih = r" (:'?[/|\(]) "
print('Happy Emoticons : ', set(re.findall(emot_seneng, twits)))
print('Sad Emoticons : ', set(re.findall(emot_sedih, twits)))


# ### Searching the most word used

# In[7]:


import nltk
from nltk.tokenize import word_tokenize

def kata_sering(text):
    token = word_tokenize(text)
    frek = nltk.FreqDist(token)
    print('There is %d different words' % len(set(token)))
    return sorted(frek, key=frek.__getitem__, reverse = True)


# In[8]:


import nltk
nltk.download('punkt')


# In[9]:


nltk.download('stopwords')


# In[10]:


nltk.download('wordnet')


# In[11]:


kata_sering(data_train.SentimentText.str.cat())[:100]


# In[12]:


from nltk.corpus import stopwords

ks = kata_sering(data_train.SentimentText.str.cat())
katasering = []
for s in ks:
    if len(katasering) == 1000:
        break
    if s in stopwords.words('english'):
        continue
    else:
        katasering.append(s)


# In[13]:


sorted(katasering)


# ### Stemming : hapus data yang artinya sama

# In[14]:


from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

def stem_tokenize(text):
    stemmer = SnowballStemmer('english')
    return[stemmer.stem(token) for token in word_tokenize(text.lower())]
                              
def lemma_tokenize(text):
    lemma = WordNetLemmatizer()
    return[lemma.lemmatize(token) for token in word_tokenize(text.lower())]


# ### Siapin data
# ##### Bag of Words

# In[15]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline


# In[16]:


# Hapus karakter gak penting (gak tau sedih apa seneng)
class TextPreProc(BaseEstimator, TransformerMixin):
    def __init__(self, use_mention = False):
        self.use_mention = use_mention
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        if self.use_mention:
            X = X.str.replace(r"@[a-zA-Z0-9_]* ", " @tags ")
        else:
            X = X.str.replace(r"@[a-zA-Z0-9_]* ", "")

        # Simpan kata setelah #
        X = X.str.replace("#", "")
        X = X.str.replace(r"[-\.\n]", "")
        
        # Hapus html
        X = X.str.replace(r"&\w+;", "")
        
        # Hapus link
        X = X.str.replace(r"https?://\S*", "")
        
        # Ganti huruf berulang hanya dengan 2 huruf
        X = X.str.replace(r"(.)\1+", r"\1\1")
        
        # Tandai emot seneng atau sedih
        X = X.str.replace(emot_seneng, ' Happy Emoticon ')
        X = X.str.replace(emot_sedih, ' Sad Emoticon ')
        X = X.str.lower()
        return X


# In[17]:


# pipeline twit pakai stemmer
from sklearn.model_selection import train_test_split

sentimen = data_train['Sentiment']
tweets = data_train['SentimentText']

vectorizer = TfidfVectorizer(tokenizer = lemma_tokenize, ngram_range = (1,2))
pipeline = Pipeline([('text_pre_processing', TextPreProc(use_mention = True)), ('vectorizer', vectorizer), ])


# ### Split data

# In[18]:


learn_data, test_data, sentiments_learning, sentiments_test = train_test_split(tweets, sentimen, test_size=0.1)
learning_data = pipeline.fit_transform(learn_data)


# ### Pilih Model

# In[19]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

reglog = LogisticRegression()
bernouli = BernoulliNB()
multinom = MultinomialNB()

models = {'Logistic Regression' : reglog, 'bernoulli' : bernouli, 'multinomial': multinom,}

for model in models.keys():
    skor = cross_val_score(models[model], learning_data, sentiments_learning, scoring = 'f1', cv = 10)
    print('===', model, '===')
    print('Scores = ', skor)
    print('Mean = ', skor.mean())
    print('Variance = ',skor.var())
    models[model].fit(learning_data, sentiments_learning)
    print('Score on the learning data (accuracy) = ', accuracy_score(models[model].predict(learning_data), sentiments_learning))
    print('')


# ##### Akurasi Bernoulli yang paling bagus
# 
# ### GridSearchCV (milih parameter terbaik)

# In[20]:


from sklearn.model_selection import GridSearchCV

grid_pipe = Pipeline([('text_pre_processing', TextPreProc()), ('vectorizer', TfidfVectorizer()), ('model', MultinomialNB()),])
parameter = [{'text_pre_processing__use_mention': [True, False], 'vectorizer__max_features': [1000, 2000, 5000, 10000, 20000, None], 'vectorizer__ngram_range': [(1,1), (1,2)],},]
grid = GridSearchCV(grid_pipe, parameter, cv = 5, scoring = 'f1')
grid.fit(learn_data, sentiments_learning)
print(grid.best_params_)


# ### Uji Model (Pakai Bernouli karena yang paling gede akurasi

# In[21]:


bernouli.fit(learning_data, sentiments_learning)


# In[22]:


data_test = pipeline.transform(test_data)
bernouli.score(data_test, sentiments_test)


# In[32]:


reglog.fit(learning_data, sentiments_learning)


# In[33]:


data_test = pipeline.transform(test_data)
reglog.score(data_test, sentiments_test)


# ### Prediksi pada data csv

# In[23]:


testing = pd.read_csv("D:\\Rosebay\\Data Mining\\Sentimen Analisis\\test.csv", encoding='ISO-8859-1', error_bad_lines=False)
testing["SentimentText"] = testing["SentimentText"].astype('str')

test_learn = pipeline.transform(testing.SentimentText)
test = pd.DataFrame(testing.ItemID, columns=('ItemID', 'SentimentText'))
test['Sentiment'] = bernouli.predict(test_learn)
print(test)


# In[24]:


testing


# ### Testing Tweet

# In[25]:


model = MultinomialNB()
model.fit(learning_data, sentiments_learning)


# In[30]:


tweet = pd.Series([input(),])
tweet = pipeline.transform(tweet)
peluang = model.predict_proba(tweet)[0]
print('Peluang twit sedih = ', peluang[0])
print('Peluang twit seneng = ', peluang[1])

