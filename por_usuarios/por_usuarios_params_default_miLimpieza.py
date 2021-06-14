#!/usr/bin/env python
# coding: utf-8

# # Proyecto Final de la Maestria Por Usuarios
# ### Cuantificación y predicción de posibles casos de depresión en usuarios de twitter utilizando machine learning.
# ### Enrique Isidoro Vázquez Ramos
# ### Datos:
# 
# Cada línea contiene 3 campos: user, klass y tweets.
# 
# - Usuarios:
#     - Hay 317 usuarios (Hay que tener en cuenta que las clases no están balanceadas)
#         - 92  considerados con depresión
#         - 225 considerados sin depresion
#     
# - Clases:
#     - klass = 1  considerados con  depresión
#     - klass = 0 sin depresión
# 
# - Tweets:
# 
#     - El campo tweets  es una lista y cada elemento es un tweet que contiene todos los datos del tweet posteado: fecha, id, texto, fecha de creacion, geolocalizacion, etc.
#     - Se juntaran todos los tweets de cada usuario para así tener un tweet único por usuario. 

# ### Código Inicial.
# ### Datos

# In[1]:


import tarfile
tar = tarfile.open("usuarios_depresion.json.tar.gz")
tar.extractall()
tar.close()


# In[2]:


from microtc.utils import tweet_iterator
data = [i for i in tweet_iterator('usuarios_depresion.json')]


# ### Extraer los tweets

# In[5]:


data[0]['tweets'][4099]['text']


# ### Extraer lo que se requiere, tres listas de dimension 317 en donde todos los tweets de cada usuario se juntaron para hacer solo un tweet unico por usuario.

# In[26]:


klases,users,t,tweets = [],[],[],[]
for i in data:
    t = []
    klases.append(int(i['klass']))
    users.append(i['user'])
    for j in i['tweets']:
        t.append(j['text'])
    tweets.append(' '.join(t))


# In[27]:


print('Data Original')
print(len(klases))
print(len(tweets))
print(len(users))


# ### Limpieza

# In[30]:


import re as reg
import unicodedata
import nltk
import re, string

textos = tweets.copy()

def QuitarAcentos(cadena):
    formaNFKD = unicodedata.normalize('NFKD', cadena)
    return u"".join([c for c in formaNFKD if not unicodedata.combining(c)])
def RP(text):
    return re.sub('[%s]' % re.escape(string.punctuation), '', text)
def RD(digito):
    return ''.join(i for i in digito if not i.isdigit())
#def deEmojify(inputString):
#    return inputString.encode('ascii', 'ignore').decode('ascii')
def reducirString(palabra):
    pattern=reg.compile(r"(.)\1{1,}",reg.DOTALL)
    string=pattern.sub(r"\1",palabra)
    return string

l1,l2,l3,l4,l5 = [],[],[],[],[]

for i in textos:
    l1.append(QuitarAcentos(i).lower())    
for i in l1:
    aux = []
    h = i.split()
    for r in h:
        if (r.startswith('http') or r.startswith('rt') or r.startswith('"http')): # Si la palabra comienza con lo señalado entonces no se toma en cuenta.
            continue
        aux.append(r)
    l2.append(' '.join(aux)) 
for i in l2:
    l3.append(RP(i))      
for i in l3:
    l4.append(RD(i))
#for i in TextoL:
#    TextoLimpio.append(deEmojify(i))
for i in l4:
    aux = i.split()
    aux2 = []
    for j in aux:
        aux2.append(reducirString(j))
    l5.append(' '.join(aux2))


# In[31]:


nltk.download('stopwords')
from nltk.corpus import stopwords
lsw = stopwords.words('spanish')

TweetsLimpios = []

for i in l5:
    aux = []
    h = i.split()
    for r in h:
        if r not in lsw: #and r not in lsw2):
            aux.append(r)
    TweetsLimpios.append(' '.join(aux))
#from nltk.stem.snowball import SnowballStemmer
#stemmer = SnowballStemmer("spanish")
#for i in QR:
#    aux = []
#    for j in i.split():
#        aux.append(stemmer.stem(j))
#    stem1.append(' '.join(aux))
print('Data Trabajada')
print(len(users))
print(len(klases))
print(len(TweetsLimpios))


# In[38]:


import numpy as np
print('Limpieza realizada')
print(np.unique(klases, return_counts=True))


# ### Tabla en pandas con la data

# In[93]:


import pandas as pd
dataE = pd.DataFrame({'Textos':TweetsLimpios, 'Klases':klases,'Usuarios':users})


# ### Modelo FastText

# In[ ]:

import fasttext
ft = fasttext.load_model('crawl-300d-2M-subword.bin')#, encoding="latin1")
print('Se cargo el modelo de fasttext')


# In[ ]:


embeddingsEF = [ ft.get_sentence_vector(v) for v  in list(dataE['Textos'])]
print('Se crearon los vectores de fasttext')


# In[ ]:


dataE = dataE.assign(fft=embeddingsEF)


# ### Modelo InferSent

# In[ ]:

from InferSent.models import InferSent
import torch
import torchvision

def InferSent_model(model_version=1):
    MODEL_PATH = 'infersent1.pkl'
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
    # If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
    if model_version == 1:
        W2V_PATH = 'glove.840B.300d.txt'  
    else: 
        W2V_PATH = 'crawl-300d-2M.vec'
    print("loading model ...  {}".format(W2V_PATH))
    model = InferSent(params_model)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.set_w2v_path(W2V_PATH)

    model.build_vocab_k_words(K=3000000)
    return model


# In[ ]:


import nltk
nltk.download('punkt')
model = InferSent_model(model_version=1)
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
print('Se cargo el modelo de infersent')


# In[ ]:


embeddingsEI = [ model.encode(sent_detector.tokenize(v), bsize=128, tokenize=False, verbose=True) for v  in list(dataE['Textos'])]
print('Se crearon los vectores de infersent')


# In[95]:


dataE = dataE.assign(infer = [i[0] for i in embeddingsEI] )


# ### Particion 70/30

# In[98]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataE, dataE['Klases'], test_size=0.30)#, random_state=42, stratify=dataE['Klases'])
print('Particion realizada')

print(dataE.groupby(['Klases']).count())
print(X_train.groupby(['Klases']).count())
print(X_test.groupby(['Klases']).count())


# ### Vectores fasttext e infersent

# In[ ]:


y_train = list(y_train)
y_test = list(y_test)

XF_train = list(X_train['fft'])
XF_test = list(X_test['fft'])

XI_train = list(X_train['infer'])
XI_test = list(X_test['infer'])


# ### SVM - fasttext/infersent

# In[ ]:


# Support Vector Machine (SVM)
from sklearn.svm import LinearSVC
#entrenamiento
svcF = LinearSVC(random_state=0)
svcI = LinearSVC(random_state=0)
svcF.fit(XF_train, y_train)
svcI.fit(XI_train, y_train)
#prueba
svcF_ = svcF.predict(XF_test)
svcI_ = svcI.predict(XI_test)
print('SVM de fasttext e infersent realizado')


# ### K-Vecinos - fasttext/infersent

# In[ ]:


# K-Vecinos Cercanos
from sklearn.neighbors import KNeighborsClassifier
#entrenamiento
knnF = KNeighborsClassifier(n_neighbors=5)
knnI = KNeighborsClassifier(n_neighbors=5)
knnF.fit(XF_train, y_train)
knnI.fit(XI_train, y_train)
#prueba
knnF_ = knnF.predict(XF_test)
knnI_ = knnI.predict(XI_test)
print('KNN de fasttext e infersent realizado')


# ### TfIdf

# In[54]:

XTM_train = list(X_train['Textos'])
XTM_test = list(X_test['Textos'])

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

#TfidfVectorizer - entrenamiento
vec = TfidfVectorizer()
X = vec.fit_transform(XTM_train)
mnb = MultinomialNB()
mnb.fit(X, y_train)

#Tfidf - prueba
XTFIDF = vec.transform(XTM_test)
mnb_ = mnb.predict(XTFIDF)
print('TfIdf realizado')


# ### microTC - TextModel

# In[66]:


#TextModel - entrenamiento
from microtc.textmodel import TextModel
from sklearn.svm import LinearSVC

textmodel = TextModel().fit(XTM_train)
lsvc = LinearSVC().fit(textmodel.transform(XTM_train), y_train)

#TextModel - prueba
lsvc_ = lsvc.predict(textmodel.transform(XTM_test))
print('microTC realizado')


# ### Metricas

# In[75]:


from sklearn.metrics import  f1_score, precision_score, recall_score, accuracy_score

def metricas(modelo,y_test,predict):
    print(modelo)
    print('Precision:',round(precision_score(y_test, predict, average=None)[0],3),'   Macro:',round(precision_score(y_test, predict, average='macro'),3),'   Micro:',round(precision_score(y_test, predict, average='micro'),3))
    print('Recall:   ',round(recall_score(y_test, predict, average=None)[0],3),'   Macro:',round(recall_score(y_test, predict, average='macro'),3),'   Micro:',round(recall_score(y_test, predict, average='micro'),3))
    print('F1:       ',round(f1_score(y_test, predict, average=None)[0],3),'   Macro:',round(f1_score(y_test, predict, average='macro'),3),'    Micro:',round(f1_score(y_test, predict, average='micro'),3))
    print('Accuracy: ',round(accuracy_score(y_test, predict),3))


# In[78]:


print(metricas('\ntfidf',y_test,mnb_))
print(metricas('\nmicroTC',y_test,lsvc_))
print(metricas('\nSVC_FastText',y_test,svcF_))
print(metricas('\nSVC_InferSent',y_test,svcI_))
print(metricas('\nKNN_FastText',y_test,knnF_))
print(metricas('\nKNN_InferSent',y_test,knnI_))


# ### Confusion Matrix

# In[101]:


from sklearn.metrics import confusion_matrix

print('\ntfidf\n',confusion_matrix(y_test, mnb_))
print('\nmicroTC\n',confusion_matrix(y_test, lsvc_))
print('\nSVC_FastText\n',confusion_matrix(y_test, svcF_))
print('\nSVC_InferSent\n',confusion_matrix(y_test, svcI_))
print('\nKNN_FastText\n',confusion_matrix(y_test, knnF_))
print('\nKNN_InferSent\n',confusion_matrix(y_test, knnI_))


# In[ ]:




