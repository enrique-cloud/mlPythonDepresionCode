# 1
# Datos
import tarfile
tar = tarfile.open("usuarios_depresion.json.tar.gz")
tar.extractall()
tar.close()

from microtc.utils import tweet_iterator
data = [i for i in tweet_iterator('usuarios_depresion.json')]

# Extraer los tweets
data[0]['tweets'][4099]['text']

# Extraer lo que se requiere, tres listas de dimension 317 en donde todos los tweets de cada usuario se juntaron para hacer solo un tweet unico por usuario.
klases,users,t,tweets = [],[],[],[]
for i in data:
    t = []
    klases.append(int(i['klass']))
    users.append(i['user'])
    for j in i['tweets']:
        t.append(j['text'])
    tweets.append(' '.join(t))

print('Data Original')
print(len(klases))
print(len(tweets))
print(len(users))


# Limpieza
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
        if (r.startswith('http') or r.startswith('rt') or r.startswith('"http')):
            continue
        aux.append(r)
    l2.append(' '.join(aux)) 
for i in l2:
    l3.append(RP(i))      
for i in l3:
    l4.append(RD(i))
for i in l4:
    aux = i.split()
    aux2 = []
    for j in aux:
        aux2.append(reducirString(j))
    l5.append(' '.join(aux2))

nltk.download('stopwords')
from nltk.corpus import stopwords
lsw = stopwords.words('spanish')

TweetsLimpios = []

for i in l5:
    aux = []
    h = i.split()
    for r in h:
        if r not in lsw:
            aux.append(r)
    TweetsLimpios.append(' '.join(aux))

print('Data Trabajada')
print(len(users))
print(len(klases))
print(len(TweetsLimpios))

import numpy as np
print('Limpieza realizada')
print(np.unique(klases, return_counts=True))

import pandas as pd
dataE = pd.DataFrame({'Textos':TweetsLimpios, 'Klases':klases,'Usuarios':users})

# Particion 70/30
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataE, dataE['Klases'], test_size=0.30)
print('Particion realizada')

print(dataE.groupby(['Klases']).count())
print(X_train.groupby(['Klases']).count())
print(X_test.groupby(['Klases']).count())

y_train = list(y_train)
y_test = list(y_test)

# TfIdf
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


# microTC - TextModel
# TextModel - entrenamiento
from microtc.textmodel import TextModel
from sklearn.svm import LinearSVC

textmodel = TextModel(docs=None, text='text',num_option='delete',usr_option='delete',
    url_option='delete',emo_option='group',hashtag_option='delete',
    ent_option='none',lc=True,del_dup=True,del_punc=True,del_diac=True,
    token_list=[[2, 1], -1, 3, 4],token_min_filter=0,token_max_filter=1,
    select_ent=False,select_suff=False, select_conn=False,
    weighting='tfidf').fit(XTM_train)

lsvc = LinearSVC().fit(textmodel.transform(XTM_train), y_train)

#TextModel - prueba
lsvc_ = lsvc.predict(textmodel.transform(XTM_test))
print('microTC realizado')


# Metricas
from sklearn.metrics import  f1_score, precision_score, recall_score, accuracy_score

def metricas(modelo,y_test,predict):
    print(modelo)
    print('Precision:',round(precision_score(y_test, predict, average=None)[0],3),
    'Macro:',round(precision_score(y_test, predict, average='macro'),3),
    'Micro:',round(precision_score(y_test, predict, average='micro'),3))
    print('Recall:   ',round(recall_score(y_test, predict, average=None)[0],3),
    'Macro:',round(recall_score(y_test, predict, average='macro'),3),
    'Micro:',round(recall_score(y_test, predict, average='micro'),3))
    print('F1:       ',round(f1_score(y_test, predict, average=None)[0],3),
    'Macro:',round(f1_score(y_test, predict, average='macro'),3),
    'Micro:',round(f1_score(y_test, predict, average='micro'),3))
    print('Accuracy: ',round(accuracy_score(y_test, predict),3))

print(metricas('\ntfidf',y_test,mnb_))
print(metricas('\nmicroTC',y_test,lsvc_))

# Confusion Matrix
from sklearn.metrics import confusion_matrix

print('\ntfidf\n',confusion_matrix(y_test, mnb_))
print('\nmicroTC\n',confusion_matrix(y_test, lsvc_))





