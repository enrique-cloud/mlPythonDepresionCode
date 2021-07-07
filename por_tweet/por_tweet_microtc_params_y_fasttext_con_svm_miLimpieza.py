# 3
import fasttext
from InferSent.models import InferSent
import nltk
nltk.download('punkt')
from sklearn.model_selection import train_test_split
import numpy as np
import math
import tarfile
from microtc.utils import tweet_iterator

tar = tarfile.open("usuarios_depresion.json.tar.gz")
tar.extractall()
tar.close()
data = [i for i in tweet_iterator('usuarios_depresion.json')]

ktu,klases,tweets,users = [],[],[],[]
for i in data:
    for j in i['tweets']:
        ktu.append([j['text'],int(i['klass']),i['user']])

from random import sample

for i in ktu:
    tweets.append(i[0])
    klases.append(i[1])
    users.append(i[2])

import re as reg
import unicodedata
import nltk
nltk.download('stopwords')
import re, string
from nltk.corpus import stopwords
lsw = stopwords.words('spanish')

text2 = tweets.copy()

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

susp2,susp3,susp5,TextoL,TextoLimpio = [],[],[],[],[]
QR,stop1,stem1,Tt,KC,Us = [],[],[],[],[],[]

for i in text2:
    susp2.append(QuitarAcentos(i).lower())    
for i in susp2:
    susp5.append(RP(i))      
for i in susp5:
    TextoL.append(RD(i))
for i in TextoL:
    aux = i.split()
    aux2 = []
    for j in aux:
        aux2.append(reducirString(j))
    QR.append(' '.join(aux2))

for i in QR:
    aux = []
    h = i.split()
    for r in h:
        if r not in lsw: #and r not in lsw2):
            aux.append(r)
    stop1.append(' '.join(aux))
    
for i in range(len(stop1)):
    if stop1[i] == '':
        continue
    else:
        Tt.append(stop1[i])
        KC.append(klases[i])
        Us.append(users[i])

print('Limpieza realizada')

import pandas as pd
dataE = pd.DataFrame({'Textos':Tt, 'Klases':KC,'Usuarios':Us})

# FastText
ft = fasttext.load_model('crawl-300d-2M-subword.bin')
print('Se cargo el modelo de fasttext')

embeddingsEF = [ ft.get_sentence_vector(v) for v  in list(dataE['Textos'])]
print('Se crearon los vectores de fasttext')

dataE = dataE.assign(fft=embeddingsEF)

# Train Test
XE = dataE
yE = dataE['Klases']
XE_data_train, XE_data_test, yE_train, yE_test = train_test_split(XE, yE, test_size=0.3)
print('se creo el train y el test')

print(XE.groupby(['Klases']).count())
print(XE_data_train.groupby(['Klases']).count())
print(XE_data_test.groupby(['Klases']).count())

yE_train = list(yE_train)
yE_test = list(yE_test)

XEF_train = list(XE_data_train['fft'])
XEF_test = list(XE_data_test['fft'])

# Support Vector Machine (SVM)
from sklearn.svm import LinearSVC
svcEF = LinearSVC(random_state=0)
svcEF.fit(XEF_train, yE_train)

y_predEF0 = svcEF.predict(XEF_test)
print('Termino SVM')

# microTC - micro Text Classification
X_trainEM = list(XE_data_train['Textos'])
X_testEM = list(XE_data_test['Textos'])

from microtc.textmodel import TextModel
textmodelEM = TextModel(docs=None, text='text',num_option='delete',usr_option='delete',
    url_option='delete',emo_option='group',hashtag_option='delete',
    ent_option='none',lc=True,del_dup=True,del_punc=True,del_diac=True,
    token_list=[[2, 1], -1, 3, 4],token_min_filter=0,token_max_filter=1,
    select_ent=False,select_suff=False, select_conn=False,
    weighting='tfidf')
textmodelEM.fit(list(X_trainEM))

from sklearn.svm import LinearSVC
svcEM = LinearSVC()
svcEM.fit(textmodelEM.transform(X_trainEM), yE_train)

y_predEM = svcEM.predict(textmodelEM.transform(X_testEM))
print('Termino microTC')

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

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

print(metricas('\nFastText_SVM',yE_test,y_predEF0))
print(metricas('\nmicroTC',yE_test,y_predEM))

from sklearn.metrics import confusion_matrix

print('\nFastText_SVM\n',confusion_matrix(yE_test, y_predEF0))
print('\nmicroTC\n',confusion_matrix(yE_test, y_predEM))

