# 1
from sklearn.model_selection import train_test_split
import numpy as np
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

for i in ktu:
    tweets.append(i[0])
    klases.append(i[1])
    users.append(i[2])

import pandas as pd
dataE = pd.DataFrame({'Textos':tweets, 'Klases':klases,'Usuarios':users})

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

XE_train = list(XE_data_train['Textos'])
XE_test = list(XE_data_test['Textos'])


# microTC - micro Text Classification
# Params editados
from microtc.textmodel import TextModel

textmodelEM = TextModel(docs=None, text='text',num_option='delete',usr_option='delete',
    url_option='delete',emo_option='group',hashtag_option='delete',
    ent_option='none',lc=True,del_dup=True,del_punc=True,del_diac=True,
    token_list=[[2, 1], -1, 3, 4],token_min_filter=0,token_max_filter=1,
    select_ent=False,select_suff=False, select_conn=False,
    weighting='tfidf')
textmodelEM.fit(list(XE_train))

from sklearn.svm import LinearSVC
svcEM = LinearSVC()
svcEM.fit(textmodelEM.transform(XE_train), yE_train)

y_predEM = svcEM.predict(textmodelEM.transform(XE_test))
print('Termino microTC params')

# Default
textmodelEMD = TextModel()
textmodelEMD.fit(list(XE_train))

svcEMD = LinearSVC()
svcEMD.fit(textmodelEMD.transform(XE_train), yE_train)

y_predEMD = svcEMD.predict(textmodelEMD.transform(XE_test))
print('Termino microTC default')


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

print(metricas('\nmicroTC_Params',yE_test,y_predEM))
print(metricas('\nmicroTC_Default',yE_test,y_predEMD))


from sklearn.metrics import confusion_matrix

print('\nmicroTC\n',confusion_matrix(yE_test, y_predEM))
print('\nmicroTC_Default\n',confusion_matrix(yE_test, y_predEMD))
