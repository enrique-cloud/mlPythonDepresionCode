
# 3
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



import tarfile
tar = tarfile.open("usuarios_depresion.json.tar.gz")
tar.extractall()
tar.close()

from microtc.utils import tweet_iterator
data = [i for i in tweet_iterator('usuarios_depresion.json')]

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

import numpy as np
print(np.unique(klases, return_counts=True))

# Tabla en pandas con la data
import pandas as pd
dataE = pd.DataFrame({'Textos':tweets, 'Klases':klases,'Usuarios':users})

# Particion 70/30
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataE, dataE['Klases'], test_size=0.30)
print('Particion realizada')

print(dataE.groupby(['Klases']).count())
print(X_train.groupby(['Klases']).count())
print(X_test.groupby(['Klases']).count())

y_train = list(y_train)
y_test = list(y_test)
XM_train = list(X_train['Textos'])
XM_test = list(X_test['Textos'])

# microTC - TextModel
# Params editados
# TextModel - entrenamiento
from microtc.textmodel import TextModel
from sklearn.svm import LinearSVC

textmodel = TextModel(docs=None, text='text',num_option='delete',usr_option='delete',
    url_option='delete',emo_option='group',hashtag_option='delete',
    ent_option='none',lc=True,del_dup=True,del_punc=True,del_diac=True,
    token_list=[[2, 1], -1, 3, 4],token_min_filter=0,token_max_filter=1,
    select_ent=False,select_suff=False, select_conn=False,
    weighting='tfidf').fit(XM_train)

lsvc = LinearSVC().fit(textmodel.transform(XM_train), y_train)

# TextModel - prueba
lsvc_ = lsvc.predict(textmodel.transform(XM_test))
print('microTC params realizado')

# Params default
# TextModel - entrenamiento
textmodelD = TextModel().fit(XM_train)

lsvcD = LinearSVC().fit(textmodelD.transform(XM_train), y_train)

# TextModel - prueba
lsvcD_ = lsvcD.predict(textmodelD.transform(XM_test))
print('microTC default realizado')


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

print(metricas('\nmicroTC_params',y_test,lsvc_))
print(metricas('\nmicroTC_default',y_test,lsvcD_))

# Confusion Matrix
from sklearn.metrics import confusion_matrix

print('\nmicroTC_params\n',confusion_matrix(y_test, lsvc_))
print('\nmicroTC_default\n',confusion_matrix(y_test, lsvcD_))





