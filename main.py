import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf
import numpy as np
from keras import layers
import pandas as pd
from tqdm import tqdm
import spacy
import re

splits = {'train': 'train.csv', 'validation': 'validation.csv'}
trainning_df = pd.read_csv("hf://datasets/Ernesto-1997/Sarcastic_spanish_dataset/" + splits["train"])
validation_df = pd.read_csv("hf://datasets/Ernesto-1997/Sarcastic_spanish_dataset/" + splits["validation"])

print('Número de Comentarios de entrenamiento Cargados: {num}'.format(num=trainning_df.shape[0]))
print('Número de Comentarios de validación Cargados: {num}'.format(num=validation_df.shape[0]))

trainning_comentarios = [list(x) for x in trainning_df[['Texto', 'Sarcasmo']].values]
validation_comentarios = [list(x) for x in validation_df[['Texto', 'Sarcasmo']].values]

nlp = spacy.load('es_core_news_md')

def normalize(comentarios, min_word=5):

    comentarios_list = []
    for comentario in tqdm(comentarios):

        # Tokenizamos el comentario
        comment = nlp(comentario[0].replace('.', ' ').replace('-', '').replace('?', ' ').replace('!', ' ').replace(',', ' ').replace('¿', ' ').replace('¡', ' ').strip())

        comment = ([word.lemma_ for word in comment if (not word.is_punct) and (not ':' in word.text)])

        if (len(comment) > min_word):
            comentarios_list.append([' '.join(comment), comentario[1]])
    return comentarios_list

X_train_norm = normalize(trainning_comentarios)
X_test_norm = normalize(validation_comentarios)

X_train = [doc[0] for doc in X_train_norm]
y_train = np.array([doc[1] for doc in X_train_norm])
X_test = [doc[0] for doc in X_test_norm]
y_test = np.array([doc[1] for doc in X_test_norm])

print('comentarios de entrenamiento: {}'.format(len(X_train)))
print('comentarios de Test: {}'.format(len(X_test)))

from sklearn.preprocessing import LabelEncoder
from keras._tf_keras.keras.preprocessing.text import one_hot

# Hacemos un one-hot encoder del texto
VOCAB_SIZE = 100000
X_train = [one_hot(doc, VOCAB_SIZE) for doc in X_train]
X_test = [one_hot(doc, VOCAB_SIZE) for doc in X_test]

encoder = LabelEncoder()
encoder.fit(y_train)

class_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
print("Mapeo de clases:", class_mapping)

y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)

MAX_WORDS = 50
pad_corpus_train = keras.utils.pad_sequences(X_train, maxlen=MAX_WORDS, padding='post')
pad_corpus_test = keras.utils.pad_sequences(X_test, maxlen=MAX_WORDS, padding='post')

from keras.api.layers import Dense, Dropout, LSTM, Embedding
from keras import Sequential

EMBEDDING_SIZE = 32

model = Sequential()
model.add(Embedding(VOCAB_SIZE, EMBEDDING_SIZE))
model.add(LSTM(EMBEDDING_SIZE))
model.add(Dropout(0.05))
model.add(Dense(2, activation='softmax'))
# Compilación del modelo
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenamiento del modelo
history = model.fit(pad_corpus_train, y_train, epochs=50, batch_size=32, validation_data=(pad_corpus_test, y_test))

# Evaluación del modelo
loss, accuracy = model.evaluate(pad_corpus_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Imprimimos la arquitectura de la red
model.summary()

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import matplotlib.pyplot as plt

# Pintamos las métricas por epoch
def plot_metric(history, name, remove_first=0):
    metric_train = np.array(history.history[name])[remove_first:]
    metric_test = np.array(history.history['val_{}'.format(name)])[remove_first:]
    acum_avg_metric_train = (np.cumsum(metric_train) / (np.arange(metric_train.shape[-1]) + 1))[remove_first:]
    acum_avg_metric_test = (np.cumsum(metric_test) / (np.arange(metric_test.shape[-1]) + 1))[remove_first:]
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title('{} - Epochs'.format(name))
    plt.plot(metric_train, label='{} Train'.format(name))
    plt.plot(metric_test, label='{} Test'.format(name))
    plt.grid()
    plt.legend(loc='upper center')
    plt.subplot(1, 2, 2)
    plt.title('AVG ACCUMULATIVE {} - Epochs'.format(name))
    plt.plot(acum_avg_metric_train, label='{} Train'.format(name))
    plt.plot(acum_avg_metric_test, label='{} Test'.format(name))
    plt.grid()
    plt.legend(loc='upper center')
    plt.show()

# Función de perdida
plot_metric(history=history, name='loss')

# Accuracy
plot_metric(history=history, name='accuracy')

model.save('ruta_al_modelo/modelo_entrenado.h5')