import tensorflow as tf
import numpy as np
import re
import tqdm
import spacy
import keras
from keras._tf_keras.keras.preprocessing.text import one_hot

nlp = spacy.load('es_core_news_md')

def normalize(comentarios, min_word=5):
    comentarios_list = []
    for comentario in tqdm.tqdm(comentarios):
        # Tokenizamos el comentario
        comment = nlp(comentario[0].replace('.', ' ').replace('-', '').replace('?', ' ').replace('!', ' ').replace(',', ' ').replace('¿', ' ').replace('¡', ' ').strip())
        comment = ([word.lemma_ for word in comment if (not word.is_punct) and (not ':' in word.text)])
        if (len(comment) > min_word):
            comentarios_list.append([' '.join(comment), comentario[1]])
    return comentarios_list

# Cargar el modelo entrenado
model = tf.keras.models.load_model('model.h5')

# Solicitar los inputs al usuario
input_text = input("Introduce el texto a procesar: ")
inputs = [[input_text, None]]  # Añadimos un valor dummy para el segundo elemento de la tupla

# Normalizar el texto de entrada
normalized_inputs = normalize(inputs)

# Extraer el texto normalizado
normalized_text = [doc[0] for doc in normalized_inputs]

# Convertir el texto a una secuencia one-hot
VOCAB_SIZE = 100000  # Poner un valor muy alto
one_hot_sequences = [one_hot(doc, VOCAB_SIZE) for doc in normalized_text]

# Aplicar padding a las secuencias para que tengan la misma longitud que las usadas en el entrenamiento
MAX_WORDS = 50
padded_sequences = keras.utils.pad_sequences(one_hot_sequences, maxlen=MAX_WORDS, padding='post')

# Realizar la predicción
predicciones = model.predict(padded_sequences)

# Aplicar softmax si es necesario (para obtener probabilidades)
# predicciones_probabilidades = tf.nn.softmax(predicciones).numpy()

print(predicciones)

if (predicciones[0][0] > 0.50):
    print("El texto no se considera como un sarcástico")
else:
    print("El texto se considera como un sarcástico")
