import tensorflow as tf
from Config import IMG_SIZE


class PCAMPooling(tf.keras.layers.Layer):
    def __init__(self, filters=1):
        super(PCAMPooling, self).__init__()
        # Convolución 1x1 para generar mapa de atención
        self.attention_conv = tf.keras.layers.Conv2D(filters, kernel_size=1, activation='sigmoid')

    def call(self, inputs):
        attention = self.attention_conv(inputs)  
        
        weighted = inputs * attention  
        sum_attention = tf.reduce_sum(attention, axis=[1, 2], keepdims=True) + 1e-8
        pooled = tf.reduce_sum(weighted, axis=[1, 2], keepdims=True) / sum_attention

        return tf.squeeze(pooled, axis=[1, 2]) 

def crear_modelo_R():

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape= (*IMG_SIZE,3)),
        # Convolución usando un filtro 3X3
        tf.keras.layers.Conv2D(
            256, (5,5), activation='relu', padding='same', strides=(2,2), use_bias=False
        ),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(
            128, (3,3), activation='relu', padding='same', strides=(1,1), use_bias=False
        ),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(
            64, (1,1), activation='relu', padding='same', strides=(1,1), use_bias=False
        ),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides=(2,2), padding='same'),

        tf.keras.layers.GlobalAveragePooling2D(),   #Aplanando la salida

        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.BatchNormalization(),     
        
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.BatchNormalization(),
        
        # Capa de salida con con función de actvación sigmoid
        tf.keras.layers.Dense(5, activation='sigmoid'),
    ])
    #model.summary()
    return model

def crear_modelo_J():
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(*IMG_SIZE,3)),

        #1 Capa de convolución
        tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same', use_bias=False,kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        

        #2 capa de convolución
        tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', use_bias=False,kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        

        #3 capa de convolución
        tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', use_bias=False,kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same', use_bias=False,kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),

        tf.keras.layers.GlobalAveragePooling2D(),   #Aplanando la salida

        #Capa oculta de 64 nodos
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.1),

        #Capa de salida multietiqueta: 5 nodos
        tf.keras.layers.Dense(5, activation='sigmoid')
    ])
    return model

def re_loss_P1(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    num_clases = tf.shape(y_pred)[-1]
    # Ajusta los pesos a la cantidad de clases
    pesos = tf.constant([1.5, 1.8, 2.3, 1.0, 0.8], dtype=tf.float32)
    pesos = pesos[:num_clases]  # Por si acaso

    mascara = tf.not_equal(y_true, -1)
    mascara = tf.cast(mascara, dtype=tf.float32)
    y_true_clean = tf.where(mascara == 1, y_true, tf.zeros_like(y_true))
    bce = tf.keras.backend.binary_crossentropy(y_true_clean, y_pred)
    class_weights = tf.reshape(pesos, (1, -1))
    bce_weigthed = bce * class_weights
    bce_masked = bce_weigthed * mascara
    return tf.reduce_sum(bce_masked) / tf.reduce_sum(mascara)

def re_loss_P(y_true, y_pred):
    pesos = tf.constant([1.5, 1.8, 2.3, 1.0, 0.8], dtype=tf.float32)
    #Se crea una lista de mascara que indica si hay un -1 o no:
    mascara = tf.not_equal(y_true, -1)   #Por ejemplo para:  y_true = [1, 0, -1, 1, 0]  -> mascara = [ True,  True, False,  True,  True ]

    #Convertir la mascara a un float entre 0 y 1, para operar con multiplicacion
    mascara = tf.cast(mascara, dtype=tf.float32)    #ej:  -> [1.0, 1.0, 0.0, 1.0, 1.0]

    #Reemplaza los -1 en y_true por ceros, sin alterar el resto, esto para evitar calculos de -1
    y_true_clean = tf.where(mascara == 1, y_true, tf.zeros_like(y_true))

    #Uso binary crossentropy entre los valores "limpios" y las predicciones, provisoriamente antes de borrar los falsos 0
    bce = tf.keras.backend.binary_crossentropy(y_true_clean, y_pred)

    class_weights = tf.constant(pesos, dtype=tf.float32)
    bce_weigthed = bce * tf.reshape(class_weights, (1, -1))  # Asegura que los pesos se apliquen correctamente a cada clase
    bce_masked = bce_weigthed * mascara   # [x, x, 0, x]

    return tf.reduce_sum(bce_masked) / tf.reduce_sum(mascara) #Devuelve el cálculo habitual del error promedio, pero sin los -1


def re_loss_(y_true, y_pred):
    #Se crea una lista de mascara que indica si hay un -1 o no:
    mascara = tf.not_equal(y_true, -1)   #Por ejemplo para:  y_true = [1, 0, -1, 1, 0]  -> mascara = [ True,  True, False,  True,  True ]

    #Convertir la mascara a un float entre 0 y 1, para operar con multiplicacion
    mascara = tf.cast(mascara, dtype=tf.float32)    #ej:  -> [1.0, 1.0, 0.0, 1.0, 1.0]

    #Reemplaza los -1 en y_true por ceros, sin alterar el resto, esto para evitar calculos de -1
    y_true_clean = tf.where(mascara == 1, y_true, tf.zeros_like(y_true))

    #Uso binary crossentropy entre los valores "limpios" y las predicciones, provisoriamente antes de borrar los falsos 0
    bce = tf.keras.backend.binary_crossentropy(y_true_clean, y_pred)

    #Aplico la máscara para ignorar el error en las posiciones donde y_true sea -1.
    masked_bce = bce * mascara   # [x, x, 0, x]
    return tf.reduce_sum(masked_bce) / tf.reduce_sum(mascara) #Devuelve el cálculo habitual del error promedio, pero sin los -1

   
