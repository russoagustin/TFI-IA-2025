import pandas as pd
import os
from Config import BASE_DIR, LABEL_COLUMS, BATCH_SIZE, IMG_SIZE
import tensorflow as tf



def crear_dataFrame_UIgnore(csv_rpath,fracc):
    df = pd.read_csv(BASE_DIR + csv_rpath) # Creo el Data Frame
    # Corrección de rutas del Data Frame
    prefix = 'CheXpert-v1.0-small/'
    # Quito el prefijo
    df['Path'] = df['Path'].apply(lambda x: x[len(prefix):] if x.startswith(prefix) else x)
    # concateno la otra parte para obtener ruta absoluta
    df['Path'] = df['Path'].apply(lambda x: BASE_DIR + x)
    # Aquí ya tengo corregidos todos los Path a las imágenes
    print(df.head()['Path'][0])
    # Enfoque U-zeroes
    df[LABEL_COLUMS] = df[LABEL_COLUMS].fillna(-1).astype('float32')
    
    df = df.sample(frac=fracc, random_state=42)
    return df



# Crear DataFrame
def crear_dataFrame_UZeroes(csv_rpath,fracc):
    df = pd.read_csv(BASE_DIR + csv_rpath) # Creo el Data Frame
    # Corrección de rutas del Data Frame
    prefix = 'CheXpert-v1.0-small/'
    # Quito el prefijo
    df['Path'] = df['Path'].apply(lambda x: x[len(prefix):] if x.startswith(prefix) else x)
    # concateno la otra parte para obtener ruta absoluta
    df['Path'] = df['Path'].apply(lambda x: BASE_DIR + x)
    # Aquí ya tengo corregidos todos los Path a las imágenes
    print(df.head()['Path'][0])
    # Enfoque U-zeroes
    df[LABEL_COLUMS] = df[LABEL_COLUMS].fillna(0).replace(-1,0).astype('float32')
    
    df = df.sample(frac=fracc, random_state=42)
    return df

# Creación del dataset
def obtener_dataset(df):

    image_paths = df['Path'].tolist()
    labels = df[LABEL_COLUMS].values
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def obtener_dataset_Aug(df):
    image_paths = df['Path'].tolist()
    labels = df[LABEL_COLUMS].values
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_and_preprocess_image_Aug, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset
 


def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3) # decodifica la imagen en un tensor de imagen (matriz de píxeles)
    image = tf.image.resize(image,[IMG_SIZE[0],IMG_SIZE[1]])        # Las redes convolucionales requieren imagenes de tamaño uniforme (por las dudas)
    image = image / 255.0                           # Normaliza los valores de [0, 255] a [0, 1]
    return image, label                             # Devuelve una tupla que tensorFlow usa como entrada y salida para el modelo


def load_and_preprocess_image_Aug(path, label, training=True):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)

    # Aumentación solo si es para entrenamiento
    if training:
        image = tf.image.random_flip_left_right(image)     # Flip horizontal
        image = tf.image.random_brightness(image, max_delta=0.1)  # Ligero cambio de brillo
        image = tf.image.random_contrast(image, 0.9, 1.1)   # Cambio de contraste
        image = tf.image.random_saturation(image, 0.9, 1.1) # Saturación (leve)
        image = tf.image.resize_with_crop_or_pad(image, 256, 256)  # Padding
        image = tf.image.random_crop(image, [IMG_SIZE[0], IMG_SIZE[1], 3])  # Recorte aleatorio centrado

    else:
        image = tf.image.resize(image, [IMG_SIZE[0], IMG_SIZE[1]])

    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def balancear_dataset(df, label_cols):

    dfs = [df]  # Incluye el original
    max_positivos = max([(df[col] == 1).sum() for col in label_cols])

    for col in label_cols:
        df_pos = df[df[col] == 1]
        n_pos = len(df_pos)
        if n_pos == 0:
            continue
        n_to_add = int(max_positivos - n_pos)
        if n_to_add > 0:
            df_extra = df_pos.sample(n=n_to_add, replace=True, random_state=42)
            dfs.append(df_extra)

    df_bal = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=42)
    return df_bal


def balancear_dataset_por_clase(df, label_cols):

    max_positivos = max([(df[col]==1).sum() for col in label_cols])
    dfs = []

    for col in label_cols:
        df_pos = df[df[col] == 1]
        n_pos = len(df_pos)

        if n_pos == 0:
            continue

        n_to_add = int(max_positivos - n_pos)
        if n_to_add > 0:
            df_extra = df_pos.sample(n=n_to_add, replace=True, random_state=42)
            df_bal_col = pd.concat([df_pos, df_extra], ignore_index=True)
        else:
            df_bal_col = df_pos

        for other_col in label_cols:
            if other_col != col:
                df_bal_col[other_col] = 0

        dfs.append(df_bal_col)

    df_bal = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=42)
    return df_bal