import matplotlib.pyplot as plt
import matplotlib.image as mpimg

BASE_DIR= '/home/lanar/.cache/kagglehub/datasets/ashery/chexpert/versions/1/'
BATCH_SIZE = 16
IMG_SIZE = (200, 200)  # Tamaño de la imagen
LABEL_COLUMS = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Pleural Effusion'
]

# Comprobar que se accede correctamente al los datos
# Ruta de la imagen
path = BASE_DIR + 'train/patient00014/study1/view1_frontal.jpg'
 #Mostrar imagen
img = mpimg.imread(path)
plt.imshow(img)
plt.axis('off')
plt.show()