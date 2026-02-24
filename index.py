import tensorflow as tf
print("¡Listo! TensorFlow versión:", tf.__version__)
print("Iniciando el nuevo proyecto de IA para Julio...")
# Aquí irá el código de tu modelo más adelante
from tensorflow.keras import layers, models
import os

# 1. Configuración de dimensiones (como las piezas de tus equipos)
# Usaremos 180x180 para que la IA vea bien los detalles de las tarjetas
img_height = 180
img_width = 180
batch_size = 32
data_dir = "data" # Aquí es donde pondrás tus carpetas de fotos

print("--- Iniciando el sistema de reconocimiento de repuestos para Julio ---")

# 2. Carga y preparación de las imágenes
# Este comando separa automáticamente tus fotos en entrenamiento y prueba
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Optimizamos la lectura de datos para que GitHub no pierda tiempo
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 3. Aumento de Datos (Data Augmentation)
# Esto ayuda a que la IA aprenda aunque tengas pocas fotos de cada pieza
data_augmentation = models.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])

# 4. Estructura de la Red Neuronal (CNN)
model = models.Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(4) # Las 4 categorías de tu taller
])

# 5. Ajustando las "tuercas" del modelo (Compilación)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 6. Entrenamiento (Las 'épocas' o vueltas de aprendizaje)
epochs = 10 
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# 7. Guardar el resultado final para tu Web-App
model.save('modelo_repuestos.h5')
print("--- Entrenamiento finalizado. Modelo guardado como 'modelo_repuestos.h5' ---")