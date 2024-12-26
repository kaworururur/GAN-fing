import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

# Функция для загрузки изображений отпечатков пальцев
def load_data(path):
    # Пример загрузки данных с помощью ImageDataGenerator
    IMG_SIZE = 64  # Размер изображений
    batch_size = 32
    
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
    dataset = datagen.flow_from_directory(
        path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode=None)  # Не нужно определять классы
    
    return dataset

# Загрузка данных
fingerprint_data = load_data('fingerprints/') #Введите путь к папке с обучающим датасетом

# Генератор
def build_generator(latent_dim):
    model = tf.keras.Sequential()
    model.add(layers.Dense(128 * 16 * 16, activation="relu", input_dim=latent_dim))
    model.add(layers.Reshape((16, 16, 128)))
    model.add(layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding="same", activation="relu"))
    model.add(layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="same", activation="relu"))
    model.add(layers.Conv2D(1, kernel_size=3, padding="same", activation="sigmoid"))

    return model

# Дискриминатор
def build_discriminator(img_shape):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, kernel_size=3, strides=2, padding="same", input_shape=img_shape))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))

    return model

# Определение размеров
img_shape = (64, 64, 1)
latent_dim = 100

# Создание моделей
generator = build_generator(latent_dim)
discriminator = build_discriminator(img_shape)

# Компиляция дискриминатора
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

discriminator.trainable = False  # Замораживаем дискриминатор при обучении GAN

# Создание GAN
gan_input = layers.Input(shape=(latent_dim,))
x = generator(gan_input)
gan_output = discriminator(x)

gan = tf.keras.Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer='adam')

discriminator.trainable = False  # Замораживаем дискриминатор при обучении GAN

# Создание GAN
gan_input = layers.Input(shape=(latent_dim,))
x = generator(gan_input)
gan_output = discriminator(x)

gan = tf.keras.Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer='adam')

def train_gan(epochs, batch_size):
    for epoch in range(epochs):
        # 1. Тренируем дискриминатор
        idx = np.random.randint(0, fingerprint_data.samples, batch_size)
        real_images = fingerprint_data[idx]
        
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_images = generator.predict(noise)
        
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        
        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # 2. Тренируем генератор
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_labels = np.ones((batch_size, 1))  # Мы хотим, чтобы генератор создавал фейковые изображения, которые проверяет как реальные
        
        g_loss = gan.train_on_batch(noise, valid_labels)
        
        # Вывод результатов каждую эпоху
        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc.: {100 * d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

# Обучение GAN
train_gan(epochs=10000, batch_size=32)

def generate_fingerprints(n_images):
    noise = np.random.normal(0, 1, (n_images, latent_dim))
    generated_images = generator.predict(noise)
    
    # Визуализация изображений
    plt.figure(figsize=(10, 10))
    for i in range(n_images):
        plt.subplot(1, n_images, i + 1)
        plt.imshow(generated_images[i].reshape(64, 64), cmap='gray')
        plt.axis('off')
    plt.show()

# Генерация изображений
generate_fingerprints(5)
