# %%
import keras
import numpy as np
from keras.datasets import fashion_mnist

# %%
train , test = fashion_mnist.load_data()

# %%
train_jamaal = {'images': train[0], 'labels': train[1]}
test_jamaal = {'images': test[0], 'labels': test[1]}

# %% [markdown]
# #Initial Exploration

# %%
print('Train Size')
print(train_jamaal["images"].shape)
print(train_jamaal['labels'].shape)

print('Test Size')
print(test_jamaal["images"].shape)
print(test_jamaal['labels'].shape)

# %%
largest_pixel_value = np.max(train_jamaal["images"])
print('Largest Train Pixel Value', largest_pixel_value)

largest_pixel_value = np.max(test_jamaal["images"])
print('Largest Test Pixel Value', largest_pixel_value)

# %%
train_jamaal["images"] = train_jamaal["images"] / largest_pixel_value
test_jamaal["images"] = test_jamaal["images"] / largest_pixel_value

# %% [markdown]
# #Data Preprocessing

# %%
import tensorflow as tf

train_jamaal['labels'] = tf.keras.utils.to_categorical(train_jamaal['labels'])
test_jamaal['labels'] = tf.keras.utils.to_categorical(test_jamaal['labels'])


# %%
print('Train Labels Size: ', train_jamaal['labels'].shape)
print('Test Labels Size: ', test_jamaal['labels'].shape)

# %% [markdown]
# #Visualization

# %%
import matplotlib.pyplot as plt

def display_image(image, label):
    plt.imshow(image, cmap='gray')
    plt.title(f'True Label: {label}')
    plt.axis('off')
    # plt.show()
    
fig = plt.figure(figsize=(8, 8))

for i in range(12):
    ax = fig.add_subplot(4, 3, i+1)
    display_image(train_jamaal['images'][i], np.argmax(train_jamaal['labels'][i]))
    
plt.tight_layout()
plt.show()

# %% [markdown]
# #Training Data Preparation

# %%
from sklearn.model_selection import train_test_split

x_train_jamaal, x_val_jamaal, y_train_jamaal, y_val_jamaal = train_test_split(train_jamaal['images'], train_jamaal['labels'], test_size=0.3, random_state=42)

print('Training Data Size:', x_train_jamaal.shape)
print('Validation Data Size:', x_val_jamaal.shape)
print('Training Labels Size:', y_train_jamaal.shape)
print('Validation Labels Size:', y_val_jamaal.shape)

# %% [markdown]
# #Build,Train,and Validate CNN Model

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

cnn_model_jamaal = Sequential()

# Add the first convolutional layer
cnn_model_jamaal.add(
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# Add max pooling layer
cnn_model_jamaal.add(MaxPooling2D(pool_size=(2, 2)))

# Add the second convolutional layer
cnn_model_jamaal.add(Conv2D(32, (3, 3), activation='relu'))

# Add max pooling layer
cnn_model_jamaal.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output from the previous layer
cnn_model_jamaal.add(Flatten())

# Add fully connected layer with 100 neurons
cnn_model_jamaal.add(Dense(100, activation='relu'))

# Add output layer
cnn_model_jamaal.add(Dense(10, activation='softmax'))

# Compile the model
cnn_model_jamaal.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

cnn_model_jamaal.summary()

# %%
from tensorflow.keras.utils import plot_model

plot_model(cnn_model_jamaal, to_file='model.png',
           show_shapes=True, show_layer_names=True)

# %%
cnn_history_jamaal = cnn_model_jamaal.fit(x_train_jamaal, y_train_jamaal, validation_data=(
    x_val_jamaal, y_val_jamaal), epochs=10, batch_size=256)

# %% [markdown]
# #Test and Analyze the model

# %%
plt.plot(cnn_history_jamaal.history['accuracy'])
plt.plot(cnn_history_jamaal.history['val_accuracy'])
plt.title('CNN Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# %%
test_loss, test_accuracy = cnn_model_jamaal.evaluate(
    test_jamaal['images'], test_jamaal['labels'])
print('Test Accuracy:', test_accuracy)

# %%
print('Training Accuracy:', cnn_history_jamaal.history['accuracy'][-1])
print('Validation Accuracy:', cnn_history_jamaal.history['val_accuracy'][-1])


# %%
cnn_predictions_jamaal = cnn_model_jamaal.predict(test_jamaal['images'])

# %%
#//create a function that plots the probability distribution of the predictions as a histogram. the function should take the true label of the image and an array with the probability distribution. probability of true labels in gree and predicted labels are in blue 
import numpy as np
import matplotlib.pyplot as plt

def plot_probability_distribution(true_label, probability_distribution):
    # Set the labels for the histogram
    labels = np.arange(len(probability_distribution))
    
    # Set the colors for true and predicted labels
    colors = ['green' if i == true_label else 'blue' for i in labels]
    
    # Plot the histogram
    plt.bar(labels, probability_distribution, color=colors)
    
    # Set the title and labels
    plt.title('Probability Distribution')
    plt.xlabel('Labels')
    plt.ylabel('Probability')
    plt.xticks(labels)
    # Show the plot
    plt.show()


plot_probability_distribution(np.argmax(test_jamaal['labels'][0]), cnn_predictions_jamaal[0])

# %%
start_index = 42
num_images = 4

for i in range(start_index, start_index + num_images):
    image = test_jamaal['images'][i]
    true_label = np.argmax(test_jamaal['labels'][i])
    prediction = cnn_predictions_jamaal[i]
    
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title(f'True Label: {true_label}')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plot_probability_distribution(true_label, prediction)
    
    plt.tight_layout()
    plt.show()


# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Get the predicted labels
predicted_labels = np.argmax(cnn_predictions_jamaal, axis=1)

# Get the true labels
true_labels = np.argmax(test_jamaal['labels'], axis=1)

# Compute the confusion matrix
confusion_mat = confusion_matrix(true_labels, predicted_labels)

# Display the confusion matrix using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mat, annot=True, cmap='Blues')
plt.title('CNN Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# %% [markdown]
# #Build, Train, Validate, Test, and Analyze RNN Model

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

rnn_model_jamaal = Sequential()

# Add LSTM layer with 128 units
rnn_model_jamaal.add(LSTM(128, input_shape=(28, 28)))

# Add output layer
rnn_model_jamaal.add(Dense(10, activation='softmax'))

# Compile the model
rnn_model_jamaal.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

rnn_model_jamaal.summary()

# %%
from tensorflow.keras.utils import plot_model

plot_model(rnn_model_jamaal, to_file='model.png',
           show_shapes=True, show_layer_names=True)

# %%
rnn_history_jamaal = rnn_model_jamaal.fit(x_train_jamaal, y_train_jamaal, validation_data=(
    x_val_jamaal, y_val_jamaal), epochs=10, batch_size=256)


# %%
plt.plot(rnn_history_jamaal.history['accuracy'])
plt.plot(rnn_history_jamaal.history['val_accuracy'])
plt.title('RNN Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# %%
test_loss, test_accuracy = rnn_model_jamaal.evaluate(test_jamaal['images'], test_jamaal['labels'])
print('Test Accuracy:', test_accuracy)


# %%
print('Training Accuracy:', rnn_history_jamaal.history['accuracy'][-1])
print('Validation Accuracy:', rnn_history_jamaal.history['val_accuracy'][-1])


# %%
rnn_predictions_jamaal = rnn_model_jamaal.predict(test_jamaal['images'])

# %%
plot_probability_distribution(
    np.argmax(test_jamaal['labels'][0]), rnn_predictions_jamaal[0])

# %%
start_index = 42
num_images = 4

for i in range(start_index, start_index + num_images):
    image = test_jamaal['images'][i]
    true_label = np.argmax(test_jamaal['labels'][i])
    prediction = rnn_predictions_jamaal[i]

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title(f'True Label: {true_label}')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plot_probability_distribution(true_label, prediction)

    plt.tight_layout()
    plt.show()

# %%
# Get the predicted labels
predicted_labels = np.argmax(rnn_predictions_jamaal, axis=1)

# Get the true labels
true_labels = np.argmax(test_jamaal['labels'], axis=1)

# Compute the confusion matrix
confusion_mat = confusion_matrix(true_labels, predicted_labels)

# Display the confusion matrix using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mat, annot=True, cmap='Blues')
plt.title('RNN Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


