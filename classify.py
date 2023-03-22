import tensorflow as tf
# #ssl import is for successfull importing of dataset from ciphar10 it is not mandatory
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#from the keras we going to use cifar10 dataset that contains 60000 images of 10 different class_names which is described below
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images/255.0 , test_images/255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

fig, ax = plt.subplots(nrows=6, ncols=6, figsize=(15,15))
for i in range(36):
    row, col = divmod(i, 6)
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    ax[row, col].imshow(train_images[i], cmap=plt.cm.binary)
    #xticks,ytics show axis values of pixel of an image from both x and y axis so we clean it to view image perfectly
    ax[row, col].set_xticks([])
    ax[row, col].set_yticks([])
    ax[row, col].set_xlabel(class_names[train_labels[i][0]])
    #Manually added padding to wspace=0.5 and hspace=0.5
    fig.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=0.5, hspace=0.5)

fig.suptitle("CIFAR10 Classification", fontsize=16)

#Create the convolutional base
#As input, a CNN takes tensors of shape (image_height, image_width, color_channels)
model = models.Sequential()
#since the given size from the first layer is 32x32 with 3 color format i.e RGB
#output_size = (input_size - kernel_size +2*padding)+strde+1
#From convolution kernel_size=3, filters = 32, input size = 32 from the output feature map size = (32-3+2*0)/1+1 = 30
#From pooling input_size =30 it will divide the input_size by the given size here it is 2 so 30/2 = 15
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#From the summary we can observe that increasing the filter value increases the total parameter count
model.summary()

#Adding dense layers on top
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()

#Compile and train the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#the number of epochs to be set till where val_accuracy should be decrease and accuracy still increase find the point and stop at that epochs proess reason is model start memorize instead of predicting it
history = model.fit(train_images, train_labels, epochs=10,  
                    validation_data=(test_images, test_labels))

#Evaluate the model
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(history.history['accuracy'], label='accuracy')
ax.plot(history.history['val_accuracy'], label = 'val_accuracy')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_ylim([0.5, 1])
ax.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

#Testing the accuracy
print("Accuracy:",test_acc)

plt.show()
