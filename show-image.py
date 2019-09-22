import matplotlib.pyplot as plt
import script as script

X_train = script.X_train
y_train = script.y_train

img_index = 8888 # <<<<<  You can update this value to look at other images
img = X_train[img_index]
print("Image Label: " + str(chr(y_train[img_index]+96)))
plt.imshow(img.reshape((28,28)))
plt.show()