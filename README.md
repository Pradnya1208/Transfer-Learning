<div align="center">
  
[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]

</div>

# <div align="center">Transfer Learning</div>
<div align="center"><img src="https://github.com/Pradnya1208/Transfer-Learning/blob/main/readme_files/intro.gif?raw=true" width="70%"></div>


## Overview:
In this project, we will learn to classify images using transfer learning from a pre-trained network.

A pre-trained model is a saved network that was previously trained on a large dataset, typically on a large-scale image-classification task. You either use the pretrained model as is or use transfer learning to customize this model to a given task.

The intuition behind transfer learning for image classification is that if a model is trained on a large and general enough dataset, this model will effectively serve as a generic model of the visual world. You can then take advantage of these learned feature maps without having to start from scratch by training a large model on a large dataset.

## Why transfer learning?
Unfortunately the Inception model seemed unable to classify images of people. The reason was the data-set used for training the Inception model, which had some confusing text-labels for classes.
he Inception model is actually quite capable of extracting useful information from an image. So we can instead train the Inception model using another data-set. But it takes several weeks using a very powerful and expensive computer to fully train the Inception model on a new data-set.

We can instead re-use the pre-trained Inception model and merely replace the layer that does the final classification. This is called Transfer Learning.

## Flowchart:
The following chart shows how the data flows when using the Inception model for Transfer Learning. First we input and process an image with the Inception model. Just prior to the final classification layer of the Inception model, we save the so-called Transfer Values to a cache-file.

The reason for using a cache-file is that it takes a long time to process an image with the Inception model.
If each image is processed more than once then we can save a lot of time by caching the transfer-values.

When all the images in the new data-set have been processed through the Inception model and the resulting transfer-values saved to a cache file, then we can use those transfer-values as the input to another neural network. We will then train the second neural network using the classes from the new data-set, so the network learns how to classify images based on the transfer-values from the Inception model.

In this way, the Inception model is used to extract useful information from the images and another neural network is then used for the actual classification.
<br>

<img src="https://github.com/Pradnya1208/Transfer-Learning/blob/main/readme_files/flowchart.PNG?raw=true" width="80%">


## Dataset:
[CIFAR 10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
<br>
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

Here are the classes in the dataset, as well as 10 random images from each:
<br>
<img src="https://github.com/Pradnya1208/Transfer-Learning/blob/main/readme_files/cifar10.PNG?raw=true">


## Implementation:

**libraries** : `matplotlib` `numpy` `pandas` `plotly` `sklearn` `keras` `tensorflow`

## Data Exploration:

### Plot a few images:
```
# Plot the images and labels using our helper-function.
plot_images(images=images, cls_true=cls_true, smooth=False)
```
<img src="https://github.com/Pradnya1208/Transfer-Learning/blob/main/readme_files/plot.PNG?raw=true" width="80%">

### Calculate transfer values using Inception model and store it in cache:
```
from inception import transfer_values_cache
transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train,
                                              images=images_scaled,
                                              model=model)

transfer_values_test = transfer_values_cache(cache_path=file_path_cache_test,
                                             images=images_scaled,
                                             model=model)
```
Above lines just show important functions for more detailes check [Notebook](https://github.com/Pradnya1208/Transfer-Learning/blob/main/Transfer%20Learning.ipynb)

### Plotting transfer values:
<img src="https://github.com/Pradnya1208/Transfer-Learning/blob/main/readme_files/transfer%20values.PNG?raw=true" width="70%">

### Analysis of transfer vlaues using PCA and t-sne:
#### PCA:

```
transfer_values_reduced = pca.fit_transform(transfer_values)
```
<img src="https://github.com/Pradnya1208/Transfer-Learning/blob/main/readme_files/pca.PNG?raw=true" width="70%">

#### T-SNE:
```
transfer_values_reduced = tsne.fit_transform(transfer_values_50d)
```
<img src="https://github.com/Pradnya1208/Transfer-Learning/blob/main/readme_files/tsne.PNG?raw=true" width="70%">


## Creating a new classifier in Tensorflow:
Now we will create another neural network in TensorFlow. This network will take as input the transfer-values from the Inception model and output the predicted classes for CIFAR-10 images.

### 1. Placeholder variables:
- First we need the array-length for transfer-values which is stored as a variable in the object for the Inception model.
```
transfer_len = model.transfer_len
```
- Now create a placeholder variable for inputting the transfer-values from the Inception model into the new network that we are building. The shape of this variable is [None, transfer_len] which means it takes an input array with an arbitrary number of samples as indicated by the keyword None and each sample has 2048 elements, equal to transfer_len.
```
x = tf.placeholder(tf.float32, shape=[None, transfer_len], name='x')
```
- Create another placeholder variable for inputting the true class-label of each image. These are so-called One-Hot encoded arrays with 10 elements, one for each possible class in the data-set.
```
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
```
- Calculate the true class as an integer. This could also be a placeholder variable.
```
y_true_cls = tf.argmax(y_true, dimension=1)
```
### 2. Neural Network:
Create the neural network for doing the classification on the CIFAR-10 data-set. This takes as input the transfer-values from the Inception model which will be fed into the placeholder variable x. The network outputs the predicted class in y_pred.
```
# Wrap the transfer-values as a Pretty Tensor object.
x_pretty = pt.wrap(x)

with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.\
        fully_connected(size=1024, name='layer_fc1').\
        softmax_classifier(num_classes=num_classes, labels=y_true)
```
### 3. Optimization method:
Create a variable for keeping track of the number of optimization iterations performed.
```
global_step = tf.Variable(initial_value=0,
                          name='global_step', trainable=False)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step)

### 4. Classification accuracy:
- The output of the network y_pred is an array with 10 elements. The class number is the index of the largest element in the array.
```
y_pred_cls = tf.argmax(y_pred, dimension=1)
```

- Create an array of booleans whether the predicted class equals the true class of each image.
```
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```
### 5. Tensorflow run:
```
session = tf.Session()
session.run(tf.global_variables_initializer())
```

There are 50,000 images (and arrays with transfer-values for the images) in the training-set. It takes a long time to calculate the gradient of the model using all these images (transfer-values). We therefore only use a small batch of images (transfer-values) in each iteration of the optimizer.
<br>

Check out the helper functions to train, optimize and find the accuracy of the model [here](https://github.com/Pradnya1208/Transfer-Learning/blob/main/Transfer%20Learning.ipynb)

### 6. Result:
```
print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)
```
#### Confusion Matrix:
```
Confusion Matrix:
[926   6  13   2   3   0   1   1  29  19] (0) airplane
[  9 921   2   5   0   1   1   1   2  58] (1) automobile
[ 18   1 883  31  32   4  22   5   1   3] (2) bird
[  7   2  19 855  23  57  24   9   2   2] (3) cat
[  5   0  21  25 896   4  24  22   2   1] (4) deer
[  2   0  12  97  18 843  10  15   1   2] (5) dog
[  2   1  16  17  17   4 940   1   2   0] (6) frog
[  8   0  10  19  28  14   1 914   2   4] (7) horse
[ 42   6   1   4   1   0   2   0 932  12] (8) ship
[  6  19   2   2   1   0   1   1   9 959] (9) truck
 (0) (1) (2) (3) (4) (5) (6) (7) (8) (9)
```




### Learnings:
`transfer learning`







## References:
[Transfer Learning essentials](https://towardsdatascience.com/how-transfer-learning-can-be-a-blessing-in-deep-learning-models-fbc576dc42)
[Transfer learning](https://www.youtube.com/watch?v=upfgTWrhkpg&list=PL9Hr9sNUjfsmEu1ZniY0XpHSzl5uihcXZ&index=16)
### Feedback

If you have any feedback, please reach out at pradnyapatil671@gmail.com


### 🚀 About Me
#### Hi, I'm Pradnya! 👋
I am an AI Enthusiast and  Data science & ML practitioner




[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]

