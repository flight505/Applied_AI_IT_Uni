# Applied Artificial Intelligence (Summer University) (Spring 2021)


Three lectures per week.
Monday / Tuesday / Thursday / friday is question day
Wednesday and Fridays are dedicated to individual work and supervision


***
### day01 Intro to the course and technology stack:
<b> The hand-ins for this session will be: </b>
- [x] 1b - Chipotle.

- [ ] 7b - Numpy exercises.

- [ ] 8b - Plotting exercises.


***
### day02 Simple Regression Models:

<b> The hand-ins for this session will be: </b>

- [ ] linear-regression-house-price-prediction.ipynb
(polynomial regression part in optional; leave neural networks for Thursday)

- [ ] Logistic_regression_heart_attack.ipynb
(tips and clarifications:
"cost" function and "loss" function have the same meaning in the context if this exercise;
You can either use PyTorch to calculate the  gradient (so don't fill the "gradient code"), or you can implement your own gradient descent using the gradient implementation shown in the slides;
Accuracy is calculated  like: correctly classified samples / all samples.
)

- An example how the optimisation works with PyTorch.

https://towardsdatascience.com/logistic-regression-on-mnist-with-pytorch-b048327f8d19

***
### day03 Classification and Deep Learning:
Notebooks for Neural Networks

https://drive.google.com/drive/folders/1M6gUQDsfNSCyH_dCuSqNdPSakj0FX2rt?usp=sharing

Implementing neural network from scratch:

https://www.kaggle.com/djoledjole/mnist-neural-network-from-scratch/edit

Example of using Keras to build and train a neural network to solve multi-class MNIST dataset classification:

https://colab.research.google.com/github/tensorflow/datasets/blob/master/docs/keras_example.ipynb

Link to Fashion-MNIST:

https://github.com/zalandoresearch/fashion-mnist

<b> The hand-ins for this session will be: </b>
- [ ] fashion-mnist-with-keras-neural-networks.ipynb

Clarifications:

* You can import data however you want. The second executable cell assumes that you saved your files into ../input folder and checks if you saved it correctly. You can change this and import Fashion-MNIST with Keras, if you want.

* Split the training data into validation and training data.

* F score should be calculated for each class


***

### day04 Dimensionality Reduction and Clustering:

<b> The hand-ins for this session will be: </b>

- [ ] PROJECT - Pizza - Dimensionality Reduction and Clustering.ipynb

The goal of this project is to check if you understand the PCA and KMeans algorithms that we covered in class.

We will walk you through computing the principal components of a matrix that shows the nutrient composition of pizzas of different brands.

***
### day05 Introduction to NLP:

Text normalization using Gensim, we will load and preprocess the data from the six_thousand_tweets.csv file using the normalization techniques we talked about in the lecture.
next text processing
For this, we will end up covering several libraries, including (but not limited to):
- BeautifulSoup4.
- PyPDF2.
- urllib.
- NLTK.

No hand-ins for this day.
***

### day06 word2vec and LDA:

<b> The hand-ins for this session will be: </b>
- [ ] Project: state-of-the-union speeches.

  In this project, we will load and process the us_president_speeches.csv file, preprocess the speeches, embed their words and do topic modeling on them.

***
### day07 Convolutional Neural Networks + Dropout and BatchNorm:

<b> The hand-ins for this session will be: </b>
- [ ] Project: PROJECT_Convolutional_networks.ipynb

Clarifications: Feel free to browse Kaggle, Tensorflow dataset database, or any other dataset. The Intel Image Classification dataset is an example dataset you might find interesting.

Avoid using MNIST-type dataset cause they are too easy to solve, pay attention that you don't take a massive dataset that will take forever to train on (like ImageNet). Also, pay attention that the dataset is not too small or has low-quality data points (too much noise, nonsense entries, etc).

***

### day08 Reinforcement Learning (Deep Q-learning):

<b> The hand-ins for this session will be: </b>
- [ ] PROJECT_reinforcement_learning.ipynb


In this project we will solve two simple environments using a Q-table and a Neural Network (Deep Q-learning).

***
### day09:
***
### day10:
***
### day11:
***
### day12:

