# Machine Learning with [Jed Borovik](https://github.com/jedborovik)

Run with `node index.js`

##### _Tuesday Feb 9_

Feed forward propagation, Back propagation

Training vs testing data
  - how well does the algorithm deal with unseen data
  - on a bigger data set we want to find our accuracy on data we've never seen before
  - testing data is withheld during the training portion

Predicting handwritten digits
  - 10 output neurons, outputs whether or not it is that digit
  - Hidden layer? No definitive number of neurons for hidden layer. Soome say avg num inputs + num outputs.
  - We'll do 25 for now - each of these 25 will have 400 inputs, so will have 401 inputs with bias
  - 401 inputs => 25 neurons => 10 neurons
  - Neuron pruning when weight is 0 (brain does this, called [synaptic pruning](https://en.wikipedia.org/wiki/Synaptic_pruning))
  - Best trained on this data is like 99% accuracy
  - Going forward is very fast, but going backwards is where we slow down

Other resources
  - https://www.coursera.org/learn/machine-learning
  - Convolutional neural network
  - Nearest neighbor clustering
  - Neuron that we created is called a __perceptron__
  - More layers to a network - usually one hidden layer is enough, depends on a lot of things
