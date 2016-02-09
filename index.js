'use strict';

var math = require('forwardjs-ml-math');
var Network = require('./network');
var mnist = require('mnistjs');

var training = mnist.training;
var testing = mnist.testing;

var maxIter = 4000;
var hiddenLayers = 50;

var network = new Network(400, hiddenLayers, 10);

console.log('Starting training:\n%s hidden layers\n%s max training iterations', hiddenLayers, maxIter);
console.time('Training time')

// train the neurons to modify the weights
for (var iter = 0; iter < maxIter; iter++) {
  var i = Math.floor(Math.random() * training.length);
  var input = training[i].input;
  var output = training[i].output;

  // hypotheses = final output
  var hs = network.forward(input);
  var outputError = math.arraySubtract(hs, output);

  network.backward(outputError);
  network.updateWeights();

  if (iter % 250 === 0) {
    console.log('Accuracy at iter %s: %s', iter, Math.round(accuracy(testing)*1000)/1000);
  }
}

console.log('Accuracy at iter %s: %s', maxIter, Math.round(accuracy(testing)*1000)/1000);
console.timeEnd('Training time')
console.log('Testing accuracy', accuracy(testing));

function accuracy(data) {
  var correct = 0;
  for (var i = 0; i < data.length; i++) {
    var input = data[i].input;
    var output = data[i].output;

    var hs = network.forward(input);

    var h = maxElem(hs);
    if (h === data[i].label) correct++;
  }
  return correct / data.length;
}

function maxElem(array) {
  var index = 0;
  for (var i = 1; i < array.length; i++) {
    if (array[index] < array[i]) index = i;
  }
  return index;
}
