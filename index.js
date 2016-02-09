'use strict';

var math = require('forwardjs-ml-math');
var Network = require('./network');
var mnist = require('mnistjs');

var training = mnist.training.slice(10000);

var network = new Network(400, 50, 10);

// train the neurons to modify the weights
for (var iter = 0; iter < 40000; iter++) {
  var i = Math.floor(Math.random() * training.length);
  var input = training[i].input;
  var output = training[i].output;

  // hypotheses = final output
  var hs = network.forward(input);
  var outputError = math.arraySubtract(hs, output);

  network.backward(outputError);

  network.updateWeights();

  if (iter % 1000 === 0) {
    console.log('accuracy at iter %s: %s', iter, accuracy(training));
  }
}

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
