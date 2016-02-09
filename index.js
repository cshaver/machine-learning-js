'use strict';

var math = require('forwardjs-ml-math');
var Network = require('./network');

var data = [
  {
    input: [0, 0],
    output: [0],
  },
  {
    input: [1, 0],
    output: [1],
  },
  {
    input: [0, 1],
    output: [1],
  },
  {
    input: [1, 1],
    output: [0],
  },
];

var network = new Network(2, 1);

// train the neurons to modify the weights
for (var iter = 0; iter < 100000; iter++) {
  var i = Math.floor(Math.random() * data.length);
  var input = data[i].input;
  var output = data[i].output;

  // hypotheses = final output
  var hs = network.forward(input);
  var outputError = math.arraySubtract(hs, output);

  network.backward(outputError);

  network.updateWeights();

  if (iter % 1000 === 0) console.log('accuracy at iter %s: %s', iter, accuracy());
}

// final prediction
for (var i = 0; i < data.length; i++) {
  var input = data[i].input;
  var output = data[i].output;

  // hypotheses = final output
  var hs = network.forward(input);

  console.log('XOR %s -> %s', data[i].input, Math.round(hs[0] * 100) / 100);
}

// console.log(andNeuron.weights.map(w => Math.round(w * 100)/100));
// console.log(orNeuron.weights.map(w => Math.round(w * 100)/100));
// console.log(outputNeuron.weights.map(w => Math.round(w * 100)/100));

function accuracy() {
  var correct = 0;
  for (var i = 0; i < data.length; i++) {
    var input = data[i].input;
    var output = data[i].output;

    var hs = network.forward(input);
    var outputError = math.arraySubtract(hs, output);

    var h = hs[0] > 0.5 ? 1 : 0;
    if (h === output[0]) correct++;
  }

  return correct / data.length;
}
