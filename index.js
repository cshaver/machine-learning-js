'use strict';

var math = require('forwardjs-ml-math');

var data = [
  {
    input: [0, 0],
    output: 0,
  },
  {
    input: [1, 0],
    output: 1,
  },
  {
    input: [0, 1],
    output: 1,
  },
  {
    input: [1, 1],
    output: 0,
  },
];

var Neuron = require('./neuron');

var neuron = new Neuron();

var andNeuron = new Neuron();
var orNeuron = new Neuron();
var outputNeuron = new Neuron();

// train the neurons to modify the weights
for (var iter = 0; iter < 100000; iter++) {
  var i = Math.floor(Math.random() * data.length);
  var input = [1].concat(data[i].input);
  var output = data[i].output;

  var output1 = andNeuron.forward(input);
  var output2 = orNeuron.forward(input);

  var input2 = [1, output1, output2];

  // hypothesis = final output
  var h = outputNeuron.forward(input2);
  var error = h - output;

  // hiddenErrors is the output's error applied to its inputs
  // inputs to outputNeuron were [andNeuron output, orNeuron output]
  // so they get fed back in that way as well
  var hiddenErrors = outputNeuron.backward(error);
  andNeuron.updateWeights(hiddenErrors[0]);
  orNeuron.updateWeights(hiddenErrors[1]);
  outputNeuron.updateWeights(error);

  if (iter % 25 === 0) console.log('accuracy at iter %s: %s', iter, accuracy());
}

// final prediction
for (var i = 0; i < data.length; i++) {
  var input = [1].concat(data[i].input);
  var output = data[i].output;

  var output1 = andNeuron.forward(input);
  var output2 = orNeuron.forward(input);

  var input2 = [1, output1, output2];

  // hypothesis = final output
  var h = outputNeuron.forward(input2);

  console.log('XOR %s -> %s', data[i].input, h);
}

console.log(andNeuron.weights);
console.log(orNeuron.weights);
console.log(outputNeuron.weights);

// // train the neurons to modify the weights
// for (var iter = 0; iter < 10000; iter++) {
//   var i = Math.floor(Math.random() * data.length);
//   var input = data[i].input;
//   var output = data[i].output;
//
//   input = [i].concat(input);
//   var h = neuron.forward(input);
//   var error = h - output;
//
//   neuron.updateWeights(error);
//
//   if (iter % 25 === 0) console.log('accuracy at iter %s: %s', iter, accuracy());
// }

function accuracy() {
  var correct = 0;
  for (var i = 0; i < data.length; i++) {
    var input = [1].concat(data[i].input);
    var output = data[i].output;

    var output1 = andNeuron.forward(input);
    var output2 = orNeuron.forward(input);

    var input2 = [1, output1, output2];

    // hypothesis = final output
    var h = outputNeuron.forward(input2);
    h = h > 0.5 ? 1 : 0;
    if (h === output) correct++;
  }

  return correct / data.length;
}
