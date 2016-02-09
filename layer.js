'use strict';

var math = require('forwardjs-ml-math');
var Neuron = require('./neuron');

module.exports = class Layer {
  constructor(size, inputs) {
    this.neurons = [];
    for (var i = 0; i < size; i++) {
      var neuron = new Neuron(inputs);
      this.neurons.push(neuron);
    }
  }

  /**
   * @param {Array} inputs
   * @return {Array}
   */
  forward(inputs) {
    var outputs = this.neurons.map(neuron =>
      neuron.forward(inputs)
    );
    return outputs;
  }

  /**
   * @param {Array} errors
   * @return {Array}
   */
  backward(errors) {
    // collecting the errors for each
    // neuron in this layer
    var allBackwardErrors = [];
    for (var i = 0; i < this.neurons.length; i++) {
      var neuron = this.neurons[i];
      var error = errors[i];
      var backwardError = neuron.backward(error);
      allBackwardErrors.push(backwardError);
    }

    // summing the errors that each of
    // the neurons in this layer want
    // to pass backward
    var totalBackwardError = allBackwardErrors[0];
    for (var i = 1; i < allBackwardErrors.length; i++) {
      totalBackwardError = math.arrayAdd(
        totalBackwardError,
        allBackwardErrors[i]
      );
    }
    return totalBackwardError;
  }

  updateWeights() {
    this.neurons.forEach(n => n.updateWeights());
  }
}
