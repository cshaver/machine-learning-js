'use strict';

var math = require('forwardjs-ml-math');

// size of steps made when adjusting weights
var stepSize = 0.1;

module.exports = class Neuron {

  constructor(n) {
    this.weights = [];
    for (var i = 0; i < n; i++) {
      var weight = Math.random() - 0.5;
      this.weights.push(weight);
    }
  }

  /**
   * @param [Array] inputs
   */
  forward(inputs) {
    this.inputs = inputs;
    this.z = math.arrayMultiply(inputs, this.weights);
    return math.sigmoid(this.z);
  }

  /**
   * @param {Number} error
   * @returns {Array}
   */
  backward(error) {
    this.error = error;
    var backErrors = this.weights.map(w => w * error);
    // dont need to change bias
    return backErrors.slice(1);
  }

  /**
   * @param {Number} error
   */
  updateWeights() {
    var deltas = this.inputs.map(input => {
      return this.error * input * math.sigmoidGradient(this.z) * stepSize;
    });

    this.weights = math.arraySubtract(this.weights, deltas);
  }
};
