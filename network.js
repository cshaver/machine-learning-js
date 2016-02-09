'use strict';

var Layer = require('./layer');

module.exports = class Network {

  /**
   * @param {Number} inputSize
   * @param {Number} hiddenlayerSize
   * @param {Number} outputLayerSize
   */
  constructor (inputSize, hiddenlayerSize, outputLayerSize) {
    this.hiddenLayer = new Layer(hiddenlayerSize, inputSize + 1);
    this.outputLayer = new Layer(outputLayerSize, hiddenlayerSize + 1);
  }

  /**
    * @param {Array} inputs
    * @return {Array}
    */
  forward(inputs) {
    var hiddenLayerInputs = [1].concat(inputs);
    var hiddenLayerOutput = this.hiddenLayer.forward(hiddenLayerInputs);

    var outputLayerInputs = [1].concat(hiddenLayerOutput);
    var output = this.outputLayer.forward(outputLayerInputs);

    return output;
  }

  backward(errors) {
    // hiddenErrors is the output's error applied to its inputs
    // inputs to outputNeuron were [andNeuron output, orNeuron output]
    // so they get fed back in that way as well
    // grab hidden errors before updating weights for outneuron so that
    // we're not misattributing error
    var hiddenErrors = this.outputLayer.backward(errors);
    this.hiddenLayer.backward(hiddenErrors);
  }

  updateWeights() {
    this.hiddenLayer.updateWeights();
    this.outputLayer.updateWeights();
  }
}
