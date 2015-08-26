var Layer = require('./Layer');
var ERROR = require('./Error');
var Promise = require('./Promise');

/**
 * Creates a Network of Layers consisting of Neurons. Each array element
 * indicates a layer.  The value indicates the number of Neurons in that Layer.
 *
 * The first element represents the number of Neurons in the input Layer.
 * The last element represents the number of Neurons in the output Layer.
 * Each element in between represents a hidden Layer with n Neurons.
 * @param {number[]} layerSizes - Number of neurons in each layer.
 * @constructor
 *
 * @example
 * // 2 inputs
 * // 1 output
 * var net = new Network([2, 1]);
 *
 * @example
 * // 16 inputs
 * // 10 neuron hidden layer
 * // 4 neuron hidden layer
 * // 1 output
 * var net = new Network([16, 10, 4, 1]);
 */
function Network(layerSizes) {
  var numInputs = _.first(layerSizes);
  var numOutputs = _.last(layerSizes);
  var hiddenLayers = _.slice(layerSizes, 1, layerSizes.length - 1);
  this.output = [];
  this.errorFn = ERROR.meanSquared;

  this.allLayers = [];
  this.hiddenLayers = [];

  // input layer
  this.inputLayer = new Layer(numInputs, true);
  this.allLayers.push(this.inputLayer);

  // hidden layers
  _.each(hiddenLayers, function(numNeurons) {
    var layer = new Layer(numNeurons, true);
    this.hiddenLayers.push(layer);
    this.allLayers.push(layer);
  }, this);

  // output layer
  this.outputLayer = new Layer(numOutputs);
  this.allLayers.push(this.outputLayer);

  // connect layers and populate allLayers
  _.each(this.allLayers, function(layer, i) {
    var next = this.allLayers[i + 1];
    if (next) {
      layer.connect(next);
    }
  }, this);
}

/**
 * Activate the network with a given set of `input` values.
 * @param {number[]} inputs - Values to activate the Network input Neurons.
 *   Values should be normalized between -1 and 1 using Util.normalize.
 * @returns {*} output values
 */
Network.prototype.activate = function(inputs) {
  // console.debug('network.activate() with:', inputs);
  var self = this;

  // this.inputLayer.activate(inputs);
  // _.invoke(this.hiddenLayers, 'activate');
  // this.output = this.outputLayer.activate();
  // return this.output;

  return new Promise(function(resolve, reject) {
    self.inputLayer.activate(inputs)
      .then(function() {
        return self.outputLayer.activate();
      })
      .then(function(prevLayerOutput) {
        self.output = prevLayerOutput;
        resolve(self.output);
      })
      .catch(function(error) {
        reject(error);
      });
  });
};

/**
 * Correct the Network to produce the specified `output`.
 * @param {number[]} output - The target output for the Network.
 * Values in the array specify the target output of the Neuron in the output
 *   layer.
 */
Network.prototype.correct = function(output) {
  // console.debug('network.correct() with:', output);
  this.outputLayer.train(output);

  // train hidden layers in reverse (last to first)
  for (var i = this.hiddenLayers.length - 1; i >= 0; i -= 1) {
    this.hiddenLayers[i].train();
  }

  this.inputLayer.train();
};

/**
 * Train the Network to produce the output from the given input.
 * @param {object[]} data - Array of objects in the form
 * `{input: [], output: []}`.
 * @param {function} [callback] - Called with the current error every
 *   `frequency`.
 * @param {number} [frequency] - How many iterations to let pass between
 *   logging the current error.
 */
Network.prototype.train = function(data, callback, frequency) {
  var self = this;

  return new Promise(function(resolve, reject) {
    console.debug('network.train()');
    // TODO: validation and help on the data.
    //  ensure it is normalized between -1 and 1
    //  ensure the input length matches the number of Network inputs
    //  ensure the output length matches the number of Network outputs
    var epochs = 50000;
    var errorThreshold = 0.001;
    var callbackFrequency = frequency || _.max([1, _.floor(epochs / 20)]);

    var epochCounter = 0;
    var sampleCounter = 0;
    var avgError = 0;

    var success = function success() {
      var successMsg = [
        'Successfully trained to an error of', avgError,
        'after', epochCounter, 'epochs.'
      ].join(' ');

      console.debug(successMsg);
      resolve(successMsg);
    };

    var fail = function fail() {
      var errorMsg = [
        'Failed to train. Error is', avgError, 'after', epochCounter, 'epochs.'
      ].join(' ');

      console.warn(errorMsg);
      reject(errorMsg);
    };

    var defaultCallback = function defaultCallback(err, epoch) {
      console.log('[training] epoch:', epoch, 'error:', err);
    };

    var nextSample = function nextSample() {
      // console.debug('network.train() nextSample()');
      var sample = data[sampleCounter++];

      // make a prediction
      self.activate(sample.input).then(function(output) {
        // console.debug('network.activate() success =>', output);
        // correct the error
        self.correct(sample.output);

        // get the error
        avgError += self.errorFn(sample.output, self.output) / data.length;

        // success
        if (avgError <= errorThreshold) {
          success();
          // fail
        } else if (epochCounter === epochs) {
          fail();
          // next sample
        } else if (sampleCounter < data.length) {
          // console.debug('next sample', sampleCounter, 'of', data.length);
          nextSample();
          // next epoch
        } else {
          avgError = 0;
          epochCounter += 1;
          sampleCounter = 0;
          console.debug('next epoch', epochCounter);

          // callback with results periodically
          if (epochCounter === 1 || epochCounter % callbackFrequency === 0) {
            (callback || defaultCallback)(avgError, epochCounter);
          }
          nextSample();
        }
      });
    };

    nextSample();
  });

  // _.each(_.range(epochs), function(index) {
  //   var n = index + 1;
  //
  //   // loop over the training data summing the error of all samples
  //   // http://www.researchgate.net/post
  //   //   /Neural_networks_and_mean-square_errors#rgw51_55cb2f1399589
  //   var avgError = _.sum(_.map(data, function(sample) {
  //     // make a prediction
  //     this.activate(sample.input);
  //
  //     // correct the error
  //     this.correct(sample.output);
  //
  //     // get the error
  //     return this.errorFn(sample.output, this.output) / data.length;
  //   }, this));
  //
  //   // callback with results periodically
  //   if (n === 1 || n % callbackFrequency === 0) {
  //     (callback || defaultCallback)(avgError, n);
  //   }
  //
  //   // success / fail
  //   if (avgError <= errorThreshold) {
  //     console.debug(
  //       'Successfully trained to an error of', avgError, 'after', n,
  // 'epochs.' ); return false; } else if (n === epochs) { console.warn(
  // 'Failed to train. Error is', avgError, 'after', n, 'epochs.' ); } },
  // this);
};

module.exports = Network;
