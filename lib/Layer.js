var INITIALIZE = require('./Initialize');
var Neuron = require('./Neuron');
var parallelizer = require('./Parallelizer');

/**
 * Creates a single dimension Layer of Neurons.
 * @param {string} numNeurons - The number of Neurons this Layer should have.
 * @param {boolean} [addBias=false] - Add a bias Neuron to this Layer.
 * @constructor
 */
function Layer(numNeurons, addBias) {
  var self = this;
  self.id = Layer.count++;
  self.neurons = [];

  // add neurons
  _.times(numNeurons, function() {
    self.neurons.push(new Neuron());
  });

  // add bias neuron
  if (addBias) {
    var biasNeuron = new Neuron();
    biasNeuron.isBias = true;
    self.neurons.push(biasNeuron);
  }
}

Layer.count = 0;

/**
 * Connects every Neuron in this Layer to each Neuron in the `target` Layer.
 * @param {Layer} targetLayer - The Layer to connect to.
 */
Layer.prototype.connect = function(targetLayer) {
  var self = this;

  _.each(self.neurons, function(source) {
    // every neuron in this layer is connected to each neuron in the next.
    // we can assume the numInputs to be the num of neurons in this layer.

    // connect to each neuron in this Layer to the targetLayer
    _.each(targetLayer.neurons, function(target) {
      source.connect(target, INITIALIZE.weight(self.neurons.length));
    });
  });
};

/**
 * Activates all the Neurons in this Layer with the given array of values.
 * @param {number[]} [values] - Map of input values for each Neuron.
 * @returns {number[]} - Array of Neuron output values.
 */
Layer.prototype.activate = function(values) {
  // console.debug('[l' + this.id + '] layer.activate() with:', values);
  var self = this;
  var data = _.map(self.neurons, function(neuron, i) {
    return {
      neuron: parallelizer.serialize(neuron),
      value: typeof values !== 'undefined' ? values[i] : undefined
    };
  });

  // parallelizer.map() => Promise
  return parallelizer.map(data, {}, function(item) {
    // parse function strings back into functions :(
    item.neuron.activate = eval('(' + item.neuron.activate + ')');
    item.neuron.activationFn = eval('(' + item.neuron.activationFn + ')');
    item.neuron.activationDerivative =
      eval('(' + item.neuron.activationDerivative + ')');

    return item.neuron.activate(item.neuron, item.value);
  })
    .then(function(results) {
      // console.debug('[l' + self.id + '] layer.activate() success', results);
      _.each(self.neurons, function(neuron, i) {
        neuron.output = results[i];
      });
      return results;
    });

  // return _.map(this.neurons, function(neuron, i) {
  //   return neuron.activate(values ? values[i] : undefined);
  // });
};

/**
 * Train the Neurons in this layer.  If target `outputs` are specified, the
 * Neurons will learn to output these values.  This is only useful for output
 * Layers.
 * @param {number[]} [outputs] - Map of target output values for each Neuron.
 */
Layer.prototype.train = function(outputs) {
  _.each(this.neurons, function(neuron, i) {
    neuron.train(outputs ? outputs[i] : undefined);
  });
};

module.exports = Layer;
