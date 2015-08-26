var Promise = require('./Promise');
var Parallelizer = {};

function activate(parNeuron, value) {
  // console.debug(
  //   'parallelizer n' + parNeuron.id + '.activate() with:', parNeuron, value
  // );
  if (parNeuron.isBias) {
    parNeuron.output = 1;
    return parNeuron.output;
  }

  //
  // Input
  //
  if (typeof value !== 'undefined') {
    parNeuron.input = value;
  } else {
    var connection;
    var parNeuronCount = typeof parNeuron.incoming !== 'undefined'
      ? parNeuron.incoming.length
      : 0;
    parNeuron.input = 0;
    for (var i = 0; i < parNeuronCount; i += 1) {
      connection = parNeuron.incoming[i];
      // we don't need to add the bias parNeuron manually here.
      // since the bias Neuron is connected like all other Neurons and it's
      // output is always 1, the weight will be added by bias.output * weight.
      parNeuron.input += connection.source.output * connection.weight;
    }
  }

  //
  // Output
  //
  // do not squash input Neurons values, pass them straight through
  parNeuron.output = parNeuron.isInput
    ? parNeuron.input
    : parNeuron.activationFn(parNeuron.input);

  return parNeuron.output;
}

/**
 * Serialize a `neuron` into an object literal for use in a web worker.
 * @param {Neuron} neuron - The Neuron to serialize.
 * @returns {object}
 */
Parallelizer.serialize = function(neuron) {
  return {
    id: neuron.id,
    activate: activate.toString(),
    activationFn: neuron.activationFn.toString(),
    activationDerivative: neuron.activationDerivative.toString(),
    isBias: neuron.isBias,
    isInput: neuron.isInput(),
    delta: neuron.delta,
    error: neuron.error,
    input: neuron.input,
    output: neuron.output,
    incoming: _.map(neuron.incoming, function(connection) {
      return {
        weight: connection.weight,
        source: {
          output: connection.source.output
        }
      };
    }),
    outgoing: _.map(neuron.outgoing, function(connection) {
      return {
        weight: connection.weight,
        target: {
          delta: connection.target.delta
        }
      };
    })
  };
};

Parallelizer.map = function(array, environment, callback) {
  // console.debug('parallelizer.map() with:', array, environment, callback);
  callback.name = callback.name || 'mapPar';

  return new Promise(function(resolve, reject) {
    var parallel = new Parallel(array, {env: environment});

    parallel.map(callback).then(function() {
      // console.debug('parallelizer.map() success:', parallel);
      resolve(parallel.data);
    }, function(error) {
      reject(error);
    });
  });
};

module.exports = Parallelizer;
