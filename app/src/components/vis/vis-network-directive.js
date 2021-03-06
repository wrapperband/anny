function visNetwork(visNetworkOptions, AnnyFactory, $rootScope) {
  return {
    replace: true,
    scope: {},
    template: '<div class="vis-network"></div>',
    link: function link(scope, elm) {
      scope.getData = function getData() {
        var nodes = []
        var edges = []

        // layers
        _.each(AnnyFactory.network.allLayers, function(layer, layerIndex) {
          // neurons
          _.each(layer.neurons, function(neuron) {
            var id = neuron.id
            var input = neuron.input.toFixed(3)
            var output = neuron.output.toFixed(3)
            var delta = neuron.delta.toFixed(6)

            nodes.push({
              id: id,
              title: [
                '<b>id:</b> ', id, '<br/>',
                '<b>delta:</b> ', delta, '<br/>',
              ].join(''),
              level: layerIndex,
              label: (neuron.isInput() ? [
                '\no:', output,
              ] : neuron.isOutput() ? [
                '\ni:', input,
                '\no:', output,
              ] : neuron.isBias ? [
                '\no:', output,
              ] : /* hidden layer */ [
                '\ni:', input,
                '\no:', output,
              ]
              ).join(' '),
              value: Math.abs(output),
              group: neuron.isBias ? 'bias' : 'normal',
            })

            // connections
            _.each(neuron.outgoing, function(connection) {
              var weight = connection.weight.toFixed(3)

              edges.push({
                from: connection.source.id,
                to: connection.target.id,
                value: Math.abs(weight),
                title: 'weight: ' + weight,
                // matches border colors in network options factory
                color: {
                  color: weight >= 0 ? 'hsl(210, 20%, 25%)' :
                    'hsl(30, 15%, 25%)',
                  hover: weight >= 0 ? 'hsl(210, 35%, 45%)' :
                    'hsl(30, 40%, 40%)',
                  highlight: weight >= 0 ? 'hsl(210, 60%, 70%)' :
                    'hsl(30, 60%, 60%)',
                },
              })
            })
          })
        })

        return {
          nodes: new vis.DataSet(nodes),
          edges: new vis.DataSet(edges),
        }
      }

      // causes a refresh of the network graph
      scope.setData = function setData() {
        scope.network.setData(scope.getData())
      }

      $rootScope.$on('anny:changed', function onChange() {
        scope.setData()
      })

      // create network
      scope.network =
        new vis.Network(elm[0], scope.getData(), visNetworkOptions)
    },
  }
}

angular.module('App.vis')
  .directive('visNetwork', visNetwork)
