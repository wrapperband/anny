/* eslint-disable no-console */
import _ from 'lodash'
import Network from './src/Network'
import Data from './src/Data'

const network = new Network([2, 1])

let lastError
let lowestError = Infinity

network.train(Data.ORGate, {
  maxEpochs: 100,
  frequency: 10,
  onSuccess: (error, epoch) => {
    console.log(`Successfully trained to ${error} error after ${epoch} epochs`)
  },
  onFail: (error, epoch) => {
    console.log(`Fail to train, error is ${error} after ${epoch} epochs`)
  },
  onProgress: (error, epoch) => {
    const sign = error < lastError ? '↓' : '↑'
    const lowest = error < lowestError ? '★' : ' '
    lowestError = Math.min(error, lowestError)
    lastError = error
    console.log(`${_.pad(epoch, 5)} ${lowest} ${sign} ${error}`)
  },
})


console.log(`
Activation Results:
  ${network.activate([0, 0])}
  ${network.activate([0, 1])}
  ${network.activate([1, 0])}
  ${network.activate([1, 1])}
`)
