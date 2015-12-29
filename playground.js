/* eslint-disable no-console */
import _ from 'lodash'
import Network from './src/Network'
import Data from './src/Data'

let fails = 0
let successes = 0

_.times(10, () => {
  const network = new Network([2, 1])

  let lastError
  let lowestError = Infinity
  console.log(_.repeat('-', 70))

  network.train(Data.ORGate, {
    maxEpochs: 100,
    frequency: 33,
    onSuccess: (error, epoch) => {
      console.log(`\nSuccessfully trained to ${error} error after ${epoch} epochs`)
      console.log(`
      Activation Results:
        ${network.activate([0, 0])}
        ${network.activate([0, 1])}
        ${network.activate([1, 0])}
        ${network.activate([1, 1])}
      `)
      successes++
    },
    onFail: (error, epoch) => {
      console.log(`Fail to train, error is ${error} after ${epoch} epochs`)
      fails++
    },
    onProgress: (error, epoch) => {
      const sign = error < lastError ? '↓' : '↑'
      const lowest = error < lowestError ? '★' : ' '
      lowestError = Math.min(error, lowestError)
      lastError = error
      console.log(`${_.pad(epoch, 5)} ${lowest} ${sign} ${error}`)
    },
  })
})

console.log(`
  Fails:     ${fails}
  Successes: ${successes}
`)
