import _ from 'lodash'

/**
 * Functions for calculating Network error.  The error is simply the difference
 * between the correct output and the actual output.
 * @namespace
 * @see Network
 * @type {object}
 */
const ERROR = {
  /**
   * @param {number[]} expected - Array of output values the Network should
   *   have produced.
   * @param {number[]} actual - Array of output values the Network actually
   *   produced.
   * @returns {number}
   */
  crossEntropy(expected, actual) {
    return -(_.sum(actual, (actVal, i) => {
      return Math.log(actVal) * expected[i]
    })) / actual.length
  },

  // These taken from: https://www.youtube.com/watch?v=U4BTzF3Wzt0

  /**
   * @param {number[]} expected - Array of output values the Network should
   *   have produced.
   * @param {number[]} actual - Array of output values the Network actually
   *   produced.
   * @returns {number}
   */
  meanSquared(expected, actual) {
    return _.sum(actual, (actVal, i) => {
      return 0.5 * Math.pow(expected[i] - actVal, 2)
    }) / actual.length
  },

  /**
   * @param {number[]} expected - Array of output values the Network should
   *   have produced.
   * @param {number[]} actual - Array of output values the Network actually
   *   produced.
   * @returns {number}
   */
  rootMeanSquared(expected, actual) {
    return Math.sqrt(ERROR.meanSquared(expected, actual))
  },

  /**
   * @param {number[]} expected - Array of output values the Network should
   *   have produced.
   * @param {number[]} actual - Array of output values the Network actually
   *   produced.
   * @returns {number}
   */
  arcTan(expected, actual) {
    return _.sum(actual, (actVal, i) => {
      return Math.atan(expected[i] - actVal)
    }) / actual.length
  },
}

export default ERROR
