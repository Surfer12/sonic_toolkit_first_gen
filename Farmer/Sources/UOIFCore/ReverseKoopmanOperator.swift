//
//  ReverseKoopmanOperator.swift
//  UOIFCore
//
//  Created by Ryan David Oates on 8/26/25.
//  Swift implementation of reverse Koopman operator for dynamical system analysis.
//  Provides mathematical framework for system behavior analysis and anomaly detection.

import Foundation
import Accelerate

/// Reverse Koopman operator implementation for dynamical system analysis
public class ReverseKoopmanOperator {

    // MARK: - Properties

    private var koopmanMatrix: [[Double]] = []
    private var eigenvalues: [ComplexNumber] = []
    private var eigenVectors: [[ComplexNumber]] = []
    private let logger = Logger()

    /// Configuration for Koopman analysis
    public struct Configuration {
        public var svdTolerance: Double = 1e-10
        public var eigenvalueThreshold: Double = 1e-8
        public var maxEigenvalues: Int = 50
        public var useComplexEigenvalues: Bool = true

        public init() {}
    }

    private var configuration: Configuration

    // MARK: - Initialization

    public init(configuration: Configuration = Configuration()) {
        self.configuration = configuration
    }

    // MARK: - Public API

    /// Compute reverse Koopman operator from time series data
    /// - Parameters:
    ///   - timeSeriesData: Array of state vectors over time
    ///   - observables: Array of observable functions
    ///   - completion: Callback with analysis results
    public func computeReverseKoopman(timeSeriesData: [[Double]],
                                     observables: [ObservableFunction],
                                     completion: @escaping (KoopmanAnalysis) -> Void) {
        logger.log("Starting reverse Koopman analysis with \(timeSeriesData.count) time steps")

        // Validate input data
        guard validateInputData(timeSeriesData, observables) else {
            logger.log("Input validation failed")
            completion(KoopmanAnalysis(
                eigenvalues: [],
                stabilityMargin: 0.0,
                reconstructionError: Double.infinity,
                koopmanMatrix: [],
                eigenDecomposition: nil
            ))
            return
        }

        // Compute observable matrix
        let observableMatrix = computeObservableMatrix(timeSeriesData, observables)
        logger.log("Computed observable matrix: \(observableMatrix.count) x \(observableMatrix.first?.count ?? 0)")

        // Compute Koopman matrix using SVD
        let koopmanMatrix = computeKoopmanMatrix(observableMatrix)
        self.koopmanMatrix = koopmanMatrix
        logger.log("Computed Koopman matrix: \(koopmanMatrix.count) x \(koopmanMatrix.first?.count ?? 0)")

        // Perform eigenvalue decomposition
        let eigenDecomposition = computeEigenDecomposition(koopmanMatrix)
        logger.log("Computed \(eigenDecomposition.eigenvalues.count) eigenvalues")

        // Calculate stability metrics
        let stabilityMargin = calculateStabilityMargin(eigenDecomposition.eigenvalues)
        let reconstructionError = calculateReconstructionError(observableMatrix, koopmanMatrix)

        logger.log("Stability margin: \(stabilityMargin)")
        logger.log("Reconstruction error: \(reconstructionError)")

        let analysis = KoopmanAnalysis(
            eigenvalues: eigenDecomposition.eigenvalues.map { $0.toComplex() },
            stabilityMargin: stabilityMargin,
            reconstructionError: reconstructionError,
            koopmanMatrix: koopmanMatrix,
            eigenDecomposition: eigenDecomposition
        )

        completion(analysis)
    }

    /// Compute Koopman matrix from observable data
    /// - Parameter observableData: Matrix of observable values
    /// - Returns: Koopman operator matrix
    public func computeKoopmanMatrix(_ observableData: [[Double]]) -> [[Double]] {
        guard observableData.count > 1 else { return [] }

        let rows = observableData[0].count
        let cols = observableData.count - 1

        var koopmanMatrix = Array(repeating: Array(repeating: 0.0, count: rows), count: rows)

        // Build data matrices for linear regression
        for i in 0..<rows {
            var xData = [Double]()
            var yData = [Double]()

            for t in 0..<cols {
                xData.append(contentsOf: observableData[t])
                yData.append(observableData[t + 1][i])
            }

            // Perform linear regression to find Koopman matrix row
            if let coefficients = linearRegression(xData, yData, numFeatures: rows) {
                koopmanMatrix[i] = coefficients
            }
        }

        return koopmanMatrix
    }

    /// Compute eigenvalues and eigenvectors of Koopman matrix
    /// - Parameter matrix: Square matrix for decomposition
    /// - Returns: Eigen decomposition results
    public func computeEigenDecomposition(_ matrix: [[Double]]) -> EigenDecomposition {
        guard !matrix.isEmpty, matrix.count == matrix[0].count else {
            return EigenDecomposition(eigenvalues: [], eigenvectors: [])
        }

        let n = matrix.count

        // Convert to flat array for LAPACK
        var flatMatrix = matrix.flatMap { $0 }
        var eigenvaluesReal = [Double](repeating: 0.0, count: n)
        var eigenvaluesImag = [Double](repeating: 0.0, count: n)
        var eigenvectors = [Double](repeating: 0.0, count: n * n)
        var work = [Double](repeating: 0.0, count: 4 * n)
        var info: Int32 = 0

        // Call LAPACK dgeev for real matrix eigenvalue decomposition
        flatMatrix.withUnsafeMutableBufferPointer { matrixPtr in
            eigenvaluesReal.withUnsafeMutableBufferPointer { eigenRealPtr in
                eigenvaluesImag.withUnsafeMutableBufferPointer { eigenImagPtr in
                    eigenvectors.withUnsafeMutableBufferPointer { eigenVecPtr in
                        work.withUnsafeMutableBufferPointer { workPtr in
                            dgeev_(
                                UnsafeMutablePointer(mutating: "N"), // JOBVL
                                UnsafeMutablePointer(mutating: "N"), // JOBVR
                                UnsafeMutablePointer(mutating: &n), // N
                                matrixPtr.baseAddress, // A
                                UnsafeMutablePointer(mutating: &n), // LDA
                                eigenRealPtr.baseAddress, // WR
                                eigenImagPtr.baseAddress, // WI
                                nil, // VL
                                UnsafeMutablePointer(mutating: &n), // LDVL
                                nil, // VR
                                UnsafeMutablePointer(mutating: &n), // LDVR
                                workPtr.baseAddress, // WORK
                                UnsafeMutablePointer(mutating: &work.count), // LWORK
                                UnsafeMutablePointer(mutating: &info) // INFO
                            )
                        }
                    }
                }
            }
        }

        guard info == 0 else {
            logger.log("Eigenvalue decomposition failed with info: \(info)")
            return EigenDecomposition(eigenvalues: [], eigenvectors: [])
        }

        // Convert results to complex numbers
        var eigenvalues: [ComplexNumber] = []
        for i in 0..<n {
            let eigenvalue = ComplexNumber(real: eigenvaluesReal[i], imaginary: eigenvaluesImag[i])
            eigenvalues.append(eigenvalue)
        }

        // For now, return empty eigenvectors as they're more complex to extract
        let eigenvectors: [[ComplexNumber]] = []

        return EigenDecomposition(eigenvalues: eigenvalues, eigenvectors: eigenvectors)
    }

    /// Calculate reconstruction error between original and predicted observables
    /// - Parameters:
    ///   - originalObservables: Original observable matrix
    ///   - koopmanMatrix: Computed Koopman matrix
    /// - Returns: Root mean square reconstruction error
    public func calculateReconstructionError(_ originalObservables: [[Double]],
                                           _ koopmanMatrix: [[Double]]) -> Double {
        guard originalObservables.count > 1, !koopmanMatrix.isEmpty else {
            return Double.infinity
        }

        var totalError = 0.0
        var count = 0

        for t in 0..<(originalObservables.count - 1) {
            let original = originalObservables[t + 1]
            let predicted = matrixVectorMultiply(koopmanMatrix, originalObservables[t])

            for i in 0..<min(original.count, predicted.count) {
                let error = original[i] - predicted[i]
                totalError += error * error
                count += 1
            }
        }

        return count > 0 ? sqrt(totalError / Double(count)) : Double.infinity
    }

    // MARK: - Private Methods

    private func validateInputData(_ timeSeriesData: [[Double]], _ observables: [ObservableFunction]) -> Bool {
        guard !timeSeriesData.isEmpty, !observables.isEmpty else {
            logger.log("Empty input data or observables")
            return false
        }

        let stateDimension = timeSeriesData[0].count
        guard stateDimension > 0 else {
            logger.log("Invalid state dimension")
            return false
        }

        // Validate all time steps have same dimension
        for (index, state) in timeSeriesData.enumerated() {
            if state.count != stateDimension {
                logger.log("Inconsistent state dimension at time step \(index)")
                return false
            }
        }

        return true
    }

    private func computeObservableMatrix(_ timeSeriesData: [[Double]],
                                       _ observables: [ObservableFunction]) -> [[Double]] {
        var observableMatrix: [[Double]] = []

        for state in timeSeriesData {
            var observablesAtState: [Double] = []

            for observable in observables {
                let value = observable.function(state)
                observablesAtState.append(value)
            }

            observableMatrix.append(observablesAtState)
        }

        return observableMatrix
    }

    private func calculateStabilityMargin(_ eigenvalues: [ComplexNumber]) -> Double {
        guard !eigenvalues.isEmpty else { return 0.0 }

        var minModulus = Double.infinity

        for eigenvalue in eigenvalues {
            let modulus = eigenvalue.modulus()
            if modulus < minModulus {
                minModulus = modulus
            }
        }

        return minModulus
    }

    private func linearRegression(_ xData: [Double], _ yData: [Double], numFeatures: Int) -> [Double]? {
        guard xData.count == yData.count, xData.count >= numFeatures else {
            return nil
        }

        // Simple least squares solution for small problems
        // For larger problems, consider using Accelerate framework
        let n = xData.count
        var sumX = [Double](repeating: 0.0, count: numFeatures)
        var sumY = 0.0
        var sumXY = [Double](repeating: 0.0, count: numFeatures)
        var sumXX = [Double](repeating: 0.0, count: numFeatures * numFeatures)

        // Build normal equations
        for i in 0..<n {
            let x = xData[i]
            let y = yData[i]
            let featureIndex = i % numFeatures

            sumX[featureIndex] += x
            sumY += y
            sumXY[featureIndex] += x * y

            for j in 0..<numFeatures {
                sumXX[featureIndex * numFeatures + j] += x * xData[(i / numFeatures) * numFeatures + j]
            }
        }

        // Solve normal equations (simplified for diagonal case)
        var coefficients = [Double](repeating: 0.0, count: numFeatures)

        for i in 0..<numFeatures {
            let denominator = sumXX[i * numFeatures + i] - sumX[i] * sumX[i] / Double(n)
            if denominator != 0 {
                coefficients[i] = (sumXY[i] - sumX[i] * sumY / Double(n)) / denominator
            }
        }

        return coefficients
    }

    private func matrixVectorMultiply(_ matrix: [[Double]], _ vector: [Double]) -> [Double] {
        guard !matrix.isEmpty, !vector.isEmpty, matrix[0].count == vector.count else {
            return []
        }

        let rows = matrix.count
        var result = [Double](repeating: 0.0, count: rows)

        for i in 0..<rows {
            for j in 0..<vector.count {
                result[i] += matrix[i][j] * vector[j]
            }
        }

        return result
    }
}

// MARK: - Supporting Types

/// Observable function for Koopman analysis
public struct ObservableFunction {
    public let name: String
    public let function: ([Double]) -> Double
    public let weights: [Double]

    public init(name: String, function: @escaping ([Double]) -> Double, weights: [Double]) {
        self.name = name
        self.function = function
        self.weights = weights
    }
}

/// Results of Koopman analysis
public struct KoopmanAnalysis {
    public let eigenvalues: [ComplexNumber]
    public let stabilityMargin: Double
    public let reconstructionError: Double
    public let koopmanMatrix: [[Double]]
    public let eigenDecomposition: EigenDecomposition?
}

/// Eigenvalue decomposition results
public struct EigenDecomposition {
    public let eigenvalues: [ComplexNumber]
    public let eigenvectors: [[ComplexNumber]]
}

/// Complex number implementation
public struct ComplexNumber {
    public let real: Double
    public let imaginary: Double

    public init(real: Double, imaginary: Double = 0.0) {
        self.real = real
        self.imaginary = imaginary
    }

    public func modulus() -> Double {
        return sqrt(real * real + imaginary * imaginary)
    }

    public func toComplex() -> ComplexNumber {
        return self
    }
}

// MARK: - Utility Classes

private class Logger {
    func log(_ message: String) {
        print("[ReverseKoopmanOperator] \(message)")
    }
}

// MARK: - LAPACK Interface

// External LAPACK function for eigenvalue decomposition
@_silgen_name("dgeev_")
private func dgeev_(
    _ jobvl: UnsafeMutablePointer<CChar>,
    _ jobvr: UnsafeMutablePointer<CChar>,
    _ n: UnsafeMutablePointer<Int32>,
    _ a: UnsafeMutablePointer<Double>,
    _ lda: UnsafeMutablePointer<Int32>,
    _ wr: UnsafeMutablePointer<Double>,
    _ wi: UnsafeMutablePointer<Double>,
    _ vl: UnsafeMutablePointer<Double>?,
    _ ldvl: UnsafeMutablePointer<Int32>,
    _ vr: UnsafeMutablePointer<Double>?,
    _ ldvr: UnsafeMutablePointer<Int32>,
    _ work: UnsafeMutablePointer<Double>,
    _ lwork: UnsafeMutablePointer<Int32>,
    _ info: UnsafeMutablePointer<Int32>
)
