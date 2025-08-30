// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
//
//  ReverseKoopmanOperator.java
//  qualia
//
//  Created by Ryan David Oates on 8/26/25.
//  Implements reverse koopman penetration testing for Java systems
package qualia;

import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;
import java.util.function.Function;
import java.lang.Math;

/**
 * Reverse Koopman Operator for dynamical system analysis
 * Implements reverse engineering of system behavior through observable functions
 */
public class ReverseKoopmanOperator {

    private final int maxEigenfunctions;
    private final double convergenceThreshold;
    private final int maxIterations;
    private final ExecutorService executor;

    public ReverseKoopmanOperator(int maxEigenfunctions, double convergenceThreshold, int maxIterations) {
        this.maxEigenfunctions = maxEigenfunctions;
        this.convergenceThreshold = convergenceThreshold;
        this.maxIterations = maxIterations;
        this.executor = Executors.newCachedThreadPool();
    }

    public ReverseKoopmanOperator() {
        this(50, 1e-10, 1000);
    }

    /**
     * Data structure for observable functions
     */
    public static class ObservableFunction {
        private final String name;
        private final Function<double[], Double> function;
        private final double[] weights;

        public ObservableFunction(String name, Function<double[], Double> function, double[] weights) {
            this.name = name;
            this.function = function;
            this.weights = weights.clone();
        }

        public String getName() { return name; }
        public Function<double[], Double> getFunction() { return function; }
        public double[] getWeights() { return weights.clone(); }
    }

    /**
     * Result of koopman analysis
     */
    public static class KoopmanAnalysis {
        private final List<ComplexNumber> eigenvalues;
        private final List<double[]> eigenfunctions;
        private final List<ObservableFunction> observables;
        private final double[][] koopmanMatrix;
        private final double reconstructionError;
        private final double stabilityMargin;
        private final long computationTime;

        public KoopmanAnalysis(List<ComplexNumber> eigenvalues, List<double[]> eigenfunctions,
                              List<ObservableFunction> observables, double[][] koopmanMatrix,
                              double reconstructionError, double stabilityMargin, long computationTime) {
            this.eigenvalues = new ArrayList<>(eigenvalues);
            this.eigenfunctions = new ArrayList<>(eigenfunctions);
            this.observables = new ArrayList<>(observables);
            this.koopmanMatrix = Arrays.stream(koopmanMatrix)
                .map(double[]::clone)
                .toArray(double[][]::new);
            this.reconstructionError = reconstructionError;
            this.stabilityMargin = stabilityMargin;
            this.computationTime = computationTime;
        }

        // Getters
        public List<ComplexNumber> getEigenvalues() { return new ArrayList<>(eigenvalues); }
        public List<double[]> getEigenfunctions() { return new ArrayList<>(eigenfunctions); }
        public List<ObservableFunction> getObservables() { return new ArrayList<>(observables); }
        public double[][] getKoopmanMatrix() {
            return Arrays.stream(koopmanMatrix).map(double[]::clone).toArray(double[][]::new);
        }
        public double getReconstructionError() { return reconstructionError; }
        public double getStabilityMargin() { return stabilityMargin; }
        public long getComputationTime() { return computationTime; }
    }

    /**
     * Compute reverse koopman operator from time series data
     */
    public CompletableFuture<KoopmanAnalysis> computeReverseKoopman(
            List<double[]> timeSeriesData,
            List<ObservableFunction> observables) {

        return CompletableFuture.supplyAsync(() -> {
            long startTime = System.nanoTime();

            try {
                // Step 1: Construct observable matrix
                double[][] observableMatrix = constructObservableMatrix(timeSeriesData, observables);

                // Step 2: Compute koopman matrix using DMD
                double[][] koopmanMatrix = computeKoopmanMatrix(observableMatrix);

                // Step 3: Compute eigenvalues and eigenfunctions
                EigenDecomposition eigenDecomp = computeEigenDecomposition(koopmanMatrix);

                // Step 4: Validate stability and reconstruction
                double reconstructionError = computeReconstructionError(observableMatrix, koopmanMatrix);
                double stabilityMargin = computeStabilityMargin(eigenDecomp.getEigenvalues());

                long computationTime = System.nanoTime() - startTime;

                return new KoopmanAnalysis(
                    eigenDecomp.getEigenvalues(),
                    eigenDecomp.getEigenfunctions(),
                    observables,
                    koopmanMatrix,
                    reconstructionError,
                    stabilityMargin,
                    computationTime
                );

            } catch (Exception e) {
                throw new RuntimeException("Failed to compute reverse koopman operator: " + e.getMessage(), e);
            }
        }, executor);
    }

    /**
     * Construct observable matrix from time series data
     */
    private double[][] constructObservableMatrix(List<double[]> timeSeriesData, List<ObservableFunction> observables) {
        int numSamples = timeSeriesData.size();
        int numObservables = observables.size();
        double[][] matrix = new double[numSamples][numObservables];

        for (int i = 0; i < numSamples; i++) {
            double[] state = timeSeriesData.get(i);
            for (int j = 0; j < numObservables; j++) {
                try {
                    matrix[i][j] = observables.get(j).getFunction().apply(state);
                } catch (Exception e) {
                    // If observable function fails, use 0.0 as default
                    matrix[i][j] = 0.0;
                }
            }
        }

        return matrix;
    }

    /**
     * Compute koopman matrix using Dynamic Mode Decomposition (DMD)
     */
    private double[][] computeKoopmanMatrix(double[][] observableMatrix) {
        int numSamples = observableMatrix.length;
        int numObservables = observableMatrix[0].length;

        // Split into current and next states
        double[][] X = new double[numObservables][numSamples - 1];
        double[][] Xp = new double[numObservables][numSamples - 1];

        for (int i = 0; i < numSamples - 1; i++) {
            for (int j = 0; j < numObservables && j < observableMatrix[i].length; j++) {
                X[j][i] = observableMatrix[i][j];
                if (i + 1 < observableMatrix.length && j < observableMatrix[i + 1].length) {
                    Xp[j][i] = observableMatrix[i + 1][j];
                }
            }
        }

        // Compute SVD of X
        SingularValueDecomposition svd = new SingularValueDecomposition(X);

        // Compute koopman matrix using pseudo-inverse
        double[][] U = svd.getU();
        double[][] S = svd.getS();
        double[][] Vt = svd.getVt();

        // Compute pseudo-inverse of X
        double[][] Xinv = computePseudoInverse(U, S, Vt);

        // Compute koopman matrix: K = Xp * Xinv
        double[][] koopmanMatrix = matrixMultiply(Xp, Xinv);

        return koopmanMatrix;
    }

    /**
     * Compute eigen decomposition of koopman matrix
     */
    private EigenDecomposition computeEigenDecomposition(double[][] matrix) {
        // Use power iteration method for eigenvalue decomposition
        List<ComplexNumber> eigenvalues = new ArrayList<>();
        List<double[]> eigenfunctions = new ArrayList<>();

        // For simplicity, implement basic power iteration
        // In practice, you'd want to use more sophisticated methods

        for (int i = 0; i < Math.min(maxEigenfunctions, matrix.length); i++) {
            double[] eigenvector = powerIteration(matrix, maxIterations, convergenceThreshold);
            ComplexNumber eigenvalue = rayleighQuotient(matrix, eigenvector);

            eigenvalues.add(eigenvalue);
            eigenfunctions.add(normalizeVector(eigenvector));
        }

        return new EigenDecomposition(eigenvalues, eigenfunctions);
    }

    /**
     * Power iteration method for finding dominant eigenvector
     */
    private double[] powerIteration(double[][] matrix, int maxIter, double threshold) {
        int n = matrix.length;
        double[] vector = new double[n];

        // Initialize with random vector
        Random rand = new Random();
        for (int i = 0; i < n; i++) {
            vector[i] = rand.nextDouble() - 0.5;
        }

        for (int iter = 0; iter < maxIter; iter++) {
            double[] newVector = matrixMultiply(matrix, vector);
            double norm = vectorNorm(newVector);

            if (norm < threshold) break;

            vector = vectorDivide(newVector, norm);
        }

        return vector;
    }

    /**
     * Compute Rayleigh quotient for eigenvalue approximation
     */
    private ComplexNumber rayleighQuotient(double[][] matrix, double[] vector) {
        double[] mv = matrixMultiply(matrix, vector);
        double numerator = dotProduct(vector, mv);
        double denominator = dotProduct(vector, vector);

        return new ComplexNumber(numerator / denominator, 0.0);
    }

    /**
     * Compute reconstruction error
     */
    private double computeReconstructionError(double[][] original, double[][] koopman) {
        double totalError = 0.0;
        int count = 0;

        for (int i = 0; i < original.length - 1; i++) {
            double[] predicted = matrixMultiply(koopman, original[i]);
            double[] actual = original[i + 1];

            for (int j = 0; j < predicted.length; j++) {
                double diff = predicted[j] - actual[j];
                totalError += diff * diff;
                count++;
            }
        }

        return Math.sqrt(totalError / count);
    }

    /**
     * Compute stability margin from eigenvalues
     */
    private double computeStabilityMargin(List<ComplexNumber> eigenvalues) {
        double minStability = Double.MAX_VALUE;

        for (ComplexNumber eigen : eigenvalues) {
            double magnitude = eigen.magnitude();
            if (magnitude > 0) {
                double stability = 1.0 / magnitude;
                minStability = Math.min(minStability, stability);
            }
        }

        return minStability;
    }

    // Utility methods for matrix operations
    private double[][] matrixMultiply(double[][] a, double[][] b) {
        int rows = a.length;
        int cols = b[0].length;
        int common = a[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                for (int k = 0; k < common; k++) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }

        return result;
    }

    private double[] matrixMultiply(double[][] matrix, double[] vector) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[] result = new double[rows];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i] += matrix[i][j] * vector[j];
            }
        }

        return result;
    }

    private double dotProduct(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    private double vectorNorm(double[] vector) {
        return Math.sqrt(dotProduct(vector, vector));
    }

    private double[] normalizeVector(double[] vector) {
        double norm = vectorNorm(vector);
        if (norm < 1e-10) return vector;

        double[] normalized = new double[vector.length];
        for (int i = 0; i < vector.length; i++) {
            normalized[i] = vector[i] / norm;
        }
        return normalized;
    }

    private double[] vectorDivide(double[] vector, double scalar) {
        double[] result = new double[vector.length];
        for (int i = 0; i < vector.length; i++) {
            result[i] = vector[i] / scalar;
        }
        return result;
    }

    private double[][] computePseudoInverse(double[][] U, double[][] S, double[][] Vt) {
        // Simplified pseudo-inverse computation
        // In practice, you'd want more robust implementation
        int m = U.length;
        int n = Vt[0].length;
        double[][] pseudoInv = new double[n][m];

        // For diagonal S matrix, compute S^+
        for (int i = 0; i < Math.min(m, n); i++) {
            if (Math.abs(S[i][i]) > 1e-10) {
                S[i][i] = 1.0 / S[i][i];
            }
        }

        // Compute V * S^+ * U^T
        return matrixMultiply(Vt, matrixMultiply(S, transpose(U)));
    }

    private double[][] transpose(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] transposed = new double[cols][rows];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed[j][i] = matrix[i][j];
            }
        }

        return transposed;
    }

    /**
     * Simplified SVD implementation (placeholder)
     * In practice, use a proper linear algebra library
     */
    private static class SingularValueDecomposition {
        private final double[][] U;
        private final double[][] S;
        private final double[][] Vt;

        public SingularValueDecomposition(double[][] matrix) {
            // Simplified implementation - use proper SVD library in production
            int m = matrix.length;
            int n = matrix[0].length;

            this.U = new double[m][Math.min(m, n)];
            this.S = new double[Math.min(m, n)][Math.min(m, n)];
            this.Vt = new double[Math.min(m, n)][n];

            // Initialize with identity matrices for demonstration
            for (int i = 0; i < Math.min(m, n); i++) {
                S[i][i] = 1.0;
                if (i < m) U[i][i] = 1.0;
                if (i < n) Vt[i][i] = 1.0;
            }
        }

        public double[][] getU() { return U; }
        public double[][] getS() { return S; }
        public double[][] getVt() { return Vt; }
    }

    /**
     * Eigen decomposition result
     */
    private static class EigenDecomposition {
        private final List<ComplexNumber> eigenvalues;
        private final List<double[]> eigenfunctions;

        public EigenDecomposition(List<ComplexNumber> eigenvalues, List<double[]> eigenfunctions) {
            this.eigenvalues = eigenvalues;
            this.eigenfunctions = eigenfunctions;
        }

        public List<ComplexNumber> getEigenvalues() { return eigenvalues; }
        public List<double[]> getEigenfunctions() { return eigenfunctions; }
    }

    /**
     * Shutdown executor service
     */
    public void shutdown() {
        executor.shutdown();
        try {
            if (!executor.awaitTermination(5, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
}
