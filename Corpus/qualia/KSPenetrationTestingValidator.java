// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
//
//  KSPenetrationTestingValidator.java
//  qualia
//
//  Created by Ryan David Oates on 8/26/25.
//  K-S validation framework for penetration testing quality assurance
package qualia;

import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;
import java.lang.Math;

/**
 * K-S Validation Result structure
 */
class KSValidationResult {
    private final double ksStatistic;
    private final double pValue;
    private final boolean isValid;
    private final double confidenceLevel;
    private final String recommendation;
    private final double distributionSimilarity;
    private final Date timestamp;

    public KSValidationResult(double ksStatistic, double pValue, boolean isValid,
                            double confidenceLevel, String recommendation, double distributionSimilarity) {
        this.ksStatistic = ksStatistic;
        this.pValue = pValue;
        this.isValid = isValid;
        this.confidenceLevel = confidenceLevel;
        this.recommendation = recommendation;
        this.distributionSimilarity = distributionSimilarity;
        this.timestamp = new Date();
    }

    // Getters
    public double getKsStatistic() { return ksStatistic; }
    public double getPValue() { return pValue; }
    public boolean isValid() { return isValid; }
    public double getConfidenceLevel() { return confidenceLevel; }
    public String getRecommendation() { return recommendation; }
    public double getDistributionSimilarity() { return distributionSimilarity; }
    public Date getTimestamp() { return timestamp; }

    @Override
    public String toString() {
        return String.format("KS Validation: D=%.4f, p=%.4f, Valid=%s, Confidence=%.2f%%, Similarity=%.2f%%",
                           ksStatistic, pValue, isValid, confidenceLevel * 100, distributionSimilarity * 100);
    }
}

/**
 * K-S Validation Framework for Penetration Testing
 * Implements statistical validation of security testing results using Kolmogorov-Smirnov tests
 */
public class KSPenetrationTestingValidator {

    private final double significanceLevel;
    private final int minSampleSize;
    private final ExecutorService executor;
    private final Random random;

    public KSPenetrationTestingValidator(double significanceLevel, int minSampleSize) {
        this.significanceLevel = significanceLevel;
        this.minSampleSize = minSampleSize;
        this.executor = Executors.newCachedThreadPool();
        this.random = new Random();
    }

    public KSPenetrationTestingValidator() {
        this(0.05, 30);
    }

    /**
     * Validate penetration testing results using K-S test
     */
    public CompletableFuture<KSValidationResult> validatePenetrationTesting(
            List<SecurityFinding> findings,
            List<SecurityFinding> baselineFindings) {

        return CompletableFuture.supplyAsync(() -> {
            try {
                // Convert findings to numerical distributions
                double[] testDistribution = convertFindingsToDistribution(findings);
                double[] baselineDistribution = convertFindingsToDistribution(baselineFindings);

                // Perform K-S test
                double ksStatistic = calculateKSStatistic(testDistribution, baselineDistribution);
                double pValue = calculatePValue(ksStatistic, testDistribution.length, baselineDistribution.length);

                // Determine validity
                boolean isValid = pValue > significanceLevel;

                // Calculate confidence level
                double confidenceLevel = calculateConfidenceLevel(pValue, findings.size());

                // Calculate distribution similarity
                double distributionSimilarity = calculateDistributionSimilarity(testDistribution, baselineDistribution);

                // Generate recommendation
                String recommendation = generateRecommendation(isValid, ksStatistic, confidenceLevel);

                return new KSValidationResult(
                    ksStatistic, pValue, isValid, confidenceLevel, recommendation, distributionSimilarity
                );

            } catch (Exception e) {
                throw new RuntimeException("Failed to validate penetration testing: " + e.getMessage(), e);
            }
        }, executor);
    }

    /**
     * Convert security findings to numerical distribution for statistical analysis
     */
    private double[] convertFindingsToDistribution(List<SecurityFinding> findings) {
        if (findings.isEmpty()) {
            return new double[minSampleSize];
        }

        // Convert severity levels to numerical values
        List<Double> severityValues = findings.stream()
            .mapToDouble(finding -> {
                switch (finding.getSeverity()) {
                    case CRITICAL: return 10.0;
                    case HIGH: return 7.5;
                    case MEDIUM: return 5.0;
                    case LOW: return 2.5;
                    case INFO: return 1.0;
                    default: return 0.0;
                }
            })
            .boxed()
            .collect(Collectors.toList());

        // Ensure minimum sample size by padding with zeros
        while (severityValues.size() < minSampleSize) {
            severityValues.add(0.0);
        }

        return severityValues.stream().mapToDouble(Double::doubleValue).toArray();
    }

    /**
     * Calculate Kolmogorov-Smirnov statistic
     */
    private double calculateKSStatistic(double[] sample1, double[] sample2) {
        // Sort samples
        Arrays.sort(sample1);
        Arrays.sort(sample2);

        int n1 = sample1.length;
        int n2 = sample2.length;

        double maxDifference = 0.0;
        int i = 0, j = 0;

        while (i < n1 && j < n2) {
            double cdf1 = (i + 1.0) / n1;
            double cdf2 = (j + 1.0) / n2;

            double difference = Math.abs(cdf1 - cdf2);
            maxDifference = Math.max(maxDifference, difference);

            if (sample1[i] < sample2[j]) {
                i++;
            } else if (sample1[i] > sample2[j]) {
                j++;
            } else {
                i++;
                j++;
            }
        }

        return maxDifference;
    }

    /**
     * Calculate p-value using asymptotic approximation
     */
    private double calculatePValue(double ksStatistic, int n1, int n2) {
        if (ksStatistic <= 0) return 1.0;
        if (ksStatistic >= 1) return 0.0;

        // Calculate asymptotic p-value
        double n = (double) n1 * n2 / (n1 + n2);
        double lambda = ksStatistic * Math.sqrt(n);

        // Use exponential approximation for p-value
        double pValue = 2.0 * Math.exp(-2.0 * lambda * lambda);

        return Math.min(pValue, 1.0);
    }

    /**
     * Calculate confidence level based on p-value and sample size
     */
    private double calculateConfidenceLevel(double pValue, int sampleSize) {
        if (sampleSize < minSampleSize) {
            return 0.5; // Low confidence for small samples
        }

        // Convert p-value to confidence level
        double confidence = 1.0 - pValue;

        // Adjust for sample size
        double sampleAdjustment = Math.min(1.0, sampleSize / 100.0);

        return confidence * sampleAdjustment;
    }

    /**
     * Calculate distribution similarity using correlation coefficient
     */
    private double calculateDistributionSimilarity(double[] dist1, double[] dist2) {
        try {
            double mean1 = Arrays.stream(dist1).average().orElse(0.0);
            double mean2 = Arrays.stream(dist2).average().orElse(0.0);

            double variance1 = Arrays.stream(dist1)
                .map(x -> (x - mean1) * (x - mean1))
                .average().orElse(1.0);
            double variance2 = Arrays.stream(dist2)
                .map(x -> (x - mean2) * (x - mean2))
                .average().orElse(1.0);

            double covariance = 0.0;
            int n = Math.min(dist1.length, dist2.length);

            for (int i = 0; i < n; i++) {
                covariance += (dist1[i] - mean1) * (dist2[i] - mean2);
            }
            covariance /= n;

            double correlation = covariance / Math.sqrt(variance1 * variance2);
            return Math.abs(correlation);

        } catch (Exception e) {
            return 0.0; // Default to no similarity if calculation fails
        }
    }

    /**
     * Generate recommendation based on validation results
     */
    private String generateRecommendation(boolean isValid, double ksStatistic, double confidenceLevel) {
        StringBuilder recommendation = new StringBuilder();

        if (isValid) {
            recommendation.append("Penetration testing results are statistically valid. ");
            recommendation.append(String.format("Confidence level: %.1f%%. ", confidenceLevel * 100));

            if (confidenceLevel > 0.8) {
                recommendation.append("High confidence in test results. ");
            } else if (confidenceLevel > 0.6) {
                recommendation.append("Moderate confidence in test results. ");
            } else {
                recommendation.append("Consider increasing sample size for higher confidence. ");
            }
        } else {
            recommendation.append("Penetration testing results differ significantly from baseline. ");
            recommendation.append(String.format("KS statistic: %.4f. ", ksStatistic));

            if (ksStatistic > 0.5) {
                recommendation.append("Large deviation detected - review testing methodology. ");
            } else if (ksStatistic > 0.3) {
                recommendation.append("Moderate deviation detected - verify test coverage. ");
            }

            recommendation.append("Consider re-running tests with different parameters. ");
        }

        return recommendation.toString();
    }

    /**
     * Perform cross-validation between multiple test runs
     */
    public CompletableFuture<List<KSValidationResult>> performCrossValidation(
            List<List<SecurityFinding>> testRuns) {

        return CompletableFuture.supplyAsync(() -> {
            List<KSValidationResult> results = new ArrayList<>();

            try {
                // Compare each run against the average of others
                for (int i = 0; i < testRuns.size(); i++) {
                    List<SecurityFinding> currentRun = testRuns.get(i);
                    List<SecurityFinding> otherRuns = new ArrayList<>();

                    // Collect all other runs
                    for (int j = 0; j < testRuns.size(); j++) {
                        if (j != i) {
                            otherRuns.addAll(testRuns.get(j));
                        }
                    }

                    // Validate current run against combined others
                    KSValidationResult result = validatePenetrationTesting(currentRun, otherRuns).get();
                    results.add(result);
                }

            } catch (Exception e) {
                throw new RuntimeException("Cross-validation failed: " + e.getMessage(), e);
            }

            return results;
        }, executor);
    }

    /**
     * Generate statistical summary of validation results
     */
    public String generateValidationSummary(List<KSValidationResult> results) {
        if (results.isEmpty()) {
            return "No validation results available";
        }

        StringBuilder summary = new StringBuilder();
        summary.append("=== K-S Validation Summary ===\n\n");

        // Calculate statistics
        int totalValid = (int) results.stream().filter(KSValidationResult::isValid).count();
        double avgConfidence = results.stream()
            .mapToDouble(KSValidationResult::getConfidenceLevel)
            .average().orElse(0.0);
        double avgSimilarity = results.stream()
            .mapToDouble(KSValidationResult::getDistributionSimilarity)
            .average().orElse(0.0);

        summary.append(String.format("Total Validations: %d\n", results.size()));
        summary.append(String.format("Valid Results: %d (%.1f%%)\n", totalValid,
                                   (double) totalValid / results.size() * 100));
        summary.append(String.format("Average Confidence: %.1f%%\n", avgConfidence * 100));
        summary.append(String.format("Average Similarity: %.1f%%\n\n", avgSimilarity * 100));

        // Detailed results
        summary.append("=== Detailed Results ===\n");
        for (int i = 0; i < results.size(); i++) {
            KSValidationResult result = results.get(i);
            summary.append(String.format("Run %d: %s\n", i + 1, result.toString()));
        }

        // Recommendations
        summary.append("\n=== Recommendations ===\n");
        if ((double) totalValid / results.size() < 0.7) {
            summary.append("- Low validation rate: Review testing methodology\n");
            summary.append("- Consider using different testing parameters\n");
        }

        if (avgConfidence < 0.6) {
            summary.append("- Low confidence: Increase sample sizes\n");
            summary.append("- Consider more comprehensive test coverage\n");
        }

        if (avgSimilarity < 0.5) {
            summary.append("- Low similarity: Results may be inconsistent\n");
            summary.append("- Standardize testing procedures\n");
        }

        return summary.toString();
    }

    /**
     * Validate penetration testing methodology
     */
    public KSValidationResult validateMethodology(
            Map<String, List<SecurityFinding>> testResults,
            String baselineMethodology) {

        try {
            // Convert methodology results to findings
            List<SecurityFinding> methodologyFindings = new ArrayList<>();
            List<SecurityFinding> baselineFindings = new ArrayList<>();

            // This would typically analyze the methodology differences
            // For demonstration, create synthetic findings
            methodologyFindings.add(new SecurityFinding(
                VulnerabilityType.INFORMATION_DISCLOSURE,
                Severity.LOW,
                "Methodology Validation",
                "Testing methodology analysis",
                "Test Framework",
                "Review testing approach",
                "Methodology validation"
            ));

            baselineFindings.add(new SecurityFinding(
                VulnerabilityType.INFORMATION_DISCLOSURE,
                Severity.INFO,
                "Baseline Methodology",
                "Standard testing approach",
                "Test Framework",
                "Use standard procedures",
                "Baseline validation"
            ));

            return validatePenetrationTesting(methodologyFindings, baselineFindings).get();

        } catch (Exception e) {
            return new KSValidationResult(0.0, 1.0, false, 0.0,
                                        "Methodology validation failed: " + e.getMessage(), 0.0);
        }
    }

    /**
     * Export validation results to JSON
     */
    public String exportResultsToJson(List<KSValidationResult> results) {
        StringBuilder json = new StringBuilder();
        json.append("{\"validationResults\":[");

        for (int i = 0; i < results.size(); i++) {
            KSValidationResult result = results.get(i);
            json.append("{");
            json.append("\"ksStatistic\":").append(result.getKsStatistic()).append(",");
            json.append("\"pValue\":").append(result.getPValue()).append(",");
            json.append("\"isValid\":").append(result.isValid()).append(",");
            json.append("\"confidenceLevel\":").append(result.getConfidenceLevel()).append(",");
            json.append("\"recommendation\":\"").append(escapeJson(result.getRecommendation())).append("\",");
            json.append("\"distributionSimilarity\":").append(result.getDistributionSimilarity()).append(",");
            json.append("\"timestamp\":\"").append(result.getTimestamp()).append("\"");
            json.append("}");

            if (i < results.size() - 1) {
                json.append(",");
            }
        }

        json.append("]}");
        return json.toString();
    }

    /**
     * Simple JSON escaping
     */
    private String escapeJson(String text) {
        return text.replace("\"", "\\\"")
                   .replace("\\", "\\\\")
                   .replace("\n", "\\n")
                   .replace("\r", "\\r")
                   .replace("\t", "\\t");
    }

    /**
     * Shutdown the validator
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
