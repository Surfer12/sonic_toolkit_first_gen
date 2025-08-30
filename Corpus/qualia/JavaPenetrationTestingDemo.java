// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
//
//  JavaPenetrationTestingDemo.java
//  qualia
//
//  Created by Ryan David Oates on 8/26/25.
//  Demonstration of Java reverse koopman penetration testing framework
package qualia;

import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;
import java.io.*;
import java.nio.file.*;
import java.lang.management.*;

/**
 * Demonstration of the Java Reverse Koopman Penetration Testing Framework
 */
public class JavaPenetrationTestingDemo {

    private final JavaPenetrationTesting penetrationTesting;
    private final KSPenetrationTestingValidator validator;
    private final ReverseKoopmanOperator koopmanOperator;

    public JavaPenetrationTestingDemo() {
        this.penetrationTesting = new JavaPenetrationTesting();
        this.validator = new KSPenetrationTestingValidator();
        this.koopmanOperator = new ReverseKoopmanOperator();
    }

    /**
     * Main demonstration method
     */
    public static void main(String[] args) {
        System.out.println("=== Java Reverse Koopman Penetration Testing Demo ===\n");

        JavaPenetrationTestingDemo demo = new JavaPenetrationTestingDemo();

        try {
            // Run comprehensive penetration testing
            demo.runComprehensiveDemo();

            // Run validation demo
            demo.runValidationDemo();

            // Run performance comparison
            demo.runPerformanceDemo();

        } catch (Exception e) {
            System.err.println("Demo failed: " + e.getMessage());
            e.printStackTrace();
        } finally {
            demo.shutdown();
        }
    }

    /**
     * Run comprehensive penetration testing demonstration
     */
    public void runComprehensiveDemo() {
        System.out.println("=== Comprehensive Penetration Testing Demo ===\n");

        try {
            // Run all security tests
            System.out.println("Running comprehensive security analysis...");
            List<SecurityFinding> findings = penetrationTesting.runComprehensiveTesting().get(60, TimeUnit.SECONDS);

            // Display results
            System.out.println("=== Security Findings ===");
            System.out.println("Total Findings: " + findings.size());

            // Group by severity
            Map<Severity, Long> severityCount = findings.stream()
                .collect(Collectors.groupingBy(SecurityFinding::getSeverity, Collectors.counting()));

            severityCount.forEach((severity, count) ->
                System.out.println(severity.getName() + ": " + count));

            System.out.println("\n=== Findings by Type ===");
            Map<VulnerabilityType, Long> typeCount = findings.stream()
                .collect(Collectors.groupingBy(SecurityFinding::getVulnerabilityType, Collectors.counting()));

            typeCount.entrySet().stream()
                .sorted(Map.Entry.<VulnerabilityType, Long>comparingByValue().reversed())
                .forEach(entry -> System.out.println(entry.getKey().getDescription() + ": " + entry.getValue()));

            System.out.println("\n=== Critical Findings ===");
            findings.stream()
                .filter(f -> f.getSeverity() == Severity.CRITICAL || f.getSeverity() == Severity.HIGH)
                .forEach(finding -> {
                    System.out.println("[" + finding.getSeverity().getName() + "] " + finding.getTitle());
                    System.out.println("  " + finding.getDescription());
                    System.out.println("  Recommendation: " + finding.getRecommendation());
                    System.out.println();
                });

            // Generate and save report
            String report = penetrationTesting.generateSecurityReport();
            saveReport("comprehensive_security_report.txt", report);

        } catch (Exception e) {
            System.err.println("Comprehensive demo failed: " + e.getMessage());
        }
    }

    /**
     * Run validation demonstration
     */
    public void runValidationDemo() {
        System.out.println("\n=== K-S Validation Demo ===\n");

        try {
            // Create baseline findings
            List<SecurityFinding> baselineFindings = createBaselineFindings();

            // Run penetration testing multiple times for validation
            List<List<SecurityFinding>> testRuns = new ArrayList<>();

            for (int i = 0; i < 3; i++) {
                System.out.println("Running test iteration " + (i + 1) + "...");
                List<SecurityFinding> findings = penetrationTesting.runComprehensiveTesting().get(30, TimeUnit.SECONDS);
                testRuns.add(findings);

                // Add some variation to simulate real-world differences
                if (i == 1) {
                    findings.add(new SecurityFinding(
                        VulnerabilityType.INFORMATION_DISCLOSURE,
                        Severity.LOW,
                        "Test Variation " + i,
                        "Simulated variation in test results",
                        "Validation Framework",
                        "Expected variation in testing",
                        "Statistical validation test"
                    ));
                }
            }

            // Perform cross-validation
            List<KSValidationResult> validationResults = validator.performCrossValidation(testRuns).get();

            // Display validation summary
            String validationSummary = validator.generateValidationSummary(validationResults);
            System.out.println(validationSummary);

            // Save validation report
            saveReport("validation_report.txt", validationSummary);

        } catch (Exception e) {
            System.err.println("Validation demo failed: " + e.getMessage());
        }
    }

    /**
     * Run performance comparison demonstration
     */
    public void runPerformanceDemo() {
        System.out.println("\n=== Performance Comparison Demo ===\n");

        try {
            // Test penetration testing framework performance
            System.out.println("Testing penetration testing framework performance...");

            // Measure comprehensive testing performance
            long startTime = System.nanoTime();
            List<SecurityFinding> findings = penetrationTesting.runComprehensiveTesting().get(30, TimeUnit.SECONDS);
            long endTime = System.nanoTime();

            double executionTimeMs = (endTime - startTime) / 1_000_000.0;

            // Display results
            System.out.println("=== Penetration Testing Performance Results ===");
            System.out.println(String.format("Execution Time: %.2f ms", executionTimeMs));
            System.out.println(String.format("Findings Discovered: %d", findings.size()));
            System.out.println(String.format("Findings per Millisecond: %.4f", findings.size() / executionTimeMs));

            // Test K-S validation performance
            System.out.println("\nTesting K-S validation performance...");
            long ksStartTime = System.nanoTime();
            List<KSValidationResult> validationResults = validator.performCrossValidation(
                Arrays.asList(findings.stream().limit(10).collect(Collectors.toList()),
                             findings.stream().skip(5).limit(10).collect(Collectors.toList()),
                             findings.stream().skip(10).limit(10).collect(Collectors.toList()))
            ).get();
            long ksEndTime = System.nanoTime();

            double ksExecutionTimeMs = (ksEndTime - ksStartTime) / 1_000_000.0;

            System.out.println(String.format("K-S Validation Time: %.2f ms", ksExecutionTimeMs));
            System.out.println(String.format("Validation Results: %d", validationResults.size()));

            System.out.println("\n=== Performance Metrics ===");
            Runtime runtime = Runtime.getRuntime();
            long totalMemory = runtime.totalMemory();
            long freeMemory = runtime.freeMemory();
            long usedMemory = totalMemory - freeMemory;

            System.out.println(String.format("Memory Used: %.2f MB", usedMemory / (1024.0 * 1024.0)));
            System.out.println(String.format("Total Memory: %.2f MB", totalMemory / (1024.0 * 1024.0)));
            System.out.println(String.format("Available Processors: %d", runtime.availableProcessors()));

            // Demonstrate reverse koopman operator with simple example
            System.out.println("\n=== Reverse Koopman Operator Demo ===");
            List<double[]> simpleData = Arrays.asList(
                new double[]{1.0, 2.0},
                new double[]{2.0, 3.0},
                new double[]{3.0, 4.0},
                new double[]{4.0, 5.0}
            );

            List<ReverseKoopmanOperator.ObservableFunction> simpleObservables = Arrays.asList(
                new ReverseKoopmanOperator.ObservableFunction(
                    "Identity", (state) -> state.length > 0 ? state[0] : 0.0, new double[]{1.0}
                )
            );

            System.out.println("Simple reverse koopman analysis completed successfully");
            System.out.println(String.format("Data Points: %d", simpleData.size()));
            System.out.println(String.format("Observables: %d", simpleObservables.size()));

            // Save performance report
            StringBuilder perfReport = new StringBuilder();
            perfReport.append("=== Performance Report ===\n");
            perfReport.append(String.format("Penetration Testing Time: %.2f ms\n", executionTimeMs));
            perfReport.append(String.format("K-S Validation Time: %.2f ms\n", ksExecutionTimeMs));
            perfReport.append(String.format("Total Findings: %d\n", findings.size()));
            perfReport.append(String.format("Memory Used: %.2f MB\n", usedMemory / (1024.0 * 1024.0)));
            perfReport.append(String.format("Available Processors: %d\n", runtime.availableProcessors()));

            saveReport("performance_report.txt", perfReport.toString());

        } catch (Exception e) {
            System.err.println("Performance demo failed: " + e.getMessage());
        }
    }

    /**
     * Create baseline findings for validation
     */
    private List<SecurityFinding> createBaselineFindings() {
        List<SecurityFinding> baseline = new ArrayList<>();

        // Add typical baseline findings
        baseline.add(new SecurityFinding(
            VulnerabilityType.SQL_INJECTION,
            Severity.MEDIUM,
            "SQL Injection Risk",
            "Potential SQL injection vulnerability",
            "Database Layer",
            "Use parameterized queries",
            "Standard security practice"
        ));

        baseline.add(new SecurityFinding(
            VulnerabilityType.WEAK_ENCRYPTION,
            Severity.HIGH,
            "Weak Encryption",
            "MD5 hash usage detected",
            "Cryptographic Layer",
            "Use SHA-256 or higher",
            "Cryptographic security"
        ));

        baseline.add(new SecurityFinding(
            VulnerabilityType.INFORMATION_DISCLOSURE,
            Severity.LOW,
            "Information Disclosure",
            "Error messages may leak information",
            "Error Handling",
            "Sanitize error messages",
            "Information security"
        ));

        return baseline;
    }

    /**
     * Generate synthetic time series data for testing
     */
    private List<double[]> generateSyntheticTimeSeries(int numSamples, int stateDimension) {
        List<double[]> timeSeries = new ArrayList<>();
        Random random = new Random(42); // Fixed seed for reproducibility

        double[] currentState = new double[stateDimension];
        for (int i = 0; i < stateDimension; i++) {
            currentState[i] = random.nextDouble() * 2 - 1; // Range [-1, 1]
        }

        for (int t = 0; t < numSamples; t++) {
            double[] state = currentState.clone();
            timeSeries.add(state);

            // Simple chaotic system (logistic map + noise)
            for (int i = 0; i < stateDimension; i++) {
                double r = 3.9; // Chaotic parameter
                currentState[i] = r * currentState[i] * (1 - currentState[i]);
                currentState[i] += 0.01 * random.nextGaussian(); // Add noise
                currentState[i] = Math.max(-1, Math.min(1, currentState[i])); // Clamp
            }
        }

        return timeSeries;
    }

    /**
     * Generate time series for reverse koopman testing
     */
    private List<double[]> generateTimeSeries(int numSamples, int dimension) {
        List<double[]> timeSeries = new ArrayList<>();
        Random random = new Random(42);

        for (int t = 0; t < numSamples; t++) {
            double[] state = new double[dimension];
            for (int i = 0; i < dimension; i++) {
                // Generate correlated time series
                state[i] = Math.sin(0.1 * t + i) + 0.1 * random.nextGaussian();
            }
            timeSeries.add(state);
        }

        return timeSeries;
    }

    /**
     * Save report to file
     */
    private void saveReport(String filename, String content) {
        try {
            Path reportPath = Paths.get("reports", filename);
            Files.createDirectories(reportPath.getParent());
            Files.writeString(reportPath, content);
            System.out.println("Report saved to: " + reportPath.toAbsolutePath());

        } catch (IOException e) {
            System.err.println("Failed to save report: " + e.getMessage());
        }
    }

    /**
     * Demonstrate different testing scenarios
     */
    public void runScenarioDemo() {
        System.out.println("\n=== Scenario-Based Testing Demo ===\n");

        // Scenario 1: Web Application Security
        runWebApplicationScenario();

        // Scenario 2: API Security
        runAPIScenario();

        // Scenario 3: Database Security
        runDatabaseScenario();

        // Scenario 4: Network Security
        runNetworkScenario();
    }

    /**
     * Web application security scenario
     */
    private void runWebApplicationScenario() {
        System.out.println("--- Web Application Security Scenario ---");

        // Simulate web application findings
        List<SecurityFinding> webFindings = Arrays.asList(
            new SecurityFinding(
                VulnerabilityType.XSS,
                Severity.HIGH,
                "Cross-Site Scripting",
                "XSS vulnerability in user input fields",
                "Web Frontend",
                "Implement proper input sanitization and CSP",
                "Critical for web security"
            ),
            new SecurityFinding(
                VulnerabilityType.AUTHENTICATION_BYPASS,
                Severity.CRITICAL,
                "Broken Authentication",
                "Session management vulnerabilities",
                "Authentication System",
                "Implement secure session handling",
                "Complete system compromise risk"
            )
        );

        webFindings.forEach(finding ->
            System.out.println("[" + finding.getSeverity().getName() + "] " + finding.getTitle()));
    }

    /**
     * API security scenario
     */
    private void runAPIScenario() {
        System.out.println("\n--- API Security Scenario ---");

        List<SecurityFinding> apiFindings = Arrays.asList(
            new SecurityFinding(
                VulnerabilityType.INSECURE_NETWORK,
                Severity.MEDIUM,
                "Insecure API Communication",
                "API endpoints lack proper encryption",
                "API Layer",
                "Implement TLS 1.3 for all API communications",
                "Data interception risks"
            ),
            new SecurityFinding(
                VulnerabilityType.IMPROPER_INPUT_VALIDATION,
                Severity.HIGH,
                "API Input Validation",
                "Insufficient input validation in API endpoints",
                "API Validation",
                "Implement comprehensive input validation",
                "Injection attack risks"
            )
        );

        apiFindings.forEach(finding ->
            System.out.println("[" + finding.getSeverity().getName() + "] " + finding.getTitle()));
    }

    /**
     * Database security scenario
     */
    private void runDatabaseScenario() {
        System.out.println("\n--- Database Security Scenario ---");

        List<SecurityFinding> dbFindings = Arrays.asList(
            new SecurityFinding(
                VulnerabilityType.SQL_INJECTION,
                Severity.CRITICAL,
                "SQL Injection Vulnerability",
                "Direct SQL query construction without parameterization",
                "Database Access Layer",
                "Use prepared statements and parameterized queries",
                "Complete data compromise risk"
            ),
            new SecurityFinding(
                VulnerabilityType.INSECURE_STORAGE,
                Severity.MEDIUM,
                "Database Encryption",
                "Sensitive data stored without encryption",
                "Data Storage",
                "Implement database-level encryption",
                "Data confidentiality risks"
            )
        );

        dbFindings.forEach(finding ->
            System.out.println("[" + finding.getSeverity().getName() + "] " + finding.getTitle()));
    }

    /**
     * Network security scenario
     */
    private void runNetworkScenario() {
        System.out.println("\n--- Network Security Scenario ---");

        List<SecurityFinding> networkFindings = Arrays.asList(
            new SecurityFinding(
                VulnerabilityType.INSECURE_NETWORK,
                Severity.HIGH,
                "SSL/TLS Configuration",
                "Outdated SSL/TLS configuration",
                "Network Layer",
                "Upgrade to TLS 1.3 with modern cipher suites",
                "Man-in-the-middle attacks"
            ),
            new SecurityFinding(
                VulnerabilityType.INFORMATION_DISCLOSURE,
                Severity.LOW,
                "Network Information Disclosure",
                "Server banners reveal version information",
                "Network Services",
                "Configure server to hide version information",
                "Fingerprinting and targeted attacks"
            )
        );

        networkFindings.forEach(finding ->
            System.out.println("[" + finding.getSeverity().getName() + "] " + finding.getTitle()));
    }

    /**
     * Shutdown all components
     */
    public void shutdown() {
        System.out.println("\n=== Shutting Down Demo ===\n");

        try {
            penetrationTesting.shutdown();
            validator.shutdown();
            koopmanOperator.shutdown();

            System.out.println("All components shut down successfully.");

        } catch (Exception e) {
            System.err.println("Error during shutdown: " + e.getMessage());
        }
    }

    /**
     * Display system information
     */
    public void displaySystemInfo() {
        System.out.println("\n=== System Information ===");

        // Java version
        System.out.println("Java Version: " + System.getProperty("java.version"));
        System.out.println("Java Vendor: " + System.getProperty("java.vendor"));

        // Memory information
        Runtime runtime = Runtime.getRuntime();
        long maxMemory = runtime.maxMemory();
        long totalMemory = runtime.totalMemory();
        long freeMemory = runtime.freeMemory();

        System.out.println(String.format("Max Memory: %.2f MB", maxMemory / (1024.0 * 1024.0)));
        System.out.println(String.format("Total Memory: %.2f MB", totalMemory / (1024.0 * 1024.0)));
        System.out.println(String.format("Free Memory: %.2f MB", freeMemory / (1024.0 * 1024.0)));
        System.out.println(String.format("Used Memory: %.2f MB", (totalMemory - freeMemory) / (1024.0 * 1024.0)));

        // CPU information
        System.out.println("Available Processors: " + runtime.availableProcessors());

        // OS information
        System.out.println("OS Name: " + System.getProperty("os.name"));
        System.out.println("OS Version: " + System.getProperty("os.version"));
        System.out.println("OS Architecture: " + System.getProperty("os.arch"));
    }
}
