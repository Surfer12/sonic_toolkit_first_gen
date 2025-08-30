// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
//
//  IntegratedSecurityDemo.java
//  qualia
//
//  Created by Ryan David Oates on 8/26/25.
//  Integrated security testing demo with GPTOSS 2.0 and Java penetration testing
package qualia;

import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;

/**
 * Integrated Security Testing Demo
 * Combines Java penetration testing with GPTOSS 2.0 AI model security assessment
 */
public class IntegratedSecurityDemo {

    private final JavaPenetrationTesting javaTesting;
    private final GPTOSSTesting gptossTesting;
    private final KSPenetrationTestingValidator validator;

    public IntegratedSecurityDemo(String gptossEndpoint, String gptossApiKey) {
        this.javaTesting = new JavaPenetrationTesting();
        this.gptossTesting = new GPTOSSTesting(gptossEndpoint, gptossApiKey);
        this.validator = new KSPenetrationTestingValidator();
    }

    /**
     * Run comprehensive integrated security assessment
     */
    public void runIntegratedSecurityAssessment() {
        System.out.println("=== Integrated Security Assessment Demo ===\n");
        System.out.println("This demo combines:");
        System.out.println("• Java Application Penetration Testing");
        System.out.println("• GPTOSS 2.0 AI Model Security Assessment");
        System.out.println("• Statistical Validation with K-S Framework\n");

        try {
            long startTime = System.nanoTime();

            // Run both testing frameworks concurrently
            CompletableFuture<List<SecurityFinding>> javaResults = javaTesting.runComprehensiveTesting();
            CompletableFuture<List<SecurityFinding>> gptossResults = gptossTesting.runComprehensiveGPTOSSTesting();

            // Wait for both to complete
            List<SecurityFinding> javaFindings = javaResults.get(120, TimeUnit.SECONDS);
            List<SecurityFinding> gptossFindings = gptossResults.get(120, TimeUnit.SECONDS);

            long endTime = System.nanoTime();
            double totalTimeMs = (endTime - startTime) / 1_000_000.0;

            // Combine all findings
            List<SecurityFinding> allFindings = new ArrayList<>();
            allFindings.addAll(javaFindings);
            allFindings.addAll(gptossFindings);

            // Display comprehensive results
            displayAssessmentResults(javaFindings, gptossFindings, allFindings, totalTimeMs);

            // Run statistical validation
            runIntegratedValidation(javaFindings, gptossFindings);

            // Generate integrated report
            generateIntegratedReport(javaFindings, gptossFindings, allFindings, totalTimeMs);

        } catch (Exception e) {
            System.err.println("Integrated assessment failed: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Display assessment results
     */
    private void displayAssessmentResults(List<SecurityFinding> javaFindings,
                                        List<SecurityFinding> gptossFindings,
                                        List<SecurityFinding> allFindings,
                                        double totalTimeMs) {

        System.out.println("=== Assessment Results ===\n");

        // Overall statistics
        System.out.println("Total Execution Time: " + String.format("%.2f seconds", totalTimeMs / 1000));
        System.out.println("Total Security Findings: " + allFindings.size());
        System.out.println("Java Application Findings: " + javaFindings.size());
        System.out.println("GPTOSS AI Model Findings: " + gptossFindings.size());
        System.out.println();

        // Severity breakdown for all findings
        Map<Severity, Long> severityCount = allFindings.stream()
            .collect(Collectors.groupingBy(SecurityFinding::getSeverity, Collectors.counting()));

        System.out.println("=== Overall Severity Breakdown ===");
        severityCount.entrySet().stream()
            .sorted(Map.Entry.<Severity, Long>comparingByValue().reversed())
            .forEach(entry -> System.out.println(entry.getKey().getName() + ": " + entry.getValue()));
        System.out.println();

        // Java findings breakdown
        if (!javaFindings.isEmpty()) {
            System.out.println("=== Java Application Security ===");
            Map<VulnerabilityType, Long> javaTypeCount = javaFindings.stream()
                .collect(Collectors.groupingBy(SecurityFinding::getVulnerabilityType, Collectors.counting()));

            javaTypeCount.entrySet().stream()
                .sorted(Map.Entry.<VulnerabilityType, Long>comparingByValue().reversed())
                .forEach(entry -> System.out.println("• " + entry.getKey().getDescription() + ": " + entry.getValue()));
            System.out.println();
        }

        // GPTOSS findings breakdown
        if (!gptossFindings.isEmpty()) {
            System.out.println("=== GPTOSS 2.0 AI Model Security ===");
            Map<VulnerabilityType, Long> gptossTypeCount = gptossFindings.stream()
                .collect(Collectors.groupingBy(SecurityFinding::getVulnerabilityType, Collectors.counting()));

            gptossTypeCount.entrySet().stream()
                .sorted(Map.Entry.<VulnerabilityType, Long>comparingByValue().reversed())
                .forEach(entry -> System.out.println("• " + entry.getKey().getDescription() + ": " + entry.getValue()));
            System.out.println();
        }

        // Critical findings
        List<SecurityFinding> criticalFindings = allFindings.stream()
            .filter(f -> f.getSeverity() == Severity.CRITICAL)
            .collect(Collectors.toList());

        if (!criticalFindings.isEmpty()) {
            System.out.println("=== Critical Security Issues ===");
            criticalFindings.forEach(finding -> {
                System.out.println("🚨 [" + finding.getVulnerabilityType().getDescription() + "] " + finding.getTitle());
                System.out.println("   " + finding.getDescription());
                System.out.println("   Impact: " + finding.getImpactAssessment());
                System.out.println();
            });
        }
    }

    /**
     * Run integrated statistical validation
     */
    private void runIntegratedValidation(List<SecurityFinding> javaFindings,
                                       List<SecurityFinding> gptossFindings) {

        System.out.println("=== Statistical Validation ===\n");

        try {
            // Create baseline findings for comparison
            List<SecurityFinding> baselineFindings = createBaselineFindings();

            // Validate Java findings
            if (!javaFindings.isEmpty()) {
                KSValidationResult javaValidation = validator.validatePenetrationTesting(javaFindings, baselineFindings).get();
                System.out.println("Java Framework Validation:");
                System.out.println("  " + javaValidation.toString());
                System.out.println();
            }

            // Validate GPTOSS findings
            if (!gptossFindings.isEmpty()) {
                KSValidationResult gptossValidation = validator.validatePenetrationTesting(gptossFindings, baselineFindings).get();
                System.out.println("GPTOSS Framework Validation:");
                System.out.println("  " + gptossValidation.toString());
                System.out.println();
            }

            // Cross-validate both frameworks
            List<List<SecurityFinding>> combinedRuns = Arrays.asList(javaFindings, gptossFindings);
            List<KSValidationResult> crossValidation = validator.performCrossValidation(combinedRuns).get();

            System.out.println("Cross-Framework Validation:");
            for (int i = 0; i < crossValidation.size(); i++) {
                System.out.println("  Framework " + (i + 1) + ": " + crossValidation.get(i).toString());
            }

            System.out.println();
            String summary = validator.generateValidationSummary(crossValidation);
            System.out.println(summary);

        } catch (Exception e) {
            System.err.println("Statistical validation failed: " + e.getMessage());
        }
    }

    /**
     * Generate integrated comprehensive report
     */
    private void generateIntegratedReport(List<SecurityFinding> javaFindings,
                                        List<SecurityFinding> gptossFindings,
                                        List<SecurityFinding> allFindings,
                                        double totalTimeMs) {

        System.out.println("=== Integrated Security Report ===\n");

        StringBuilder report = new StringBuilder();
        report.append("INTEGRATED SECURITY ASSESSMENT REPORT\n");
        report.append("=====================================\n\n");
        report.append("Generated: ").append(new Date()).append("\n");
        report.append("Assessment Duration: ").append(String.format("%.2f seconds", totalTimeMs / 1000)).append("\n\n");

        // Executive Summary
        report.append("EXECUTIVE SUMMARY\n");
        report.append("----------------\n");
        report.append("Total Security Findings: ").append(allFindings.size()).append("\n");
        report.append("Java Application Vulnerabilities: ").append(javaFindings.size()).append("\n");
        report.append("AI Model Vulnerabilities: ").append(gptossFindings.size()).append("\n\n");

        // Critical Issues Summary
        List<SecurityFinding> criticalIssues = allFindings.stream()
            .filter(f -> f.getSeverity() == Severity.CRITICAL)
            .collect(Collectors.toList());

        if (!criticalIssues.isEmpty()) {
            report.append("CRITICAL ISSUES REQUIRING IMMEDIATE ATTENTION:\n");
            for (SecurityFinding finding : criticalIssues) {
                report.append("• ").append(finding.getTitle()).append("\n");
                report.append("  ").append(finding.getImpactAssessment()).append("\n\n");
            }
        }

        // Detailed Findings by Framework
        if (!javaFindings.isEmpty()) {
            report.append("JAVA APPLICATION SECURITY FINDINGS\n");
            report.append("----------------------------------\n");
            appendFindingsBySeverity(report, javaFindings, "Java Application");
            report.append("\n");
        }

        if (!gptossFindings.isEmpty()) {
            report.append("GPTOSS 2.0 AI MODEL SECURITY FINDINGS\n");
            report.append("------------------------------------\n");
            appendFindingsBySeverity(report, gptossFindings, "AI Model");
            report.append("\n");
        }

        // Recommendations
        report.append("RECOMMENDATIONS\n");
        report.append("---------------\n");

        // Java-specific recommendations
        if (!javaFindings.isEmpty()) {
            report.append("Java Application Security:\n");
            report.append("• Implement comprehensive input validation and sanitization\n");
            report.append("• Use parameterized queries to prevent SQL injection\n");
            report.append("• Implement proper authentication and session management\n");
            report.append("• Use secure cryptographic algorithms (AES-256, SHA-256)\n");
            report.append("• Implement proper error handling without information disclosure\n");
            report.append("• Regular dependency vulnerability scanning\n\n");
        }

        // AI-specific recommendations
        if (!gptossFindings.isEmpty()) {
            report.append("AI Model Security:\n");
            report.append("• Implement prompt sanitization and filtering\n");
            report.append("• Strengthen jailbreak protection mechanisms\n");
            report.append("• Implement differential privacy for training data\n");
            report.append("• Regular model integrity and backdoor detection\n");
            report.append("• Secure API key management and rotation\n");
            report.append("• Implement rate limiting and adversarial input detection\n\n");
        }

        // Save integrated report
        saveReport("integrated_security_report.txt", report.toString());

        System.out.println("Integrated security report saved to: integrated_security_report.txt");
        System.out.println("Report includes comprehensive findings from both Java and AI model assessments.");
    }

    /**
     * Append findings by severity to report
     */
    private void appendFindingsBySeverity(StringBuilder report, List<SecurityFinding> findings, String framework) {
        for (Severity severity : Arrays.asList(Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO)) {
            List<SecurityFinding> severityFindings = findings.stream()
                .filter(f -> f.getSeverity() == severity)
                .collect(Collectors.toList());

            if (!severityFindings.isEmpty()) {
                report.append(severity.getName().toUpperCase()).append(" (").append(severityFindings.size()).append("):\n");

                for (SecurityFinding finding : severityFindings) {
                    report.append("• ").append(finding.getTitle()).append("\n");
                    report.append("  ").append(finding.getDescription()).append("\n");
                    report.append("  Recommendation: ").append(finding.getRecommendation()).append("\n\n");
                }
            }
        }
    }

    /**
     * Create baseline findings for validation
     */
    private List<SecurityFinding> createBaselineFindings() {
        List<SecurityFinding> baseline = new ArrayList<>();

        // Add typical baseline security findings
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
     * Save report to file
     */
    private void saveReport(String filename, String content) {
        try {
            java.nio.file.Path reportPath = java.nio.file.Paths.get("reports", filename);
            java.nio.file.Files.createDirectories(reportPath.getParent());
            java.nio.file.Files.writeString(reportPath, content);
            System.out.println("Report saved to: " + reportPath.toAbsolutePath());

        } catch (java.io.IOException e) {
            System.err.println("Failed to save report: " + e.getMessage());
        }
    }

    /**
     * Run scenario-based testing
     */
    public void runScenarioBasedAssessment() {
        System.out.println("\n=== Scenario-Based Security Assessment ===\n");

        List<CompletableFuture<Void>> scenarios = Arrays.asList(
            runWebApplicationScenario(),
            runAPIScenario(),
            runDatabaseScenario(),
            runAIScenario()
        );

        try {
            CompletableFuture.allOf(scenarios.toArray(new CompletableFuture[0])).get(60, TimeUnit.SECONDS);
            System.out.println("\nScenario-based assessment completed successfully.");
        } catch (Exception e) {
            System.err.println("Scenario assessment failed: " + e.getMessage());
        }
    }

    /**
     * Web application security scenario
     */
    private CompletableFuture<Void> runWebApplicationScenario() {
        return CompletableFuture.runAsync(() -> {
            System.out.println("Running Web Application Security Scenario...");

            // Run Java penetration testing focused on web security
            try {
                List<SecurityFinding> javaFindings = javaTesting.runComprehensiveTesting().get(30, TimeUnit.SECONDS);
                List<SecurityFinding> webFindings = javaFindings.stream()
                    .filter(f -> f.getLocation().contains("Web") || f.getLocation().contains("HTTP"))
                    .collect(Collectors.toList());

                System.out.println("Web Application Findings: " + webFindings.size());

            } catch (Exception e) {
                System.err.println("Web application scenario failed: " + e.getMessage());
            }
        });
    }

    /**
     * API security scenario
     */
    private CompletableFuture<Void> runAPIScenario() {
        return CompletableFuture.runAsync(() -> {
            System.out.println("Running API Security Scenario...");

            try {
                List<SecurityFinding> javaFindings = javaTesting.runComprehensiveTesting().get(30, TimeUnit.SECONDS);
                List<SecurityFinding> apiFindings = javaFindings.stream()
                    .filter(f -> f.getLocation().contains("API") || f.getLocation().contains("Network"))
                    .collect(Collectors.toList());

                System.out.println("API Security Findings: " + apiFindings.size());

            } catch (Exception e) {
                System.err.println("API scenario failed: " + e.getMessage());
            }
        });
    }

    /**
     * Database security scenario
     */
    private CompletableFuture<Void> runDatabaseScenario() {
        return CompletableFuture.runAsync(() -> {
            System.out.println("Running Database Security Scenario...");

            try {
                List<SecurityFinding> javaFindings = javaTesting.runComprehensiveTesting().get(30, TimeUnit.SECONDS);
                List<SecurityFinding> dbFindings = javaFindings.stream()
                    .filter(f -> f.getLocation().contains("Database") || f.getLocation().contains("SQL"))
                    .collect(Collectors.toList());

                System.out.println("Database Security Findings: " + dbFindings.size());

            } catch (Exception e) {
                System.err.println("Database scenario failed: " + e.getMessage());
            }
        });
    }

    /**
     * AI model security scenario
     */
    private CompletableFuture<Void> runAIScenario() {
        return CompletableFuture.runAsync(() -> {
            System.out.println("Running AI Model Security Scenario...");

            try {
                List<SecurityFinding> gptossFindings = gptossTesting.runComprehensiveGPTOSSTesting().get(60, TimeUnit.SECONDS);
                System.out.println("AI Model Security Findings: " + gptossFindings.size());

                // Show AI-specific vulnerabilities
                long criticalAIFindings = gptossFindings.stream()
                    .filter(f -> f.getSeverity() == Severity.CRITICAL)
                    .count();

                System.out.println("Critical AI Vulnerabilities: " + criticalAIFindings);

            } catch (Exception e) {
                System.err.println("AI scenario failed: " + e.getMessage());
            }
        });
    }

    /**
     * Main demo method
     */
    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Usage: java IntegratedSecurityDemo <gptoss-endpoint> <gptoss-api-key>");
            System.out.println("Example: java IntegratedSecurityDemo https://api.gptoss.com/v1 sk-your-api-key-here");
            System.exit(1);
        }

        String gptossEndpoint = args[0];
        String gptossApiKey = args[1];

        System.out.println("=== Integrated Security Testing Framework ===\n");
        System.out.println("GPTOSS Endpoint: " + gptossEndpoint);
        System.out.println("API Key: " + (gptossApiKey.startsWith("sk-") ? "Valid format" : "Invalid format"));
        System.out.println();

        IntegratedSecurityDemo demo = new IntegratedSecurityDemo(gptossEndpoint, gptossApiKey);

        try {
            // Run comprehensive integrated assessment
            demo.runIntegratedSecurityAssessment();

            // Run scenario-based testing
            demo.runScenarioBasedAssessment();

        } catch (Exception e) {
            System.err.println("Demo failed: " + e.getMessage());
            e.printStackTrace();
        } finally {
            demo.shutdown();
        }
    }

    /**
     * Shutdown all components
     */
    public void shutdown() {
        System.out.println("\n=== Shutting Down Integrated Demo ===\n");

        try {
            javaTesting.shutdown();
            gptossTesting.shutdown();
            validator.shutdown();

            System.out.println("All components shut down successfully.");

        } catch (Exception e) {
            System.err.println("Error during shutdown: " + e.getMessage());
        }
    }
}
