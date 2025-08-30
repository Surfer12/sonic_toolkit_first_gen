// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
//
//  SecurityDashboard.java
//  qualia
//
//  Created by Ryan David Oates on 8/26/25.
//  Visual dashboard for security assessment results
package qualia;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import java.util.List;
import java.util.concurrent.*;
import java.util.stream.Collectors;
import org.jfree.chart.*;
import org.jfree.chart.plot.*;
import org.jfree.data.category.*;
import org.jfree.data.general.*;

/**
 * Security Dashboard - Visual Interface for Penetration Testing Results
 * Provides real-time visualization of security findings and assessment progress
 */
public class SecurityDashboard extends JFrame {

    private final JavaPenetrationTesting javaTesting;
    private final GPTOSSTesting gptossTesting;
    private final KSPenetrationTestingValidator validator;

    // UI Components
    private JTabbedPane tabbedPane;
    private JPanel mainPanel;
    private JPanel statusPanel;
    private JProgressBar progressBar;
    private JTextArea logArea;
    private JButton startAssessmentButton;
    private JButton stopAssessmentButton;
    private JButton generateReportButton;

    // Charts
    private ChartPanel severityChart;
    private ChartPanel vulnerabilityChart;
    private ChartPanel timelineChart;

    // Data
    private List<SecurityFinding> currentFindings;
    private boolean assessmentRunning;

    public SecurityDashboard() {
        this(null, null);
    }

    public SecurityDashboard(String gptossEndpoint, String gptossApiKey) {
        this.javaTesting = new JavaPenetrationTesting();
        this.gptossTesting = (gptossEndpoint != null && gptossApiKey != null) ?
            new GPTOSSTesting(gptossEndpoint, gptossApiKey) : null;
        this.validator = new KSPenetrationTestingValidator();
        this.currentFindings = new ArrayList<>();
        this.assessmentRunning = false;

        initializeComponents();
        setupLayout();
        setupEventHandlers();
    }

    /**
     * Initialize all UI components
     */
    private void initializeComponents() {
        setTitle("Reverse Koopman Penetration Testing Dashboard");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(1200, 800);
        setLocationRelativeTo(null);

        // Initialize main components
        tabbedPane = new JTabbedPane();
        mainPanel = new JPanel(new BorderLayout());
        statusPanel = createStatusPanel();
        progressBar = new JProgressBar(0, 100);
        logArea = new JTextArea(10, 50);
        logArea.setEditable(false);
        logArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));

        // Initialize buttons
        startAssessmentButton = new JButton("Start Assessment");
        stopAssessmentButton = new JButton("Stop Assessment");
        stopAssessmentButton.setEnabled(false);
        generateReportButton = new JButton("Generate Report");
        generateReportButton.setEnabled(false);

        // Initialize charts
        severityChart = createSeverityChart();
        vulnerabilityChart = createVulnerabilityChart();
        timelineChart = createTimelineChart();
    }

    /**
     * Setup the main layout
     */
    private void setupLayout() {
        // Create main dashboard tab
        JPanel dashboardPanel = createDashboardPanel();
        tabbedPane.addTab("Dashboard", dashboardPanel);

        // Create detailed findings tab
        JPanel findingsPanel = createFindingsPanel();
        tabbedPane.addTab("Detailed Findings", findingsPanel);

        // Create charts tab
        JPanel chartsPanel = createChartsPanel();
        tabbedPane.addTab("Analytics", chartsPanel);

        // Create logs tab
        JPanel logsPanel = createLogsPanel();
        tabbedPane.addTab("Logs", logsPanel);

        // Create settings tab
        JPanel settingsPanel = createSettingsPanel();
        tabbedPane.addTab("Settings", settingsPanel);

        add(tabbedPane, BorderLayout.CENTER);

        // Add status bar
        add(statusPanel, BorderLayout.SOUTH);
    }

    /**
     * Create the main dashboard panel
     */
    private JPanel createDashboardPanel() {
        JPanel panel = new JPanel(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);

        // Title
        JLabel titleLabel = new JLabel("Security Assessment Dashboard");
        titleLabel.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 24));
        gbc.gridx = 0; gbc.gridy = 0; gbc.gridwidth = 3;
        panel.add(titleLabel, gbc);

        // Control buttons
        JPanel buttonPanel = new JPanel(new FlowLayout());
        buttonPanel.add(startAssessmentButton);
        buttonPanel.add(stopAssessmentButton);
        buttonPanel.add(generateReportButton);

        gbc.gridx = 0; gbc.gridy = 1; gbc.gridwidth = 3;
        panel.add(buttonPanel, gbc);

        // Progress bar
        gbc.gridx = 0; gbc.gridy = 2; gbc.gridwidth = 3; gbc.fill = GridBagConstraints.HORIZONTAL;
        panel.add(progressBar, gbc);

        // Summary statistics
        JPanel statsPanel = createStatsPanel();
        gbc.gridx = 0; gbc.gridy = 3; gbc.gridwidth = 3; gbc.fill = GridBagConstraints.BOTH; gbc.weightx = 1.0; gbc.weighty = 0.3;
        panel.add(statsPanel, gbc);

        // Quick charts
        JPanel quickChartsPanel = createQuickChartsPanel();
        gbc.gridx = 0; gbc.gridy = 4; gbc.gridwidth = 3; gbc.fill = GridBagConstraints.BOTH; gbc.weighty = 0.7;
        panel.add(quickChartsPanel, gbc);

        return panel;
    }

    /**
     * Create statistics panel
     */
    private JPanel createStatsPanel() {
        JPanel panel = new JPanel(new GridLayout(1, 4, 10, 0));

        // Total findings card
        JPanel totalCard = createStatCard("Total Findings", "0", Color.BLUE);
        panel.add(totalCard);

        // Critical findings card
        JPanel criticalCard = createStatCard("Critical", "0", Color.RED);
        panel.add(criticalCard);

        // High findings card
        JPanel highCard = createStatCard("High", "0", Color.ORANGE);
        panel.add(highCard);

        // Assessment status card
        JPanel statusCard = createStatCard("Status", "Ready", Color.GRAY);
        panel.add(statusCard);

        return panel;
    }

    /**
     * Create a statistics card
     */
    private JPanel createStatCard(String title, String value, Color color) {
        JPanel card = new JPanel(new BorderLayout());
        card.setBorder(BorderFactory.createCompoundBorder(
            BorderFactory.createLineBorder(color, 2),
            BorderFactory.createEmptyBorder(10, 10, 10, 10)
        ));
        card.setBackground(Color.WHITE);

        JLabel valueLabel = new JLabel(value, SwingConstants.CENTER);
        valueLabel.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 24));
        valueLabel.setForeground(color);

        JLabel titleLabel = new JLabel(title, SwingConstants.CENTER);
        titleLabel.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 12));

        card.add(valueLabel, BorderLayout.CENTER);
        card.add(titleLabel, BorderLayout.SOUTH);

        return card;
    }

    /**
     * Create quick charts panel
     */
    private JPanel createQuickChartsPanel() {
        JPanel panel = new JPanel(new GridLayout(1, 2, 10, 0));

        // Severity distribution chart
        JPanel severityPanel = new JPanel(new BorderLayout());
        severityPanel.setBorder(BorderFactory.createTitledBorder("Severity Distribution"));
        severityPanel.add(severityChart, BorderLayout.CENTER);
        panel.add(severityPanel);

        // Vulnerability types chart
        JPanel vulnPanel = new JPanel(new BorderLayout());
        vulnPanel.setBorder(BorderFactory.createTitledBorder("Vulnerability Types"));
        vulnPanel.add(vulnerabilityChart, BorderLayout.CENTER);
        panel.add(vulnPanel);

        return panel;
    }

    /**
     * Create findings panel
     */
    private JPanel createFindingsPanel() {
        JPanel panel = new JPanel(new BorderLayout());

        // Findings table
        String[] columnNames = {"Severity", "Type", "Title", "Location", "Timestamp"};
        JTable findingsTable = new JTable(new String[0][0], columnNames);
        findingsTable.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        findingsTable.getTableHeader().setReorderingAllowed(false);

        JScrollPane scrollPane = new JScrollPane(findingsTable);
        panel.add(scrollPane, BorderLayout.CENTER);

        // Details panel
        JPanel detailsPanel = new JPanel(new BorderLayout());
        detailsPanel.setBorder(BorderFactory.createTitledBorder("Finding Details"));
        detailsPanel.setPreferredSize(new Dimension(-1, 200));

        JTextArea detailsArea = new JTextArea();
        detailsArea.setEditable(false);
        detailsArea.setWrapStyleWord(true);
        detailsArea.setLineWrap(true);

        JScrollPane detailsScroll = new JScrollPane(detailsArea);
        detailsPanel.add(detailsScroll, BorderLayout.CENTER);

        panel.add(detailsPanel, BorderLayout.SOUTH);

        return panel;
    }

    /**
     * Create charts panel
     */
    private JPanel createChartsPanel() {
        JPanel panel = new JPanel(new GridLayout(2, 2, 10, 10));

        // Severity pie chart
        JPanel severityPanel = new JPanel(new BorderLayout());
        severityPanel.setBorder(BorderFactory.createTitledBorder("Severity Distribution"));
        severityPanel.add(severityChart, BorderLayout.CENTER);
        panel.add(severityPanel);

        // Vulnerability bar chart
        JPanel vulnPanel = new JPanel(new BorderLayout());
        vulnPanel.setBorder(BorderFactory.createTitledBorder("Vulnerability Types"));
        vulnPanel.add(vulnerabilityChart, BorderLayout.CENTER);
        panel.add(vulnPanel);

        // Timeline chart
        JPanel timelinePanel = new JPanel(new BorderLayout());
        timelinePanel.setBorder(BorderFactory.createTitledBorder("Findings Timeline"));
        timelinePanel.add(timelineChart, BorderLayout.CENTER);
        panel.add(timelinePanel);

        // Placeholder for additional charts
        JPanel additionalPanel = new JPanel(new BorderLayout());
        additionalPanel.setBorder(BorderFactory.createTitledBorder("K-S Validation Results"));
        additionalPanel.add(new JLabel("K-S Validation charts will be displayed here", SwingConstants.CENTER), BorderLayout.CENTER);
        panel.add(additionalPanel);

        return panel;
    }

    /**
     * Create logs panel
     */
    private JPanel createLogsPanel() {
        JPanel panel = new JPanel(new BorderLayout());

        // Log controls
        JPanel controlsPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        JButton clearLogsButton = new JButton("Clear Logs");
        JButton saveLogsButton = new JButton("Save Logs");
        controlsPanel.add(clearLogsButton);
        controlsPanel.add(saveLogsButton);

        panel.add(controlsPanel, BorderLayout.NORTH);

        // Log area
        JScrollPane scrollPane = new JScrollPane(logArea);
        scrollPane.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_ALWAYS);
        panel.add(scrollPane, BorderLayout.CENTER);

        // Setup log button handlers
        clearLogsButton.addActionListener(e -> logArea.setText(""));
        saveLogsButton.addActionListener(e -> saveLogsToFile());

        return panel;
    }

    /**
     * Create settings panel
     */
    private JPanel createSettingsPanel() {
        JPanel panel = new JPanel(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);
        gbc.anchor = GridBagConstraints.WEST;

        // GPTOSS Settings
        gbc.gridx = 0; gbc.gridy = 0;
        panel.add(new JLabel("GPTOSS 2.0 Settings:"), gbc);

        gbc.gridx = 0; gbc.gridy = 1;
        panel.add(new JLabel("Endpoint:"), gbc);
        JTextField endpointField = new JTextField("http://localhost:8000", 20);
        gbc.gridx = 1;
        panel.add(endpointField, gbc);

        gbc.gridx = 0; gbc.gridy = 2;
        panel.add(new JLabel("API Key:"), gbc);
        JPasswordField apiKeyField = new JPasswordField(20);
        gbc.gridx = 1;
        panel.add(apiKeyField, gbc);

        // Assessment Settings
        gbc.gridx = 0; gbc.gridy = 3; gbc.gridwidth = 2;
        panel.add(new JSeparator(), gbc);

        gbc.gridwidth = 1;
        gbc.gridx = 0; gbc.gridy = 4;
        panel.add(new JLabel("Assessment Settings:"), gbc);

        JCheckBox javaTestingCheck = new JCheckBox("Java Application Testing", true);
        JCheckBox gptossTestingCheck = new JCheckBox("GPTOSS AI Testing", gptossTesting != null);
        JCheckBox validationCheck = new JCheckBox("K-S Validation", true);

        gbc.gridx = 0; gbc.gridy = 5; gbc.gridwidth = 2;
        panel.add(javaTestingCheck, gbc);
        gbc.gridy = 6;
        panel.add(gptossTestingCheck, gbc);
        gbc.gridy = 7;
        panel.add(validationCheck, gbc);

        // Save settings button
        JButton saveSettingsButton = new JButton("Save Settings");
        gbc.gridx = 0; gbc.gridy = 8; gbc.gridwidth = 2;
        panel.add(saveSettingsButton, gbc);

        return panel;
    }

    /**
     * Create severity distribution chart
     */
    private ChartPanel createSeverityChart() {
        DefaultPieDataset dataset = new DefaultPieDataset();
        dataset.setValue("Critical", 0);
        dataset.setValue("High", 0);
        dataset.setValue("Medium", 0);
        dataset.setValue("Low", 0);
        dataset.setValue("Info", 0);

        JFreeChart chart = ChartFactory.createPieChart(
            "Findings by Severity",
            dataset,
            true,
            true,
            false
        );

        PiePlot plot = (PiePlot) chart.getPlot();
        plot.setSectionPaint("Critical", Color.RED);
        plot.setSectionPaint("High", Color.ORANGE);
        plot.setSectionPaint("Medium", Color.YELLOW);
        plot.setSectionPaint("Low", Color.BLUE);
        plot.setSectionPaint("Info", Color.GRAY);

        return new ChartPanel(chart);
    }

    /**
     * Create vulnerability types chart
     */
    private ChartPanel createVulnerabilityChart() {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();

        JFreeChart chart = ChartFactory.createBarChart(
            "Vulnerability Types",
            "Type",
            "Count",
            dataset,
            PlotOrientation.VERTICAL,
            true,
            true,
            false
        );

        return new ChartPanel(chart);
    }

    /**
     * Create timeline chart
     */
    private ChartPanel createTimelineChart() {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();

        JFreeChart chart = ChartFactory.createLineChart(
            "Findings Over Time",
            "Time",
            "Findings",
            dataset,
            PlotOrientation.VERTICAL,
            true,
            true,
            false
        );

        return new ChartPanel(chart);
    }

    /**
     * Create status panel
     */
    private JPanel createStatusPanel() {
        JPanel panel = new JPanel(new BorderLayout());
        panel.setBorder(BorderFactory.createLoweredBevelBorder());

        JLabel statusLabel = new JLabel("Ready to start assessment");
        statusLabel.setName("statusLabel");

        panel.add(statusLabel, BorderLayout.WEST);

        // Add progress indicator
        JProgressBar miniProgress = new JProgressBar();
        miniProgress.setVisible(false);
        miniProgress.setName("miniProgress");
        panel.add(miniProgress, BorderLayout.EAST);

        return panel;
    }

    /**
     * Setup event handlers
     */
    private void setupEventHandlers() {
        startAssessmentButton.addActionListener(e -> startAssessment());
        stopAssessmentButton.addActionListener(e -> stopAssessment());
        generateReportButton.addActionListener(e -> generateReport());
    }

    /**
     * Start security assessment
     */
    private void startAssessment() {
        if (assessmentRunning) return;

        assessmentRunning = true;
        startAssessmentButton.setEnabled(false);
        stopAssessmentButton.setEnabled(true);
        progressBar.setValue(0);

        logMessage("Starting comprehensive security assessment...");

        // Run assessment in background
        SwingWorker<Void, String> worker = new SwingWorker<>() {
            @Override
            protected Void doInBackground() throws Exception {
                try {
                    publish("Running Java application security testing...");

                    // Run Java penetration testing
                    CompletableFuture<List<SecurityFinding>> javaResults = javaTesting.runComprehensiveTesting();
                    List<SecurityFinding> javaFindings = javaResults.get(30, TimeUnit.SECONDS);

                    publish("Java testing completed: " + javaFindings.size() + " findings");

                    // Run GPTOSS testing if available
                    List<SecurityFinding> gptossFindings = new ArrayList<>();
                    if (gptossTesting != null) {
                        publish("Running GPTOSS 2.0 AI model security testing...");
                        CompletableFuture<List<SecurityFinding>> gptossResults = gptossTesting.runComprehensiveGPTOSSTesting();
                        gptossFindings = gptossResults.get(60, TimeUnit.SECONDS);
                        publish("GPTOSS testing completed: " + gptossFindings.size() + " findings");
                    }

                    // Combine all findings
                    currentFindings.clear();
                    currentFindings.addAll(javaFindings);
                    currentFindings.addAll(gptossFindings);

                    publish("Total findings: " + currentFindings.size());
                    publish("Running K-S validation...");

                    // Run validation
                    if (!javaFindings.isEmpty()) {
                        List<SecurityFinding> baselineFindings = createBaselineFindings();
                        KSValidationResult validation = validator.validatePenetrationTesting(javaFindings, baselineFindings).get();
                        publish("K-S Validation: " + validation.toString());
                    }

                    publish("Assessment completed successfully");

                } catch (Exception ex) {
                    publish("Assessment failed: " + ex.getMessage());
                    ex.printStackTrace();
                }

                return null;
            }

            @Override
            protected void process(List<String> chunks) {
                for (String message : chunks) {
                    logMessage(message);
                    updateProgress();
                }
            }

            @Override
            protected void done() {
                assessmentRunning = false;
                startAssessmentButton.setEnabled(true);
                stopAssessmentButton.setEnabled(false);
                generateReportButton.setEnabled(true);
                progressBar.setValue(100);
                logMessage("Assessment completed");

                // Update charts with new data
                updateCharts();
                updateStatsCards();
            }
        };

        worker.execute();
    }

    /**
     * Stop assessment
     */
    private void stopAssessment() {
        if (!assessmentRunning) return;

        logMessage("Stopping assessment...");
        assessmentRunning = false;
        startAssessmentButton.setEnabled(true);
        stopAssessmentButton.setEnabled(false);
        progressBar.setValue(0);
    }

    /**
     * Generate security report
     */
    private void generateReport() {
        if (currentFindings.isEmpty()) {
            JOptionPane.showMessageDialog(this, "No findings to report. Run an assessment first.",
                                        "No Data", JOptionPane.WARNING_MESSAGE);
            return;
        }

        try {
            // Generate comprehensive report
            String report = generateComprehensiveReport();

            // Save to file
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setSelectedFile(new java.io.File("security_report.txt"));

            if (fileChooser.showSaveDialog(this) == JFileChooser.APPROVE_OPTION) {
                java.io.File file = fileChooser.getSelectedFile();
                java.nio.file.Files.writeString(file.toPath(), report);
                logMessage("Report saved to: " + file.getAbsolutePath());
                JOptionPane.showMessageDialog(this, "Report saved successfully!",
                                            "Success", JOptionPane.INFORMATION_MESSAGE);
            }

        } catch (Exception ex) {
            logMessage("Failed to generate report: " + ex.getMessage());
            JOptionPane.showMessageDialog(this, "Failed to generate report: " + ex.getMessage(),
                                        "Error", JOptionPane.ERROR_MESSAGE);
        }
    }

    /**
     * Generate comprehensive report
     */
    private String generateComprehensiveReport() {
        StringBuilder report = new StringBuilder();
        report.append("REVERSE KOOPMAN PENETRATION TESTING REPORT\n");
        report.append("==========================================\n\n");
        report.append("Generated: ").append(new Date()).append("\n");
        report.append("Total Findings: ").append(currentFindings.size()).append("\n\n");

        // Executive Summary
        Map<Severity, Long> severityCount = currentFindings.stream()
            .collect(Collectors.groupingBy(SecurityFinding::getSeverity, Collectors.counting()));

        report.append("EXECUTIVE SUMMARY\n");
        report.append("----------------\n");
        severityCount.forEach((severity, count) ->
            report.append(severity.getName()).append(": ").append(count).append("\n"));

        // Detailed Findings
        report.append("\nDETAILED FINDINGS\n");
        report.append("-----------------\n");

        for (SecurityFinding finding : currentFindings) {
            report.append("\n[").append(finding.getSeverity().getName()).append("] ")
                  .append(finding.getTitle()).append("\n");
            report.append("Type: ").append(finding.getVulnerabilityType().getDescription()).append("\n");
            report.append("Location: ").append(finding.getLocation()).append("\n");
            report.append("Description: ").append(finding.getDescription()).append("\n");
            report.append("Recommendation: ").append(finding.getRecommendation()).append("\n");
            report.append("Impact: ").append(finding.getImpactAssessment()).append("\n");
        }

        return report.toString();
    }

    /**
     * Update charts with current findings
     */
    private void updateCharts() {
        // Update severity chart
        DefaultPieDataset severityDataset = new DefaultPieDataset();
        Map<Severity, Long> severityCount = currentFindings.stream()
            .collect(Collectors.groupingBy(SecurityFinding::getSeverity, Collectors.counting()));

        severityCount.forEach((severity, count) ->
            severityDataset.setValue(severity.getName(), count));

        ((PiePlot) severityChart.getChart().getPlot()).setDataset(severityDataset);

        // Update vulnerability chart
        DefaultCategoryDataset vulnDataset = new DefaultCategoryDataset();
        Map<VulnerabilityType, Long> vulnCount = currentFindings.stream()
            .collect(Collectors.groupingBy(SecurityFinding::getVulnerabilityType, Collectors.counting()));

        vulnCount.forEach((type, count) ->
            vulnDataset.addValue(count, "Findings", type.getDescription()));

        ((CategoryPlot) vulnerabilityChart.getChart().getPlot()).setDataset(vulnDataset);
    }

    /**
     * Update statistics cards
     */
    private void updateStatsCards() {
        // This would update the stat cards with current data
        // Implementation depends on the specific card components
        logMessage("Statistics updated with " + currentFindings.size() + " findings");
    }

    /**
     * Update progress bar
     */
    private void updateProgress() {
        if (progressBar.getValue() < 90) {
            progressBar.setValue(progressBar.getValue() + 10);
        }
    }

    /**
     * Log message to the log area
     */
    private void logMessage(String message) {
        SwingUtilities.invokeLater(() -> {
            logArea.append("[" + new Date() + "] " + message + "\n");
            logArea.setCaretPosition(logArea.getDocument().getLength());
        });
    }

    /**
     * Save logs to file
     */
    private void saveLogsToFile() {
        try {
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setSelectedFile(new java.io.File("assessment_logs.txt"));

            if (fileChooser.showSaveDialog(this) == JFileChooser.APPROVE_OPTION) {
                java.io.File file = fileChooser.getSelectedFile();
                java.nio.file.Files.writeString(file.toPath(), logArea.getText());
                JOptionPane.showMessageDialog(this, "Logs saved successfully!",
                                            "Success", JOptionPane.INFORMATION_MESSAGE);
            }

        } catch (Exception ex) {
            JOptionPane.showMessageDialog(this, "Failed to save logs: " + ex.getMessage(),
                                        "Error", JOptionPane.ERROR_MESSAGE);
        }
    }

    /**
     * Create baseline findings for validation
     */
    private List<SecurityFinding> createBaselineFindings() {
        List<SecurityFinding> baseline = new ArrayList<>();
        baseline.add(new SecurityFinding(
            VulnerabilityType.SQL_INJECTION,
            Severity.MEDIUM,
            "SQL Injection Risk",
            "Potential SQL injection vulnerability",
            "Database Layer",
            "Use parameterized queries",
            "Standard security practice"
        ));
        return baseline;
    }

    /**
     * Main method to launch the dashboard
     */
    public static void main(String[] args) {
        // Set system look and feel
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeel());
        } catch (Exception e) {
            // Use default look and feel
        }

        SwingUtilities.invokeLater(() -> {
            String endpoint = args.length > 0 ? args[0] : null;
            String apiKey = args.length > 1 ? args[1] : null;

            new SecurityDashboard(endpoint, apiKey).setVisible(true);
        });
    }
}
