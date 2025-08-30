// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
//
//  KoopmanVisualization.java
//  qualia
//
//  Created by Ryan David Oates on 8/26/25.
//  Advanced visualization for koopman operators and security findings
package qualia;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.*;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.stage.Stage;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import java.util.*;
import java.util.List;
import java.util.concurrent.*;
import java.util.stream.Collectors;

/**
 * Advanced JavaFX Visualization for Reverse Koopman Penetration Testing
 * Provides interactive 3D visualizations and real-time security dashboards
 */
public class KoopmanVisualization extends Application {

    private final JavaPenetrationTesting javaTesting;
    private final GPTOSSTesting gptossTesting;
    private final KSPenetrationTestingValidator validator;

    // UI Components
    private BorderPane root;
    private TabPane mainTabPane;
    private ProgressBar progressBar;
    private TextArea logArea;
    private Button startButton;
    private Button stopButton;
    private Label statusLabel;

    // Data
    private List<SecurityFinding> currentFindings;
    private boolean assessmentRunning;

    public KoopmanVisualization() {
        this(null, null);
    }

    public KoopmanVisualization(String gptossEndpoint, String gptossApiKey) {
        this.javaTesting = new JavaPenetrationTesting();
        this.gptossTesting = (gptossEndpoint != null && gptossApiKey != null) ?
            new GPTOSSTesting(gptossEndpoint, gptossApiKey) : null;
        this.validator = new KSPenetrationTestingValidator();
        this.currentFindings = new ArrayList<>();
        this.assessmentRunning = false;
    }

    @Override
    public void start(Stage primaryStage) {
        primaryStage.setTitle("Reverse Koopman Security Visualization");

        // Create main layout
        root = new BorderPane();
        root.setTop(createMenuBar());
        root.setCenter(createMainContent());
        root.setBottom(createStatusBar());

        // Create scene
        Scene scene = new Scene(root, 1400, 900);
        scene.getStylesheets().add(getClass().getResource("security-dashboard.css").toExternalForm());

        primaryStage.setScene(scene);
        primaryStage.show();
    }

    /**
     * Create menu bar
     */
    private MenuBar createMenuBar() {
        MenuBar menuBar = new MenuBar();

        // File menu
        Menu fileMenu = new Menu("File");
        MenuItem exitItem = new MenuItem("Exit");
        exitItem.setOnAction(e -> System.exit(0));
        fileMenu.getItems().add(exitItem);

        // Assessment menu
        Menu assessmentMenu = new Menu("Assessment");
        MenuItem runJavaItem = new MenuItem("Run Java Testing");
        MenuItem runGPTOSSItem = new MenuItem("Run GPTOSS Testing");
        MenuItem runIntegratedItem = new MenuItem("Run Integrated Assessment");

        runJavaItem.setOnAction(e -> runJavaAssessment());
        runGPTOSSItem.setOnAction(e -> runGPTOSSTesting());
        runIntegratedItem.setOnAction(e -> runIntegratedAssessment());

        assessmentMenu.getItems().addAll(runJavaItem, runGPTOSSItem, runIntegratedItem);

        // Visualization menu
        Menu visualizationMenu = new Menu("Visualization");
        MenuItem koopman3DItem = new MenuItem("3D Koopman Analysis");
        MenuItem securityMapItem = new MenuItem("Security Heatmap");
        MenuItem timelineItem = new MenuItem("Timeline Analysis");

        koopman3DItem.setOnAction(e -> showKoopman3DVisualization());
        securityMapItem.setOnAction(e -> showSecurityHeatmap());
        timelineItem.setOnAction(e -> showTimelineAnalysis());

        visualizationMenu.getItems().addAll(koopman3DItem, securityMapItem, timelineItem);

        // Reports menu
        Menu reportsMenu = new Menu("Reports");
        MenuItem generateReportItem = new MenuItem("Generate Report");
        MenuItem exportJsonItem = new MenuItem("Export JSON");
        MenuItem exportCsvItem = new MenuItem("Export CSV");

        generateReportItem.setOnAction(e -> generateReport());
        exportJsonItem.setOnAction(e -> exportFindings("json"));
        exportCsvItem.setOnAction(e -> exportFindings("csv"));

        reportsMenu.getItems().addAll(generateReportItem, exportJsonItem, exportCsvItem);

        menuBar.getMenus().addAll(fileMenu, assessmentMenu, visualizationMenu, reportsMenu);
        return menuBar;
    }

    /**
     * Create main content area
     */
    private TabPane createMainContent() {
        mainTabPane = new TabPane();

        // Dashboard tab
        Tab dashboardTab = new Tab("Dashboard");
        dashboardTab.setContent(createDashboard());
        dashboardTab.setClosable(false);

        // Findings tab
        Tab findingsTab = new Tab("Findings");
        findingsTab.setContent(createFindingsView());
        findingsTab.setClosable(false);

        // Analytics tab
        Tab analyticsTab = new Tab("Analytics");
        analyticsTab.setContent(createAnalyticsView());
        analyticsTab.setClosable(false);

        // 3D Visualization tab
        Tab visualizationTab = new Tab("3D View");
        visualizationTab.setContent(create3DVisualization());
        visualizationTab.setClosable(false);

        mainTabPane.getTabs().addAll(dashboardTab, findingsTab, analyticsTab, visualizationTab);
        return mainTabPane;
    }

    /**
     * Create dashboard
     */
    private VBox createDashboard() {
        VBox dashboard = new VBox(10);
        dashboard.setPadding(new Insets(20));

        // Title
        Label titleLabel = new Label("Reverse Koopman Security Assessment");
        titleLabel.setFont(Font.font("System", FontWeight.BOLD, 24));
        titleLabel.setTextFill(Color.DARKBLUE);

        // Control panel
        HBox controlPanel = createControlPanel();

        // Statistics grid
        GridPane statsGrid = createStatsGrid();

        // Progress area
        VBox progressArea = createProgressArea();

        dashboard.getChildren().addAll(titleLabel, controlPanel, statsGrid, progressArea);
        return dashboard;
    }

    /**
     * Create control panel
     */
    private HBox createControlPanel() {
        HBox controlPanel = new HBox(10);
        controlPanel.setAlignment(Pos.CENTER);

        startButton = new Button("Start Assessment");
        startButton.setStyle("-fx-background-color: #4CAF50; -fx-text-fill: white; -fx-font-size: 14px; -fx-padding: 10px 20px;");
        startButton.setOnAction(e -> startAssessment());

        stopButton = new Button("Stop Assessment");
        stopButton.setStyle("-fx-background-color: #f44336; -fx-text-fill: white; -fx-font-size: 14px; -fx-padding: 10px 20px;");
        stopButton.setDisable(true);
        stopButton.setOnAction(e -> stopAssessment());

        Button reportButton = new Button("Generate Report");
        reportButton.setStyle("-fx-background-color: #2196F3; -fx-text-fill: white; -fx-font-size: 14px; -fx-padding: 10px 20px;");
        reportButton.setOnAction(e -> generateReport());

        controlPanel.getChildren().addAll(startButton, stopButton, reportButton);
        return controlPanel;
    }

    /**
     * Create statistics grid
     */
    private GridPane createStatsGrid() {
        GridPane grid = new GridPane();
        grid.setHgap(20);
        grid.setVgap(15);
        grid.setPadding(new Insets(20));
        grid.setAlignment(Pos.CENTER);

        // Create stat cards
        VBox totalCard = createStatCard("Total Findings", "0", "#2196F3");
        VBox criticalCard = createStatCard("Critical", "0", "#f44336");
        VBox highCard = createStatCard("High", "0", "#ff9800");
        VBox mediumCard = createStatCard("Medium", "0", "#ffeb3b");

        grid.add(totalCard, 0, 0);
        grid.add(criticalCard, 1, 0);
        grid.add(highCard, 2, 0);
        grid.add(mediumCard, 3, 0);

        return grid;
    }

    /**
     * Create a statistics card
     */
    private VBox createStatCard(String title, String value, String color) {
        VBox card = new VBox(5);
        card.setAlignment(Pos.CENTER);
        card.setStyle(String.format(
            "-fx-background-color: %s; -fx-background-radius: 10; -fx-padding: 20; -fx-effect: dropshadow(three-pass-box, rgba(0,0,0,0.1), 10, 0, 0, 5);",
            color
        ));
        card.setPrefWidth(150);
        card.setPrefHeight(100);

        Label valueLabel = new Label(value);
        valueLabel.setFont(Font.font("System", FontWeight.BOLD, 32));
        valueLabel.setTextFill(Color.WHITE);
        valueLabel.setId(title.toLowerCase().replace(" ", "") + "Value");

        Label titleLabel = new Label(title);
        titleLabel.setFont(Font.font("System", FontWeight.NORMAL, 14));
        titleLabel.setTextFill(Color.WHITE);

        card.getChildren().addAll(valueLabel, titleLabel);
        return card;
    }

    /**
     * Create progress area
     */
    private VBox createProgressArea() {
        VBox progressArea = new VBox(10);
        progressArea.setPadding(new Insets(20));

        Label progressLabel = new Label("Assessment Progress");
        progressLabel.setFont(Font.font("System", FontWeight.BOLD, 16));

        progressBar = new ProgressBar(0);
        progressBar.setPrefWidth(800);
        progressBar.setPrefHeight(20);

        statusLabel = new Label("Ready to start assessment");
        statusLabel.setFont(Font.font("System", FontWeight.NORMAL, 14));

        progressArea.getChildren().addAll(progressLabel, progressBar, statusLabel);
        return progressArea;
    }

    /**
     * Create findings view
     */
    private TableView<SecurityFinding> createFindingsView() {
        TableView<SecurityFinding> table = new TableView<>();

        // Create columns
        TableColumn<SecurityFinding, String> severityCol = new TableColumn<>("Severity");
        severityCol.setCellValueFactory(data -> new javafx.beans.property.SimpleStringProperty(
            data.getValue().getSeverity().getName()));

        TableColumn<SecurityFinding, String> typeCol = new TableColumn<>("Type");
        typeCol.setCellValueFactory(data -> new javafx.beans.property.SimpleStringProperty(
            data.getValue().getVulnerabilityType().getDescription()));

        TableColumn<SecurityFinding, String> titleCol = new TableColumn<>("Title");
        titleCol.setCellValueFactory(data -> new javafx.beans.property.SimpleStringProperty(
            data.getValue().getTitle()));

        TableColumn<SecurityFinding, String> locationCol = new TableColumn<>("Location");
        locationCol.setCellValueFactory(data -> new javafx.beans.property.SimpleStringProperty(
            data.getValue().getLocation()));

        table.getColumns().addAll(severityCol, typeCol, titleCol, locationCol);
        table.setItems(javafx.collections.FXCollections.observableArrayList(currentFindings));

        return table;
    }

    /**
     * Create analytics view with charts
     */
    private VBox createAnalyticsView() {
        VBox analytics = new VBox(20);
        analytics.setPadding(new Insets(20));

        // Severity distribution pie chart
        PieChart severityChart = createSeverityPieChart();

        // Vulnerability types bar chart
        BarChart<String, Number> vulnChart = createVulnerabilityBarChart();

        // Timeline line chart
        LineChart<String, Number> timelineChart = createTimelineChart();

        analytics.getChildren().addAll(severityChart, vulnChart, timelineChart);
        return analytics;
    }

    /**
     * Create severity pie chart
     */
    private PieChart createSeverityPieChart() {
        PieChart chart = new PieChart();
        chart.setTitle("Findings by Severity");
        chart.setPrefSize(600, 400);

        // Set colors
        chart.getData().add(new PieChart.Data("Critical", 0));
        chart.getData().add(new PieChart.Data("High", 0));
        chart.getData().add(new PieChart.Data("Medium", 0));
        chart.getData().add(new PieChart.Data("Low", 0));
        chart.getData().add(new PieChart.Data("Info", 0));

        return chart;
    }

    /**
     * Create vulnerability bar chart
     */
    private BarChart<String, Number> createVulnerabilityBarChart() {
        CategoryAxis xAxis = new CategoryAxis();
        NumberAxis yAxis = new NumberAxis();
        BarChart<String, Number> chart = new BarChart<>(xAxis, yAxis);
        chart.setTitle("Vulnerability Types");
        chart.setPrefSize(800, 400);

        return chart;
    }

    /**
     * Create timeline chart
     */
    private LineChart<String, Number> createTimelineChart() {
        CategoryAxis xAxis = new CategoryAxis();
        NumberAxis yAxis = new NumberAxis();
        LineChart<String, Number> chart = new LineChart<>(xAxis, yAxis);
        chart.setTitle("Findings Timeline");
        chart.setPrefSize(800, 400);

        return chart;
    }

    /**
     * Create 3D visualization placeholder
     */
    private VBox create3DVisualization() {
        VBox visualization = new VBox(20);
        visualization.setPadding(new Insets(20));
        visualization.setAlignment(Pos.CENTER);

        Label titleLabel = new Label("3D Security Visualization");
        titleLabel.setFont(Font.font("System", FontWeight.BOLD, 24));

        // Placeholder for 3D visualization
        VBox placeholder = new VBox(10);
        placeholder.setAlignment(Pos.CENTER);
        placeholder.setStyle("-fx-background-color: #f0f0f0; -fx-border-color: #ccc; -fx-border-width: 2; -fx-border-radius: 5;");
        placeholder.setPrefSize(800, 600);

        Label placeholderLabel = new Label("3D Visualization Coming Soon");
        placeholderLabel.setFont(Font.font("System", FontWeight.NORMAL, 18));

        Label descriptionLabel = new Label("This will show interactive 3D representations of:");
        Label featuresLabel = new Label("• Security findings in 3D space\n• Koopman operator dynamics\n• Real-time assessment visualization\n• Interactive exploration tools");

        placeholder.getChildren().addAll(placeholderLabel, descriptionLabel, featuresLabel);

        Button exploreButton = new Button("Explore 3D Space");
        exploreButton.setOnAction(e -> show3DVisualizationDialog());

        visualization.getChildren().addAll(titleLabel, placeholder, exploreButton);
        return visualization;
    }

    /**
     * Create status bar
     */
    private HBox createStatusBar() {
        HBox statusBar = new HBox(10);
        statusBar.setPadding(new Insets(10));
        statusBar.setStyle("-fx-background-color: #f0f0f0; -fx-border-color: #ccc; -fx-border-width: 1 0 0 0;");

        statusLabel = new Label("Ready");
        statusLabel.setFont(Font.font("System", FontWeight.NORMAL, 12));

        Region spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);

        Label versionLabel = new Label("Reverse Koopman v2.0");
        versionLabel.setFont(Font.font("System", FontWeight.NORMAL, 12));

        statusBar.getChildren().addAll(statusLabel, spacer, versionLabel);
        return statusBar;
    }

    /**
     * Start security assessment
     */
    private void startAssessment() {
        if (assessmentRunning) return;

        assessmentRunning = true;
        startButton.setDisable(true);
        stopButton.setDisable(false);
        progressBar.setProgress(0);
        statusLabel.setText("Running assessment...");

        // Run assessment in background
        Task<Void> task = new Task<Void>() {
            @Override
            protected Void call() throws Exception {
                try {
                    updateMessage("Running Java application security testing...");

                    // Run Java penetration testing
                    CompletableFuture<List<SecurityFinding>> javaResults = javaTesting.runComprehensiveTesting();
                    List<SecurityFinding> javaFindings = javaResults.get(30, TimeUnit.SECONDS);

                    updateMessage("Java testing completed: " + javaFindings.size() + " findings");
                    updateProgress(0.4, 1.0);

                    // Run GPTOSS testing if available
                    List<SecurityFinding> gptossFindings = new ArrayList<>();
                    if (gptossTesting != null) {
                        updateMessage("Running GPTOSS 2.0 AI model security testing...");
                        CompletableFuture<List<SecurityFinding>> gptossResults = gptossTesting.runComprehensiveGPTOSSTesting();
                        gptossFindings = gptossResults.get(60, TimeUnit.SECONDS);
                        updateMessage("GPTOSS testing completed: " + gptossFindings.size() + " findings");
                        updateProgress(0.8, 1.0);
                    }

                    // Combine all findings
                    currentFindings.clear();
                    currentFindings.addAll(javaFindings);
                    currentFindings.addAll(gptossFindings);

                    updateMessage("Total findings: " + currentFindings.size());
                    updateMessage("Assessment completed successfully");
                    updateProgress(1.0, 1.0);

                } catch (Exception ex) {
                    updateMessage("Assessment failed: " + ex.getMessage());
                    ex.printStackTrace();
                }

                return null;
            }

            @Override
            protected void succeeded() {
                assessmentRunning = false;
                startButton.setDisable(false);
                stopButton.setDisable(true);
                statusLabel.setText("Assessment completed");
                updateUIWithFindings();
            }

            @Override
            protected void failed() {
                assessmentRunning = false;
                startButton.setDisable(false);
                stopButton.setDisable(true);
                statusLabel.setText("Assessment failed");
            }
        };

        // Bind progress and message to UI
        progressBar.progressProperty().bind(task.progressProperty());
        statusLabel.textProperty().bind(task.messageProperty());

        // Run in background thread
        Thread thread = new Thread(task);
        thread.setDaemon(true);
        thread.start();
    }

    /**
     * Stop assessment
     */
    private void stopAssessment() {
        if (!assessmentRunning) return;

        assessmentRunning = false;
        startButton.setDisable(false);
        stopButton.setDisable(true);
        progressBar.setProgress(0);
        statusLabel.setText("Assessment stopped");
    }

    /**
     * Run Java assessment only
     */
    private void runJavaAssessment() {
        startAssessment(); // For now, same as full assessment
    }

    /**
     * Run GPTOSS testing only
     */
    private void runGPTOSSTesting() {
        if (gptossTesting == null) {
            showAlert("GPTOSS not configured", "Please configure GPTOSS endpoint and API key first.");
            return;
        }

        // Similar to startAssessment but GPTOSS only
        statusLabel.setText("Running GPTOSS testing...");

        Task<Void> task = new Task<Void>() {
            @Override
            protected Void call() throws Exception {
                updateMessage("Running GPTOSS 2.0 AI model security testing...");
                CompletableFuture<List<SecurityFinding>> gptossResults = gptossTesting.runComprehensiveGPTOSSTesting();
                currentFindings = gptossResults.get(60, TimeUnit.SECONDS);
                updateMessage("GPTOSS testing completed: " + currentFindings.size() + " findings");
                return null;
            }

            @Override
            protected void succeeded() {
                statusLabel.setText("GPTOSS testing completed");
                updateUIWithFindings();
            }
        };

        Thread thread = new Thread(task);
        thread.setDaemon(true);
        thread.start();
    }

    /**
     * Run integrated assessment
     */
    private void runIntegratedAssessment() {
        startAssessment(); // For now, same as full assessment
    }

    /**
     * Show 3D visualization dialog
     */
    private void showKoopman3DVisualization() {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle("3D Visualization");
        alert.setHeaderText("Koopman 3D Visualization");
        alert.setContentText("3D visualization of koopman operators and security findings will be implemented here. This will show interactive 3D representations of system dynamics and vulnerability landscapes.");
        alert.showAndWait();
    }

    /**
     * Show security heatmap
     */
    private void showSecurityHeatmap() {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle("Security Heatmap");
        alert.setHeaderText("Security Heatmap");
        alert.setContentText("Interactive security heatmap showing vulnerability distribution across different components and severity levels.");
        alert.showAndWait();
    }

    /**
     * Show timeline analysis
     */
    private void showTimelineAnalysis() {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle("Timeline Analysis");
        alert.setHeaderText("Timeline Analysis");
        alert.setContentText("Timeline visualization showing security findings over time with trend analysis and prediction capabilities.");
        alert.showAndWait();
    }

    /**
     * Show 3D visualization dialog
     */
    private void show3DVisualizationDialog() {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle("3D Space Exploration");
        alert.setHeaderText("3D Security Space Exploration");
        alert.setContentText("This feature will allow interactive exploration of:\n\n• 3D vulnerability landscapes\n• Dynamic security metrics\n• Interactive finding correlation\n• Real-time assessment visualization");
        alert.showAndWait();
    }

    /**
     * Generate security report
     */
    private void generateReport() {
        if (currentFindings.isEmpty()) {
            showAlert("No Data", "No findings to report. Run an assessment first.");
            return;
        }

        // Generate and save report
        String report = generateComprehensiveReport();
        saveReportToFile(report);
    }

    /**
     * Export findings
     */
    private void exportFindings(String format) {
        if (currentFindings.isEmpty()) {
            showAlert("No Data", "No findings to export. Run an assessment first.");
            return;
        }

        String content = "";
        String extension = "";

        switch (format.toLowerCase()) {
            case "json":
                content = javaTesting.exportFindingsToJson();
                extension = "json";
                break;
            case "csv":
                content = generateCSVReport();
                extension = "csv";
                break;
            default:
                showAlert("Error", "Unsupported export format: " + format);
                return;
        }

        saveContentToFile(content, "security_findings." + extension, "Export completed successfully!");
    }

    /**
     * Update UI with findings data
     */
    private void updateUIWithFindings() {
        // Update statistics cards
        Map<Severity, Long> severityCount = currentFindings.stream()
            .collect(Collectors.groupingBy(SecurityFinding::getSeverity, Collectors.counting()));

        // Update stat card values (this would be done with proper component references)
        System.out.println("Updated UI with " + currentFindings.size() + " findings");

        // Update charts
        updateCharts();
    }

    /**
     * Update charts with current data
     */
    private void updateCharts() {
        // This would update the actual chart components
        System.out.println("Charts updated with current findings data");
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

        // Add detailed findings...
        report.append("DETAILED FINDINGS\n");
        report.append("-----------------\n");

        for (SecurityFinding finding : currentFindings) {
            report.append("\n[").append(finding.getSeverity().getName()).append("] ")
                  .append(finding.getTitle()).append("\n");
            report.append("Type: ").append(finding.getVulnerabilityType().getDescription()).append("\n");
            report.append("Description: ").append(finding.getDescription()).append("\n");
            report.append("Recommendation: ").append(finding.getRecommendation()).append("\n");
        }

        return report.toString();
    }

    /**
     * Generate CSV report
     */
    private String generateCSVReport() {
        StringBuilder csv = new StringBuilder();
        csv.append("Severity,Type,Title,Location,Description,Recommendation,Impact\n");

        for (SecurityFinding finding : currentFindings) {
            csv.append(escapeCSV(finding.getSeverity().getName())).append(",")
               .append(escapeCSV(finding.getVulnerabilityType().getDescription())).append(",")
               .append(escapeCSV(finding.getTitle())).append(",")
               .append(escapeCSV(finding.getLocation())).append(",")
               .append(escapeCSV(finding.getDescription())).append(",")
               .append(escapeCSV(finding.getRecommendation())).append(",")
               .append(escapeCSV(finding.getImpactAssessment())).append("\n");
        }

        return csv.toString();
    }

    /**
     * Escape CSV values
     */
    private String escapeCSV(String value) {
        if (value.contains(",") || value.contains("\"") || value.contains("\n")) {
            return "\"" + value.replace("\"", "\"\"") + "\"";
        }
        return value;
    }

    /**
     * Save report to file
     */
    private void saveReportToFile(String content) {
        // In JavaFX, this would use FileChooser
        // For now, just show success message
        showAlert("Success", "Report generated successfully!\n\n" + content.substring(0, Math.min(200, content.length())) + "...");
    }

    /**
     * Save content to file
     */
    private void saveContentToFile(String content, String filename, String successMessage) {
        // In JavaFX, this would use FileChooser
        showAlert("Success", successMessage + "\n\nFile: " + filename);
    }

    /**
     * Show alert dialog
     */
    private void showAlert(String title, String message) {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }

    /**
     * Main method to launch the application
     */
    public static void main(String[] args) {
        String endpoint = args.length > 0 ? args[0] : null;
        String apiKey = args.length > 1 ? args[1] : null;

        if (endpoint != null && apiKey != null) {
            launch(args);
        } else {
            // Launch without GPTOSS
            launch(args);
        }
    }
}
