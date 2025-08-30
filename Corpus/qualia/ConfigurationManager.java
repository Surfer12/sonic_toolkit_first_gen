// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
//
//  ConfigurationManager.java
//  qualia
//
//  Created by Ryan David Oates on 8/26/25.
//  Configuration management for security testing framework

package qualia;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.function.*;
import java.util.stream.Collectors;

/**
 * Configuration Management System for Qualia Security Framework
 *
 * Supports multiple configuration formats:
 * - Properties files (.properties)
 * - YAML files (.yml/.yaml)
 * - Environment variables
 * - Dynamic reloading
 * - Configuration validation
 */
public class ConfigurationManager {

    private static final ConfigurationManager instance = new ConfigurationManager();
    private final Properties properties;
    private final Map<String, Object> yamlConfig;
    private final Map<String, String> environmentVariables;
    private final AtomicLong lastModified;
    private final ScheduledExecutorService reloadExecutor;
    private volatile boolean initialized = false;

    // Configuration file paths
    private static final String PROPERTIES_FILE = "config/application.properties";
    private static final String YAML_FILE = "config/application.yml";
    private static final long RELOAD_INTERVAL_MS = 30000; // 30 seconds

    // Configuration validation schemas
    private static final Map<String, ValidationRule> VALIDATION_RULES = new HashMap<>();

    static {
        // Initialize validation rules
        VALIDATION_RULES.put("security.scan.timeout", new ValidationRule(
            value -> Integer.parseInt(value) > 0 && Integer.parseInt(value) <= 300000,
            "Security scan timeout must be between 1 and 300000 ms"
        ));

        VALIDATION_RULES.put("security.scan.threads", new ValidationRule(
            value -> Integer.parseInt(value) > 0 && Integer.parseInt(value) <= 100,
            "Security scan threads must be between 1 and 100"
        ));

        VALIDATION_RULES.put("database.connectionTimeout", new ValidationRule(
            value -> Integer.parseInt(value) > 0 && Integer.parseInt(value) <= 60000,
            "Database connection timeout must be between 1 and 60000 ms"
        ));

        VALIDATION_RULES.put("validation.ks.confidenceThreshold", new ValidationRule(
            value -> Double.parseDouble(value) > 0.0 && Double.parseDouble(value) < 1.0,
            "K-S confidence threshold must be between 0.0 and 1.0"
        ));

        VALIDATION_RULES.put("performance.memory.threshold", new ValidationRule(
            value -> Double.parseDouble(value) > 0.0 && Double.parseDouble(value) < 1.0,
            "Memory threshold must be between 0.0 and 1.0"
        ));
    }

    private ConfigurationManager() {
        this.properties = new Properties();
        this.yamlConfig = new ConcurrentHashMap<>();
        this.environmentVariables = new ConcurrentHashMap<>();
        this.lastModified = new AtomicLong(0);
        this.reloadExecutor = Executors.newScheduledThreadPool(1);

        initializeConfiguration();
        startReloadScheduler();
    }

    /**
     * Get singleton instance
     */
    public static ConfigurationManager getInstance() {
        return instance;
    }

    /**
     * Initialize configuration from all sources
     */
    private void initializeConfiguration() {
        try {
            loadEnvironmentVariables();
            loadPropertiesFile();
            loadYamlFile();
            validateConfiguration();
            initialized = true;

            System.out.println("✅ Configuration initialized successfully");
            logConfigurationSummary();

        } catch (Exception e) {
            System.err.println("❌ Configuration initialization failed: " + e.getMessage());
            throw new RuntimeException("Failed to initialize configuration", e);
        }
    }

    /**
     * Load environment variables
     */
    private void loadEnvironmentVariables() {
        System.getenv().forEach((key, value) -> {
            if (key.startsWith("QUALIA_") || key.startsWith("DATABASE_") ||
                key.startsWith("REDIS_") || key.startsWith("GPTOSS_")) {
                environmentVariables.put(key, value);
            }
        });
    }

    /**
     * Load properties file
     */
    private void loadPropertiesFile() {
        Path propertiesPath = Paths.get(PROPERTIES_FILE);
        if (Files.exists(propertiesPath)) {
            try (InputStream is = Files.newInputStream(propertiesPath)) {
                properties.load(is);
                lastModified.set(Files.getLastModifiedTime(propertiesPath).toMillis());
                System.out.println("✅ Loaded properties file: " + PROPERTIES_FILE);
            } catch (IOException e) {
                System.err.println("⚠️ Failed to load properties file: " + e.getMessage());
            }
        } else {
            System.out.println("ℹ️ Properties file not found: " + PROPERTIES_FILE);
        }
    }

    /**
     * Load YAML file (simplified implementation)
     */
    private void loadYamlFile() {
        Path yamlPath = Paths.get(YAML_FILE);
        if (Files.exists(yamlPath)) {
            try {
                // For this implementation, we'll use a simple approach
                // In production, you'd use a YAML library like SnakeYAML
                List<String> lines = Files.readAllLines(yamlPath);
                parseYamlLines(lines);
                System.out.println("✅ Loaded YAML file: " + YAML_FILE);
            } catch (IOException e) {
                System.err.println("⚠️ Failed to load YAML file: " + e.getMessage());
            }
        } else {
            System.out.println("ℹ️ YAML file not found: " + YAML_FILE);
        }
    }

    /**
     * Simple YAML parser (production would use proper library)
     */
    private void parseYamlLines(List<String> lines) {
        String currentSection = "";
        for (String line : lines) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;

            if (line.endsWith(":")) {
                currentSection = line.substring(0, line.length() - 1);
            } else if (line.contains(": ")) {
                String[] parts = line.split(": ", 2);
                String key = currentSection.isEmpty() ? parts[0] :
                           currentSection + "." + parts[0];
                String value = parts[1].replaceAll("^\"|\"$", ""); // Remove quotes
                yamlConfig.put(key, parseValue(value));
            }
        }
    }

    /**
     * Parse YAML value to appropriate type
     */
    private Object parseValue(String value) {
        if (value.equalsIgnoreCase("true")) return true;
        if (value.equalsIgnoreCase("false")) return false;
        if (value.matches("-?\\d+")) return Integer.parseInt(value);
        if (value.matches("-?\\d+\\.\\d+")) return Double.parseDouble(value);
        if (value.startsWith("[") && value.endsWith("]")) {
            return Arrays.asList(value.substring(1, value.length() - 1)
                              .split(",")).stream()
                              .map(String::trim)
                              .collect(Collectors.toList());
        }
        return value;
    }

    /**
     * Validate configuration against rules
     */
    private void validateConfiguration() throws ConfigurationException {
        List<String> errors = new ArrayList<>();

        // Validate all configuration sources
        validateSource(properties, "properties", errors);
        validateSource(yamlConfig, "yaml", errors);
        validateSource(environmentVariables, "environment", errors);

        if (!errors.isEmpty()) {
            throw new ConfigurationException("Configuration validation failed: " + errors);
        }
    }

    /**
     * Validate configuration source
     */
    private void validateSource(Map<?, ?> source, String sourceName, List<String> errors) {
        for (Object key : source.keySet()) {
            String keyStr = key.toString();
            if (VALIDATION_RULES.containsKey(keyStr)) {
                ValidationRule rule = VALIDATION_RULES.get(keyStr);
                String value = source.get(key).toString();
                if (!rule.validator.test(value)) {
                    errors.add(String.format("[%s] %s: %s", sourceName, keyStr, rule.errorMessage));
                }
            }
        }
    }

    /**
     * Start automatic reload scheduler
     */
    private void startReloadScheduler() {
        reloadExecutor.scheduleAtFixedRate(
            this::checkForConfigurationUpdates,
            RELOAD_INTERVAL_MS,
            RELOAD_INTERVAL_MS,
            TimeUnit.MILLISECONDS
        );
    }

    /**
     * Check for configuration file updates
     */
    private void checkForConfigurationUpdates() {
        try {
            Path propertiesPath = Paths.get(PROPERTIES_FILE);
            if (Files.exists(propertiesPath)) {
                long currentModified = Files.getLastModifiedTime(propertiesPath).toMillis();
                if (currentModified > lastModified.get()) {
                    System.out.println("🔄 Configuration file updated, reloading...");
                    loadPropertiesFile();
                    validateConfiguration();
                    lastModified.set(currentModified);
                    System.out.println("✅ Configuration reloaded successfully");
                }
            }
        } catch (Exception e) {
            System.err.println("❌ Configuration reload failed: " + e.getMessage());
        }
    }

    /**
     * Get configuration value with priority: Environment > Properties > YAML > Default
     */
    public String getString(String key, String defaultValue) {
        return getString(key).orElse(defaultValue);
    }

    public Optional<String> getString(String key) {
        // Priority: Environment > Properties > YAML
        String envKey = "QUALIA_" + key.toUpperCase().replace(".", "_");
        if (environmentVariables.containsKey(envKey)) {
            return Optional.of(environmentVariables.get(envKey));
        }

        String propValue = properties.getProperty(key);
        if (propValue != null) {
            return Optional.of(propValue);
        }

        if (yamlConfig.containsKey(key)) {
            return Optional.of(yamlConfig.get(key).toString());
        }

        return Optional.empty();
    }

    /**
     * Get integer configuration value
     */
    public int getInt(String key, int defaultValue) {
        return getString(key).map(Integer::parseInt).orElse(defaultValue);
    }

    /**
     * Get boolean configuration value
     */
    public boolean getBoolean(String key, boolean defaultValue) {
        return getString(key).map(Boolean::parseBoolean).orElse(defaultValue);
    }

    /**
     * Get double configuration value
     */
    public double getDouble(String key, double defaultValue) {
        return getString(key).map(Double::parseDouble).orElse(defaultValue);
    }

    /**
     * Get list configuration value
     */
    @SuppressWarnings("unchecked")
    public List<String> getList(String key, List<String> defaultValue) {
        if (yamlConfig.containsKey(key)) {
            Object value = yamlConfig.get(key);
            if (value instanceof List) {
                return (List<String>) value;
            }
        }
        return defaultValue;
    }

    /**
     * Log configuration summary
     */
    private void logConfigurationSummary() {
        System.out.println("📊 Configuration Summary:");
        System.out.println("  Properties loaded: " + properties.size());
        System.out.println("  YAML config keys: " + yamlConfig.size());
        System.out.println("  Environment vars: " + environmentVariables.size());
        System.out.println("  Validation rules: " + VALIDATION_RULES.size());
    }

    /**
     * Shutdown configuration manager
     */
    public void shutdown() {
        reloadExecutor.shutdown();
        try {
            if (!reloadExecutor.awaitTermination(5, TimeUnit.SECONDS)) {
                reloadExecutor.shutdownNow();
            }
        } catch (InterruptedException e) {
            reloadExecutor.shutdownNow();
        }
    }

    /**
     * Validation rule class
     */
    private static class ValidationRule {
        final Predicate<String> validator;
        final String errorMessage;

        ValidationRule(Predicate<String> validator, String errorMessage) {
            this.validator = validator;
            this.errorMessage = errorMessage;
        }
    }

    /**
     * Configuration exception
     */
    public static class ConfigurationException extends Exception {
        public ConfigurationException(String message) {
            super(message);
        }
    }
}
