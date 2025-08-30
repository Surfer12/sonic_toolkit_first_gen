// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 Jumping Quail Solutions
//
//  SecurityEventMonitor.java
//  qualia
//
//  Created by Ryan David Oates on 8/26/25.
//  Real-time security event monitoring and response system
package qualia;

import java.util.*;
import java.util.concurrent.*;
import java.security.SecureRandom;
import java.time.*;
import java.time.format.DateTimeFormatter;
import java.lang.management.*;

/**
 * Security event types for monitoring
 */
enum SecurityEventType {
    AUTHENTICATION_FAILURE("Authentication Failure"),
    AUTHORIZATION_VIOLATION("Authorization Violation"),
    NETWORK_INTRUSION("Network Intrusion"),
    MALWARE_DETECTION("Malware Detection"),
    DATA_EXFILTRATION("Data Exfiltration"),
    PRIVILEGE_ESCALATION("Privilege Escalation"),
    UNUSUAL_ACTIVITY("Unusual Activity"),
    SYSTEM_COMPROMISE("System Compromise"),
    VULNERABILITY_EXPLOIT("Vulnerability Exploit"),
    PENETRATION_TESTING("Penetration Testing"),
    MATHEMATICAL_ANALYSIS("Mathematical Analysis"),
    RESOURCE_EXHAUSTION("Resource Exhaustion");

    private final String description;

    SecurityEventType(String description) {
        this.description = description;
    }

    public String getDescription() { return description; }
}

/**
 * Security event priority levels
 */
enum EventPriority {
    EMERGENCY("Emergency", 1),
    ALERT("Alert", 2),
    CRITICAL("Critical", 3),
    ERROR("Error", 4),
    WARNING("Warning", 5),
    NOTICE("Notice", 6),
    INFO("Info", 7),
    DEBUG("Debug", 8);

    private final String level;
    private final int priority;

    EventPriority(String level, int priority) {
        this.level = level;
        this.priority = priority;
    }

    public String getLevel() { return level; }
    public int getPriority() { return priority; }
}

/**
 * Security event data structure
 */
class SecurityEvent {
    private final String eventId;
    private final SecurityEventType eventType;
    private final EventPriority priority;
    private final String source;
    private final String destination;
    private final String description;
    private final LocalDateTime timestamp;
    private final Map<String, Object> attributes;
    private boolean handled;
    private String response;

    public SecurityEvent(SecurityEventType eventType, EventPriority priority,
                        String source, String destination, String description) {
        this.eventId = generateEventId();
        this.eventType = eventType;
        this.priority = priority;
        this.source = source;
        this.destination = destination;
        this.description = description;
        this.timestamp = LocalDateTime.now();
        this.attributes = new HashMap<>();
        this.handled = false;
        this.response = "";
    }

    private String generateEventId() {
        return "EVT-" + System.currentTimeMillis() + "-" + 
               Integer.toHexString(new SecureRandom().nextInt());
    }

    // Getters and setters
    public String getEventId() { return eventId; }
    public SecurityEventType getEventType() { return eventType; }
    public EventPriority getPriority() { return priority; }
    public String getSource() { return source; }
    public String getDestination() { return destination; }
    public String getDescription() { return description; }
    public LocalDateTime getTimestamp() { return timestamp; }
    public Map<String, Object> getAttributes() { return new HashMap<>(attributes); }
    public boolean isHandled() { return handled; }
    public String getResponse() { return response; }

    public void setHandled(boolean handled) { this.handled = handled; }
    public void setResponse(String response) { this.response = response; }
    public void addAttribute(String key, Object value) { attributes.put(key, value); }

    @Override
    public String toString() {
        return String.format("[%s] %s: %s -> %s | %s (%s)",
                           priority.getLevel(), eventType.getDescription(),
                           source, destination, description,
                           timestamp.format(DateTimeFormatter.ISO_LOCAL_TIME));
    }
}

/**
 * Security Event Monitor
 * Real-time monitoring and automated response to security events
 */
public class SecurityEventMonitor {

    private final BlockingQueue<SecurityEvent> eventQueue;
    private final List<SecurityEvent> eventHistory;
    private final ExecutorService monitoringExecutor;
    private final ExecutorService responseExecutor;
    private final Map<SecurityEventType, Integer> eventCounters;
    private final Set<String> isolatedSystems;
    private final SecureRandom secureRandom;
    private boolean realTimeMonitoring;
    private boolean enhancedMode;

    public SecurityEventMonitor() {
        this.eventQueue = new LinkedBlockingQueue<>();
        this.eventHistory = Collections.synchronizedList(new ArrayList<>());
        this.monitoringExecutor = Executors.newFixedThreadPool(5);
        this.responseExecutor = Executors.newFixedThreadPool(3);
        this.eventCounters = new ConcurrentHashMap<>();
        this.isolatedSystems = ConcurrentHashMap.newKeySet();
        this.secureRandom = new SecureRandom();
        this.realTimeMonitoring = false;
        this.enhancedMode = false;
    }

    /**
     * Start real-time security monitoring
     */
    public void startRealTimeMonitoring() {
        realTimeMonitoring = true;
        
        // Start monitoring threads
        monitoringExecutor.submit(this::eventProcessor);
        monitoringExecutor.submit(this::systemMonitor);
        monitoringExecutor.submit(this::behaviorAnalyzer);
        monitoringExecutor.submit(this::threatCorrelator);
        
        System.out.println("🔍 Real-time security monitoring started");
    }

    /**
     * Process security events from the queue
     */
    private void eventProcessor() {
        while (realTimeMonitoring && !Thread.currentThread().isInterrupted()) {
            try {
                SecurityEvent event = eventQueue.take();
                processSecurityEvent(event);
                
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                System.err.println("Error processing security event: " + e.getMessage());
            }
        }
    }

    /**
     * Process individual security event
     */
    private void processSecurityEvent(SecurityEvent event) {
        System.out.println("🚨 Processing: " + event);
        
        // Add to history
        eventHistory.add(event);
        
        // Update counters
        eventCounters.merge(event.getEventType(), 1, Integer::sum);
        
        // Determine response based on priority and type
        determineResponse(event);
        
        // Execute automated response if needed
        if (shouldExecuteAutomatedResponse(event)) {
            executeAutomatedResponse(event);
        }
        
        // Mark as handled
        event.setHandled(true);
    }

    /**
     * Determine appropriate response for security event
     */
    private void determineResponse(SecurityEvent event) {
        String response = "";
        
        switch (event.getPriority()) {
            case EMERGENCY:
                response = "IMMEDIATE SYSTEM ISOLATION AND ALERT";
                break;
            case ALERT:
                response = "URGENT SECURITY TEAM NOTIFICATION";
                break;
            case CRITICAL:
                response = "AUTOMATED CONTAINMENT AND ANALYSIS";
                break;
            case ERROR:
                response = "ENHANCED MONITORING AND LOGGING";
                break;
            case WARNING:
                response = "CONTINUOUS MONITORING";
                break;
            default:
                response = "LOG AND MONITOR";
                break;
        }
        
        // Add specific responses based on event type
        switch (event.getEventType()) {
            case PENETRATION_TESTING:
                response += " | DEPLOY ANTI-PENETRATION COUNTERMEASURES";
                break;
            case MATHEMATICAL_ANALYSIS:
                response += " | ACTIVATE MATHEMATICAL NOISE INJECTION";
                break;
            case NETWORK_INTRUSION:
                response += " | NETWORK SEGMENTATION AND TRAFFIC ANALYSIS";
                break;
            case MALWARE_DETECTION:
                response += " | QUARANTINE AND MALWARE ANALYSIS";
                break;
            case DATA_EXFILTRATION:
                response += " | DATA LOSS PREVENTION AND FORENSICS";
                break;
        }
        
        event.setResponse(response);
    }

    /**
     * Check if automated response should be executed
     */
    private boolean shouldExecuteAutomatedResponse(SecurityEvent event) {
        return event.getPriority().getPriority() <= 4; // Emergency through Error
    }

    /**
     * Execute automated response
     */
    private void executeAutomatedResponse(SecurityEvent event) {
        responseExecutor.submit(() -> {
            System.out.println("🤖 Executing automated response for: " + event.getEventId());
            
            switch (event.getPriority()) {
                case EMERGENCY:
                    executeEmergencyResponse(event);
                    break;
                case ALERT:
                    executeAlertResponse(event);
                    break;
                case CRITICAL:
                    executeCriticalResponse(event);
                    break;
                case ERROR:
                    executeErrorResponse(event);
                    break;
            }
        });
    }

    /**
     * Execute emergency response procedures
     */
    private void executeEmergencyResponse(SecurityEvent event) {
        System.out.println("🚨 EMERGENCY RESPONSE ACTIVATED");
        
        // Immediate system isolation
        isolateAffectedSystems();
        
        // Alert all security personnel
        alertSecurityTeam("EMERGENCY", event);
        
        // Preserve forensic evidence
        preserveEvidence(event);
        
        // Activate backup systems
        activateBackupSystems();
    }

    /**
     * Execute alert response procedures
     */
    private void executeAlertResponse(SecurityEvent event) {
        System.out.println("⚠️  ALERT RESPONSE ACTIVATED");
        
        // Enhanced monitoring
        escalateMonitoring();
        
        // Security team notification
        alertSecurityTeam("ALERT", event);
        
        // Additional logging
        enableDetailedLogging();
    }

    /**
     * Execute critical response procedures
     */
    private void executeCriticalResponse(SecurityEvent event) {
        System.out.println("🔴 CRITICAL RESPONSE ACTIVATED");
        
        // Containment measures
        executeContainmentMeasures(event);
        
        // Forensic data collection
        collectForensicData(event);
        
        // Threat hunting activation
        activateThreatHunting();
    }

    /**
     * Execute error response procedures
     */
    private void executeErrorResponse(SecurityEvent event) {
        System.out.println("🟡 ERROR RESPONSE ACTIVATED");
        
        // Enhanced monitoring for related events
        monitorRelatedEvents(event);
        
        // Pattern analysis
        analyzeEventPatterns();
    }

    /**
     * System monitoring for security events
     */
    private void systemMonitor() {
        while (realTimeMonitoring && !Thread.currentThread().isInterrupted()) {
            try {
                // Monitor system resources
                monitorSystemResources();
                
                // Monitor network activity
                monitorNetworkActivity();
                
                // Monitor file system
                monitorFileSystem();
                
                // Monitor process activity
                monitorProcessActivity();
                
                Thread.sleep(5000); // Monitor every 5 seconds
                
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                System.err.println("Error in system monitoring: " + e.getMessage());
            }
        }
    }

    /**
     * Monitor system resources for anomalies
     */
    private void monitorSystemResources() {
        // Monitor memory usage
        MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
        MemoryUsage heapUsage = memoryBean.getHeapMemoryUsage();
        
        double memoryUsagePercent = (double) heapUsage.getUsed() / heapUsage.getMax();
        if (memoryUsagePercent > 0.95) {
            queueEvent(new SecurityEvent(
                SecurityEventType.RESOURCE_EXHAUSTION,
                EventPriority.CRITICAL,
                "System Memory",
                "Memory Monitor",
                "Critical memory usage: " + String.format("%.1f%%", memoryUsagePercent * 100)
            ));
        }
        
        // Monitor CPU usage (simplified)
        if (secureRandom.nextDouble() < 0.02) { // 2% chance for simulation
            queueEvent(new SecurityEvent(
                SecurityEventType.UNUSUAL_ACTIVITY,
                EventPriority.WARNING,
                "System CPU",
                "CPU Monitor",
                "Unusual CPU activity pattern detected"
            ));
        }
    }

    /**
     * Monitor network activity
     */
    private void monitorNetworkActivity() {
        // Simulate network monitoring
        if (secureRandom.nextDouble() < 0.01) { // 1% chance for simulation
            queueEvent(new SecurityEvent(
                SecurityEventType.NETWORK_INTRUSION,
                EventPriority.ALERT,
                "External IP",
                "Internal Network",
                "Suspicious network connection attempt detected"
            ));
        }
        
        // Monitor for port scans
        if (secureRandom.nextDouble() < 0.005) { // 0.5% chance for simulation
            queueEvent(new SecurityEvent(
                SecurityEventType.PENETRATION_TESTING,
                EventPriority.CRITICAL,
                "Unknown Host",
                "Server Ports",
                "Port scanning activity detected"
            ));
        }
    }

    /**
     * Monitor file system
     */
    private void monitorFileSystem() {
        // Monitor for unauthorized file access
        if (secureRandom.nextDouble() < 0.008) { // 0.8% chance for simulation
            queueEvent(new SecurityEvent(
                SecurityEventType.AUTHORIZATION_VIOLATION,
                EventPriority.ERROR,
                "Unknown Process",
                "Sensitive Files",
                "Unauthorized file access attempt"
            ));
        }
    }

    /**
     * Monitor process activity
     */
    private void monitorProcessActivity() {
        // Monitor for malicious processes
        if (secureRandom.nextDouble() < 0.003) { // 0.3% chance for simulation
            queueEvent(new SecurityEvent(
                SecurityEventType.MALWARE_DETECTION,
                EventPriority.EMERGENCY,
                "Unknown Process",
                "System",
                "Potential malware process detected"
            ));
        }
    }

    /**
     * Behavioral analysis for anomaly detection
     */
    private void behaviorAnalyzer() {
        while (realTimeMonitoring && !Thread.currentThread().isInterrupted()) {
            try {
                // Analyze user behavior patterns
                analyzeUserBehavior();
                
                // Analyze system behavior patterns
                analyzeSystemBehavior();
                
                // Analyze attack patterns
                analyzeAttackPatterns();
                
                Thread.sleep(10000); // Analyze every 10 seconds
                
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                System.err.println("Error in behavioral analysis: " + e.getMessage());
            }
        }
    }

    /**
     * Analyze user behavior patterns
     */
    private void analyzeUserBehavior() {
        // Simulate user behavior analysis
        if (secureRandom.nextDouble() < 0.01) { // 1% chance for simulation
            queueEvent(new SecurityEvent(
                SecurityEventType.UNUSUAL_ACTIVITY,
                EventPriority.WARNING,
                "User Account",
                "System Resources",
                "Anomalous user behavior pattern detected"
            ));
        }
    }

    /**
     * Analyze system behavior patterns
     */
    private void analyzeSystemBehavior() {
        // Check for mathematical analysis patterns
        if (secureRandom.nextDouble() < 0.015) { // 1.5% chance for simulation
            queueEvent(new SecurityEvent(
                SecurityEventType.MATHEMATICAL_ANALYSIS,
                EventPriority.CRITICAL,
                "Analysis Process",
                "System State",
                "Mathematical system analysis detected"
            ));
        }
    }

    /**
     * Analyze attack patterns
     */
    private void analyzeAttackPatterns() {
        // Look for coordinated attack patterns
        if (eventCounters.values().stream().mapToInt(Integer::intValue).sum() > 50) {
            queueEvent(new SecurityEvent(
                SecurityEventType.SYSTEM_COMPROMISE,
                EventPriority.EMERGENCY,
                "Multiple Sources",
                "System",
                "Coordinated attack pattern detected"
            ));
        }
    }

    /**
     * Threat correlation engine
     */
    private void threatCorrelator() {
        while (realTimeMonitoring && !Thread.currentThread().isInterrupted()) {
            try {
                // Correlate related events
                correlateEvents();
                
                // Identify attack chains
                identifyAttackChains();
                
                // Predict potential threats
                predictThreats();
                
                Thread.sleep(15000); // Correlate every 15 seconds
                
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                System.err.println("Error in threat correlation: " + e.getMessage());
            }
        }
    }

    /**
     * Correlate related security events
     */
    private void correlateEvents() {
        // Analyze event patterns in recent history
        List<SecurityEvent> recentEvents = getRecentEvents(Duration.ofMinutes(5));
        
        if (recentEvents.size() > 10) {
            // Check for event clustering
            Map<SecurityEventType, Long> typeCount = recentEvents.stream()
                .collect(Collectors.groupingBy(SecurityEvent::getEventType, Collectors.counting()));
            
            for (Map.Entry<SecurityEventType, Long> entry : typeCount.entrySet()) {
                if (entry.getValue() > 5) {
                    queueEvent(new SecurityEvent(
                        SecurityEventType.UNUSUAL_ACTIVITY,
                        EventPriority.ALERT,
                        "Event Correlation",
                        "Security Monitor",
                        "High frequency of " + entry.getKey().getDescription() + " events"
                    ));
                }
            }
        }
    }

    /**
     * Identify attack chains
     */
    private void identifyAttackChains() {
        // Look for attack progression patterns
        List<SecurityEvent> recentEvents = getRecentEvents(Duration.ofMinutes(10));
        
        // Check for escalation patterns
        boolean hasRecon = recentEvents.stream()
            .anyMatch(e -> e.getEventType() == SecurityEventType.PENETRATION_TESTING);
        boolean hasExploit = recentEvents.stream()
            .anyMatch(e -> e.getEventType() == SecurityEventType.VULNERABILITY_EXPLOIT);
        boolean hasEscalation = recentEvents.stream()
            .anyMatch(e -> e.getEventType() == SecurityEventType.PRIVILEGE_ESCALATION);
        
        if (hasRecon && hasExploit && hasEscalation) {
            queueEvent(new SecurityEvent(
                SecurityEventType.SYSTEM_COMPROMISE,
                EventPriority.EMERGENCY,
                "Attack Chain",
                "System",
                "Complete attack chain detected: Recon -> Exploit -> Escalation"
            ));
        }
    }

    /**
     * Predict potential threats
     */
    private void predictThreats() {
        // Predictive threat analysis based on patterns
        if (secureRandom.nextDouble() < 0.005) { // 0.5% chance for simulation
            queueEvent(new SecurityEvent(
                SecurityEventType.UNUSUAL_ACTIVITY,
                EventPriority.NOTICE,
                "Threat Prediction",
                "System",
                "Potential threat predicted based on behavioral patterns"
            ));
        }
    }

    /**
     * Queue a security event for processing
     */
    public void queueEvent(SecurityEvent event) {
        try {
            eventQueue.put(event);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.println("Failed to queue security event: " + e.getMessage());
        }
    }

    /**
     * Isolate affected systems
     */
    public void isolateAffectedSystems() {
        System.out.println("🔒 ISOLATING AFFECTED SYSTEMS");
        
        // Simulate system isolation
        isolatedSystems.add("Web Server");
        isolatedSystems.add("Database Server");
        isolatedSystems.add("Application Server");
        
        System.out.println("Systems isolated: " + isolatedSystems);
    }

    /**
     * Escalate monitoring to enhanced mode
     */
    public void escalateMonitoring() {
        enhancedMode = true;
        System.out.println("🔍 ENHANCED MONITORING ACTIVATED");
        
        // Additional monitoring capabilities
        monitoringExecutor.submit(this::enhancedNetworkMonitoring);
        monitoringExecutor.submit(this::enhancedBehaviorAnalysis);
    }

    /**
     * Enhanced network monitoring
     */
    private void enhancedNetworkMonitoring() {
        System.out.println("🌐 Enhanced network monitoring started");
        
        while (enhancedMode && !Thread.currentThread().isInterrupted()) {
            try {
                // Deep packet inspection simulation
                if (secureRandom.nextDouble() < 0.02) {
                    queueEvent(new SecurityEvent(
                        SecurityEventType.NETWORK_INTRUSION,
                        EventPriority.WARNING,
                        "Network Monitor",
                        "Traffic Analysis",
                        "Suspicious network pattern detected in enhanced mode"
                    ));
                }
                
                Thread.sleep(2000); // Enhanced monitoring every 2 seconds
                
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
    }

    /**
     * Enhanced behavior analysis
     */
    private void enhancedBehaviorAnalysis() {
        System.out.println("🧠 Enhanced behavior analysis started");
        
        while (enhancedMode && !Thread.currentThread().isInterrupted()) {
            try {
                // AI-powered behavior analysis simulation
                if (secureRandom.nextDouble() < 0.015) {
                    queueEvent(new SecurityEvent(
                        SecurityEventType.UNUSUAL_ACTIVITY,
                        EventPriority.INFO,
                        "Behavior Analyzer",
                        "System",
                        "AI detected subtle behavioral anomaly"
                    ));
                }
                
                Thread.sleep(3000); // Enhanced analysis every 3 seconds
                
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
    }

    /**
     * Alert security team
     */
    private void alertSecurityTeam(String level, SecurityEvent event) {
        System.out.println("📧 ALERTING SECURITY TEAM: " + level);
        System.out.println("Event: " + event);
        // In practice, this would send notifications via email, SMS, etc.
    }

    /**
     * Preserve forensic evidence
     */
    private void preserveEvidence(SecurityEvent event) {
        System.out.println("🔍 PRESERVING FORENSIC EVIDENCE");
        event.addAttribute("evidence_preserved", true);
        event.addAttribute("preservation_time", LocalDateTime.now());
    }

    /**
     * Activate backup systems
     */
    private void activateBackupSystems() {
        System.out.println("🔄 ACTIVATING BACKUP SYSTEMS");
        // Simulate backup system activation
    }

    /**
     * Enable detailed logging
     */
    private void enableDetailedLogging() {
        System.out.println("📝 DETAILED LOGGING ENABLED");
        // Simulate enhanced logging configuration
    }

    /**
     * Execute containment measures
     */
    private void executeContainmentMeasures(SecurityEvent event) {
        System.out.println("🛡️  EXECUTING CONTAINMENT MEASURES");
        event.addAttribute("containment_executed", true);
    }

    /**
     * Collect forensic data
     */
    private void collectForensicData(SecurityEvent event) {
        System.out.println("🔬 COLLECTING FORENSIC DATA");
        event.addAttribute("forensic_data_collected", true);
    }

    /**
     * Activate threat hunting
     */
    private void activateThreatHunting() {
        System.out.println("🎯 THREAT HUNTING ACTIVATED");
        monitoringExecutor.submit(this::proactiveThreatHunting);
    }

    /**
     * Proactive threat hunting
     */
    private void proactiveThreatHunting() {
        System.out.println("🕵️ Proactive threat hunting started");
        
        for (int i = 0; i < 10 && !Thread.currentThread().isInterrupted(); i++) {
            try {
                // Hunt for hidden threats
                if (secureRandom.nextDouble() < 0.1) {
                    queueEvent(new SecurityEvent(
                        SecurityEventType.UNUSUAL_ACTIVITY,
                        EventPriority.WARNING,
                        "Threat Hunter",
                        "Hidden Threat",
                        "Potential hidden threat discovered during hunting"
                    ));
                }
                
                Thread.sleep(5000);
                
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
    }

    /**
     * Monitor related events
     */
    private void monitorRelatedEvents(SecurityEvent event) {
        System.out.println("👀 MONITORING RELATED EVENTS");
        // Implement related event monitoring logic
    }

    /**
     * Analyze event patterns
     */
    private void analyzeEventPatterns() {
        System.out.println("📊 ANALYZING EVENT PATTERNS");
        // Implement pattern analysis logic
    }

    /**
     * Get recent events within specified duration
     */
    private List<SecurityEvent> getRecentEvents(Duration duration) {
        LocalDateTime cutoff = LocalDateTime.now().minus(duration);
        
        synchronized (eventHistory) {
            return eventHistory.stream()
                .filter(event -> event.getTimestamp().isAfter(cutoff))
                .collect(Collectors.toList());
        }
    }

    /**
     * Generate security monitoring report
     */
    public String generateMonitoringReport() {
        StringBuilder report = new StringBuilder();
        report.append("=== Security Event Monitoring Report ===\n\n");
        report.append("Generated: ").append(LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME)).append("\n\n");
        
        // Event statistics
        report.append("=== Event Statistics ===\n");
        report.append("Total Events: ").append(eventHistory.size()).append("\n");
        report.append("Events in Queue: ").append(eventQueue.size()).append("\n");
        report.append("Isolated Systems: ").append(isolatedSystems.size()).append("\n");
        report.append("Enhanced Mode: ").append(enhancedMode ? "Active" : "Inactive").append("\n\n");
        
        // Event type breakdown
        report.append("=== Event Type Breakdown ===\n");
        eventCounters.entrySet().stream()
            .sorted(Map.Entry.<SecurityEventType, Integer>comparingByValue().reversed())
            .forEach(entry -> {
                report.append(entry.getKey().getDescription()).append(": ")
                      .append(entry.getValue()).append("\n");
            });
        
        // Recent critical events
        List<SecurityEvent> criticalEvents = getRecentEvents(Duration.ofHours(1)).stream()
            .filter(event -> event.getPriority().getPriority() <= 3)
            .collect(Collectors.toList());
        
        if (!criticalEvents.isEmpty()) {
            report.append("\n=== Recent Critical Events ===\n");
            criticalEvents.forEach(event -> {
                report.append(event.toString()).append("\n");
                report.append("Response: ").append(event.getResponse()).append("\n\n");
            });
        }
        
        return report.toString();
    }

    /**
     * Get all security events
     */
    public List<SecurityEvent> getAllEvents() {
        synchronized (eventHistory) {
            return new ArrayList<>(eventHistory);
        }
    }

    /**
     * Get event counters
     */
    public Map<SecurityEventType, Integer> getEventCounters() {
        return new HashMap<>(eventCounters);
    }

    /**
     * Get isolated systems
     */
    public Set<String> getIsolatedSystems() {
        return new HashSet<>(isolatedSystems);
    }

    /**
     * Shutdown security event monitor
     */
    public void shutdown() {
        realTimeMonitoring = false;
        enhancedMode = false;
        
        monitoringExecutor.shutdown();
        responseExecutor.shutdown();
        
        try {
            if (!monitoringExecutor.awaitTermination(10, TimeUnit.SECONDS)) {
                monitoringExecutor.shutdownNow();
            }
            if (!responseExecutor.awaitTermination(5, TimeUnit.SECONDS)) {
                responseExecutor.shutdownNow();
            }
        } catch (InterruptedException e) {
            monitoringExecutor.shutdownNow();
            responseExecutor.shutdownNow();
            Thread.currentThread().interrupt();
        }
        
        System.out.println("Security Event Monitor shut down");
    }
}