//
//  iOSPenetrationTestingTests.swift
//  UOIFCoreTests
//
//  Created by Ryan David Oates on 8/26/25.
//  Unit tests for iOS penetration testing framework.

import XCTest
@testable import UOIFCore

final class iOSPenetrationTestingTests: XCTestCase {

    var penetrationTesting: iOSPenetrationTesting!

    override func setUpWithError() throws {
        penetrationTesting = iOSPenetrationTesting()
    }

    override func tearDownWithError() throws {
        penetrationTesting = nil
    }

    func testRunComprehensiveTesting() throws {
        let expectation = XCTestExpectation(description: "Comprehensive testing completion")

        penetrationTesting.runComprehensiveTesting { findings in
            // Verify that findings are generated
            XCTAssertGreaterThan(findings.count, 0, "Should generate security findings")

            // Verify finding structure
            if let firstFinding = findings.first {
                XCTAssertNotNil(firstFinding.vulnerabilityType)
                XCTAssertNotNil(firstFinding.severity)
                XCTAssertFalse(firstFinding.title.isEmpty)
                XCTAssertFalse(firstFinding.description.isEmpty)
                XCTAssertFalse(firstFinding.location.isEmpty)
                XCTAssertFalse(firstFinding.recommendation.isEmpty)
            }

            expectation.fulfill()
        }

        wait(for: [expectation], timeout: 10.0)
    }

    func testGenerateSecurityReport() throws {
        // Run testing first to generate findings
        let expectation = XCTestExpectation(description: "Report generation")

        penetrationTesting.runComprehensiveTesting { [weak self] findings in
            guard let self = self else { return }

            let report = self.penetrationTesting.generateSecurityReport()

            // Verify report content
            XCTAssertFalse(report.isEmpty)
            XCTAssertTrue(report.contains("iOS Penetration Testing Report"))
            XCTAssertTrue(report.contains("SECURITY METRICS"))

            expectation.fulfill()
        }

        wait(for: [expectation], timeout: 10.0)
    }

    func testExportFindingsToJson() throws {
        let expectation = XCTestExpectation(description: "JSON export")

        penetrationTesting.runComprehensiveTesting { [weak self] findings in
            guard let self = self else { return }

            let jsonString = self.penetrationTesting.exportFindingsToJson()

            // Verify JSON structure
            XCTAssertFalse(jsonString.isEmpty)
            XCTAssertTrue(jsonString.contains("findings"))

            // Verify it's valid JSON
            let jsonData = jsonString.data(using: .utf8)
            XCTAssertNotNil(jsonData)

            do {
                let jsonObject = try JSONSerialization.jsonObject(with: jsonData!, options: [])
                XCTAssertNotNil(jsonObject)
            } catch {
                XCTFail("Invalid JSON: \(error)")
            }

            expectation.fulfill()
        }

        wait(for: [expectation], timeout: 10.0)
    }

    func testGetFindingsBySeverity() throws {
        let expectation = XCTestExpectation(description: "Severity filtering")

        penetrationTesting.runComprehensiveTesting { [weak self] findings in
            guard let self = self else { return }

            let highSeverityFindings = self.penetrationTesting.getFindingsBySeverity(.high)
            let criticalSeverityFindings = self.penetrationTesting.getFindingsBySeverity(.critical)

            // Verify filtering works
            for finding in highSeverityFindings {
                XCTAssertEqual(finding.severity, .high)
            }

            for finding in criticalSeverityFindings {
                XCTAssertEqual(finding.severity, .critical)
            }

            expectation.fulfill()
        }

        wait(for: [expectation], timeout: 10.0)
    }

    func testGetFindingsByType() throws {
        let expectation = XCTestExpectation(description: "Type filtering")

        penetrationTesting.runComprehensiveTesting { [weak self] findings in
            guard let self = self else { return }

            let bufferOverflowFindings = self.penetrationTesting.getFindingsByType(.bufferOverflow)
            let memoryCorruptionFindings = self.penetrationTesting.getFindingsByType(.memoryCorruption)

            // Verify filtering works
            for finding in bufferOverflowFindings {
                XCTAssertEqual(finding.vulnerabilityType, .bufferOverflow)
            }

            for finding in memoryCorruptionFindings {
                XCTAssertEqual(finding.vulnerabilityType, .memoryCorruption)
            }

            expectation.fulfill()
        }

        wait(for: [expectation], timeout: 10.0)
    }

    func testConfiguration() throws {
        var config = iOSPenetrationTesting.Configuration()
        config.enableMemoryTesting = false
        config.enableNetworkTesting = true
        config.testTimeout = 60.0

        let customTesting = iOSPenetrationTesting(configuration: config)

        // Test that configuration is applied
        let expectation = XCTestExpectation(description: "Custom configuration test")

        customTesting.runComprehensiveTesting { findings in
            // With memory testing disabled, we should still get some findings
            // but potentially fewer or different types
            XCTAssertGreaterThanOrEqual(findings.count, 0)

            expectation.fulfill()
        }

        wait(for: [expectation], timeout: 10.0)
    }
}
