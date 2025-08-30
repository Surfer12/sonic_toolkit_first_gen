#!/usr/bin/env python3
"""
Cross-Framework Communication Protocol Implementation

This module provides comprehensive communication protocols between different
scientific computing framework components, enabling seamless integration
and data exchange across:

1. Hybrid UQ Framework (Python)
2. Corpus/Qualia Security Framework (Java)
3. Farmer iOS Framework (Swift)
4. Data Processing Pipeline (Python)

Author: Scientific Computing Toolkit Team
Date: 2025
License: GPL-3.0-only
"""

import json
import asyncio
import aiohttp
import websockets
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import hmac
import secrets
from pathlib import Path
import subprocess
import threading
import queue
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FrameworkMessage:
    """Standardized message format for cross-framework communication."""

    message_id: str
    timestamp: str
    source_framework: str
    target_framework: str
    message_type: str
    payload: Dict[str, Any]
    signature: Optional[str] = None
    correlation_id: Optional[str] = None
    priority: str = "normal"
    ttl_seconds: int = 300

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat() + "Z"
        if not self.message_id:
            self.message_id = f"msg_{int(time.time()*1000)}_{secrets.token_hex(4)}"


@dataclass
class FrameworkEndpoint:
    """Configuration for framework communication endpoints."""

    framework_name: str
    protocol: str  # 'http', 'websocket', 'grpc', 'local'
    host: str
    port: int
    path: str = ""
    authentication: Dict[str, Any] = None
    ssl_enabled: bool = False
    timeout_seconds: int = 30

    @property
    def url(self) -> str:
        """Construct full endpoint URL."""
        scheme = "https" if self.ssl_enabled else "http"
        return f"{scheme}://{self.host}:{self.port}{self.path}"


class CrossFrameworkCommunicator:
    """Main communication hub for cross-framework integration."""

    def __init__(self, config_path: str = "communication_config.json"):
        self.config_path = Path(config_path)
        self.endpoints: Dict[str, FrameworkEndpoint] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.message_queue = queue.Queue()
        self.running = False
        self.session_token = secrets.token_hex(32)

        # Load configuration
        self.load_configuration()

        # Initialize communication channels
        self.http_session = None
        self.websocket_connections = {}

    def load_configuration(self):
        """Load communication configuration from file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                self.load_endpoints(config.get('endpoints', {}))
        else:
            self.create_default_configuration()

    def load_endpoints(self, endpoint_config: Dict[str, Any]):
        """Load framework endpoints from configuration."""
        for name, config in endpoint_config.items():
            self.endpoints[name] = FrameworkEndpoint(
                framework_name=name,
                protocol=config.get('protocol', 'http'),
                host=config.get('host', 'localhost'),
                port=config.get('port', 8080),
                path=config.get('path', ''),
                authentication=config.get('authentication', {}),
                ssl_enabled=config.get('ssl_enabled', False),
                timeout_seconds=config.get('timeout_seconds', 30)
            )

    def create_default_configuration(self):
        """Create default communication configuration."""
        default_config = {
            "endpoints": {
                "hybrid_uq": {
                    "protocol": "http",
                    "host": "localhost",
                    "port": 5000,
                    "path": "/api/v1",
                    "ssl_enabled": False,
                    "timeout_seconds": 30
                },
                "corpus_qualia": {
                    "protocol": "http",
                    "host": "localhost",
                    "port": 8080,
                    "path": "/api/qualia",
                    "ssl_enabled": False,
                    "timeout_seconds": 60
                },
                "farmer_ios": {
                    "protocol": "websocket",
                    "host": "localhost",
                    "port": 9001,
                    "path": "/ws",
                    "ssl_enabled": False,
                    "timeout_seconds": 30
                },
                "data_pipeline": {
                    "protocol": "local",
                    "host": "localhost",
                    "port": 0,
                    "path": "data_output",
                    "ssl_enabled": False,
                    "timeout_seconds": 30
                }
            },
            "security": {
                "enable_encryption": True,
                "signature_algorithm": "HMAC-SHA256",
                "session_timeout_minutes": 60,
                "max_message_size_kb": 1024
            },
            "routing": {
                "enable_message_routing": True,
                "dead_letter_queue": "failed_messages.jsonl",
                "retry_attempts": 3,
                "retry_delay_seconds": 5
            }
        }

        # Save default configuration
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)

        self.load_endpoints(default_config['endpoints'])

    async def initialize(self):
        """Initialize communication channels."""
        logger.info("Initializing cross-framework communication...")

        # Initialize HTTP session
        self.http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )

        # Initialize WebSocket connections
        await self.initialize_websockets()

        logger.info("Cross-framework communication initialized successfully")

    async def initialize_websockets(self):
        """Initialize WebSocket connections for real-time communication."""
        for name, endpoint in self.endpoints.items():
            if endpoint.protocol == "websocket":
                try:
                    uri = f"ws://{endpoint.host}:{endpoint.port}{endpoint.path}"
                    connection = await websockets.connect(uri)
                    self.websocket_connections[name] = connection
                    logger.info(f"WebSocket connection established: {name}")
                except Exception as e:
                    logger.warning(f"Failed to connect WebSocket for {name}: {e}")

    def register_message_handler(self, message_type: str, handler: Callable):
        """Register a handler for specific message types."""
        self.message_handlers[message_type] = handler
        logger.info(f"Registered handler for message type: {message_type}")

    def sign_message(self, message: FrameworkMessage) -> str:
        """Create cryptographic signature for message authenticity."""
        message_data = json.dumps({
            'message_id': message.message_id,
            'timestamp': message.timestamp,
            'source_framework': message.source_framework,
            'target_framework': message.target_framework,
            'message_type': message.message_type,
            'payload': message.payload,
            'correlation_id': message.correlation_id
        }, sort_keys=True)

        signature = hmac.new(
            self.session_token.encode(),
            message_data.encode(),
            hashlib.sha256
        ).hexdigest()

        return signature

    def verify_message_signature(self, message: FrameworkMessage) -> bool:
        """Verify message signature for authenticity."""
        if not message.signature:
            return False

        expected_signature = self.sign_message(message)
        return hmac.compare_digest(message.signature, expected_signature)

    async def send_message(self, message: FrameworkMessage) -> Dict[str, Any]:
        """Send message to target framework."""
        # Sign message
        message.signature = self.sign_message(message)

        target_endpoint = self.endpoints.get(message.target_framework)
        if not target_endpoint:
            raise ValueError(f"Unknown target framework: {message.target_framework}")

        try:
            if target_endpoint.protocol == "http":
                return await self.send_http_message(message, target_endpoint)
            elif target_endpoint.protocol == "websocket":
                return await self.send_websocket_message(message, target_endpoint)
            elif target_endpoint.protocol == "local":
                return await self.send_local_message(message, target_endpoint)
            else:
                raise ValueError(f"Unsupported protocol: {target_endpoint.protocol}")

        except Exception as e:
            logger.error(f"Failed to send message to {message.target_framework}: {e}")
            # Add to retry queue
            self.message_queue.put(message)
            return {"status": "failed", "error": str(e)}

    async def send_http_message(self, message: FrameworkMessage,
                               endpoint: FrameworkEndpoint) -> Dict[str, Any]:
        """Send message via HTTP."""
        url = f"{endpoint.url}/messages"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.session_token}",
            "X-Message-ID": message.message_id
        }

        async with self.http_session.post(url, json=asdict(message), headers=headers) as response:
            if response.status == 200:
                result = await response.json()
                return {"status": "success", "response": result}
            else:
                error_text = await response.text()
                return {"status": "error", "code": response.status, "message": error_text}

    async def send_websocket_message(self, message: FrameworkMessage,
                                    endpoint: FrameworkEndpoint) -> Dict[str, Any]:
        """Send message via WebSocket."""
        connection = self.websocket_connections.get(message.target_framework)
        if not connection:
            return {"status": "error", "message": "WebSocket connection not available"}

        try:
            await connection.send(json.dumps(asdict(message)))
            # Wait for response (implement timeout)
            response = await asyncio.wait_for(
                connection.recv(),
                timeout=endpoint.timeout_seconds
            )
            result = json.loads(response)
            return {"status": "success", "response": result}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def send_local_message(self, message: FrameworkMessage,
                                endpoint: FrameworkEndpoint) -> Dict[str, Any]:
        """Send message to local framework component."""
        # For local communication, use subprocess or direct function calls
        if message.target_framework == "data_pipeline":
            return await self.send_data_pipeline_message(message)
        else:
            return {"status": "error", "message": "Local framework not implemented"}

    async def send_data_pipeline_message(self, message: FrameworkMessage) -> Dict[str, Any]:
        """Send message to data processing pipeline."""
        try:
            # Execute data pipeline processing
            result = subprocess.run([
                "python3", "data_output/data_flow_processor.py",
                "--message", json.dumps(asdict(message))
            ], capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                return {"status": "success", "output": result.stdout}
            else:
                return {"status": "error", "message": result.stderr}

        except subprocess.TimeoutExpired:
            return {"status": "error", "message": "Data pipeline timeout"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def receive_messages(self):
        """Receive and process incoming messages."""
        while self.running:
            try:
                # Process queued messages
                if not self.message_queue.empty():
                    message = self.message_queue.get_nowait()
                    await self.process_message(message)

                # Check WebSocket connections for new messages
                for name, connection in self.websocket_connections.items():
                    try:
                        message_data = await asyncio.wait_for(
                            connection.recv(),
                            timeout=0.1
                        )
                        message = FrameworkMessage(**json.loads(message_data))
                        await self.process_message(message)
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        logger.warning(f"Error receiving from {name}: {e}")

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in message reception loop: {e}")

    async def process_message(self, message: FrameworkMessage):
        """Process incoming message."""
        logger.info(f"Processing message: {message.message_id} from {message.source_framework}")

        # Verify signature
        if not self.verify_message_signature(message):
            logger.warning(f"Invalid signature for message: {message.message_id}")
            return

        # Check TTL
        message_age = time.time() - datetime.fromisoformat(message.timestamp[:-1]).timestamp()
        if message_age > message.ttl_seconds:
            logger.warning(f"Message expired: {message.message_id}")
            return

        # Route to appropriate handler
        handler = self.message_handlers.get(message.message_type)
        if handler:
            try:
                await handler(message)
            except Exception as e:
                logger.error(f"Error in message handler: {e}")
        else:
            logger.warning(f"No handler for message type: {message.message_type}")

    async def start(self):
        """Start the communication system."""
        self.running = True
        await self.initialize()

        # Start message reception loop
        asyncio.create_task(self.receive_messages())

        # Start retry mechanism
        asyncio.create_task(self.retry_failed_messages())

        logger.info("Cross-framework communication system started")

    async def stop(self):
        """Stop the communication system."""
        self.running = False

        # Close HTTP session
        if self.http_session:
            await self.http_session.close()

        # Close WebSocket connections
        for connection in self.websocket_connections.values():
            await connection.close()

        logger.info("Cross-framework communication system stopped")

    async def retry_failed_messages(self):
        """Retry sending failed messages."""
        while self.running:
            try:
                # Process retry queue
                if not self.message_queue.empty():
                    message = self.message_queue.get_nowait()

                    # Check retry count
                    retry_count = getattr(message, 'retry_count', 0)
                    if retry_count < 3:
                        message.retry_count = retry_count + 1
                        logger.info(f"Retrying message: {message.message_id} (attempt {retry_count + 1})")

                        # Wait before retry
                        await asyncio.sleep(5 * (retry_count + 1))

                        # Retry sending
                        result = await self.send_message(message)
                        if result.get("status") == "failed":
                            self.message_queue.put(message)
                    else:
                        logger.error(f"Message failed permanently: {message.message_id}")
                        # Save to dead letter queue
                        self.save_to_dead_letter_queue(message)

                await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"Error in retry mechanism: {e}")

    def save_to_dead_letter_queue(self, message: FrameworkMessage):
        """Save failed message to dead letter queue."""
        dead_letter_file = Path("failed_messages.jsonl")
        with open(dead_letter_file, 'a') as f:
            f.write(json.dumps(asdict(message)) + '\n')

    # Framework-specific communication methods

    async def send_hybrid_uq_prediction(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send prediction request to Hybrid UQ framework."""
        message = FrameworkMessage(
            message_id="",
            timestamp="",
            source_framework="communication_hub",
            target_framework="hybrid_uq",
            message_type="prediction_request",
            payload=input_data,
            correlation_id=secrets.token_hex(8)
        )

        result = await self.send_message(message)
        return result

    async def send_security_assessment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send security assessment request to Corpus/Qualia."""
        message = FrameworkMessage(
            message_id="",
            timestamp="",
            source_framework="communication_hub",
            target_framework="corpus_qualia",
            message_type="security_assessment",
            payload=data,
            correlation_id=secrets.token_hex(8)
        )

        result = await self.send_message(message)
        return result

    async def send_data_processing_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send data processing request to pipeline."""
        message = FrameworkMessage(
            message_id="",
            timestamp="",
            source_framework="communication_hub",
            target_framework="data_pipeline",
            message_type="data_processing",
            payload=data,
            correlation_id=secrets.token_hex(8)
        )

        result = await self.send_message(message)
        return result


# Message handlers for different frameworks

async def handle_hybrid_uq_response(message: FrameworkMessage):
    """Handle response from Hybrid UQ framework."""
    logger.info(f"Received Hybrid UQ response: {message.message_id}")
    # Process prediction results
    payload = message.payload

    if payload.get("status") == "success":
        predictions = payload.get("predictions", [])
        uncertainties = payload.get("uncertainty", [])
        psi_confidence = payload.get("psi_confidence", [])

        logger.info(f"Prediction completed: {len(predictions)} samples")
        logger.info(f"Average confidence: {sum(psi_confidence)/len(psi_confidence):.3f}")

        # Forward to data pipeline for further processing
        # Implementation would send to data_pipeline
    else:
        logger.error(f"Hybrid UQ prediction failed: {payload.get('error', 'Unknown error')}")


async def handle_security_assessment_response(message: FrameworkMessage):
    """Handle response from security assessment."""
    logger.info(f"Received security assessment: {message.message_id}")

    payload = message.payload
    if payload.get("status") == "completed":
        findings = payload.get("findings", [])
        logger.info(f"Security assessment completed: {len(findings)} findings")

        # Process security findings
        for finding in findings:
            severity = finding.get("severity", "unknown")
            if severity in ["high", "critical"]:
                logger.warning(f"Critical security finding: {finding.get('description', '')}")

        # Store results in data pipeline
        # Implementation would send to data_pipeline
    else:
        logger.error(f"Security assessment failed: {payload.get('error', 'Unknown error')}")


async def handle_data_processing_response(message: FrameworkMessage):
    """Handle response from data processing pipeline."""
    logger.info(f"Received data processing response: {message.message_id}")

    payload = message.payload
    if payload.get("status") == "completed":
        processed_files = payload.get("processed_files", [])
        results_path = payload.get("results_path", "")

        logger.info(f"Data processing completed: {len(processed_files)} files processed")
        logger.info(f"Results saved to: {results_path}")

        # Update monitoring dashboard
        # Implementation would update monitoring system
    else:
        logger.error(f"Data processing failed: {payload.get('error', 'Unknown error')}")


def create_communication_hub() -> CrossFrameworkCommunicator:
    """Create and configure the communication hub."""
    hub = CrossFrameworkCommunicator()

    # Register message handlers
    hub.register_message_handler("prediction_response", handle_hybrid_uq_response)
    hub.register_message_handler("security_assessment_response", handle_security_assessment_response)
    hub.register_message_handler("data_processing_response", handle_data_processing_response)

    return hub


async def main():
    """Main function to demonstrate cross-framework communication."""
    hub = create_communication_hub()

    try:
        await hub.start()

        # Example: Send prediction request to Hybrid UQ
        prediction_data = {
            "inputs": [[[0.1, 0.2], [0.3, 0.4]]],
            "grid_metrics": {"dx": 1.0, "dy": 1.0},
            "return_diagnostics": True
        }

        result = await hub.send_hybrid_uq_prediction(prediction_data)
        print(f"Prediction result: {result}")

        # Example: Send security assessment
        security_data = {
            "target": "sample_application",
            "scan_type": "comprehensive",
            "timeout": 300
        }

        result = await hub.send_security_assessment(security_data)
        print(f"Security assessment result: {result}")

        # Keep running for a while to receive responses
        await asyncio.sleep(10)

    finally:
        await hub.stop()


if __name__ == "__main__":
    asyncio.run(main())
