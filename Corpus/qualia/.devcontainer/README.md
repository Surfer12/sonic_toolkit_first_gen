# Dev Container Setup for Reverse Koopman Penetration Testing Framework

This dev container provides a complete development environment for the Reverse Koopman Penetration Testing Framework with GPTOSS 2.0 integration.

## 🚀 Quick Start

### Using VS Code

1. **Install Requirements:**
   - VS Code with Dev Containers extension
   - Docker Desktop
   - At least 4GB RAM available for containers

2. **Open in Dev Container:**
   ```bash
   # Open the project in VS Code
   code /path/to/reverse-koopman-pentest

   # When prompted, click "Reopen in Container"
   # Or use Command Palette: "Dev Containers: Reopen in Container"
   ```

3. **Verify Setup:**
   ```bash
   # Check Java installation
   java -version  # Should show OpenJDK 21

   # Compile the framework
   javac -d out *.java

   # Run Java penetration testing demo
   java -cp out qualia.JavaPenetrationTestingDemo
   ```

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# Access the main container
docker-compose exec reverse-koopman-pentest bash

# Inside the container
cd /app
javac -d out *.java
java -cp out qualia.JavaPenetrationTestingDemo
```

## 🏗️ Container Architecture

### Services

```
reverse-koopman-pentest (main)
├── Java 21 application
├── Reverse Koopman operators
├── Penetration testing frameworks
└── GPTOSS 2.0 integration

postgres
├── SQL injection testing database
└── Vulnerable table structures

redis
├── Session management testing
└── Caching vulnerability assessment

mock-gptoss
├── AI model security testing
└── Prompt injection simulation

elasticsearch + logstash + kibana (optional)
└── Security findings monitoring
```

## 🛠️ Development Workflow

### 1. Code Development

```bash
# Compile all Java files
javac -d out *.java

# Run specific tests
java -cp out qualia.JavaPenetrationTestingDemo
java -cp out qualia.GPTOSSTesting
java -cp out qualia.IntegratedSecurityDemo
```

### 2. Testing Against Real Services

```bash
# Test against PostgreSQL
docker-compose exec reverse-koopman-pentest bash
# Inside container, connect to postgres:5432

# Test against Redis
docker-compose exec reverse-koopman-pentest bash
# Inside container, connect to redis:6379 with password

# Test against Mock GPTOSS
curl http://mock-gptoss:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gptoss-2.0","messages":[{"role":"user","content":"Hello"}]}'
```

### 3. Multi-Service Integration Testing

```bash
# Start monitoring stack
docker-compose --profile monitoring up -d

# Run integrated security assessment
java -cp out qualia.IntegratedSecurityDemo http://mock-gptoss:8000 sk-test-key

# Check results in Kibana
open http://localhost:5601
```

## 📊 Available Services

| Service | Port | Purpose |
|---------|------|---------|
| Main App | Container Internal | Java penetration testing |
| PostgreSQL | 5432 | SQL injection testing |
| Redis | 6379 | Session/cache testing |
| Mock GPTOSS | 8000 | AI model security testing |
| Elasticsearch | 9200 | Log indexing |
| Kibana | 5601 | Dashboard/monitoring |

## 🔧 Configuration

### Environment Variables

```bash
# Database connections (from container)
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=pentest_db
POSTGRES_USER=pentest_user
POSTGRES_PASSWORD=pentest_password

# Redis connection
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=pentest_redis_password

# GPTOSS API
GPTOSS_ENDPOINT=http://mock-gptoss:8000
GPTOSS_API_KEY=sk-test-key-for-development
```

### Volume Mounts

```bash
/app                    # Source code
/app/reports           # Generated reports
/app/logs             # Application logs
/app/data             # Test data and configurations
```

## 🧪 Testing Scenarios

### 1. Java Application Security Testing

```java
JavaPenetrationTesting testing = new JavaPenetrationTesting();
List<SecurityFinding> findings = testing.runComprehensiveTesting().get();
```

### 2. GPTOSS 2.0 AI Model Security Testing

```java
GPTOSSTesting gptoss = new GPTOSSTesting("http://mock-gptoss:8000", "sk-test-key");
List<SecurityFinding> aiFindings = gptoss.runComprehensiveGPTOSSTesting().get();
```

### 3. Integrated Security Assessment

```java
IntegratedSecurityDemo demo = new IntegratedSecurityDemo("http://mock-gptoss:8000", "sk-test-key");
demo.runIntegratedSecurityAssessment();
```

## 📈 Monitoring and Logging

### ELK Stack Integration

```bash
# Start monitoring services
docker-compose --profile monitoring up -d

# View logs in Kibana
open http://localhost:5601

# Search security findings
# Index: pentest-findings-*
# Query: severity:CRITICAL OR severity:HIGH
```

### Log Configuration

```xml
<!-- logstash.conf -->
input {
  file {
    path => "/app/logs/*.log"
    start_position => "beginning"
  }
}

filter {
  json {
    source => "message"
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "pentest-findings-%{+YYYY.MM.dd}"
  }
}
```

## 🔐 Security Considerations

### Development Environment Security

- **Isolated Network**: All services run in isolated Docker network
- **Non-Root User**: Application runs as non-privileged user
- **Minimal Attack Surface**: Only necessary services exposed
- **Test Data Only**: No production data in development environment

### API Key Management

```bash
# Use environment variables for sensitive data
export GPTOSS_API_KEY="sk-your-development-key"
export DATABASE_PASSWORD="development-password"
```

## 🐛 Troubleshooting

### Common Issues

1. **Port Conflicts:**
   ```bash
   # Check what's using ports
   lsof -i :5432,6379,8000,9200,5601

   # Stop conflicting services or change ports in docker-compose.yml
   ```

2. **Memory Issues:**
   ```bash
   # Increase Docker memory allocation
   # Docker Desktop > Settings > Resources > Memory
   ```

3. **Compilation Errors:**
   ```bash
   # Clean and rebuild
   rm -rf out/
   javac -d out *.java
   ```

### Health Checks

```bash
# Check all services
docker-compose ps

# View logs
docker-compose logs reverse-koopman-pentest
docker-compose logs postgres
docker-compose logs mock-gptoss

# Restart services
docker-compose restart
```

## 📚 Learning Resources

### Documentation Links

- [Reverse Koopman Operators](https://en.wikipedia.org/wiki/Koopman_operator)
- [Java Security Best Practices](https://owasp.org/www-project-top-ten/)
- [Docker Dev Containers](https://containers.dev/)
- [GPTOSS 2.0 Documentation](https://gptoss.com/docs)

### Example Workflows

1. **Security Research:**
   ```bash
   # Run comprehensive assessment
   java -cp out qualia.IntegratedSecurityDemo

   # Analyze results in reports/
   cat reports/integrated_security_report.txt
   ```

2. **AI Model Testing:**
   ```bash
   # Test specific AI vulnerabilities
   java -cp out qualia.GPTOSSTesting

   # Monitor results in ELK stack
   open http://localhost:5601
   ```

3. **Database Security:**
   ```bash
   # Connect to test database
   psql -h localhost -U pentest_user -d pentest_db

   # Test SQL injection scenarios
   java -cp out qualia.JavaPenetrationTestingDemo
   ```

## 🤝 Contributing

1. **Make Changes:** Edit files in the container
2. **Test Changes:** Run tests in the container
3. **Commit Changes:** Git operations work normally
4. **Push Changes:** Standard Git workflow

### Development Guidelines

- Use the provided Java style configuration
- Run tests before committing
- Update documentation for new features
- Follow security best practices
- Test against all services in the stack

## 📄 License

This development environment configuration is part of the Reverse Koopman Penetration Testing Framework, licensed under GPL-3.0-only.

---

## 🎯 Next Steps

1. **Explore the Codebase:** Familiarize yourself with the framework structure
2. **Run the Demos:** Execute the provided demonstrations
3. **Customize Tests:** Adapt the framework for your specific needs
4. **Contribute Back:** Share improvements and bug fixes

Happy coding! 🔒✨
