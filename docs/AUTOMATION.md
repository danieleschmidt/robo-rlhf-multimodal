# Automation and Metrics Guide

This document describes the automated systems, metrics tracking, and monitoring capabilities implemented for the robo-rlhf-multimodal project.

## 📊 Metrics Framework

### Overview
The project implements a comprehensive metrics framework that tracks:
- Code quality and technical debt
- Development productivity and collaboration
- System reliability and performance
- Security posture and compliance
- ML-specific metrics (model performance, data quality)
- Business metrics and user satisfaction

### Configuration
All metrics are configured in `.github/project-metrics.json` which defines:
- Thresholds and targets for each metric category
- Automated vs. manual metric collection
- Reporting frequencies (daily, weekly, monthly, quarterly)
- Alert conditions and notification channels

## 🤖 Automation Scripts

### 1. Metrics Collection (`scripts/collect_metrics.py`)

Comprehensive metrics collection from multiple sources:

```bash
# Collect all metrics
./scripts/collect_metrics.py --github-token $GITHUB_TOKEN

# Generate markdown report
./scripts/collect_metrics.py --format markdown --output metrics-report.md

# Verbose logging
./scripts/collect_metrics.py --verbose
```

**Features:**
- Git repository metrics (commits, contributors, branches)
- Code quality metrics (coverage, complexity, file counts)
- Dependency analysis (security vulnerabilities, outdated packages)
- Test execution metrics (performance benchmarks)
- Docker image metrics (sizes, counts)
- GitHub API metrics (stars, PRs, workflow runs)

**Collected Metrics:**
- Commits per week
- Test coverage percentage
- Code complexity scores
- Security vulnerabilities count
- Outdated dependencies
- CI/CD success rates
- Performance benchmarks

### 2. Code Quality Monitor (`scripts/automation/code_quality_monitor.py`)

Automated code quality monitoring with alerting:

```bash
# Run all quality checks
./scripts/automation/code_quality_monitor.py

# Generate markdown report with alerts
./scripts/automation/code_quality_monitor.py --format markdown

# Send Slack notifications
./scripts/automation/code_quality_monitor.py --webhook-url $SLACK_WEBHOOK

# Fail CI on quality issues
./scripts/automation/code_quality_monitor.py --fail-on-issues
```

**Quality Checks:**
- **Test Coverage**: Validates coverage meets threshold (default: 80%)
- **Code Complexity**: Identifies functions exceeding complexity limits (default: 10)
- **Code Duplication**: Detects duplicate code blocks (threshold: 5 blocks)
- **Maintainability Index**: Measures code maintainability (threshold: 20)
- **Security Issues**: Scans for security vulnerabilities using Bandit
- **Linting Issues**: Checks code style using Flake8

**Alert Conditions:**
- Coverage below threshold
- High complexity functions detected
- Security vulnerabilities found
- Excessive code duplication
- Low maintainability scores

### 3. Dependency Updater (`scripts/automation/dependency_updater.py`)

Automated dependency management and security monitoring:

```bash
# Check dependencies and vulnerabilities
./scripts/automation/dependency_updater.py --action check

# Generate dependency update (minor versions)
./scripts/automation/dependency_updater.py --action update --update-type minor

# Check for security vulnerabilities only
./scripts/automation/dependency_updater.py --action check --format markdown
```

**Features:**
- Outdated package detection with semantic versioning classification
- Security vulnerability scanning using Safety and pip-audit
- Automated requirements.txt generation
- GitHub PR creation for dependency updates
- Severity-based vulnerability categorization

**Update Types:**
- **patch**: Only patch version updates (1.0.1 → 1.0.2)
- **minor**: Minor and patch updates (1.0.1 → 1.1.0)
- **major**: All updates including major versions (1.0.1 → 2.0.0)
- **security**: Security-focused updates only

## 📈 Monitoring and Alerting

### Prometheus Integration
The metrics collection integrates with Prometheus for time-series data:

```yaml
# Example Prometheus scrape config
- job_name: 'robo-rlhf-metrics'
  static_configs:
    - targets: ['localhost:8001']
  scrape_interval: 60s
  metrics_path: /metrics
```

### Grafana Dashboards
Pre-configured dashboards for different audiences:

1. **Engineering Dashboard**: Code quality, CI/CD, technical debt
2. **Operations Dashboard**: System health, performance, security
3. **Business Dashboard**: User engagement, feature adoption, growth

### Alert Configuration
Alerts are configured in `.github/project-metrics.json`:

```json
{
  "alerts": {
    "critical": {
      "security_breach": {
        "threshold": "any",
        "notification_channels": ["slack", "email", "sms"],
        "escalation_time_minutes": 5
      }
    },
    "warning": {
      "low_coverage": {
        "threshold": "code_coverage < 80%",
        "notification_channels": ["slack"],
        "escalation_time_minutes": 1440
      }
    }
  }
}
```

## 🔄 Automation Workflows

### Daily Automation
- Dependency vulnerability scanning
- Performance benchmark execution
- Backup verification
- Log analysis and cleanup

### Weekly Automation
- Code quality report generation
- Dependency update checks
- Capacity planning reviews
- Security posture assessments

### Monthly Automation
- Full security audits
- Performance optimization reviews
- Technical debt planning
- Disaster recovery testing

## 🚀 CI/CD Integration

### Pre-commit Hooks
Automated quality gates before commits:
```bash
# Install pre-commit hooks
pre-commit install

# Run all hooks
pre-commit run --all-files
```

### GitHub Actions Integration
The automation scripts integrate with GitHub Actions workflows:

```yaml
# Example workflow step
- name: Run Code Quality Check
  run: |
    ./scripts/automation/code_quality_monitor.py \
      --format json \
      --output quality-report.json \
      --webhook-url ${{ secrets.SLACK_WEBHOOK_URL }} \
      --fail-on-issues

- name: Collect Metrics
  run: |
    ./scripts/collect_metrics.py \
      --github-token ${{ secrets.GITHUB_TOKEN }} \
      --output metrics.json
```

### Quality Gates
Pre-defined quality gates for different stages:

**Pre-commit:**
- Code formatting (Black, isort)
- Linting (Flake8, MyPy)
- Security scan (Bandit)
- Unit tests

**Pre-merge:**
- Code review approval (2+ reviewers)
- All tests pass (unit, integration)
- Coverage threshold met (80%+)
- Security scan passed
- Performance benchmarks met

**Pre-deployment:**
- Integration tests pass
- Security validation complete
- Performance validation passed
- Rollback plan verified

## 📊 Metrics Categories

### Development Metrics
- **Code Quality**: Coverage, complexity, duplication, maintainability
- **Productivity**: Commits/week, PR review time, feature lead time
- **Collaboration**: PR approval rate, review participation, documentation coverage

### Operations Metrics
- **Reliability**: Uptime, MTTR, MTBF, error rates
- **Performance**: Response times, throughput, resource utilization
- **Security**: Vulnerability counts, remediation time, scan frequency

### ML-Specific Metrics
- **Model Performance**: Training loss, validation accuracy, inference latency
- **Data Quality**: Freshness, completeness, drift detection
- **Experimentation**: Completion rates, hypothesis validation, reproducibility

### Business Metrics
- **User Satisfaction**: NPS scores, retention rates, feature adoption
- **Growth**: MAU growth, feature usage trends, API usage

## 🔧 Configuration and Customization

### Metric Thresholds
Customize thresholds in `.github/project-metrics.json`:

```json
{
  "metrics": {
    "development": {
      "code_quality": {
        "coverage_threshold": 85,
        "complexity_threshold": 8,
        "maintainability_index_threshold": 25
      }
    }
  }
}
```

### Notification Channels
Configure multiple notification channels:

```json
{
  "notifications": {
    "slack": {
      "webhook_url": "https://hooks.slack.com/...",
      "channels": {
        "alerts": "#alerts",
        "metrics": "#metrics",
        "quality": "#code-quality"
      }
    },
    "email": {
      "smtp_server": "smtp.company.com",
      "recipients": ["team@company.com"]
    }
  }
}
```

### Custom Metrics
Add custom metrics by extending the collection scripts:

```python
def collect_custom_metrics(self) -> Dict[str, Any]:
    """Collect custom project-specific metrics."""
    metrics = {}
    
    # Example: API endpoint count
    api_count = self._count_api_endpoints()
    metrics['api_endpoints'] = api_count
    
    # Example: Model accuracy from latest training
    model_accuracy = self._get_latest_model_accuracy()
    metrics['model_accuracy'] = model_accuracy
    
    return metrics
```

## 📋 Best Practices

### Metrics Collection
1. **Automate Everything**: Prefer automated over manual metrics
2. **Set Realistic Thresholds**: Based on team capacity and project goals
3. **Track Trends**: Focus on trends rather than absolute values
4. **Regular Reviews**: Monthly metric review and threshold adjustment

### Quality Monitoring
1. **Fail Fast**: Catch issues early in the development cycle
2. **Actionable Alerts**: Only alert on actionable issues
3. **Context Matters**: Include context in alerts and reports
4. **Continuous Improvement**: Regularly update checks and thresholds

### Automation Scripts
1. **Idempotent Operations**: Scripts should be safe to run multiple times
2. **Error Handling**: Graceful handling of failures and edge cases
3. **Logging**: Comprehensive logging for debugging and auditing
4. **Configuration-Driven**: Use configuration files instead of hard-coded values

## 🔍 Troubleshooting

### Common Issues

1. **Metrics Collection Failures**
   ```bash
   # Check tool availability
   which coverage radon bandit safety
   
   # Verify configuration
   python -c "import json; print(json.load(open('.github/project-metrics.json')))"
   ```

2. **GitHub API Rate Limits**
   ```bash
   # Check rate limit status
   curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/rate_limit
   ```

3. **Notification Delivery Issues**
   ```bash
   # Test webhook
   curl -X POST -H 'Content-type: application/json' \
     --data '{"text":"Test message"}' \
     $SLACK_WEBHOOK_URL
   ```

### Debug Mode
Run scripts with verbose logging:
```bash
./scripts/collect_metrics.py --verbose
./scripts/automation/code_quality_monitor.py --verbose
```

### Manual Validation
Validate automated results manually:
```bash
# Manual coverage check
coverage report

# Manual security scan
bandit -r robo_rlhf

# Manual dependency check
pip list --outdated
safety check
```

## 🚀 Getting Started

1. **Install Dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

2. **Configure Metrics**
   ```bash
   # Review and customize thresholds
   vim .github/project-metrics.json
   ```

3. **Set Up Notifications**
   ```bash
   # Add webhook URLs to environment
   export SLACK_WEBHOOK_URL="https://hooks.slack.com/..."
   export GITHUB_TOKEN="ghp_..."
   ```

4. **Run Initial Collection**
   ```bash
   # Collect baseline metrics
   ./scripts/collect_metrics.py --output baseline-metrics.json
   
   # Run quality checks
   ./scripts/automation/code_quality_monitor.py --format markdown
   ```

5. **Integrate with CI/CD**
   ```bash
   # Copy workflow examples
   cp docs/workflows/examples/* .github/workflows/
   ```

The automation framework provides comprehensive monitoring and quality assurance for the robo-rlhf-multimodal project, enabling data-driven development decisions and maintaining high code quality standards.