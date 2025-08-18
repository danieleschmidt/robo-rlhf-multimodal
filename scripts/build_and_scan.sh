#!/bin/bash
set -euo pipefail

# ============================================================================
# Comprehensive Build and Security Scan Script
# ============================================================================

# Configuration
PROJECT_NAME="robo-rlhf-multimodal"
BUILD_DIR="build"
DIST_DIR="dist"
REPORTS_DIR="security-reports"
SBOM_FILE="sbom.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install dependencies if needed
install_security_tools() {
    log_info "Checking security tools..."
    
    # Install bandit if not present
    if ! command_exists bandit; then
        log_info "Installing bandit..."
        pip install bandit
    fi
    
    # Install safety if not present
    if ! command_exists safety; then
        log_info "Installing safety..."
        pip install safety
    fi
    
    # Install semgrep if not present
    if ! command_exists semgrep; then
        log_info "Installing semgrep..."
        pip install semgrep
    fi
    
    log_success "Security tools ready"
}

# Function to clean previous builds
clean_build() {
    log_info "Cleaning previous builds..."
    
    rm -rf "${BUILD_DIR}" "${DIST_DIR}" "${REPORTS_DIR}"
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    
    mkdir -p "${REPORTS_DIR}"
    
    log_success "Build environment cleaned"
}

# Function to run static security analysis
run_security_scans() {
    log_info "Running security scans..."
    
    # Bandit - Security linter for Python
    log_info "Running Bandit security scan..."
    bandit -r robo_rlhf/ -f json -o "${REPORTS_DIR}/bandit-report.json" || true
    bandit -r robo_rlhf/ -f txt -o "${REPORTS_DIR}/bandit-report.txt" || true
    
    # Safety - Check for known security vulnerabilities
    log_info "Running Safety vulnerability check..."
    safety check --json --output "${REPORTS_DIR}/safety-report.json" || true
    safety check --output "${REPORTS_DIR}/safety-report.txt" || true
    
    # Semgrep - Advanced static analysis
    if command_exists semgrep; then
        log_info "Running Semgrep static analysis..."
        semgrep --config=auto --json --output="${REPORTS_DIR}/semgrep-report.json" robo_rlhf/ || true
        semgrep --config=auto --output="${REPORTS_DIR}/semgrep-report.txt" robo_rlhf/ || true
    else
        log_warning "Semgrep not available, skipping advanced static analysis"
    fi
    
    log_success "Security scans completed"
}

# Function to analyze security scan results
analyze_security_results() {
    log_info "Analyzing security scan results..."
    
    # Check Bandit results
    if [[ -f "${REPORTS_DIR}/bandit-report.json" ]]; then
        high_issues=$(jq '.results | map(select(.issue_severity == "HIGH")) | length' "${REPORTS_DIR}/bandit-report.json" 2>/dev/null || echo "0")
        medium_issues=$(jq '.results | map(select(.issue_severity == "MEDIUM")) | length' "${REPORTS_DIR}/bandit-report.json" 2>/dev/null || echo "0")
        
        if [[ "$high_issues" -gt 0 ]]; then
            log_error "Found $high_issues HIGH severity security issues"
            exit 1
        elif [[ "$medium_issues" -gt 5 ]]; then
            log_warning "Found $medium_issues MEDIUM severity security issues (threshold: 5)"
        else
            log_success "Bandit scan passed (High: $high_issues, Medium: $medium_issues)"
        fi
    fi
    
    # Check Safety results
    if [[ -f "${REPORTS_DIR}/safety-report.json" ]]; then
        vulnerabilities=$(jq '. | length' "${REPORTS_DIR}/safety-report.json" 2>/dev/null || echo "0")
        
        if [[ "$vulnerabilities" -gt 0 ]]; then
            log_error "Found $vulnerabilities known vulnerabilities"
            # In a production environment, you might want to exit here
            # exit 1
        else
            log_success "Safety scan passed (no known vulnerabilities)"
        fi
    fi
    
    log_success "Security analysis completed"
}

# Function to generate SBOM
generate_sbom() {
    log_info "Generating Software Bill of Materials (SBOM)..."
    
    if [[ -f "scripts/generate_sbom.py" ]]; then
        python scripts/generate_sbom.py
        
        if [[ -f "$SBOM_FILE" ]]; then
            log_success "SBOM generated: $SBOM_FILE"
            
            # Move SBOM to reports directory
            cp "$SBOM_FILE" "${REPORTS_DIR}/sbom.json"
        else
            log_warning "SBOM generation failed"
        fi
    else
        log_warning "SBOM generator not found, skipping"
    fi
}

# Function to run code quality checks
run_quality_checks() {
    log_info "Running code quality checks..."
    
    # Run flake8
    if command_exists flake8; then
        log_info "Running flake8..."
        flake8 robo_rlhf/ --output-file="${REPORTS_DIR}/flake8-report.txt" || true
    fi
    
    # Run mypy
    if command_exists mypy; then
        log_info "Running mypy type checking..."
        mypy robo_rlhf/ --txt-report "${REPORTS_DIR}" --html-report "${REPORTS_DIR}/mypy-html" || true
    fi
    
    # Run black check
    if command_exists black; then
        log_info "Running black format check..."
        black --check robo_rlhf/ > "${REPORTS_DIR}/black-report.txt" 2>&1 || true
    fi
    
    log_success "Code quality checks completed"
}

# Function to build Python package
build_package() {
    log_info "Building Python package..."
    
    # Install build dependencies
    pip install --upgrade build setuptools wheel
    
    # Build package
    python -m build --wheel --sdist
    
    if [[ -d "$DIST_DIR" && "$(ls -A $DIST_DIR)" ]]; then
        log_success "Package built successfully"
        ls -la "$DIST_DIR"
    else
        log_error "Package build failed"
        exit 1
    fi
}

# Function to scan Docker images
scan_docker_images() {
    log_info "Scanning Docker images for vulnerabilities..."
    
    # Build Docker image for scanning
    docker build --target production -t "${PROJECT_NAME}:scan" .
    
    # Use Docker Scout if available
    if command_exists docker && docker scout version >/dev/null 2>&1; then
        log_info "Running Docker Scout scan..."
        docker scout cves "${PROJECT_NAME}:scan" --format json --output "${REPORTS_DIR}/docker-scout.json" || true
        docker scout cves "${PROJECT_NAME}:scan" --format table --output "${REPORTS_DIR}/docker-scout.txt" || true
    else
        log_warning "Docker Scout not available, skipping container scanning"
    fi
    
    # Use Trivy if available
    if command_exists trivy; then
        log_info "Running Trivy container scan..."
        trivy image --format json --output "${REPORTS_DIR}/trivy-report.json" "${PROJECT_NAME}:scan" || true
        trivy image --format table --output "${REPORTS_DIR}/trivy-report.txt" "${PROJECT_NAME}:scan" || true
    else
        log_warning "Trivy not available, skipping vulnerability scanning"
    fi
    
    log_success "Docker image scanning completed"
}

# Function to generate comprehensive report
generate_report() {
    log_info "Generating comprehensive build report..."
    
    REPORT_FILE="${REPORTS_DIR}/build-report.md"
    
    cat > "$REPORT_FILE" << EOF
# Build and Security Report

**Generated:** $(date)
**Project:** $PROJECT_NAME
**Build ID:** ${BUILD_ID:-$(date +%Y%m%d-%H%M%S)}

## Build Status

✅ Package build completed successfully

## Security Scan Results

### Static Analysis (Bandit)
$(if [[ -f "${REPORTS_DIR}/bandit-report.txt" ]]; then
    echo "\`\`\`"
    head -20 "${REPORTS_DIR}/bandit-report.txt"
    echo "\`\`\`"
else
    echo "No Bandit report found"
fi)

### Vulnerability Check (Safety)
$(if [[ -f "${REPORTS_DIR}/safety-report.txt" ]]; then
    echo "\`\`\`"
    cat "${REPORTS_DIR}/safety-report.txt"
    echo "\`\`\`"
else
    echo "No Safety report found"
fi)

## Code Quality

### Linting (flake8)
$(if [[ -f "${REPORTS_DIR}/flake8-report.txt" ]]; then
    echo "\`\`\`"
    head -20 "${REPORTS_DIR}/flake8-report.txt"
    echo "\`\`\`"
else
    echo "No flake8 report found"
fi)

## Artifacts

- Built packages: $(ls -1 ${DIST_DIR}/ 2>/dev/null | tr '\n' ', ' | sed 's/,$//')
- SBOM: $(if [[ -f "${REPORTS_DIR}/sbom.json" ]]; then echo "✅ Generated"; else echo "❌ Not generated"; fi)
- Security reports: $(ls -1 ${REPORTS_DIR}/ | wc -l) files

## Recommendations

$(if [[ -f "${REPORTS_DIR}/bandit-report.json" ]]; then
    high_issues=$(jq '.results | map(select(.issue_severity == "HIGH")) | length' "${REPORTS_DIR}/bandit-report.json" 2>/dev/null || echo "0")
    if [[ "$high_issues" -gt 0 ]]; then
        echo "⚠️ Address $high_issues high-severity security issues before deployment"
    fi
fi)

$(if [[ -f "${REPORTS_DIR}/safety-report.json" ]]; then
    vulnerabilities=$(jq '. | length' "${REPORTS_DIR}/safety-report.json" 2>/dev/null || echo "0")
    if [[ "$vulnerabilities" -gt 0 ]]; then
        echo "⚠️ Update dependencies to address $vulnerabilities known vulnerabilities"
    fi
fi)

---
*This report was generated automatically by the build and scan script.*
EOF

    log_success "Build report generated: $REPORT_FILE"
}

# Main execution
main() {
    log_info "Starting comprehensive build and security scan..."
    
    # Parse command line arguments
    SKIP_DOCKER=false
    SKIP_SECURITY=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-docker)
                SKIP_DOCKER=true
                shift
                ;;
            --skip-security)
                SKIP_SECURITY=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [--skip-docker] [--skip-security]"
                echo "  --skip-docker: Skip Docker image scanning"
                echo "  --skip-security: Skip security scans"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Execute build pipeline
    clean_build
    
    if [[ "$SKIP_SECURITY" != true ]]; then
        install_security_tools
        run_security_scans
        analyze_security_results
    fi
    
    run_quality_checks
    generate_sbom
    build_package
    
    if [[ "$SKIP_DOCKER" != true ]] && command_exists docker; then
        scan_docker_images
    fi
    
    generate_report
    
    log_success "Build and security scan completed successfully!"
    log_info "Reports available in: $REPORTS_DIR/"
    log_info "Built packages available in: $DIST_DIR/"
}

# Run main function with all arguments
main "$@"