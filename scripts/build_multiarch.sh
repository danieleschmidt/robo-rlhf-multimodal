#!/bin/bash
set -euo pipefail

# ============================================================================
# Multi-Architecture Docker Build Script
# ============================================================================

# Configuration
PROJECT_NAME="robo-rlhf-multimodal"
REGISTRY="${DOCKER_REGISTRY:-}"
TAG="${IMAGE_TAG:-latest}"
PLATFORMS="linux/amd64,linux/arm64"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Check if buildx is available
check_buildx() {
    if ! docker buildx version >/dev/null 2>&1; then
        log_error "Docker Buildx is required for multi-architecture builds"
        exit 1
    fi
    
    log_success "Docker Buildx is available"
}

# Setup buildx builder
setup_builder() {
    local builder_name="multiarch-builder"
    
    log_info "Setting up buildx builder..."
    
    # Create builder if it doesn't exist
    if ! docker buildx ls | grep -q "$builder_name"; then
        docker buildx create --name "$builder_name" --driver docker-container --bootstrap
    fi
    
    # Use the builder
    docker buildx use "$builder_name"
    
    # Inspect builder to ensure platforms are available
    docker buildx inspect --bootstrap
    
    log_success "Buildx builder ready"
}

# Build function for a specific target
build_target() {
    local target="$1"
    local image_suffix="$2"
    local push_flag="$3"
    
    local full_image_name="${PROJECT_NAME}${image_suffix}"
    
    if [[ -n "$REGISTRY" ]]; then
        full_image_name="${REGISTRY}/${full_image_name}"
    fi
    
    log_info "Building ${target} for platforms: ${PLATFORMS}"
    
    # Build command
    local build_cmd="docker buildx build"
    build_cmd+=" --platform ${PLATFORMS}"
    build_cmd+=" --target ${target}"
    build_cmd+=" --tag ${full_image_name}:${TAG}"
    
    # Add labels
    build_cmd+=" --label org.opencontainers.image.title=${PROJECT_NAME}"
    build_cmd+=" --label org.opencontainers.image.description='End-to-end pipeline for multimodal RLHF in robotics'"
    build_cmd+=" --label org.opencontainers.image.version=${TAG}"
    build_cmd+=" --label org.opencontainers.image.created=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    build_cmd+=" --label org.opencontainers.image.source=https://github.com/danieleschmidt/robo-rlhf-multimodal"
    build_cmd+=" --label org.opencontainers.image.licenses=MIT"
    
    # Add push flag if requested
    if [[ "$push_flag" == "true" ]]; then
        build_cmd+=" --push"
    else
        build_cmd+=" --load"
    fi
    
    build_cmd+=" ."
    
    log_info "Executing: $build_cmd"
    
    # Execute build
    if eval "$build_cmd"; then
        log_success "Built ${target} successfully"
    else
        log_error "Failed to build ${target}"
        return 1
    fi
}

# Generate Docker metadata
generate_metadata() {
    log_info "Generating Docker metadata..."
    
    # Create buildinfo
    cat > buildinfo.json << EOF
{
  "project": "${PROJECT_NAME}",
  "version": "${TAG}",
  "platforms": "${PLATFORMS}",
  "built_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "git_branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')",
  "builder": "$(whoami)@$(hostname)"
}
EOF
    
    log_success "Metadata generated: buildinfo.json"
}

# Manifest inspection
inspect_manifests() {
    local image_name="$1"
    
    if [[ -n "$REGISTRY" ]]; then
        image_name="${REGISTRY}/${image_name}"
    fi
    
    log_info "Inspecting manifest for ${image_name}:${TAG}"
    
    # Check if image exists and inspect
    if docker buildx imagetools inspect "${image_name}:${TAG}" >/dev/null 2>&1; then
        docker buildx imagetools inspect "${image_name}:${TAG}"
    else
        log_warning "Cannot inspect ${image_name}:${TAG} - image may not be pushed"
    fi
}

# Security scan for multi-arch images
security_scan() {
    local image_name="$1"
    
    if [[ -n "$REGISTRY" ]]; then
        image_name="${REGISTRY}/${image_name}"
    fi
    
    log_info "Running security scan for ${image_name}:${TAG}"
    
    # Scan each platform if tools are available
    for platform in $(echo "$PLATFORMS" | tr ',' ' '); do
        log_info "Scanning ${platform}..."
        
        # Use Trivy if available
        if command -v trivy >/dev/null 2>&1; then
            trivy image --platform "$platform" "${image_name}:${TAG}" || true
        fi
        
        # Use Docker Scout if available
        if docker scout version >/dev/null 2>&1; then
            docker scout cves --platform "$platform" "${image_name}:${TAG}" || true
        fi
    done
}

# Main build function
main() {
    local push_images=false
    local scan_images=false
    local targets="production development gpu-production"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --push)
                push_images=true
                shift
                ;;
            --scan)
                scan_images=true
                shift
                ;;
            --platforms)
                PLATFORMS="$2"
                shift 2
                ;;
            --tag)
                TAG="$2"
                shift 2
                ;;
            --registry)
                REGISTRY="$2"
                shift 2
                ;;
            --targets)
                targets="$2"
                shift 2
                ;;
            --help|-h)
                cat << EOF
Multi-Architecture Docker Build Script

Usage: $0 [OPTIONS]

OPTIONS:
    --push              Push images to registry
    --scan              Run security scans on images
    --platforms PLAT    Target platforms (default: linux/amd64,linux/arm64)
    --tag TAG           Image tag (default: latest)
    --registry REG      Container registry
    --targets TARGETS   Build targets (default: production development gpu-production)
    --help, -h          Show this help

EXAMPLES:
    # Build for local use
    $0

    # Build and push to registry
    $0 --push --registry ghcr.io/danieleschmidt

    # Build specific platforms
    $0 --platforms linux/amd64 --tag v1.0.0

    # Build with security scanning
    $0 --scan --tag latest
EOF
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    log_info "Starting multi-architecture Docker build"
    log_info "Platforms: $PLATFORMS"
    log_info "Tag: $TAG"
    log_info "Registry: ${REGISTRY:-'(local)'}"
    log_info "Push: $push_images"
    
    # Setup
    check_buildx
    setup_builder
    generate_metadata
    
    # Build each target
    for target in $targets; do
        case "$target" in
            "production")
                build_target "production" "" "$push_images"
                ;;
            "development")
                build_target "development" "-dev" "$push_images"
                ;;
            "gpu-production")
                build_target "gpu-production" "-gpu" "$push_images"
                ;;
            *)
                log_warning "Unknown target: $target"
                ;;
        esac
    done
    
    # Post-build tasks
    if [[ "$push_images" == "true" ]]; then
        for target in $targets; do
            case "$target" in
                "production")
                    inspect_manifests "${PROJECT_NAME}"
                    ;;
                "development")
                    inspect_manifests "${PROJECT_NAME}-dev"
                    ;;
                "gpu-production")
                    inspect_manifests "${PROJECT_NAME}-gpu"
                    ;;
            esac
        done
    fi
    
    # Security scanning
    if [[ "$scan_images" == "true" ]]; then
        for target in $targets; do
            case "$target" in
                "production")
                    security_scan "${PROJECT_NAME}"
                    ;;
                "development")
                    security_scan "${PROJECT_NAME}-dev"
                    ;;
                "gpu-production")
                    security_scan "${PROJECT_NAME}-gpu"
                    ;;
            esac
        done
    fi
    
    log_success "Multi-architecture build completed successfully!"
    
    if [[ "$push_images" == "true" && -n "$REGISTRY" ]]; then
        log_info "Images pushed to registry:"
        for target in $targets; do
            case "$target" in
                "production")
                    echo "  ${REGISTRY}/${PROJECT_NAME}:${TAG}"
                    ;;
                "development")
                    echo "  ${REGISTRY}/${PROJECT_NAME}-dev:${TAG}"
                    ;;
                "gpu-production")
                    echo "  ${REGISTRY}/${PROJECT_NAME}-gpu:${TAG}"
                    ;;
            esac
        done
    fi
}

# Execute main function
main "$@"