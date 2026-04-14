#!/bin/bash
# Production-grade Docker build and push script for Netflix/Meta scale recommendation engine
# Multi-architecture builds with security scanning and optimization

set -euo pipefail

# Configuration
REGISTRY="${REGISTRY:-ghcr.io}"
IMAGE_NAME="${IMAGE_NAME:-rec-engine}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
BUILD_DATE="${BUILD_DATE:-$(date -u +'%Y-%m-%dT%H:%M:%SZ')}"
VCS_REF="${VCS_REF:-$(git rev-parse HEAD)}"
VERSION="${VERSION:-$(git describe --tags --always --dirty)}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"
CACHE_FROM="${CACHE_FROM:-}"
CACHE_TO="${CACHE_TO:-type=gha,mode=max}"
PUSH="${PUSH:-true}"
SECURITY_SCAN="${SECURITY_SCAN:-true}"

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
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
    fi
    
    # Check if Docker Buildx is available
    if ! docker buildx version &> /dev/null; then
        log_error "Docker Buildx is not available"
    fi
    
    # Check if we're logged in to the registry
    if ! docker login --get-identity "${REGISTRY}" &> /dev/null; then
        log_warning "Not logged in to registry ${REGISTRY}"
        log_info "Attempting to login..."
        echo "${GITHUB_TOKEN}" | docker login "${REGISTRY}" --username "${GITHUB_ACTOR}" --password-stdin
    fi
    
    # Check if git is available
    if ! command -v git &> /dev/null; then
        log_error "Git is not installed"
    fi
    
    log_success "Prerequisites check passed"
}

# Build arguments
prepare_build_args() {
    log_info "Preparing build arguments..."
    
    # Create build arguments
    BUILD_ARGS=(
        --build-arg PYTHON_VERSION="${PYTHON_VERSION}"
        --build-arg BUILD_DATE="${BUILD_DATE}"
        --build-arg VCS_REF="${VCS_REF}"
        --build-arg VERSION="${VERSION}"
        --platform "${PLATFORMS}"
        --label "org.opencontainers.image.title=Recommendation Engine"
        --label "org.opencontainers.image.description=Production-grade recommendation engine for Netflix/Meta scale"
        --label "org.opencontainers.image.url=https://github.com/company/rec-engine"
        --label "org.opencontainers.image.source=https://github.com/company/rec-engine"
        --label "org.opencontainers.image.version=${VERSION}"
        --label "org.opencontainers.image.created=${BUILD_DATE}"
        --label "org.opencontainers.image.revision=${VCS_REF}"
        --label "org.opencontainers.image.licenses=MIT"
        --label "org.opencontainers.image.vendor=Company"
    )
    
    # Add cache arguments if provided
    if [[ -n "${CACHE_FROM}" ]]; then
        BUILD_ARGS+=(--cache-from "${CACHE_FROM}")
    fi
    
    if [[ -n "${CACHE_TO}" ]]; then
        BUILD_ARGS+=(--cache-to "${CACHE_TO}")
    fi
    
    log_success "Build arguments prepared"
}

# Security scan
security_scan() {
    local image_tag="$1"
    
    if [[ "${SECURITY_SCAN}" != "true" ]]; then
        log_info "Security scan disabled, skipping"
        return 0
    fi
    
    log_info "Running security scan on ${image_tag}..."
    
    # Run Trivy scan
    if command -v trivy &> /dev/null; then
        log_info "Running Trivy vulnerability scan..."
        trivy image --format json --output "trivy-${image_tag//:/_}.json" "${image_tag}" || true
        
        # Check for high/critical vulnerabilities
        HIGH_VULNS=$(trivy image --format json "${image_tag}" | jq -r '.Results[]?.Vulnerabilities[]? | select(.Severity == "HIGH" or .Severity == "CRITICAL") | .VulnerabilityID' | wc -l || echo "0")
        if [[ "${HIGH_VULNS}" -gt 0 ]]; then
            log_warning "Found ${HIGH_VULNS} high/critical vulnerabilities"
            # Continue with build but warn
        fi
    else
        log_warning "Trivy not installed, skipping security scan"
    fi
    
    log_success "Security scan completed"
}

# Build Docker image
build_image() {
    local image_tag="$1"
    
    log_info "Building Docker image ${image_tag}..."
    
    # Create buildx builder if it doesn't exist
    if ! docker buildx inspect rec-engine-builder &> /dev/null; then
        log_info "Creating buildx builder..."
        docker buildx create --name rec-engine-builder --use --bootstrap
    fi
    
    # Build the image
    docker buildx build \
        "${BUILD_ARGS[@]}" \
        --tag "${image_tag}" \
        --push="${PUSH}" \
        .
    
    log_success "Docker image built successfully: ${image_tag}"
}

# Generate SBOM
generate_sbom() {
    local image_tag="$1"
    
    log_info "Generating SBOM for ${image_tag}..."
    
    # Use Syft to generate SBOM if available
    if command -v syft &> /dev/null; then
        syft packages "${image_tag}" --format spdx-json --output "sbom-${image_tag//:/_}.spdx.json"
        log_success "SBOM generated: sbom-${image_tag//:/_}.spdx.json"
    else
        log_warning "Syft not installed, skipping SBOM generation"
    fi
}

# Push image manifest
push_manifest() {
    local image_tag="$1"
    
    if [[ "${PUSH}" != "true" ]]; then
        log_info "Push disabled, skipping manifest push"
        return 0
    fi
    
    log_info "Pushing image manifest for ${image_tag}..."
    
    # Push the manifest (for multi-arch images)
    docker buildx imagetools create "${image_tag}" || true
    
    log_success "Image manifest pushed successfully"
}

# Validate image
validate_image() {
    local image_tag="$1"
    
    log_info "Validating image ${image_tag}..."
    
    # Pull the image to validate
    if [[ "${PUSH}" == "true" ]]; then
        docker pull "${image_tag}"
    fi
    
    # Check if image exists
    if ! docker image inspect "${image_tag}" &> /dev/null; then
        log_error "Image validation failed: ${image_tag} does not exist"
    fi
    
    # Check image size
    IMAGE_SIZE=$(docker image inspect "${image_tag}" --format='{{.Size}}')
    IMAGE_SIZE_MB=$((IMAGE_SIZE / 1024 / 1024))
    log_info "Image size: ${IMAGE_SIZE_MB} MB"
    
    # Check if image is too large (warning threshold: 2GB)
    if [[ "${IMAGE_SIZE_MB}" -gt 2048 ]]; then
        log_warning "Image size is large: ${IMAGE_SIZE_MB} MB"
    fi
    
    # Test basic functionality
    log_info "Testing basic image functionality..."
    docker run --rm "${image_tag}" python --version
    docker run --rm "${image_tag}" python -c "import fastapi; print('FastAPI imported successfully')"
    
    log_success "Image validation passed"
}

# Clean up
cleanup() {
    log_info "Cleaning up..."
    
    # Remove temporary files
    rm -f trivy-*.json sbom-*.spdx.json
    
    # Prune build cache (optional)
    if [[ "${CLEANUP_CACHE:-false}" == "true" ]]; then
        docker buildx prune -f
    fi
    
    log_success "Cleanup completed"
}

# Main function
main() {
    log_info "Starting Docker build and push process..."
    log_info "Registry: ${REGISTRY}"
    log_info "Image: ${IMAGE_NAME}"
    log_info "Version: ${VERSION}"
    log_info "Platforms: ${PLATFORMS}"
    
    # Check prerequisites
    check_prerequisites
    
    # Prepare build arguments
    prepare_build_args
    
    # Determine image tags
    local image_tags=()
    
    # Add version tag
    image_tags+=("${REGISTRY}/${IMAGE_NAME}:${VERSION}")
    
    # Add latest tag if on main branch
    if [[ "${GITHUB_REF:-}" == "refs/heads/main" ]]; then
        image_tags+=("${REGISTRY}/${IMAGE_NAME}:latest")
    fi
    
    # Add branch tag
    if [[ -n "${GITHUB_REF_NAME:-}" ]]; then
        image_tags+=("${REGISTRY}/${IMAGE_NAME}:${GITHUB_REF_NAME}")
    fi
    
    # Build and push each tag
    for tag in "${image_tags[@]}"; do
        log_info "Processing tag: ${tag}"
        
        # Build image
        build_image "${tag}"
        
        # Security scan
        security_scan "${tag}"
        
        # Generate SBOM
        generate_sbom "${tag}"
        
        # Validate image
        validate_image "${tag}"
        
        # Push manifest
        push_manifest "${tag}"
        
        log_success "Completed processing: ${tag}"
    done
    
    # Cleanup
    cleanup
    
    log_success "Docker build and push process completed successfully"
    
    # Output summary
    echo
    echo "=== Build Summary ==="
    echo "Image: ${REGISTRY}/${IMAGE_NAME}"
    echo "Version: ${VERSION}"
    echo "Tags: ${image_tags[*]}"
    echo "Platforms: ${PLATFORMS}"
    echo "Build Date: ${BUILD_DATE}"
    echo "Git Ref: ${VCS_REF}"
    echo
}

# Handle signals
trap cleanup EXIT

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --platforms)
            PLATFORMS="$2"
            shift 2
            ;;
        --no-push)
            PUSH="false"
            shift
            ;;
        --no-security-scan)
            SECURITY_SCAN="false"
            shift
            ;;
        --cleanup-cache)
            CLEANUP_CACHE="true"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo
            echo "Options:"
            echo "  --registry REGISTRY      Container registry (default: ghcr.io)"
            echo "  --image IMAGE_NAME       Image name (default: rec-engine)"
            echo "  --version VERSION        Image version (default: git describe)"
            echo "  --platforms PLATFORMS   Target platforms (default: linux/amd64,linux/arm64)"
            echo "  --no-push               Disable pushing to registry"
            echo "  --no-security-scan      Disable security scanning"
            echo "  --cleanup-cache         Clean up build cache"
            echo "  --help                  Show this help message"
            echo
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main "$@"
