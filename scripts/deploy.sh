#!/bin/bash

# Pokemon RL Bot Deployment Script
# Supports Docker, cloud platforms, and production deployments

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Global variables
DEPLOYMENT_TYPE=""
PLATFORM=""
PROJECT_NAME="pokemon-rl-bot"
VERSION=$(date +%Y%m%d-%H%M%S)
DOCKER_IMAGE="$PROJECT_NAME:$VERSION"
DOCKER_REGISTRY=""

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--type)
                DEPLOYMENT_TYPE="$2"
                shift 2
                ;;
            -p|--platform)
                PLATFORM="$2"
                shift 2
                ;;
            -r|--registry)
                DOCKER_REGISTRY="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                DOCKER_IMAGE="$PROJECT_NAME:$VERSION"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Show help message
show_help() {
    echo "Pokemon RL Bot Deployment Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --type TYPE        Deployment type: docker, aws, gcp, azure, local"
    echo "  -p, --platform PLATFORM   Platform: gpu, cpu"
    echo "  -r, --registry REGISTRY    Docker registry URL"
    echo "  -v, --version VERSION      Version tag (default: timestamp)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -t docker -p gpu                    # Build GPU Docker image"
    echo "  $0 -t aws -r your-registry.com         # Deploy to AWS"
    echo "  $0 -t gcp -p cpu                       # Deploy to Google Cloud Platform"
    echo "  $0 -t local                            # Local production deployment"
}

# Check dependencies
check_dependencies() {
    print_status "Checking deployment dependencies..."
    
    local missing_deps=()
    
    case $DEPLOYMENT_TYPE in
        docker|aws|gcp|azure)
            if ! command -v docker &> /dev/null; then
                missing_deps+=("docker")
            fi
            ;;
    esac
    
    case $DEPLOYMENT_TYPE in
        aws)
            if ! command -v aws &> /dev/null; then
                missing_deps+=("aws-cli")
            fi
            ;;
        gcp)
            if ! command -v gcloud &> /dev/null; then
                missing_deps+=("gcloud")
            fi
            ;;
        azure)
            if ! command -v az &> /dev/null; then
                missing_deps+=("azure-cli")
            fi
            ;;
    esac
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        exit 1
    fi
    
    print_success "Dependencies check passed"
}

# Build Docker image
build_docker_image() {
    print_status "Building Docker image: $DOCKER_IMAGE"
    
    # Determine Dockerfile based on platform
    local dockerfile="Dockerfile"
    if [[ "$PLATFORM" == "gpu" ]]; then
        dockerfile="Dockerfile.gpu"
    fi
    
    if [[ ! -f "$dockerfile" ]]; then
        print_error "Dockerfile not found: $dockerfile"
        exit 1
    fi
    
    # Build the image
    docker build \
        -f "$dockerfile" \
        -t "$DOCKER_IMAGE" \
        --build-arg VERSION="$VERSION" \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        .
    
    print_success "Docker image built: $DOCKER_IMAGE"
    
    # Tag as latest
    docker tag "$DOCKER_IMAGE" "$PROJECT_NAME:latest"
    print_status "Tagged as latest: $PROJECT_NAME:latest"
}

# Push Docker image to registry
push_docker_image() {
    if [[ -z "$DOCKER_REGISTRY" ]]; then
        print_warning "No registry specified, skipping push"
        return 0
    fi
    
    print_status "Pushing Docker image to registry: $DOCKER_REGISTRY"
    
    local registry_image="$DOCKER_REGISTRY/$DOCKER_IMAGE"
    local registry_latest="$DOCKER_REGISTRY/$PROJECT_NAME:latest"
    
    # Tag for registry
    docker tag "$DOCKER_IMAGE" "$registry_image"
    docker tag "$DOCKER_IMAGE" "$registry_latest"
    
    # Push to registry
    docker push "$registry_image"
    docker push "$registry_latest"
    
    print_success "Docker image pushed to registry"
}

# Deploy to AWS
deploy_aws() {
    print_status "Deploying to AWS..."
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        print_error "AWS credentials not configured"
        exit 1
    fi
    
    # Create ECS task definition
    create_ecs_task_definition() {
        local task_def='{
            "family": "'$PROJECT_NAME'",
            "networkMode": "awsvpc",
            "requiresCompatibilities": ["FARGATE"],
            "cpu": "2048",
            "memory": "4096",
            "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
            "containerDefinitions": [
                {
                    "name": "'$PROJECT_NAME'",
                    "image": "'$DOCKER_REGISTRY/$DOCKER_IMAGE'",
                    "portMappings": [
                        {
                            "containerPort": 5000,
                            "protocol": "tcp"
                        }
                    ],
                    "environment": [
                        {
                            "name": "ENV",
                            "value": "production"
                        }
                    ],
                    "logConfiguration": {
                        "logDriver": "awslogs",
                        "options": {
                            "awslogs-group": "/ecs/'$PROJECT_NAME'",
                            "awslogs-region": "us-east-1",
                            "awslogs-stream-prefix": "ecs"
                        }
                    }
                }
            ]
        }'
        
        echo "$task_def" > aws-task-definition.json
        
        # Register task definition
        aws ecs register-task-definition \
            --cli-input-json file://aws-task-definition.json \
            --region us-east-1
        
        print_success "ECS task definition created"
    }
    
    # Create or update ECS service
    create_ecs_service() {
        local cluster_name="$PROJECT_NAME-cluster"
        local service_name="$PROJECT_NAME-service"
        
        # Check if cluster exists
        if ! aws ecs describe-clusters \
            --clusters "$cluster_name" \
            --region us-east-1 &> /dev/null; then
            
            print_status "Creating ECS cluster: $cluster_name"
            aws ecs create-cluster \
                --cluster-name "$cluster_name" \
                --region us-east-1
        fi
        
        # Create service if it doesn't exist
        if ! aws ecs describe-services \
            --cluster "$cluster_name" \
            --services "$service_name" \
            --region us-east-1 &> /dev/null; then
            
            print_status "Creating ECS service: $service_name"
            aws ecs create-service \
                --cluster "$cluster_name" \
                --service-name "$service_name" \
                --task-definition "$PROJECT_NAME" \
                --desired-count 1 \
                --launch-type FARGATE \
                --network-configuration "awsvpcConfiguration={subnets=[subnet-12345],securityGroups=[sg-12345],assignPublicIp=ENABLED}" \
                --region us-east-1
        else
            print_status "Updating ECS service: $service_name"
            aws ecs update-service \
                --cluster "$cluster_name" \
                --service "$service_name" \
                --task-definition "$PROJECT_NAME" \
                --region us-east-1
        fi
        
        print_success "ECS service deployed"
    }
    
    create_ecs_task_definition
    create_ecs_service
    
    print_success "AWS deployment completed"
}

# Deploy to Google Cloud Platform
deploy_gcp() {
    print_status "Deploying to Google Cloud Platform..."
    
    # Check GCP authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n 1 &> /dev/null; then
        print_error "GCP authentication required"
        print_status "Run: gcloud auth login"
        exit 1
    fi
    
    local project_id=$(gcloud config get-value project)
    if [[ -z "$project_id" ]]; then
        print_error "GCP project not set"
        print_status "Run: gcloud config set project YOUR_PROJECT_ID"
        exit 1
    fi
    
    print_status "Using GCP project: $project_id"
    
    # Build and push to Google Container Registry
    local gcr_image="gcr.io/$project_id/$PROJECT_NAME:$VERSION"
    
    docker tag "$DOCKER_IMAGE" "$gcr_image"
    docker push "$gcr_image"
    
    # Deploy to Cloud Run
    print_status "Deploying to Cloud Run..."
    
    gcloud run deploy "$PROJECT_NAME" \
        --image "$gcr_image" \
        --platform managed \
        --region us-central1 \
        --allow-unauthenticated \
        --port 5000 \
        --memory 4Gi \
        --cpu 2 \
        --max-instances 10 \
        --set-env-vars ENV=production
    
    print_success "GCP deployment completed"
}

# Deploy to Azure
deploy_azure() {
    print_status "Deploying to Azure..."
    
    # Check Azure authentication
    if ! az account show &> /dev/null; then
        print_error "Azure authentication required"
        print_status "Run: az login"
        exit 1
    fi
    
    local resource_group="$PROJECT_NAME-rg"
    local container_group="$PROJECT_NAME-cg"
    local registry_name="${PROJECT_NAME}registry"
    
    # Create resource group
    print_status "Creating resource group: $resource_group"
    az group create \
        --name "$resource_group" \
        --location eastus
    
    # Create container registry
    print_status "Creating container registry: $registry_name"
    az acr create \
        --resource-group "$resource_group" \
        --name "$registry_name" \
        --sku Basic \
        --admin-enabled true
    
    # Push image to ACR
    local acr_server=$(az acr show --name "$registry_name" --query loginServer --output tsv)
    local acr_image="$acr_server/$PROJECT_NAME:$VERSION"
    
    az acr login --name "$registry_name"
    docker tag "$DOCKER_IMAGE" "$acr_image"
    docker push "$acr_image"
    
    # Deploy to Container Instances
    print_status "Deploying to Azure Container Instances..."
    
    local acr_username=$(az acr credential show --name "$registry_name" --query username --output tsv)
    local acr_password=$(az acr credential show --name "$registry_name" --query passwords[0].value --output tsv)
    
    az container create \
        --resource-group "$resource_group" \
        --name "$container_group" \
        --image "$acr_image" \
        --registry-login-server "$acr_server" \
        --registry-username "$acr_username" \
        --registry-password "$acr_password" \
        --dns-name-label "$PROJECT_NAME" \
        --ports 5000 \
        --cpu 2 \
        --memory 4 \
        --environment-variables ENV=production
    
    print_success "Azure deployment completed"
}

# Local production deployment
deploy_local() {
    print_status "Setting up local production deployment..."
    
    # Create production directory structure
    local prod_dir="/opt/$PROJECT_NAME"
    
    print_status "Creating production directory: $prod_dir"
    sudo mkdir -p "$prod_dir"/{config,logs,models,data}
    sudo chown -R $USER:$USER "$prod_dir"
    
    # Copy application files
    print_status "Copying application files..."
    cp -r src "$prod_dir/"
    cp -r scripts "$prod_dir/"
    cp requirements.txt "$prod_dir/"
    cp config.json "$prod_dir/"
    
    # Create virtual environment
    print_status "Setting up production virtual environment..."
    cd "$prod_dir"
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    
    # Create systemd service
    print_status "Creating systemd service..."
    
    local service_file="/etc/systemd/system/$PROJECT_NAME.service"
    
    sudo tee "$service_file" > /dev/null << EOF
[Unit]
Description=Pokemon RL Bot Web Dashboard
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$prod_dir
Environment=PATH=$prod_dir/venv/bin
ExecStart=$prod_dir/venv/bin/python scripts/monitor.py --config config.json --host 0.0.0.0 --port 5000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    # Enable and start service
    sudo systemctl daemon-reload
    sudo systemctl enable "$PROJECT_NAME"
    sudo systemctl start "$PROJECT_NAME"
    
    print_success "Local production deployment completed"
    print_status "Service status: sudo systemctl status $PROJECT_NAME"
    print_status "View logs: sudo journalctl -u $PROJECT_NAME -f"
    print_status "Dashboard URL: http://localhost:5000"
}

# Create deployment configuration
create_deployment_config() {
    print_status "Creating deployment configuration..."
    
    cat > deployment-config.json << EOF
{
    "deployment": {
        "type": "$DEPLOYMENT_TYPE",
        "platform": "$PLATFORM",
        "version": "$VERSION",
        "timestamp": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')",
        "docker_image": "$DOCKER_IMAGE",
        "registry": "$DOCKER_REGISTRY"
    },
    "application": {
        "name": "$PROJECT_NAME",
        "port": 5000,
        "health_check": "/api/status"
    }
}
EOF
    
    print_success "Deployment configuration created: deployment-config.json"
}

# Main deployment function
main() {
    echo
    print_status "🚀 Pokemon RL Bot Deployment Script"
    echo "=================================="
    
    # Parse arguments
    parse_args "$@"
    
    # Validate deployment type
    if [[ -z "$DEPLOYMENT_TYPE" ]]; then
        print_error "Deployment type required"
        show_help
        exit 1
    fi
    
    # Set default platform
    if [[ -z "$PLATFORM" ]]; then
        PLATFORM="cpu"
    fi
    
    print_status "Deployment type: $DEPLOYMENT_TYPE"
    print_status "Platform: $PLATFORM"
    print_status "Version: $VERSION"
    
    # Change to script directory
    cd "$(dirname "$0")/.."
    
    # Check dependencies
    check_dependencies
    
    # Create deployment configuration
    create_deployment_config
    
    # Execute deployment based on type
    case $DEPLOYMENT_TYPE in
        docker)
            build_docker_image
            push_docker_image
            ;;
        aws)
            build_docker_image
            push_docker_image
            deploy_aws
            ;;
        gcp)
            build_docker_image
            deploy_gcp
            ;;
        azure)
            build_docker_image
            deploy_azure
            ;;
        local)
            deploy_local
            ;;
        *)
            print_error "Unknown deployment type: $DEPLOYMENT_TYPE"
            exit 1
            ;;
    esac
    
    print_success "Deployment completed successfully! 🎉"
    
    # Show next steps
    echo
    print_status "Next steps:"
    case $DEPLOYMENT_TYPE in
        docker)
            print_status "Run locally: docker run -p 5000:5000 $DOCKER_IMAGE"
            ;;
        aws|gcp|azure)
            print_status "Check cloud console for deployment status and URL"
            ;;
        local)
            print_status "Access dashboard at: http://localhost:5000"
            ;;
    esac
}

# Handle interrupts gracefully
trap 'print_error "Deployment interrupted by user"; exit 1' INT TERM

# Run main function
main "$@"
