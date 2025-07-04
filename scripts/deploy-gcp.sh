#!/bin/bash

# Pokemon RL Bot - Google Cloud Platform Deployment
# Simple script to deploy your bot to GCP with GPU support

set -e

echo "🎮 Pokemon RL Bot - GCP Deployment"
echo "=================================="

# Configuration - Using your exact GCP settings
PROJECT_ID="pokebot-45"
INSTANCE_NAME="pokemon-rl-bot"
ZONE="us-central1-a"
MACHINE_TYPE="n1-standard-4"
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1
DISK_SIZE="10"  # Size in GB (matching your 10GB configuration)
IMAGE_FAMILY="debian-12"
IMAGE_PROJECT="debian-cloud"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() { echo -e "${YELLOW}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if gcloud is installed
check_gcloud() {
    if ! command -v gcloud &> /dev/null; then
        print_error "gcloud CLI not found. Install from: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
    
    # Check if authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        print_error "Not authenticated with gcloud. Run: gcloud auth login"
        exit 1
    fi
    
    print_success "gcloud CLI ready"
}

# Set up project
setup_project() {
    if [ -z "$PROJECT_ID" ]; then
        print_status "Available projects:"
        gcloud projects list --format="table(projectId,name)"
        echo
        read -p "Enter your Google Cloud Project ID: " PROJECT_ID
    fi
    
    gcloud config set project $PROJECT_ID
    print_success "Using project: $PROJECT_ID"
}

# Enable required APIs
enable_apis() {
    print_status "Enabling required APIs..."
    gcloud services enable compute.googleapis.com
    gcloud services enable storage.googleapis.com
    print_success "APIs enabled"
}

# Create the VM instance
create_instance() {
    print_status "Creating Pokemon RL Bot instance..."
    
    # Check if instance already exists
    if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE &>/dev/null; then
        print_error "Instance $INSTANCE_NAME already exists in zone $ZONE"
        read -p "Delete and recreate? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --quiet
        else
            exit 1
        fi
    fi
    
    # Create startup script
    cat > startup-script.sh << 'EOF'
#!/bin/bash
# Pokemon RL Bot startup script - Debian 12 compatible

# Update system
apt-get update -y

# Install Python and pip
apt-get install -y python3-pip python3-venv python3-dev git curl wget

# Install build tools for Python packages
apt-get install -y build-essential cmake

# Install VBA-M emulator (might need to compile from source on Debian 12)
apt-get install -y libsdl2-dev libsfml-dev libgl1-mesa-dev libglu1-mesa-dev
# Try package manager first
apt-get install -y visualboyadvance-m || {
    echo "VBA-M not available in repos, will install manually later"
}

# Install virtual display for headless emulator  
apt-get install -y xvfb x11vnc fluxbox

# Install NVIDIA drivers (if GPU instance)
apt-get install -y nvidia-driver nvidia-cuda-toolkit || {
    echo "GPU drivers not available, continuing with CPU-only setup"
}

# Create project directory
mkdir -p /home/pokemon-rl-bot
cd /home/pokemon-rl-bot

# Set up virtual display service
cat > /etc/systemd/system/pokemon-display.service << 'DISPLAY_EOF'
[Unit]
Description=Pokemon RL Bot Virtual Display
After=network.target

[Service]
Type=simple
User=root
Environment=DISPLAY=:99
ExecStart=/usr/bin/Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
DISPLAY_EOF

# Enable and start the display service
systemctl enable pokemon-display
systemctl start pokemon-display

# Create environment file for all users
echo "export DISPLAY=:99" >> /etc/environment

# Create a test script to verify emulator works
cat > /home/pokemon-rl-bot/test_emulator.sh << 'TEST_EOF'
#!/bin/bash
export DISPLAY=:99
echo "Testing VBA-M with virtual display..."
if command -v vbam &> /dev/null; then
    echo "✅ VBA-M installed"
    if [ -f "roms/pokemon_leaf_green.gba" ]; then
        echo "🎮 ROM file found, testing emulator..."
        timeout 5s vbam roms/pokemon_leaf_green.gba &
        sleep 2
        if pgrep vbam > /dev/null; then
            echo "✅ VBA-M running successfully!"
            pkill vbam
        else
            echo "❌ VBA-M failed to start"
        fi
    else
        echo "⚠️ No ROM file yet"
    fi
else
    echo "❌ VBA-M not found - will need manual installation"
fi
TEST_EOF

chmod +x /home/pokemon-rl-bot/test_emulator.sh

echo "Pokemon RL Bot VM ready!" > /tmp/startup-complete
EOF

    # Create the instance (matching your exact Google Cloud configuration)
    gcloud compute instances create $INSTANCE_NAME \
        --project=$PROJECT_ID \
        --zone=$ZONE \
        --machine-type=$MACHINE_TYPE \
        --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
        --metadata=enable-osconfig=TRUE \
        --maintenance-policy=TERMINATE \
        --provisioning-model=STANDARD \
        --service-account=604999196487-compute@developer.gserviceaccount.com \
        --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/trace.append \
        --enable-display-device \
        --create-disk=auto-delete=yes,boot=yes,device-name=$INSTANCE_NAME,image=projects/$IMAGE_PROJECT/global/images/debian-12-bookworm-v20250610,mode=rw,size=10,type=pd-balanced \
        --no-shielded-secure-boot \
        --shielded-vtpm \
        --shielded-integrity-monitoring \
        --labels=goog-ops-agent-policy=v2-x86-template-1-4-0,goog-ec-src=vm_add-gcloud \
        --reservation-affinity=any \
        --metadata-from-file startup-script=startup-script.sh \
        --tags=pokemon-rl-bot
    
    print_success "Instance created: $INSTANCE_NAME"
    
    # Ensure the instance has the correct tags for firewall rules
    gcloud compute instances add-tags $INSTANCE_NAME --zone=$ZONE --tags=pokemon-rl-bot || echo "Tags already applied"
    
    # Clean up
    rm -f startup-script.sh
}

# Create GPU instance (if needed later)
create_instance_gpu() {
    print_status "Creating Pokemon RL Bot GPU instance..."
    
    # Check if instance already exists
    if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE &>/dev/null; then
        print_error "Instance $INSTANCE_NAME already exists in zone $ZONE"
        read -p "Delete and recreate? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --quiet
        else
            exit 1
        fi
    fi
    
    # Use same startup script as CPU version
    cat > startup-script.sh << 'EOF'
#!/bin/bash
# Pokemon RL Bot startup script - Debian 12 compatible with GPU
apt-get update -y
apt-get install -y python3-pip python3-venv python3-dev git curl wget build-essential cmake
apt-get install -y libsdl2-dev libsfml-dev libgl1-mesa-dev libglu1-mesa-dev
apt-get install -y visualboyadvance-m || echo "VBA-M not available in repos"
apt-get install -y xvfb x11vnc fluxbox
apt-get install -y nvidia-driver nvidia-cuda-toolkit || echo "GPU drivers not available"
mkdir -p /home/pokemon-rl-bot
echo "export DISPLAY=:99" >> /etc/environment
echo "Pokemon RL Bot GPU VM ready!" > /tmp/startup-complete
EOF
    
    # Create GPU instance
    gcloud compute instances create $INSTANCE_NAME \
        --project=$PROJECT_ID \
        --zone=$ZONE \
        --machine-type=$MACHINE_TYPE \
        --accelerator=type=$GPU_TYPE,count=$GPU_COUNT \
        --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
        --maintenance-policy=TERMINATE \
        --provisioning-model=STANDARD \
        --service-account=604999196487-compute@developer.gserviceaccount.com \
        --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/trace.append \
        --enable-display-device \
        --create-disk=auto-delete=yes,boot=yes,device-name=$INSTANCE_NAME,image=projects/$IMAGE_PROJECT/global/images/debian-12-bookworm-v20250610,mode=rw,size=$DISK_SIZE,type=pd-balanced \
        --shielded-vtpm \
        --shielded-integrity-monitoring \
        --metadata-from-file startup-script=startup-script.sh \
        --tags=pokemon-rl-bot
    
    print_success "GPU Instance created: $INSTANCE_NAME"
    rm -f startup-script.sh
}

# Create firewall rule for web dashboard
setup_firewall() {
    print_status "Setting up firewall for web dashboard..."
    
    if ! gcloud compute firewall-rules describe pokemon-rl-dashboard &>/dev/null; then
        gcloud compute firewall-rules create pokemon-rl-dashboard \
            --allow tcp:7500 \
            --source-ranges 0.0.0.0/0 \
            --target-tags pokemon-rl-bot \
            --description "Allow access to Pokemon RL Bot dashboard"
        print_success "Firewall rule created"
    else
        print_status "Firewall rule already exists"
    fi
}

# Upload code to instance
upload_code() {
    print_status "Uploading code to instance..."
    
    # Wait for instance to be ready
    print_status "Waiting for instance to start..."
    gcloud compute instances wait-until-running $INSTANCE_NAME --zone=$ZONE
    
    # Create tarball of code (excluding unnecessary files)
    tar -czf pokemon-rl-bot.tar.gz \
        --exclude='.git' \
        --exclude='venv' \
        --exclude='__pycache__' \
        --exclude='logs' \
        --exclude='models' \
        --exclude='*.pyc' \
        --exclude='test_screenshot.png' \
        --exclude='dashboard_screenshot.png' \
        .
    
    # Upload code
    gcloud compute scp pokemon-rl-bot.tar.gz $INSTANCE_NAME:/home/pokemon-rl-bot/ --zone=$ZONE
    
    # Extract and setup on remote
    gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
        cd /home/pokemon-rl-bot && 
        tar -xzf pokemon-rl-bot.tar.gz &&
        sudo chown -R \$(whoami):\$(whoami) . &&
        echo 'Code uploaded successfully'
    "
    
    # Clean up local tarball
    rm -f pokemon-rl-bot.tar.gz
    
    print_success "Code uploaded to instance"
}

# Start the bot
start_bot() {
    print_status "Starting Pokemon RL Bot on cloud instance..."
    
    # Get external IP
    EXTERNAL_IP=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
    
    # Setup and start the bot via SSH (Simple Python approach)
    gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
        cd /home/pokemon-rl-bot &&
        # Create Python virtual environment
        python3 -m venv venv &&
        source venv/bin/activate &&
        # Install dependencies
        pip install -r requirements.txt &&
        # Start the dashboard with virtual display
        export DISPLAY=:99 &&
        nohup python scripts/simple_dashboard.py > bot.log 2>&1 &
        echo 'Bot started successfully!'
    "
    
    print_success "Pokemon RL Bot deployed successfully!"
    echo
    echo "🎉 Deployment Complete!"
    echo "======================"  
    echo "📍 Instance: $INSTANCE_NAME"
    echo "🌐 External IP: $EXTERNAL_IP"
    echo "📱 Dashboard: http://$EXTERNAL_IP:7500"
    echo "🔑 SSH Access: gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
    echo
    echo "⚠️  Remember to stop the instance when not using it to save costs:"
    echo "   gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE"
    echo
    echo "🎮 The bot is running with a virtual display for VBA-M"
    echo "📊 Monitor training progress at the dashboard URL above"
}

# Show help
show_help() {
    echo "Pokemon RL Bot GCP Deployment"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  deploy     Deploy bot to GCP (CPU instance)"
    echo "  deploy-gpu Deploy bot to GCP (GPU instance - costs more)"
    echo "  start      Start existing instance"
    echo "  stop       Stop instance"
    echo "  status     Show instance status"
    echo "  ssh        SSH into instance"
    echo "  logs       Show bot logs"
    echo "  test-emulator  Test VBA-M emulator on cloud"
    echo "  delete     Delete instance (careful!)"
    echo ""
}

# Main execution
main() {
    case "${1:-deploy}" in
        deploy)
            check_gcloud
            setup_project
            enable_apis
            create_instance
            setup_firewall
            upload_code
            start_bot
            ;;
        deploy-gpu)
            check_gcloud
            setup_project
            enable_apis
            # Set GPU configuration
            GPU_TYPE="nvidia-tesla-t4"
            GPU_COUNT=1
            MACHINE_TYPE="n1-standard-4"
            create_instance_gpu
            setup_firewall
            upload_code
            start_bot
            ;;
        start)
            gcloud compute instances start $INSTANCE_NAME --zone=$ZONE
            ;;
        stop)
            gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE
            ;;
        status)
            gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --format="table(name,status,machineType,zone)"
            ;;
        ssh)
            gcloud compute ssh $INSTANCE_NAME --zone=$ZONE
            ;;
        logs)
            gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="cd /home/titoebiwonjumi && tail -f bot.log"
            ;;
        test-emulator)
            print_status "Testing VBA-M emulator on cloud instance..."
            gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="cd /home/titoebiwonjumi && ./test_cloud_emulator.sh"
            ;;
        delete)
            read -p "Are you sure you want to delete the instance? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE
            fi
            ;;
        help|*)
            show_help
            ;;
    esac
}

main "$@"
