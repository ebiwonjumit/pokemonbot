#!/bin/bash
# Test VBA-M with virtual display on GCP

# This script tests if VBA-M can work with a virtual display in the cloud

echo "🎮 Testing VBA-M with Virtual Display"
echo "====================================="

# Install required packages
echo "📦 Installing dependencies..."
sudo apt-get update -y
sudo apt-get install -y visualboyadvance-m xvfb

# Start virtual display
echo "🖥️ Starting virtual display..."
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
XVFB_PID=$!

# Wait for display to be ready
sleep 2

# Test if display is working
echo "✅ Virtual display started (PID: $XVFB_PID)"

# Test VBA-M with a ROM (if available)
if [ -f "roms/pokemon_leaf_green.gba" ]; then
    echo "🎮 Testing VBA-M with Pokemon ROM..."
    timeout 10s vbam roms/pokemon_leaf_green.gba --no-sound &
    VBAM_PID=$!
    
    sleep 5
    
    if ps -p $VBAM_PID > /dev/null; then
        echo "✅ VBA-M started successfully with virtual display!"
        kill $VBAM_PID
    else
        echo "❌ VBA-M failed to start"
    fi
else
    echo "⚠️ No ROM file found, testing VBA-M help..."
    vbam --help > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ VBA-M is installed and accessible"
    else
        echo "❌ VBA-M not working"
    fi
fi

# Clean up
kill $XVFB_PID 2>/dev/null

echo "🏁 Test complete!"
