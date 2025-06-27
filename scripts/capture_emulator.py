#!/usr/bin/env python3
import subprocess
import os
import time
import base64

def capture_screen():
    """Capture the current screen and return as base64 encoded image"""
    try:
        os.environ['DISPLAY'] = ':99'
        
        # Capture screen with scrot
        subprocess.run(['scrot', '/tmp/current_screen.png'], 
                      capture_output=True, check=True, timeout=5)
        
        # Read and encode the image
        with open('/tmp/current_screen.png', 'rb') as f:
            image_data = f.read()
            
        # Convert to base64 for web display
        base64_image = base64.b64encode(image_data).decode('utf-8')
        return f'data:image/png;base64,{base64_image}'
        
    except Exception as e:
        print(f'Screen capture error: {e}')
        return None

def capture_emulator_window():
    """Try to capture just the emulator window"""
    try:
        os.environ['DISPLAY'] = ':99'
        
        # Get list of windows
        result = subprocess.run(['xwininfo', '-root', '-tree'], 
                              capture_output=True, text=True, timeout=5)
        
        # Look for VBA-M window
        for line in result.stdout.split('\n'):
            if 'vbam' in line.lower() or 'visualboy' in line.lower():
                # Extract window ID
                window_id = line.split()[0]
                print(f"Found emulator window: {window_id}")
                
                # Capture specific window
                subprocess.run(['scrot', '-u', '/tmp/emulator_screen.png'], 
                              capture_output=True, check=True, timeout=5)
                
                with open('/tmp/emulator_screen.png', 'rb') as f:
                    image_data = f.read()
                    
                base64_image = base64.b64encode(image_data).decode('utf-8')
                return f'data:image/png;base64,{base64_image}'
        
        # Fallback to full screen capture
        return capture_screen()
        
    except Exception as e:
        print(f'Window capture error: {e}')
        return capture_screen()

if __name__ == '__main__':
    # Test the capture
    print('Testing screen capture...')
    result = capture_screen()
    if result:
        print('✅ Screen capture successful')
        print(f'📊 Image data length: {len(result)} characters')
    else:
        print('❌ Screen capture failed')
    
    # Test window capture
    print('Testing window capture...')
    result = capture_emulator_window()
    if result:
        print('✅ Window capture successful')
    else:
        print('❌ Window capture failed')
