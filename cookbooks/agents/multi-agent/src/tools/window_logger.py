import subprocess
import sys
import os
import tempfile
import time
import math
from threading import Lock

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class WindowLogger:
    """Manages separate terminal windows for agent logging with smart positioning"""
    
    def __init__(self):
        self.windows = {}
        self.temp_dir = tempfile.mkdtemp()
        self.lock = Lock()
        self.log_files = {}
        self.expected_window_count = 1
        self.screen_width = 1920
        self.screen_height = 1080
        self._get_screen_dimensions()
        print(f"[WindowLogger] Initialized with screen: {self.screen_width}x{self.screen_height}")
    
    def _get_screen_dimensions(self):
        """Get screen dimensions using AppleScript"""
        try:
            script = '''
    tell application "System Events"
        set screenSize to size of first desktop
        return item 1 of screenSize & "," & item 2 of screenSize
    end tell
    '''
            result = subprocess.run(['osascript', '-e', script], 
                                capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                width, height = result.stdout.strip().split(',')
                self.screen_width = int(float(width))
                self.screen_height = int(float(height))
                print(f"[WindowLogger] Detected screen: {self.screen_width}x{self.screen_height}")
                return
        except Exception as e:
            print(f"[WindowLogger] Screen detection error: {e}")
        
        print(f"[WindowLogger] Using fallback: {self.screen_width}x{self.screen_height}")
    
    def _get_adaptive_spacing(self):
        """Calculate spacing and margins based on screen size"""
        # Base calculations on screen size - use percentages
        base_padding = max(10, int(self.screen_width * 0.01))  # 1% of screen width, min 10px
        title_bar_height = max(60, int(self.screen_height * 0.08))  # 8% of screen height
        dock_height = max(60, int(self.screen_height * 0.08))  # 8% of screen height
        
        # Minimum window sizes based on screen size
        min_window_width = max(300, int(self.screen_width * 0.15))  # 15% of screen width
        min_window_height = max(200, int(self.screen_height * 0.2))  # 20% of screen height
        
        return {
            'padding': base_padding,
            'title_bar_height': title_bar_height,
            'dock_height': dock_height,
            'min_width': min_window_width,
            'min_height': min_window_height
        }
    
    def set_expected_window_count(self, count: int):
        """Set the expected number of windows for optimal positioning"""
        self.expected_window_count = max(1, count)
        print(f"[WindowLogger] Set expected window count: {self.expected_window_count}")
    
    def _calculate_grid_layout(self, window_count: int):
        """Calculate optimal grid layout for given number of windows"""
        cols = math.ceil(math.sqrt(window_count))
        rows = math.ceil(window_count / cols)
        return cols, rows
    
    def _calculate_window_position(self, window_index: int):
        """Calculate position and size for a specific window - fully adaptive"""
        cols, rows = self._calculate_grid_layout(self.expected_window_count)
        spacing = self._get_adaptive_spacing()
        
        print(f"[WindowLogger] Grid layout for {self.expected_window_count} windows: {cols}x{rows}")
        print(f"[WindowLogger] Adaptive spacing: padding={spacing['padding']}, title_bar={spacing['title_bar_height']}, dock={spacing['dock_height']}")
        
        total_padding_width = spacing['padding'] * (cols + 1)
        total_padding_height = spacing['padding'] * (rows + 1)
        
        available_width = self.screen_width - total_padding_width
        available_height = self.screen_height - spacing['title_bar_height'] - spacing['dock_height'] - total_padding_height
        
        calculated_width = available_width // cols
        calculated_height = available_height // rows
        
        window_width = max(spacing['min_width'], calculated_width)
        window_height = max(spacing['min_height'], calculated_height)
        
        # If windows are too big due to minimums, adjust screen usage
        total_needed_width = window_width * cols + total_padding_width
        total_needed_height = window_height * rows + spacing['title_bar_height'] + spacing['dock_height'] + total_padding_height
        
        if total_needed_width > self.screen_width or total_needed_height > self.screen_height:
            
            width_scale = self.screen_width / total_needed_width if total_needed_width > self.screen_width else 1.0
            height_scale = self.screen_height / total_needed_height if total_needed_height > self.screen_height else 1.0
            scale = min(width_scale, height_scale) * 0.95  # 5% safety margin
            
            window_width = int(window_width * scale)
            window_height = int(window_height * scale)
            spacing['padding'] = int(spacing['padding'] * scale)
        
        row = window_index // cols
        col = window_index % cols
        
        # Calculate actual screen coordinates
        x = spacing['padding'] + col * (window_width + spacing['padding'])
        y = spacing['title_bar_height'] + spacing['padding'] + row * (window_height + spacing['padding'])
        
        position = {
            'x': x,
            'y': y,
            'width': window_width,
            'height': window_height
        }
        
        print(f"[WindowLogger] Window {window_index} position: x={x}, y={y}, w={window_width}, h={window_height}")
        return position

    def create_window(self, agent_name: str):
        """Create a new terminal window for an agent with smart positioning"""
        with self.lock:
            if agent_name in self.windows:
                print(f"[WindowLogger] Window for {agent_name} already exists")
                return 
            
            print(f"[WindowLogger] Creating window for {agent_name}")
            
            window_index = len(self.windows)
            position = self._calculate_window_position(window_index)
            
            # Create a log file for this agent
            log_path = os.path.join(self.temp_dir, f"{agent_name}.log")
            self.log_files[agent_name] = log_path
            
            with open(log_path, 'w') as f:
                f.write(f"=== {agent_name} Agent Log ===\n")
                f.write("Starting agent...\n")
                f.flush()
            
            script_content = f'''#!/bin/bash
echo "=== {agent_name} Agent Window ==="
echo "Position: {position['x']},{position['y']} Size: {position['width']}x{position['height']}"
echo "Monitoring agent activity..."
echo ""

# Monitor log file and display updates
tail -f "{log_path}" &
TAIL_PID=$!

# Simple monitoring loop - Python handles window closing
while true; do
    if [ -f "{log_path}" ] && grep -q "AGENT_COMPLETED\\|AGENT_ERROR" "{log_path}" 2>/dev/null; then
        echo ""
        echo "🎉 Agent {agent_name} completed!"
        echo "Python will close this window..."
        
        # Clean exit - let Python handle the window closing
        kill $TAIL_PID 2>/dev/null || true
        exit 0
    fi
    sleep 1
done
'''
            
            script_path = os.path.join(self.temp_dir, f"{agent_name}_logger.sh")
            with open(script_path, 'w') as f:
                f.write(script_content)
            os.chmod(script_path, 0o755)
            
            # Open new Terminal window with positioning
            applescript = f'''
tell application "Terminal"
    activate
    set newTab to do script "bash {script_path}"
    set custom title of front window to "🤖 {agent_name} Agent"
    
    -- Wait a moment for window to be created
    delay 0.5
    
    -- Set window bounds: left, top, right, bottom
    set bounds of front window to {{{position['x']}, {position['y']}, {position['x'] + position['width']}, {position['y'] + position['height']}}}
end tell
'''
            
            print(f"[WindowLogger] Running AppleScript for {agent_name}")
            try:
                result = subprocess.run(['osascript', '-e', applescript], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode != 0:
                    print(f"[WindowLogger] AppleScript error: {result.stderr}")
                else:
                    print(f"[WindowLogger] Successfully positioned window for {agent_name}")
            except Exception as e:
                print(f"[WindowLogger] Error running AppleScript: {e}")
            
            self.windows[agent_name] = {
                'log_path': log_path,
                'script_path': script_path,
                'created': True,
                'position': position
            }
    
    def log_to_window(self, agent_name: str, message: str):
        """Send a log message to the agent's window"""
        with self.lock:  # Use lock to prevent race conditions
            if agent_name not in self.windows:
                print(f"[WindowLogger] Warning: No window found for {agent_name}, creating one...")
                self.create_window(agent_name)
        
        log_path = self.windows[agent_name]['log_path']
        timestamp = time.strftime("%H:%M:%S")
        
        try:
            with open(log_path, 'a') as f:
                f.write(f"[{timestamp}] {message}\n")
                f.flush()
        except Exception as e:
            print(f"Error writing to log for {agent_name}: {e}")
    
    def force_close_window(self, agent_name: str):
        """Force close a specific agent window using AppleScript"""
        if agent_name not in self.windows:
            return
        
        print(f"[WindowLogger] Force closing window for {agent_name}")

        script = f'tell application "Terminal" to close (first window whose custom title contains "🤖 {agent_name} Agent")'
        
        result = subprocess.run(['osascript', '-e', script], 
                                capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            print(f"[WindowLogger] Successfully closed window for {agent_name}")
        else:
            print(f"[WindowLogger] Failed to close window: {result.stderr}")
               
           
    
    def mark_agent_completed(self, agent_name: str):
        """Mark an agent as completed to trigger window closure"""
        self.log_to_window(agent_name, "")
        self.log_to_window(agent_name, "="*50)
        self.log_to_window(agent_name, "✅ TASK COMPLETED SUCCESSFULLY!")
        self.log_to_window(agent_name, "="*50)
        self.log_to_window(agent_name, "AGENT_COMPLETED")
        
        time.sleep(1)
        self.force_close_window(agent_name)
    
    def mark_agent_error(self, agent_name: str, error_msg: str):
        """Mark an agent as having an error to trigger window closure"""
        self.log_to_window(agent_name, "")
        self.log_to_window(agent_name, "="*50)
        self.log_to_window(agent_name, f"❌ ERROR: {error_msg}")
        self.log_to_window(agent_name, "="*50)
        self.log_to_window(agent_name, "AGENT_ERROR")
        
        time.sleep(1)
        self.force_close_window(agent_name)
    
    def cleanup(self):
        """Clean up temporary files"""

        for agent_name in self.windows:
            if os.path.exists(self.windows[agent_name]['log_path']):
                with open(self.windows[agent_name]['log_path'], 'a') as f:
                    f.write("\nAGENT_COMPLETED\n")
                    f.flush()
        
        time.sleep(3)
        
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

# Global instance
window_logger = WindowLogger() 