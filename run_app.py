#!/usr/bin/env python3
"""
Simple script to run the Streamlit application.
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit application."""
    
    # Check if streamlit is available
    try:
        import streamlit
        print("✅ Streamlit is available")
    except ImportError:
        print("❌ Streamlit not found. Please install it with: pip install streamlit")
        sys.exit(1)
    
    # Set environment variables for better performance
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    
    # Run the app
    print("🚀 Starting Value Comparison Tool...")
    print("📱 Open your browser and go to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'app.py',
            '--server.port=8501',
            '--server.address=0.0.0.0',
            '--server.headless=true'
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()