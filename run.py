import subprocess
import time

def run_fastapi():
    # Command to run FastAPI 
    subprocess.Popen(["fastapi", "dev", 'src/api/api.py'])

def run_streamlit():
    # Command to run Streamlit
    subprocess.Popen(["streamlit", "run", "src/gui/gui.py"])

if __name__ == "__main__":
    # Start FastAPI server
    print("Starting FastAPI...")
    run_fastapi()
    
    # Give FastAPI time to start before running Streamlit
    time.sleep(2)
    
    # Start Streamlit app
    print("Starting Streamlit...")
    run_streamlit()
    
    # Keep the script running
    while True:
        time.sleep(1)