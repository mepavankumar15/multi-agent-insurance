import os
import sys
import subprocess

# Disable file watcher (causes torch errors)
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

print("Starting Streamlit app... (using system Python)")

# Use plain 'python' from PATH instead of Anaconda
cmd = [
    "python", "-m", "streamlit", "run", "app.py",
    "--server.port", "8501",
    "--logger.level", "error"
]

subprocess.run(cmd)
