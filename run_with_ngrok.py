import subprocess
import threading
import time
import sys
from pyngrok import ngrok

# Paste your token here
ngrok.set_auth_token("2rX6LE5Y9hL5q8h2zLR5ExJ2DTg_5vg5rW8YFQT7uWFkCRKfk")

from pyngrok import ngrok
import time
import threading
import subprocess
from IPython.display import HTML

def run_app():
    print("Starting Flask app...")
    subprocess.call(["python", "app.py"])

thread = threading.Thread(target=run_app)
thread.start()

time.sleep(5)
public_url = ngrok.connect(5000)
print("Ngrok tunnel available at:", public_url.public_url)

# Create a clickable link in the Colab output
HTML(f"""
<a href="{public_url.public_url}" target="_blank" style="font-size:18px;">
  Open your Apple-inspired Iris Predictor
</a>
""")
