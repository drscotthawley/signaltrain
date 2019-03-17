#app.py
# This is for deploying the demo to Heroku
import subprocess
import atexit
from flask import render_template, Flask
import os

template_dir = os.path.abspath('./demo')
app = Flask(__name__, template_folder=template_dir)

port = os.environ['PORT']  # Heroku requires that you use $PORT
bokeh_process = subprocess.Popen(
    ['bokeh', 'serve',f'--port={port}', '--address=0.0.0.0', '--allow-websocket-origin=signaltrain.herokuapp.com','--use-xheaders','bokeh_sliders.py'], cwd="./demo", stdout=subprocess.PIPE)

@atexit.register
def kill_server():
    bokeh_process.kill()

@app.route("/")
def index():
    return render_template('index.html')

if __name__ == "__main__":
    print("Starting app")
    app.run(debug=True)
