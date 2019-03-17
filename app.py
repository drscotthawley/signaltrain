#app.py
# this is for deploying the demo to Heroku
import subprocess
import atexit
from flask import render_template, render_template_string, Flask
from bokeh.client import pull_session
import os
os.chdir('demo')


template_dir = os.path.abspath('./demo')
app = Flask(__name__, template_folder=template_dir)

local = False
if local:
    port = 8000
    bokeh_process = subprocess.Popen(
        ['bokeh', 'serve',f'--allow-websocket-origin=localhost:{port}','bokeh_sliders.py'], cwd='demo', stdout=subprocess.PIPE)
else:
    bokeh_process = subprocess.Popen(
        ['bokeh', 'serve','--allow-websocket-origin=signaltrain.herokuapp.com','bokeh_sliders.py'], cwd='demo', stdout=subprocess.PIPE)

@atexit.register
def kill_server():
    bokeh_process.kill()

@app.route("/")
def index():
    #session=pull_session(app_path="demo/")
    #bokeh_script=autoload_server(None,app_path="/demo",session_id=session.id)
    #return render_template("index.html")#, bokeh_script=bokeh_script)
    return render_template('index.html')

if __name__ == "__main__":
    print("STARTED")
    app.run(debug=True)
