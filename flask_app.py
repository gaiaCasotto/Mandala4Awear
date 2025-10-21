# ---------- NEW: Flask ingest server ----------
from flask import Flask, request, jsonify
from eeg_filereader import LiveEEGStreamFeeder
from threading import Thread
import numpy as np

'''===================== FLASK app ===================='''
# ---------- NEW: Flask app factory ----------
def make_app(feeder: LiveEEGStreamFeeder):
    app = Flask(__name__)

    @app.get("/")
    def home():
        return "Awear Test app"

    @app.get("/health")
    def health():
        return jsonify(status="ok", buffered=len(feeder.buf), capacity=feeder.maxlen, fs=feeder.fs)

    @app.post("/ingest")
    def ingest():
        """
        Body: application/json
        {
          "samples": [0.001, -0.002, ...]  # batch of floats
        }
        """
        try:
            payload = request.get_json(force=True, silent=False)
            if not payload or "samples" not in payload:
                return jsonify(error="Missing 'samples' in JSON"), 400
            samples = payload["samples"]
            feeder.push(samples)
            return jsonify(ok=True, received=len(np.asarray(samples).ravel()), buffered=len(feeder.buf))
        except Exception as e:
            return jsonify(error=str(e)), 400

    return app

def start_server(app, host: str, port: int):
    th = Thread(target=lambda: app.run(host=host, port=port, threaded=True, use_reloader=False), daemon=True)
    th.start()
    return th

# ------- end of Flask --------