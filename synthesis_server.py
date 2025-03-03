import torch
from TTS.api import TTS
import sounddevice as sd
import numpy as np
import pyaudio
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

tts_model = None
sample_rate = 24000
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.route("/load_model", methods=["POST"])
def load_model():
    global tts_model

    # Get the model name from request, default to xtts_v2
    model_name = request.json.get("model", "tts_models/multilingual/multi-dataset/xtts_v2")

    try:
        # Load the TTS model
        tts_model = TTS(model_name).to(device)
        return jsonify({"status": "success", "message": f"Model {model_name} loaded."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/translate", methods=["POST"])
def translate():
    global tts_model

    if tts_model is None:
        return jsonify({"status": "error", "message": "No model loaded. Call /load_model first."}), 400

    text = request.json.get("text")
    if not text:
        return jsonify({"status": "error", "message": "No text provided."}), 400

    try:
        wav = tts_model.tts(text=text, speaker_wav="./audio/johns_voice.wav", language="en")

        # Play the generated audio
        sd.play(wav, samplerate=sample_rate)
        sd.wait()

        # wav_np = np.array(wav, dtype=np.float32)
        # wav_int16 = (wav_np * 32767).astype(np.int16)
        # p = pyaudio.PyAudio()
        # stream = p.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
        # stream.write(wav_int16.tobytes())
        # stream.stop_stream()
        # stream.close()
        # p.terminate()

        return jsonify({"status": "success", "message": "Audio played successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
