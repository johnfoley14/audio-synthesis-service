import torch, time, pyaudio, numpy as np, sounddevice as sd
from TTS.api import TTS
from flask import Flask, request, jsonify
from flask_cors import CORS
from threading import Thread, Lock
import itertools

synthesis_results = {}         # Maps sequence_id -> wav
synthesis_lock = Lock()
synthesis_counter = itertools.count(start=0)
next_to_play = 0

app = Flask(__name__)
CORS(app)

tts_model = None
sample_rate = 24000
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.route("/load_model", methods=["POST"])
def load_model():
    global tts_model

    model_name = request.json.get("model", "tts_models/multilingual/multi-dataset/xtts_v2")

    if tts_model is not None and model_name == tts_model.model_name:
        return jsonify({"status": "success", "message": f"Model {model_name} already loaded."})

    try:
        tts_model = TTS(model_name).to(device)

        return jsonify({"status": "success", "message": f"Model {model_name} loaded and warmed up."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def playback_worker():
    global next_to_play
    while True:
        with synthesis_lock:
            if next_to_play in synthesis_results:
                wav = synthesis_results.pop(next_to_play)
                next_to_play += 1
            else:
                wav = None

        if wav:
            try:
                sd.play(wav, samplerate=sample_rate)
                sd.wait()
            except Exception as e:
                print("Playback error:", e)
        else:
            time.sleep(0.01)  # Small sleep to prevent busy waiting

@app.route("/synthesis", methods=["POST"])
def synthesis():
    if tts_model is None:
        return jsonify({"status": "error", "message": "No model loaded."}), 400

    text = request.json.get("transcript")
    if not text:
        return jsonify({"status": "error", "message": "No text provided."}), 400

    sequence_id = next(synthesis_counter)

    def synthesize(text, sid):
        try:
            wav = tts_model.tts(text=text, speaker_wav="./audio/johns_voice.wav", language="en")
            with synthesis_lock:
                synthesis_results[sid] = wav
                print(f"sequence id {sid} added to playback queue")
        except Exception as e:
            print(f"Synthesis error for seq {sid}:", e)

    Thread(target=synthesize, args=(text, sequence_id), daemon=True).start()
    return jsonify({"status": "success", "message": "Synthesis started and will be played in order."})

Thread(target=playback_worker, daemon=True).start()


if __name__ == "__main__":
    app.run(debug=True)
