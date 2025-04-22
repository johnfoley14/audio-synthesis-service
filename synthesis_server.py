import torch, time, pyaudio, numpy as np, sounddevice as sd
from TTS.api import TTS
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

tts_model = None
sample_rate = 24000
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.route("/load_model", methods=["POST"])
def load_model():
    global tts_model

    # Get the model name from request, default to xtts_v2
    model_name = request.json.get("model", "tts_models/multilingual/multi-dataset/xtts_v2")
    if not model_name:
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        
    if tts_model is not None and model_name == tts_model.model_name:
        return jsonify({"status": "success", "message": f"Model {model_name} already loaded."})
    
    try:
        tts_model = TTS(model_name).to(device)
        return jsonify({"status": "success", "message": f"Model {model_name} loaded."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/synthesis", methods=["POST"])
def synthesis():
    global tts_model

    if tts_model is None:
        return jsonify({"status": "error", "message": "No model loaded. Call /load_model first."}), 400
    
    text = request.json.get("transcript")
    if not text:
        return jsonify({"status": "error", "message": "No text provided."}), 400
    start_time = time.time()

    try:
        print("Synthesizing audio...")
        # wav = tts_model.tts(text=text)
        wav = tts_model.tts(text=text, speaker_wav="./audio/johns_voice.wav", language="en")
        print("Audio synthesis complete.")
        # Play the generated audio
        sd.play(wav, samplerate=sample_rate)
        sd.wait()

        time_taken = time.time() - start_time
        app.logger.info(f"Translated text to audio in {time_taken:.2f}s")
        return jsonify({"status": "success", "message": "Audio played successfully."})
    except Exception as e:
        print(e)
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
