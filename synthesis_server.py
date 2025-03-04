import torch, time, pyaudio, numpy as np, sounddevice as sd
from TTS.api import TTS
from flask import Flask, request, jsonify

app = Flask(__name__)

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

@app.route("/translate", methods=["POST"])
def translate():
    global tts_model

    if tts_model is None:
        return jsonify({"status": "error", "message": "No model loaded. Call /load_model first."}), 400

    text = request.json.get("text")
    if not text:
        return jsonify({"status": "error", "message": "No text provided."}), 400
    start_time = time.time()

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

        time_taken = time.time() - start_time
        app.logger.info(f"Translated text to audio in {time_taken:.2f}s")
        return jsonify({"status": "success", "message": "Audio played successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
