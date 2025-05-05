import subprocess, os, glob

input_file = '../../transcription-service/tests/test_audio.mp3'
bitrate_kbps = 24  # Typical VoIP bitrate

chunk_dir = 'chunks'
frame_ms = 20
frame_sec = frame_ms / 1000

# Step 1: Create chunk dir
os.makedirs(chunk_dir, exist_ok=True)

# Step 2: Split input into 20 ms chunks
subprocess.run([
    'ffmpeg',
    '-i', input_file,
    '-f', 'segment',
    '-segment_time', str(frame_sec),
    '-c', 'copy',
    f'{chunk_dir}/chunk_%04d.wav'
], check=True)

# Step 3: Encode each chunk with Opus and count size
total_bytes = 0
for chunk_file in sorted(glob.glob(f'{chunk_dir}/chunk_*.wav')):
    encoded_file = './output.opus'
    subprocess.run([
        'ffmpeg',
        '-y',
        '-i', chunk_file,
        '-c:a', 'libopus',
        '-b:a', f'{bitrate_kbps}k',
        encoded_file
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    total_bytes += os.path.getsize(encoded_file)

# Step 4: Report bandwidth
duration_sec = len(glob.glob(f'{chunk_dir}/chunk_*.wav')) * frame_sec
bandwidth_kbps = (total_bytes * 8) / duration_sec / 1000

print(f"Simulated stream length: {duration_sec:.2f} seconds")
print(f"Total Opus output size: {total_bytes / 1024:.2f} KB")
print(f"Estimated streaming bandwidth: {bandwidth_kbps:.2f} kbps")
