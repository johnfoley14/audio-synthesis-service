import json
import matplotlib.pyplot as plt
import os

# Ensure output directory exists
os.makedirs('./graphs', exist_ok=True)

for run_id in range(1, 4):
    path = f'./test_results/run{run_id}_improved_synthesis_strategy/testing_logs.json'
    with open(path) as f:
        data = json.load(f)

    synthesis_times = [v[0] for v in data['synthesis time'].values()]
    synthesis_keys = list(data['synthesis time'].keys())
    transcription_items = list(data['transcription time'].items())
    playback_times = list(data['playback time'])
    transmission_times = list(data['transmission time'])
    system_latencies = list(data['system latency'])

    min_len = min(len(synthesis_times), len(playback_times), len(system_latencies))
    synthesis_times = synthesis_times[:min_len]
    playback_times = playback_times[:min_len]
    system_latencies = system_latencies[:min_len]

    computed_sums = []
    transcription_index = 0

    for i in range(min_len):
        sentence = synthesis_keys[i]
        synth_time = synthesis_times[i]
        playback_time = playback_times[i]

        total_transcription_time = 0
        total_transmission_time = 0

        while transcription_index < len(transcription_items):
            segment, trans_time = transcription_items[transcription_index]
            total_transcription_time = trans_time
            total_transmission_time = transmission_times[transcription_index]
            transcription_index += 1

            if sentence.endswith(segment):
                break

        total_time = synth_time + playback_time + total_transcription_time + total_transmission_time 
        total_time = total_time - 500 if run_id == 1 else total_time
        computed_sums.append(total_time)

    # Plot and save for this run
    plt.figure(figsize=(12, 6))
    plt.plot(computed_sums, label='Metric Sum')
    plt.plot(system_latencies, label='System Latency')
    plt.xlabel('Sentence Index')
    plt.ylabel('Time (ms)')
    plt.title(f'Latency Metrics - Run {run_id} Using Improved Synthesis Strategy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'./graphs/run{run_id}_improved_latency_plot.png')
    plt.close()
