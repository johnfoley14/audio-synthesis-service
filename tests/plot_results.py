import matplotlib.pyplot as plt, numpy as np, json
from scipy import stats

files = ['./test_results/run1/testing_logs.json', './test_results/run2/testing_logs.json', './test_results/run3/testing_logs.json']
labels = ['Run 1', 'Run 2', 'Run 3']

def plot_data(files, labels, key, ylabel, title):
    plt.figure(figsize=(10, 6))
    for file, label in zip(files, labels):
        with open(file) as f:
            data = json.load(f)
            values = list(data[key].values()) if isinstance(data[key], dict) else data[key]
            plt.plot(range(len(values)), values, label=label)

    plt.xlabel('Index')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'./graphs/{key.replace(" ", "_")}.png')
    plt.close()

# Plot synthesis time
plot_data(files, labels, 'synthesis time', 'Synthesis Time (ms)', 'Sentence Synthesis Time Comparison')

# Plot transcription time
plot_data(files, labels, 'transcription time', 'Transcription Time (ms)', 'Sentence Transcription Time Comparison')

# Plot system latency
plot_data(files, labels, 'system latency', 'System Latency (ms)', 'System Latency Comparison')

metrics = ['synthesis time', 'transcription time', 'system latency']

def process_metric(files, metric):
    data = []
    for file in files:
        with open(file) as f:
            json_data = json.load(f)
            data.extend(json_data[metric].values() if isinstance(json_data[metric], dict) else json_data[metric])

    # Calculations
    n = len(data)
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)
    sem = std_dev / np.sqrt(n)
    ci = stats.t.interval(0.95, df=n-1, loc=mean, scale=sem)

    print(f"\n=== {metric.capitalize()} ({n} entries) ===")
    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std_dev}")
    print(f"95% Confidence Interval: {ci}")

    # Plot histogram
    plt.hist(data, bins=40, edgecolor='black')
    plt.title(f'Distribution of {metric.capitalize()}')
    plt.xlabel('Time (ms)')
    plt.ylabel('Frequency')
    plt.grid(True)
    text = (
        f"Mean: {mean:.2f} ms\n"
        f"Std Dev: {std_dev:.2f} ms\n"
        f"95% CI: ({ci[0]:.2f}, {ci[1]:.2f}) ms"
    )
    plt.figtext(0.5, -0.1, text, ha='center', fontsize=10)

    # Save and close
    plt.tight_layout()
    plt.savefig(f'./graphs/{metric.replace(" ", "_")}_histogram.png', bbox_inches='tight')
    plt.close()

# Process each metric
for metric in metrics:
    process_metric(files, metric)

