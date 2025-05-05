import os
import diff_match_patch as dmp_module
from jiwer import wer

# Load reference transcript
with open('./annotation_formatted.txt', 'r', encoding='utf-8') as f:
    reference = f.read().strip()

# Loop over each run folder
for i in range(1, 4):
    run_folder = f'./test_results/run{i}'
    hyp_file = os.path.join(run_folder, f'output_transcript.txt')
    with open(hyp_file, 'r') as file:
        text = file.read().replace('\n', ' ')

    with open(f'{run_folder}/transcript_formatted.txt', 'w') as file:
        file.write(text)

    hypothesis = text.strip()
    
    # Generate diff HTML
    dmp = dmp_module.diff_match_patch()
    diffs = dmp.diff_main(reference, hypothesis)

    dmp.diff_cleanupSemantic(diffs)
    html_diff = dmp.diff_prettyHtml(diffs)

    # Save diff HTML
    diff_output_path = os.path.join(run_folder, 'diff_output.html')
    with open(diff_output_path, 'w', encoding='utf-8') as f:
        f.write(html_diff)

    # Compute and save WER
    error_rate = wer(reference, hypothesis)
    wer_output_path = os.path.join(run_folder, 'wer.txt')
    with open(wer_output_path, 'w', encoding='utf-8') as f:
        f.write(f"WER: {error_rate:.4f}")

    print(f"Run{i}: WER={error_rate:.4f}, HTML diff saved to {diff_output_path}")
