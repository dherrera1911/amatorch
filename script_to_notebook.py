# This script takes as input a .py file and outputs a formatted python notebook
from nbformat import v3, v4


def uncomment_flagged_blocks(text):
    """ This function takes a text file that was read into
    python, and removes the comments of flagged, commented code
    blocks. Function written by ChatGPT"""
    uncomment_start = "##UNCOMMENT_FOR_COLAB_START##"
    uncomment_end = "##UNCOMMENT_FOR_COLAB_END##"
    lines = text.split('\n')
    in_uncomment_block = False
    new_lines = []
    for line in lines:
        if line.strip() == uncomment_start:
            in_uncomment_block = True
        elif line.strip() == uncomment_end:
            in_uncomment_block = False
        elif in_uncomment_block:
            new_lines.append(line.lstrip('#'))
        else:
            new_lines.append(line)
    return '\n'.join(new_lines)


fileName = [
        "ama_basic_tutorial",
        "ama_batch_size_analysis_dsp",
        "ama_sequential_training_dsp",
        "ama_manifold_geometry_analysis",
        "ama_filter_learning_robustness",
        "normalized_cov_approximation",
        ]

for f in range(len(fileName)):
    inputFile = './scripts/tutorials/' + fileName[f] + '.py'
    outputFile = './notebooks/' + fileName[f] + '.ipynb'
    # Open the file
    fpin = open(inputFile)
    text = fpin.read()
    # Uncomment block text for COLAB
    text = uncomment_flagged_blocks(text)
    # Convert to notebook
    nbook = v3.reads_py(text)
    nbook = v4.upgrade(nbook)  # Upgrade v3 to v4
    jsonform = v4.writes(nbook) + "\n"
    # Write notebook
    fpout = open(outputFile, "w")
    fpout.write(jsonform)
    fpout.close()


