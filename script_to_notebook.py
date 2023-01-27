# This script takes as input a .py file and outputs a formatted python notebook
from IPython.nbformat import v3, v4

#inputFile = "./scripts/ama_batch_size_analysis_dsp.py"
#outputFile = "./notebooks/ama_batch_size_analysis_dsp.ipynb"
#inputFile = "./scripts/ama_sequential_training_dsp.py"
#outputFile = "./notebooks/ama_sequential_training.ipynb"
inputFile = "./scripts/ama_manifold_geometry_analysis.py"
outputFile = "./notebooks/ama_manifold_geometry_analysis.ipynb"

fpin = open(inputFile)
text = fpin.read()

nbook = v3.reads_py(text)
nbook = v4.upgrade(nbook)  # Upgrade v3 to v4

jsonform = v4.writes(nbook) + "\n"

fpout = open(outputFile, "w")
fpout.write(jsonform)

