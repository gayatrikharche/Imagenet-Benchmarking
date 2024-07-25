#!/bin/bash
echo "Hello OSPool from Job $1 running on `hostname`"

# Create a data directory
mkdir -p data

# Combine the parts into the original zip file
cat chunks/ILSVRC.zip.part* > ILSVRC.zip

# Unzip the combined file
unzip -q ILSVRC.zip -d data

# Run the PyTorch model
python main.py --save-model --epochs 20

# Clean up
rm -r data
rm ILSVRC.zip
rm chunks/ILSVRC.zip.part*
