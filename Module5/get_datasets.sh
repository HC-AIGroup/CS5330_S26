#!/bin/bash

echo "Downloading CIFAR-10..."

URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
FILE="cifar-10-python.tar.gz"

# Download
curl -L -o $FILE $URL

# Extract
echo "Extracting..."
tar -xzvf $FILE

# Remove compressed file
echo "Cleaning up..."
rm $FILE

echo "Done! CIFAR-10 is ready."
