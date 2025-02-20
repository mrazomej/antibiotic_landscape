#!/bin/bash

# Create the destination directory if it doesn't exist
mkdir -p ./fig

# Copy the main and supplementary folders from the source to destination
cp -r /Users/mrazo/git/antibiotic_landscape/fig/main ./fig/
cp -r /Users/mrazo/git/antibiotic_landscape/fig/supplementary ./fig/

# Print success message
echo "Successfully copied fig/main and fig/supplementary to ./fig/"