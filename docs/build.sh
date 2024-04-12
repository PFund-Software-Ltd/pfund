#!/bin/bash

# Replace the placeholder in the YAML file with the actual Google Analytics ID
sed -i "s/{GA_TRACKING_ID_PLACEHOLDER}/$GA_TRACKING_ID/" _config.yml

# Build
jupyter-book build docs/ --all