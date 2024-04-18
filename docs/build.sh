#!/bin/bash

CONFIG_PATH="docs/_config.yml"

# echo "Google Analytics ID: $GA_TRACKING_ID"
# echo "Current Directory: $(pwd)"

# echo "Before replacement:"
# grep "google_analytics_id" $CONFIG_PATH

# Replace the placeholder in the YAML file with the actual Google Analytics ID
sed -i "s/{GA_TRACKING_ID_PLACEHOLDER}/$GA_TRACKING_ID/" $CONFIG_PATH

# echo "After replacement:"
# grep "google_analytics_id" $CONFIG_PATH

# Build
jupyter-book build docs/ --all