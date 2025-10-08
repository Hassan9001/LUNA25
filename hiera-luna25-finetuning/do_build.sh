#!/usr/bin/env bash

# Portions of this code are adapted from luna25-baseline-public
# Source: https://github.com/DIAGNijmegen/luna25-baseline-public/blob/main/do_build.sh
# License: Apache License 2.0 (see https://github.com/DIAGNijmegen/luna25-baseline-public/blob/main/LICENSE)

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="finetune-hiera-3d-latest"


# Check if an argument is provided
if [ "$#" -eq 1 ]; then
    DOCKER_IMAGE_TAG="$1"
fi

# Note: the build-arg is JUST for the workshop
docker build "$SCRIPT_DIR" \
  --platform=linux/amd64 \
  --tag "$DOCKER_IMAGE_TAG" 2>&1