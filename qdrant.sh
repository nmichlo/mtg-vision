#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

touch "${SCRIPT_DIR}/data/qdrant/custom_config.yaml"

docker run -p 6333:6333 \
    -v "${SCRIPT_DIR}/data/qdrant/data:/qdrant/storage" \
    -v "${SCRIPT_DIR}/data/qdrant/snapshots:/qdrant/snapshots" \
    -v "${SCRIPT_DIR}/data/qdrant/custom_config.yaml:/qdrant/config/production.yaml" \
    qdrant/qdrant
