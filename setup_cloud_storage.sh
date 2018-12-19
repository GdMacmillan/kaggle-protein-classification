#!/bin/bash

# move to Dockerfile
pip install --upgrade google-cloud-storage

# move to docker_entrypoint.sh (will require an image rebuild)
echo $GOOGLE_APPLICATION_CREDENTIALS_JSON_FILE > /workspace/gcs_credentials.json
