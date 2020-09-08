#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
REPO_ROOT="$( cd "$DIR/../../../.." >/dev/null 2>&1 && pwd )"

TEST_DATA="$1"

if [[ -z "$TEST_DATA" ]]; then
  echo "usage $0 <PATH_TO_TEST_DATA>" >&2
  echo "You can download real test data from gs://sgkit-data/validation/hapmap_JPT_CHB_r23a_filtered.zip" >&2
  echo "sgkit-data GCS bucket uses Requester Pays config: https://cloud.google.com/storage/docs/requester-pays" >&2
  exit 1
fi

if [[ -z "$RUNNING_IN_SGKIT_PC_RELATE_VALIDATION_DOCKER" ]]; then
  # Note: to speed up the process up, we could:
  #       * build the docker image and push it to the GHCR/GCR
  #       * or add docker layer caching https://github.com/marketplace/actions/docker-layer-caching
  #
  #       For now, we just build new docker image each time, if we want to run
  #       validation more often than weekly or just want to get results
  #       faster we could add any of the above in the future.
  echo "Building validation docker image, this will take about ~20 minutes ..."
  docker build -t sgkit_pc_relate_validation -f "$DIR/Dockerfile" "$DIR"
  docker run --rm \
  -v $DIR:/work \
  -v $REPO_ROOT:/code \
  -v $TEST_DATA:/test_data.zip \
  -e RUNNING_IN_SGKIT_PC_RELATE_VALIDATION_DOCKER=1 \
  sgkit_pc_relate_validation /work/run.sh "$1"
else
  echo "Running inside docker, will crunch data ..."
  unzip test_data.zip
  /work/convert_plink_to_gds.R hapmap_JPT_CHB_r23a_filtered hapmap_JPT_CHB_r23a_filtered.gds
  /work/pc_relate.R hapmap_JPT_CHB_r23a_filtered.gds
  cp /work/validate_pc_relate.py .
  pip3 install -r /code/requirements.txt
  PYTHONPATH=/code:$PYTHONPATH pytest ./validate_pc_relate.py
fi
