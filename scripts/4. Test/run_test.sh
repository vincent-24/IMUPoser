#!/bin/bash

# Get the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

combos='global'

for combo in $combos
do 
  echo Testing combo $combo 
  python 1.\ Test\ Model.py --combo_id $combo --experiment "IMUPoserGlobalModel_llm_test" --checkpoint "$(cd "$PROJECT_ROOT" && python -c 'import sys; sys.path.append("."); import constants; print(constants.TEST_CHECKPOINT)')" 
done
