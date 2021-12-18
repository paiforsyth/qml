VENV=../ide_venv # point this to any location at which to store the virudal environment.  Should be outside the main repo
python -m venv "${VENV}"
"${VENV}/bin/pip" install -r <(./pants dependencies :: |
  xargs ./pants filter --target-type=python_requirement |
  xargs ./pants peek |
  jq -r '.[]["requirements"][]')
