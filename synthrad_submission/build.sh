#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
REPO_ROOT="$( cd "$SCRIPTPATH/.." ; pwd -P )"

docker build -t synthrad_algorithm -f "$SCRIPTPATH/Dockerfile" "$REPO_ROOT"

# on windows run
# docker build -t synthrad_algorithm -f C:\Users\nicol\Coding\UZH\fullbody-sCT\algorithm_template\Dockerfile C:\Users\nicol\Coding\UZH\fullbody-sCT
