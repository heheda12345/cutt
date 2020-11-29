#!/bin/bash

case $(hostname -s) in
  nico*)
    echo "nico cluster"
    source /opt/spack/share/spack/setup-env.sh
    spack load cuda@10.2.89 /v5oqq5n
    ;;
esac
