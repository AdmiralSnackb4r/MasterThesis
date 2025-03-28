#!/usr/bin/env bash
echo "Script executed from: ${PWD}"
cd ./box_intersections_cpu && python setup.py build_ext
