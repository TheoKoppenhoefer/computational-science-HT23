#!/bin/bash

# Script for modifying math in the dia-diagrams.
sed -i 's/\\ensuremath{\\backslash}/\\/g' *.tex
sed -i 's/\\\$/\$/g' *.tex
sed -i 's/\\\_/\_/g' *.tex
sed -i 's/\\{/{/g' *.tex
sed -i 's/\\}/}/g' *.tex
sed -i 's/||\(.*\)||/\$\\norm\{\1\}\$/g' *.tex
