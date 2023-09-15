#!/bin/bash

# ensure the paths work
sed -i 's/addplot graphics.\{0,300\}] {\.\.\/\.\.\/Plots\//addplot graphics$1] {/' *temp_state_10000.pgf
sed -i 's/addplot graphics.\{0,300\}] {/&\.\.\/\.\.\/Plots\//' *temp_state_10000.pgf

# remove the legends
sed -i '/\\addlegend/d' Energies_maxwell_distribution_MC_step_fast_{0..3}.pgf
