#!/bin/bash

# ensure the paths work
# sed -i 's/addplot graphics.\{0,300\}] {\.\.\/\.\.\/Plots\//addplot graphics$1] {/' *temp_state_10000.pgf
# sed -i 's/addplot graphics.\{0,300\}] {/&\.\.\/\.\.\/Plots\//' *temp_state_10000.pgf

# remove the legends
# sed -i '/\\addlegend/d' Energies_maxwell_distribution_MC_step_fast_*.pgf

# general style
sed -i 's/\\begin{axis}\[/\\begin{axis}\[\n ytick={0,0.2,0.4,0.6,0.8,1},\n x tick label style = {rotate=70},\n y post scale=3, \n transpose legend,/' {N,O,T}_*.pgf
sed -i 's/legend style={fill opacity=0.8, draw opacity=1, text opacity=1, draw=lightgray204},/legend style={fill opacity=0.8, draw opacity=1, text opacity=1, draw=lightgray204, anchor=south east, at={(0.97,0.03)}},/' {N,O,T}_*.pgf
sed -i 's/at={(0.97,0.03)}/at={(axis cs:5,1.1)}/' {N,O,T}_*.pgf
perl -i -pe 's/anchor=south east/anchor=south west,
    legend columns=4,
    \/tikz\/every even column\/.append style={column sep=1.0cm},/' {N,O,T}_*.pgf

# align the plots
sed -i 's/\\begin{tikzpicture}/\\begin{tikzpicture}\[baseline\]/' {N,O,T}_*.pgf

# add legend entries
perl -i -pe 's/\\addlegendentry{Nanog overexpressed}/\\addlegendentry{Nanog overexpressed}
\\addlegendimage{area legend, draw=green, fill=green, opacity=0.2}
\\addlegendentry{Oct4 overexpressed}

\\addlegendimage{area legend, draw=orange, fill=orange, opacity=0.2}
\\addlegendentry{Tet1 overexpressed}

\\addlegendimage{area legend, draw=gray, fill=gray, opacity=0.2}
\\addlegendentry{LIF active}/' N_*.pgf

# remove axis label for Oct4 and Tet1
sed -i '/ylabel={Expression level},/d' {O,T}_*.pgf
sed -i '/xlabel={Time},/d' {O,T}_*.pgf

# remove legends for Oct4 and Tet1
sed -i '/\\addlegend/d' {O,T}_*.pgf

