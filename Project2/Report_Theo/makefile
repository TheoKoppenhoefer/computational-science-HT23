# The following make file was inspired by https://scaron.info/blog/makefiles-for-latex.html

ALLTEX=$(wildcard *.tex ../Plots/*.pgf)
ALLCACHE=$(wildcard cache/* cache)
ALLPYTH=$(wildcard ../*.py)
MAIN=Report_TheoKoppenhoefer.tex
LATEX=rubber --unsafe
SHELL=/bin/bash

all: $(ALLTEX) $(ALLCACHE) ## Build full thesis (LaTeX + figures)
	rubber --pdf --unsafe $(MAIN)				# main run
	biber $(MAIN:.tex=)			 # bibliography
	$(LATEX) $(MAIN)				# incremental run
	evince $(MAIN:.tex=).pdf               # show result
   
../Figures/FlowchartTranscriptionFactors.tex : ../Figures/FlowchartTranscriptionFactors.dia
	dia --export=FlowchartTranscriptionFactors.tex FlowchartTranscriptionFactors.dia
	bash ../Figures/modify_diagrams.sh

clean:  ## Clean LaTeX and output figure files
	rubber --clean $(MAIN)
	rm cache/*

watch:  ## Recompile on any update of LaTeX or SVG sources
	@while true; do		  \
		inotifywait -e modify,delete,delete_self $(MAIN) $(ALLCACHE);	\
		sleep 0.01;			\
		make all; echo "\n----------\n"; sleep 0.1; done

allplots: $(ALLPYTH) ## create the plots
	cd ..; /usr/bin/env /opt/anaconda3/envs/numbaEnvironment/bin/python main.py;
	cd ../Plots; ./modify_plots.sh;
	make clean;
