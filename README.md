# micromet-net
A tool for analyzing Solar Orbiter / RPW electrical waveforms and classifing them as dust or no dust.

# The overview:
1. Load .cdf file from SoAr using sunpy.soar
2. Extract triggered signals and read meta data
3. Read mode info (3xmonopole, XLD1 mode, XLD1 highreq)
4. Preprocess the data according to mode
5. Calculate convolution of antenna signals with wavelet
6. Use trained CNN to predict dust/no dust
7. Outpout the CNN prediction to a file

# User manual:
1. Edit the paths and the days of interest in const.py to your liking.
2. Use Snakemake to run all the steps that are needed:
	1. Download the .csv data for further processing. May be ommitted if the data is present already. Call `snakemake -j 1 -R download`.
	2. Analyze the .csv data and produce the classification files. Call `snakemake -j 8 -R analyze`. The `8` stand for the number of parallel tasks. There is no point in making this number bigger than the core count of your system and perhaps you may want to use even fewer.
