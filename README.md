# micromet-net
A tool for analyzing Solar Orbiter / RPW electrical waveforms and classifing them as dust or no dust.

1 - Load .cdf file from rpw.lesia.... 
2 - Extract triggered signals and read meta data
3 - Read mode info (3xmonopole, XLD1 mode, XLD1 highreq)
4 - Preprocess the data according to mode
5 - Calculate convolution of antenna signals with wavelet 
6 - Use trained CNN to predict dust/no dust
7 - Outpout the CNN prediction
