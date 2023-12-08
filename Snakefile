from const import CDF_TSWF_E_LOCATION
from const import CNN_FLAGS_LOCATION
from const import YYYYMMDD_MIN
from const import YYYYMMDD_MAX
from download import main as down_main

# Define input and output directories for the classification
classification_datadir = CDF_TSWF_E_LOCATION
classification_flagsdir = CNN_FLAGS_LOCATION

# Find all the inputs
tswf_e_data, = glob_wildcards(CDF_TSWF_E_LOCATION+"{sample}.cdf")

rule default:
	run:
		pass
	#	shell("echo rule all")

# Define the final rule that requests all the outputs
rule analyze:
    input: expand(CNN_FLAGS_LOCATION+"{output}.csv", output=tswf_e_data)
	run:
		shell("echo analyze done")

# Define the rule to do the classification job of each of the files
rule cnn_classify:
    input: file = CDF_TSWF_E_LOCATION+"{sample}.cdf"
    output:
        file = CNN_FLAGS_LOCATION+"{sample}.csv"
    conda:
        "environment.yml"
    shell:
        """
        python main.py "{input.file}" "{output.file}"
        """

rule download:
	run:
		down_main(YYYYMMDD_MIN,YYYYMMDD_MAX)