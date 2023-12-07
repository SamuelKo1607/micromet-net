from paths import cdf_tswf_e_location
from paths import cnn_flags_location

# Define input and output directories for the classification
classification_datadir = cdf_tswf_e_location
classification_flagsdir = cnn_flags_location

# Find all the inputs
tswf_e_data, = glob_wildcards(cdf_tswf_e_location+"{sample}.cdf")

# Define the final rule that requests all the outputs
rule all:
    input:
        expand(cnn_flags_location+"{output}.csv", output=tswf_e_data)
    conda:
        "environment.yml"
    shell:
        """
        python aggregate.py
        """

# Define the rule to do the classification job of each of the files
rule mamp_generate_stats:
    input:
        file = cdf_tswf_e_location+"{sample}.cdf"
    output:
        file = cnn_flags_location+"{sample}.csv"
    conda:
        "environment.yml"
    shell:
        """
        python main.py "{input.file}" "{output.file}"
        """
