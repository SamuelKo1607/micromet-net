import os

# Paths
CDF_TSWF_E_LOCATION = os.path.join("data","rpw_tds_surv_tswf_e","")
PREPROCESSED_LOCATION = os.path.join("data","preprocessed","")
CNN_FLAGS_LOCATION = os.path.join("data","cnn_flags","")	

# Constants
YYYYMMDD_MIN = "20000101"
YYYYMMDD_MAX = "20200620"

#%%
if __name__ == "__main__":
    print("Locs:")
    print(CDF_TSWF_E_LOCATION)
    print(PREPROCESSED_LOCATION)
    print(CNN_FLAGS_LOCATION)
    print("Consts:")
    print(YYYYMMDD_MIN)
    print(YYYYMMDD_MAX)
