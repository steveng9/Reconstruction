import pandas as pd
import numpy as np

DATA_SIZE = 10_000
train_file = "./....csv"
cols =
epsilon = 1


################################________________________________________________________
################# SMARTNOISE ___________________________________________________________
#################   - MST or AIM________________________________________________________
################################________________________________________________________
################################________________________________________________________
from snsynth import Synthesizer


deid_name = "mst"
# deid_name = "aim"

train = pd.read_csv(train_file)
train.set_index("ID", inplace=True)

synthesizer = Synthesizer.create(deid_name, epsilon=epsilon)

# Train the MST synthesizer on the first dataset
synthesizer.fit(
    train,
    preprocessor_eps=epsilon*0.3,
    categorical_columns=cols["categorical"],
    continuous_columns=cols["continuous"],
    ordinal_columns=cols["ordinal"],
)

# Create Sample
sample = synthesizer.sample(DATA_SIZE)
# Set their data types to int64
for col in sample.columns:
    sample[col] = (
        sample[col].astype(np.int64)
        if sample[col].dtype == np.float64
        else sample[col]
    )

sample.to_csv("....<name> .csv", index=False)



################################________________________________________________________
################# SYNTHCITY     ________________________________________________________
#################   - ARF       ________________________________________________________
################################________________________________________________________
################################________________________________________________________


from synthcity.plugins import Plugins
from synthcity.plugins.core.constraints import Constraints







################################________________________________________________________
################# SDV           ________________________________________________________
#################   - TVAE      ________________________________________________________
################################________________________________________________________
################################________________________________________________________

from sdv.metadata import Metadata
from sdv.single_table import TVAESynthesizer
from sdv.lite import SingleTablePreset


train = pd.read_csv(train_file)
train.set_index("ID", inplace=True)

# Load Metadata
metadata = Metadata.detect_from_dataframe(
    data=train, table_name="tvae_gt_data_qid_1" # TODO rename table
)

for col in cols["categorical"]:
    if col in train.columns:
        metadata.update_column(
            column_name=col,
            sdtype="categorical",
        )

metadata.update_column(
    column_name="EDUC",
    sdtype="categorical",
)

synthesizer = TVAESynthesizer(metadata)
synthesizer.fit(train)

# Sample data from the synthesizer
sample = synthesizer.sample(
    num_rows=DATA_SIZE
)






################################________________________________________________________
################# SDCMicro      ________________________________________________________
#################   - CellSuppr.________________________________________________________
#################   - RankSwap  ________________________________________________________
################################________________________________________________________
################################________________________________________________________





################################________________________________________________________
################# R             ________________________________________________________
#################   - Synthpop  ________________________________________________________
################################________________________________________________________
################################________________________________________________________



