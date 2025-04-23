import sys

import pandas as pd
from os import listdir
from os.path import isfile, join

def main():
    mypath = "/Users/golobs/Documents/GradSchool/NIST-CRC-25/NIST_Red-Team_Problems1-24_v2/"
    mypath = "/Users/golobs/Documents/GradSchool/NIST-CRC-25/25_PracticeProblem/"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith("Deid.csv")]
    for f in onlyfiles:
        df = pd.read_csv(mypath + f)
        print(f, df.shape)
        # print()





def calculate_reconstruction_score(df_original, df_reconstructed):
    print('\n\n\n\n')
    total_records = len(df_original)

    for col in df_original.columns:
        value_counts = df_original[col].value_counts()
        rarity_scores = df_original[col].map(total_records / value_counts)
        max_score = rarity_scores.sum()

        score = ( (df_original[col].values == df_reconstructed[col].values) * rarity_scores ).sum()
        print(col, score / max_score * 100)


main()