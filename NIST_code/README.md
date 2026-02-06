# CRC Data Generation Notebooks

If you have any questions about the scripts, synthetic data generators, or other questions about the CRC Red Team datasets, please send me an email at [dj.streat@knexus.ai](mailto:dj.streat@knexus.ai)

## Summary
The files in this directory are the scripts and notebooks used to generate the datasets for the 2025 NIST CRC Red Team Challenge.

The main file you should check out is [generate_deid_data_from_strata_COMPO.ipynb](generate_deid_data_from_strata_COMPO.ipynb), as most of the synthetic data generation is done there, in Python. Some of the algorithms use libraries from R, and in those cases there is a corresponding R markdown notebook.

Most of the code follows a pretty straightforward data processing pattern: Load Data -> Preprocess -> Run through synthesizer -> Output dataset.
 
Unnecessary comments are from most places, aside from ones that I feel make sense in context (Loading one set of files vs. another set of files for synthesis.)

## Algorithms
- SYNTHPOP -- [arizona_Rsynthpop.Rmd](arizona_Rsynthpop.Rmd)
- SMARTNOISE AIM -- [generate_deid_data_from_strata_COMPO.ipynb](generate_deid_data_from_strata_COMPO.ipynb)
- SMARTNOSIE MST -- [generate_deid_data_from_strata_COMPO.ipynb](generate_deid_data_from_strata_COMPO.ipynb)
- SDCMICRO CELL SUPRESSION -- [arizona_sdcmicro_cell_suppression.Rmd](arizona_sdcmicro_cell_suppression.Rmd)
- SYNTHCITY ARF -- [crc_synthcity.py](crc_synthcity.py)
- SDV TVAE -- [generate_deid_data_from_strata_COMPO.ipynb](generate_deid_data_from_strata_COMPO.ipynb)
- RANKSWAP -- [arizona_sdcmicro_rankswap.Rmd](arizona_sdcmicro_rankswap.Rmd)

## Data Feature Sets

The "Match 3" dataset has a total of 98 features.

Quasi-Identifiers: (always "SEX" and "RACE")
- (1) "AGEMARR", "GQTYPE", "IND", "MTONGUE", "VETSTAT"
- (2) "BPL", "FARM", "HISPAN", "LABFORCE", "MIGRATE5"
### Default Feature Set (50 Features)

| NAME     | QID? | DESCRIPTION                                                                                     |
| -------- | ---- | ----------------------------------------------------------------------------------------------- |
| AGE      |      | reports the person's age                                                                        |
| AGEMARR  | 1    | reports the respondent's age at first marriage.                                                 |
| BPL      | 2    | indicates where the person was born (state, territory or country)                               |
| CHBORN   |      | number of children ever born to each woman                                                      |
| CITIZEN  |      | reports the citizenship status of respondents                                                   |
| CITY     |      | City of residence                                                                               |
| CLASSWKR |      | indicates whether self-employed                                                                 |
| COUNTY   |      | County                                                                                          |
| DURUNEMP |      | How many weeks has respondant been unemployed                                                   |
| EDUC     |      | highest year of school or degree completed                                                      |
| EMPSTAT  |      | indicates the respondant's employment status                                                    |
| FAMSIZE  |      | counts the number of own family members residing with each individual                           |
| FARM     | 2    | indicates farm households                                                                       |
| FBPL     |      | father's birthplace                                                                             |
| GQ       |      | indicates whether the person lived in group quarters (ex: dorm, barracks, prison, nursing home) |
| GQFUNDS  |      | the funding source for each group quarters.                                                     |
| GQTYPE   | 1    | indicates group quarters type                                                                   |
| HISPAN   | 2    | identifies persons of Hispanic/Spanish/Latino origin                                            |
| HRSWORK1 |      | total number of hours the respondent was at work during the previous week                       |
| INCNONWG |      | indicates non-wage income                                                                       |
| INCWAGE  |      | Each respondent's total pre-tax wage and salary income                                          |
| IND      | 1    | Person's industry                                                                               |
| LABFORCE | 2    | indicates whether a person participated in the labor force.                                     |
| MARRNO   |      | indicates number of marriages respondant has had                                                |
| MARST    |      | gives each person's current marital status                                                      |
| MBPL     |      | mother's birthplace                                                                             |
| METAREA  |      | identifies the metro area of residence                                                          |
| METRO    |      | indicates whether the household resided within a metropolitan area                              |
| MIGCITY5 |      | the city where the respondent resided 5 years ago                                               |
| MIGRATE5 | 2    | indicates whether a preson had migrated                                                         |
| MIGTYPE5 |      | indicates whether the respondent lived in a metropolitan area five years ago                    |
| MTONGUE  | 1    | reports first language (mother tongue)                                                          |
| NATIVITY |      | indicates whether respondents were native-born or foreign-born                                  |
| NCHLT5   |      | counts the number of own children age 4 and under residing with each individual                 |
| OCC      |      | indicate's person's occupation                                                                  |
| OWNERSHP |      | indicates whether the housing unit was rented or owned                                          |
| RACE     | X    | indicates persons race                                                                          |
| RENT     |      | How much household pays in rent                                                                 |
| SAMEPLAC |      | indicates whether individuals were living in the same community 5 years ago                     |
| SCHOOL   |      | indicates whether the respondent attended school                                                |
| SEX      | X    | reports whether the person was male or female.                                                  |
| SSENROLL |      | indicates whether the respondant has a SSN                                                      |
| URBAN    |      | indicates whether a household's location was urban or rural.                                    |
| VALUEH   |      | How much the respondant's house is worth                                                        |
| VETCHILD |      | indicates whether father was a veteran                                                          |
| VETPER   |      | reports the period of military service                                                          |
| VETSTAT  | 1    | indicates veteran status                                                                        |
| VETWWI   |      | indicates WWI veteran status                                                                    |
| WARD     |      | identifies the political ward where the household was enumerated                                |
| WKSWORK1 |      | number of weeks worked the previous year                                                        |
### Reduced Feature Set (25 Features)

| NAME     | QID? | DESCRIPTION                                                                                     |
| -------- | ---- | ----------------------------------------------------------------------------------------------- |
| AGE      |      | reports the person's age                                                                        |
| AGEMARR  | 1    | reports the respondent's age at first marriage.                                                 |
| BPL      | 2    | indicates where the person was born (state, territory or country)                               |
| CITIZEN  |      | reports the citizenship status of respondents                                                   |
| DURUNEMP |      | How many weeks has respondant been unemployed                                                   |
| EDUC     |      | highest year of school or degree completed                                                      |
| EMPSTAT  |      | indicates the respondant's employment status                                                    |
| FAMSIZE  |      | counts the number of own family members residing with each individual                           |
| FARM     | 2    | indicates farm households                                                                       |
| GQ       |      | indicates whether the person lived in group quarters (ex: dorm, barracks, prison, nursing home) |
| GQTYPE   | 1    | indicates group quarters type                                                                   |
| HISPAN   | 2    | identifies persons of Hispanic/Spanish/Latino origin                                            |
| INCWAGE  |      | Each respondent's total pre-tax wage and salary income                                          |
| IND      | 1    | Person's industry                                                                               |
| LABFORCE | 2    | indicates whether a person participated in the labor force.                                     |
| MARST    |      | gives each person's current marital status                                                      |
| MIGRATE5 | 2    | indicates whether a preson had migrated                                                         |
| MTONGUE  | 1    | reports first language (mother tongue)                                                          |
| NATIVITY |      | indicates whether respondents were native-born or foreign-born                                  |
| OWNERSHP |      | indicates whether the housing unit was rented or owned                                          |
| RACE     | X    | indicates persons race                                                                          |
| SEX      | X    | reports whether the person was male or female.                                                  |
| URBAN    |      | indicates whether a household's location was urban or rural.                                    |
| VETSTAT  | 1    | indicates veteran status                                                                        |
| WKSWORK1 |      | number of weeks worked the previous year                                                        |
