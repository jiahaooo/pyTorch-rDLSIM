# An unofficial implementation of rDL-SIM

Note: This repository is an unofficial reproduction of rDL-SIM (Qiao et al., Nat. Biotechnol. 2023) and may contain errors or incomplete details. If you use this codebase in your research, please review the code carefully, verify its correctness for your use case, and make corrections if necessary. 

### Running This Code

- Install of dependencies required following [requirements](requirements.txt)

- See [SIR configuration](SIR_core/ReadMe.txt) to configure binary reconstruction package (for python 3.8 and 3.10).

- Run Demo/Step1 to Demo/Step5

### json path

This code is written and tested in PyCharm, Windows and Linux. If the error ".mrc files cannot be found" occurs when running the code in other manners, it indicates that the relative path of the .jsons file is incorrect. Users can modify the json files in the SIR_options and SSR_options by removing the "..\\" to resolve this issue.

## Citation
Please cite rDL-SIM (Qiao et al., Nat. Biotechnol. 2023) by following the original [publication](https://www.nature.com/articles/s41587-022-01471-3)
