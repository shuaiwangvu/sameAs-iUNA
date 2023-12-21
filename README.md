# iUNA

This is the main repository of code for our paper
**Refining Large Integrated Identity Graphs using the Unique Name Assumption**.

--

This repository consists of several folders. If you are looking for the final simplifed code, please go straight to the directory **`/algorithm`**

The paper is limited to the scope of relations specified in the folder `/sources`. These are details of the label-like relations and comment-like relations that are mentioned in the paper. The files contain not only the relations but also the number of triples. We select only those that are popular. 

--
Other folder consists of Python scripts that are related to different sections of the paper regarding data extraction, validation, testing, plotting, analysis, etc. 

- `/extractor_scripts` are all the scripts we used for the extraction of information from the LOD-a-lot and the raw files of LOD Laundromat.
- `/script_for_data` the folder consists of two files for the generation of dta. 
- `/script_for_analysis` consists of scripts that were used for general analysis. 
- `/plot_scripts` also consists of scripts for analysis but are mostly to generate images for the paper
- `/images` the generated images are stored here. 
- `/script_for_metalink` are the files corresponding to the metalink. 
- `validating_scripts` are the Python scripts used for the validation of the UNAs. 

Some more scripts used for testing are included in `/testing_scripts`. The log files can be outdated. Please refer to the following repository for data. 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7765113.svg)](https://doi.org/10.5281/zenodo.7765113)

Please report errors to Shuai Wang at shuai.wang@vu.nl. Thank you!

> Written with [StackEdit](https://stackedit.io/).
