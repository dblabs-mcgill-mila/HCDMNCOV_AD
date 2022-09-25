# Miami Plots of brain-phenotypic associations

We are here provinding the individual numerical values that underlie the summary data displayed in main Figs. 1-3 and Sup. Figs. 1-10.

* The p-values (in negative decimal logarithms) for each brain-phenotypic link that passed the FDR threshold in either males or females in the phenome-wide profiling of a given mode of HC-DN co-variation can be found in [hits](https://github.com/dblabs-mcgill-mila/HCDMNCOV_AD/tree/master/Miami_Plots/hits)
* The accompanying code that generated these p-values is shown in this [ipython notebook](https://github.com/dblabs-mcgill-mila/HCDMNCOV_AD/blob/master/Miami_Plots/hits/Miami_Plot_original_analysis_2022.ipynb)
* The absolute difference in association strength between males and females at a formal 5-95% population confidence interval for a given mode is shown in [bootstrap_analysis/plots](https://github.com/dblabs-mcgill-mila/HCDMNCOV_AD/tree/master/Miami_Plots/bootstrap_analysis/plots)
* Other information on the bootstrap distributions (e.g., mean, median, 25th and 75th percentiles) that served to generate Sup. Figs. 8-10 can be found in [bootstrap_analysis/tables](https://github.com/dblabs-mcgill-mila/HCDMNCOV_AD/tree/master/Miami_Plots/bootstrap_analysis/tables)
* The python code generating the bootstrap distributions is splitted in a [part 1](https://github.com/dblabs-mcgill-mila/HCDMNCOV_AD/blob/master/Miami_Plots/bootstrap_analysis/Miami_bootstrap_2022_updated_compute_canada.py) and a [part 2](https://github.com/dblabs-mcgill-mila/HCDMNCOV_AD/blob/master/Miami_Plots/bootstrap_analysis/Manhattan_Plot_bootstrap_part_2_08.27.2022.ipynb)
* [Part 1](https://github.com/dblabs-mcgill-mila/HCDMNCOV_AD/blob/master/Miami_Plots/bootstrap_analysis/Miami_bootstrap_2022_updated_compute_canada.py) serve to generate the 1,000 bootstrap iterations for each HC-DN co-variation mode 
* [Part 2](https://github.com/dblabs-mcgill-mila/HCDMNCOV_AD/blob/master/Miami_Plots/bootstrap_analysis/Manhattan_Plot_bootstrap_part_2_08.27.2022.ipynb) contains the code that shows the way in which the plotted mean and errors were derived for Sup. Figs. 8-10. 
