The code for this paper (https://arxiv.org/pdf/2301.13334) consists of four Python scripts as well as a supporting module ``estimators.py``, which contains the estimators studied in our experiments. The scripts ``uc_salary.py`` and ``lognormal_laplace.py`` are used to produce the first pair of experiments in Figures 1-2 and Table 1. The scripts ``kv_vs_unbiased.py`` and ``kv_vs_unbiased_real.py`` are used to produce the second set of experiments in Figure 3. Each script either generates synthetic data or loads a user-specified CSV dataset and produces plots visualizing the behaviour of one or more estimators on the data.

When running any of the following scripts a second time, use the "load results" field instead of "save results" to immediately load and display results without recomputing them.

All scripts display the results in a viewing window and also produce a PDF whose name is user-specified in the "load results"/"save results" field.

To modify experimental paramaters, look for the following marker in the code:\
``# === EXPERIMENT PARAMS ===``

The workflow for reproducing our results is as follows.

**Figures 1-2 and Table 1**

\> python lognormal_laplace.py
\> load results:  
\> save results: lognormal_laplace

\> python uc_salary.py
\> salary dataset path: uc_salaries_2011.csv
\> load results:
\> save results: uc_salary

**Figure 3**

\> python kv_vs_unbiased.py
\> load results:
\> save results: kv_vs_unbiased

\> python kv_vs_unbiased_real.py
\> height dataset path: weights_heights.csv
\> load results:
\> save results: kv_vs_unbiased_real
