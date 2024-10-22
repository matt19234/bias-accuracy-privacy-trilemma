The code for this paper (https://arxiv.org/pdf/2301.13334) consists of four Python scripts as well as a supporting module ``estimators.py``, which contains the estimators studied in our experiments. The scripts ``uc_salary.py`` and ``lognormal_laplace.py`` are used to produce the first pair of experiments in Figures 1-2 and Table 1. The scripts ``kv_vs_unbiased.py`` and ``kv_vs_unbiased_real.py`` are used to produce the second set of experiments in Figure 3. Each script either generates synthetic data or loads a user-specified CSV dataset and produces plots visualizing the behaviour of one or more estimators on the data.

When running any of the following scripts a second time, use the "load results" field instead of "save results" to immediately load and display results without recomputing them.

All scripts display the results in a viewing window and also produce a PDF whose name is user-specified in the "load results"/"save results" field.

To modify experimental paramaters, look for the following marker in the code:\
``# === EXPERIMENT PARAMS ===``

Dependencies including version numbers are listed in ``requirements.txt``. They can be installed using\
``pip install -r requirements.txt``\
or alternatively by running\
``pip install numpy scipy tqdm matplotlib pandas``\
which should yield compatible versions of the dependencies (our code is not reliant on version-specific features).

The two datasets we use are included in this repository.
- ``data/uc_salaries_2011.csv`` was downloaded from https://transparentcalifornia.com/salaries/2011/university-of-california/ (free account required).
- ``data/weights_heights.csv`` was downloaded from http://wiki.stat.ucla.edu/socr/index.php/SOCR_Data_Dinov_020108_HeightsWeights#Complete_Data. This link appears to be broken now. A backup can be found at https://wiki.socr.umich.edu/index.php/SOCR_Data_Dinov_020108_HeightsWeights.

After setting up the environment, the workflow for reproducing our results is as follows.

**Figures 1-2 and Table 1**

```
> python lognormal_laplace.py
> load results:  
> save results: results/lognormal_laplace
```

Figure 1 is shown on screen and can now be found at ``results/lognormal_laplace.pdf``.

```
> python uc_salary.py
> salary dataset path: data/uc_salaries_2011.csv
> load results:
> save results: results/uc_salary
```

Figure 2 is shown on screen and can now be found at ``results/uc_salary.pdf``. The data for Table 1 are located in the script text output.

**Figure 3**

```
> python kv_vs_unbiased.py
> load results:
> save results: results/kv_vs_unbiased
```

Figure 3-1 is shown on screen and can now be found at ``results/kv_vs_unbiased.pdf``.

```
> python kv_vs_unbiased_real.py
> height dataset path: data/weights_heights.csv
> load results:
> save results: results/kv_vs_unbiased_real
```

Figure 3-2 is shown on screen and can now be found at ``results/kv_vs_unbiased_real.pdf``.
