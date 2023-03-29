# Cloud Feedbacks in Radiative-Convective Equilibrium

This repository contains the data and scripts used in a paper submitted to JAMES.

## DATA

| Data | Description |
|:-----|:------------|
|CFB295300_domainavg_allmodels_meancrk_taulim00.json|decomposed cloud feedbacks for 295 K to 300 K for all tau bins|
|CFB295300_domainavg_allmodels_meancrk_taulim03.json|decomposed cloud feedbacks for 295 K to 300 K for all but the thinnest tau bins|
|CFB295305_domainavg_allmodels_meancrk_taulim00.json|decomposed cloud feedbacks for 300 K to 305 K for all tau bins|
|CFB295305_domainavg_allmodels_meancrk_taulim03.json|decomposed cloud feedbacks for 300 K to 305 K for all but the thinnest tau bins|
|CFB300305_domainavg_allmodels_meancrk_taulim00.json|decomposed cloud feedbacks for 300 K to 305 K for all tau bins|
|CFB300305_domainavg_allmodels_meancrk_taulim03.json|decomposed cloud feedbacks for 300 K to 305 K for all but the thinnest tau bins|
|cloudradiativekernels_RCE_small.tar.gz|cloud radiative kernels
|isccphistograms_RCE_small.tar.gz|ISCCP histograms
|rrtmg_lw_rcemip.nc|atmospheric composition data used in RRTMG, from Wing et al. (2018)|

## SCRIPTS

| Data | Description |
|:-----|:------------|
|cloud_kernel.py|create a cloud radiative kernel, requires rrtmg_lw_rcemip.nc|
|isccp_hist.py|calculate histogram from ISCCP simulator output|
|tau_ctp_tbr.py|offline ISCCP simulator|
