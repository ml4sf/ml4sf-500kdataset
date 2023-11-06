# A dataset constisted of 469 784 molecules with up to 28 heavy atoms

- The ability of a support vector machine trained on this 
  dataset to screen for singlet fisson chromophores can be
  tested on this web-application: https://singletfission.chem.uni-sofia.bg/ 

## Repository contents:
- `datasets` directory contains the datset in a cimpressed form. Decompression
can be done with the following commands: 
```
sudo apt install lrzip
lrunzip Dataset470K.csv.lrz
```
 - `svm` and `dtree` directories provide an example of using the dataset to build
a model for prescreening potential singlet-fission chromophores based on their diradical 
character
- `cluster-analysis` directory contains results of k-means cluster analysis of the compounds
whose diradical character is equal or greater than 0.13 (the diradical character of antracene,
estimated by the computational protocol used to compute the diradicl character of the compounds 
in the dataset)

## Citation:
L. Borislavov, M. Nedyalkova, A. Tadjer, O. Aydemir, J. Romanova *J. Phys. Chem. Lett.*, 
**2023**, *4*, 10103–10112, <https://doi.org/10.1021/acs.jpclett.3c02365>
