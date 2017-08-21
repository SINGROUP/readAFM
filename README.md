readAFM Readme
==============

The Project
-----------

The goal of this project was explore the possibilities of using Machine Learning to analyze data obtained from Non Contact Atomic Force Microscopy (NC-AFM).

The title of this git-repository, readAFM, is intentional similar to [MechAFM](https://github.com/SINGROUP/MechAFM), while MechAFM simulates an AFM image with a mechanical model of the tip atoms, readAFM is a first attempt to understand the output "images" of an AFM, in a sense "read" them.



How to run
----------

To run minimal_AFM.py, first install the following packages: h5py, numpy, tensorflow. Then copy src/ folder.

```
cp -a src scratch/minimal_testrun
```

Put the correct path to the hdf5 database in the parameters.in file. Run minimal_AFM.py

```
python minimal_AFM.py <NameOfThisRun> -i <name of parameter file>
```
e.g.:

```
python minimalAFM.py myRun -i parameters.in
```

The simulation will run now. Various files will be created, all including the filename:

* out\_minimal_myRun.log contains logging information, most importantly the time/step and the current accuracy. Try
``` cat out\_minimal_myRun.log | grep accuracy ```
* savemyRun/ is a folder containing the *.ckpt file to restore the variables and the files for Tensorboard
* viewfile_myRun.hdf5 contains the validation input and output of the last step. It also contains parameters dictionary of the run.

  
Contents of this Repo
---------------------

This report contains a brief description of the most important steps in the project, especially

* src/
..* readAFMHDF5.py: Responsible for everything database related. Main-class is AFMdata, that is initialized by AFMdata('path/to/db.hdf5'[, shape=<shape of db>]). Calculates labels ('solutions') for the xyz files, can return batches for training and validation, can be used to add labels to a db.hdf5-file.
..* minimal_AFM.py: Main script. Parses the inputparameters and calls all the other modules, makes it easier to exchange single modules.
..* model.py: Contains the function create_model(), that creates a the CNN.
..* train_model.py: Contains a function that trains the model.
..* eval_model.py: Restores and evluates the model.
..* utils.py: Contains auxiliary functions.
* databaseCode/
* docs/
* rnnCode_wontWork/



Architecture of the project
---------------------------





So far this only contains the minimal_AFM.py for the first version of the CNN and the readAFMDATAfile.py for the class containing and delivering the data.

