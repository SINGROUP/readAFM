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
` cat out_minimal_myRun.log | grep accuracy `
* savemyRun/ is a folder containing the *.ckpt file to restore the variables and the files for Tensorboard
* viewfile_myRun.hdf5 contains the validation input and output of the last step. It also contains parameters dictionary of the run.

  
Contents of this Repo
---------------------

This report contains a brief description of the most important steps in the project, especially

* src/: All the source code for the CNN.
    - readAFMHDF5.py: Responsible for everything database related. Main-class is AFMdata, that is initialized by `AFMdata('path/to/db.hdf5'[, shape=<shape of db>])`. Calculates labels ('solutions') for the xyz files, can return batches for training and validation, can be used to add labels to a db.hdf5-file.
    - minimal_AFM.py: Main script. Parses the inputparameters and calls all the other modules, makes it easier to exchange single modules.
    - model.py: Contains the function create_model(), that creates a the CNN.
    - train_model.py: Contains a function that trains the model.
    - eval_model.py: Restores and evluates the model.
    - utils.py: Contains auxiliary functions.
* databaseCode/: Mainly Dhananjays folder, see README there!
* docs/: A report about this project, contains pdf- and tex-files as well as the figures.
* rnnCode_wontWork/: Dhananjays failed attempt to do the dimension reduction with a RNN instead of a FC layer.



About the modules
-----------------

### minimal_AFM.py
Main script. Contains the default parameters, then parses the specified input file (default: parameters.in). Loads define\_model from model.py, train\_model from train\_model.py and eval\_model from eval\_model.py. Then, according to the 'train' parameter, the model is either trained or evaluated. One can easily change the model-module and load the new model from there.


### parameters.in
Use the following syntax: the name of the parameter seperated from its value by a colon `:`, use `#` to comment out lines.

``` 
parameterName		: parameterValue 
```

e.g.


```
# Example Parameters
savePath		: None
DBPath			: ../AFMDB_version_01.hdf5
DBShape			: [81,81,41,1]
outChannels		: 1
LearningRate		: 0.001
```

### module.py
Contains the function define\_model, defines a Tensorflow graph, that is later to be trained or evaluated.

### train_model.py
The trains the model according to the parameters specified in the input file.

### eval_model.py
Contains a function eval_model that only evaluates the model, without training it.

### readAFMHDF5.py
Contains all the database related code.

### readAFMDATAfile.py 
From a time when the db was not compiled into a hdf5 file, but a folder of binaries. Only for 'historical' reasons.

### run\_minimalCNN_gpu.slrm
Script for running minimal\_AFM.py on triton. Change the slurm-relevant parameters in the file and either replace $1 and $2 or run like this:

```
sbatch run\_minimalCNN_gpu.slrm runName parameterfile.in
```

### start_job.sh
Template script to start multiple jobs. Replace the respective value in parameters.in with a dummy string and sed will replace it with a special value.

### utils.py
Contains auxiliary functions.

### view_results.py
Code snippets and functions to visualize arrays saved in .npy or .hdf5 files.
















