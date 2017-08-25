readAFM Readme
==============

The Project
-----------

The goal of this project was explore the possibilities of using Machine Learning to analyze data obtained from Non Contact Atomic Force Microscopy (NC-AFM).

The title of this git-repository, readAFM, is intentional similar to [MechAFM](https://github.com/SINGROUP/MechAFM), while MechAFM simulates an AFM image with a mechanical model of the tip atoms, readAFM is a first attempt to understand the output "images" of an AFM, in a sense "read" them.



How to run
----------

To run minimal_AFM.py, first install the following packages: h5py, numpy, tensorflow. 

If you plan to modify the model (layer size,...) it is adviseable to copy the src/ folder.

```
cp -a src scratch/minimal_testrun
```

Put the correct path to the hdf5 database in the parameters.in file. For details about the parameters.in file have a look at that section. Run minimal_AFM.py:

```
python minimal_AFM.py <NameOfThisRun> -i <name of parameter file>
```
e.g.:

```
python minimalAFM.py myRun -i parameters.in
```
parameters.in is the default, so in this case the following would be sufficient:

```
python minimalAFM.py myRun
```

The simulation will run now. Various files will be created:

* out\_minimal_myRun.log contains logging information, most importantly the time/step and the current accuracy. Try
` cat out_minimal_myRun.log | grep accuracy `
* savemyRun/ is a folder containing the *.ckpt file to restore the variables and the files for Tensorboard
* viewfile_myRun.hdf5 contains the validation input and output of the last step. It also contains parameters dictionary of the run.

Tensorboard is a tool included in the Tensorflow package to make information about the training process etc. easier accessible. To open it go to the parent folder of savemyRun/ and run:

```
tensorboard -logdir=.
```

And point your browser to the indicated URL. In tensorboard there are two values displayed, accuracy is the squared difference per image, and cost is the cost used for training. Accuracy and cost are the same only if the parameter 'costWeight' is 1.0. The IMAGES section is also very interesting, the regexp "image/0" (and so on) is quite handy.

If you use triton, you might want to forward the url using ssh's -L handle:

```
ssh -L 16006:127.0.0.1:6006 reischt1@triton.aalto.fi
```
And then point your browser to 127.0.0.1:16006.

Input
-----

### parameters.in
Here are the parameters needed to run, and their default:

#### train
`Default: True `
If True, minimal_AFM.py trains the model, if False it is only evaluated once.

#### restorePath
`Default: None`
If None, the variables (weights, biases, filters) are initialized randomly, if a path to a .ckpt file is specified then the variables are restored from that file. Attention: if you use it dont forget to specify the filename including the step number: CNN\_minimal\_TR1\_myRun.ckpt-5000

#### saveName
`Default: "CNN_minimal_TR1_{}.ckpt".format(args.name)`
If None, no variables (weights, biases, filters) are saved. If a name is specified, a file will be created that can then later be restored with the restorePath parameter. The default is CNN\_minimal\_TR1\_<run name>.ckpt. The file will be saved in the directory specified under logdir.

#### DBPath
`Default: ../AFMDB_version_01.hdf5`
Path to the database containing the input data arrays.

#### DBShape
`Default: [41,41,41,1]`
Shape of the database, [xsize, ysize, zsize, inchannels].

#### outChannels
`Default: 1`
Output channels, has to match the labels in the DB (or the RuntimeSol.method):
xymap\_projection - 5-12
xymap\_collapsed, single\_atom - 1

#### logdir
`Default: './save{}/'.format(args.name)`
Location where the .ckpt-file and the tensorboard events are saved. Point tensorboard to this file to review the (training)run.

#### viewPath
`Default: './viewfile_{}.hdf5'.format(args.name)`
Path and filename where to save the viewfile.

#### logPath
`Default: './out_minimal_{}.log'.format(args.name)`
Name for the logfile, which contains logging information, most importantly the time/step and the current accuracy.

#### trainstepsNumber
`Default: 1`
Number of trainsteps to be performed.

#### trainbatchSize
`Default: 1`
Size of batch during training.

#### testbatchSize
`Default: 1`
Batchsize for validation.

#### LearningRate
`Default: 0.001`
Learning rate parameter passed to tf.AdamOptimizer.

#### costWeight
`Default: 1.0`
Parameter to give an extra penalty for false negatives. See in src/train_model.py to understand it. 

#### useRuntimeSolution
`Default: False`
If True, the labels are calculated every step.

#### RuntimeSol.method
`Default: xymap_collapsed`
Method how to calculate the solutions, options:
* xymap\_projection: use the 'outChannel' index to distinguish between elements
* xymap\_collapsed: use only one 'outChannel'
* singleAtom: especially for the case of a single atom, does not move the object to the COM

#### RuntimeSol.COMposition
`Default: [0.,0.,0.]`
For symmetrical molecules MechAFM sometimes has a bug when the COM is centered at [.0,.0,.0], so there is an option to move it away. If you move the COM in MechAFM, please change this parameter.

#### RuntimeSol.sigmabasexy
`Default: 1.0`
'Diameter' of the Gaussian in the xy-plane.

#### RuntimeSol.sigmabasez
`Default: 17.0`
'Size' of the Gaussian in the xy-plane.

#### RuntimeSol.amplificationFactor
`Default: 1.0`

#### numberTBImages
`Default: 5`
Number of images for tensorboard to store.

#### logEvery
`Default: 100`
How often tensorboard should store the current status. Logs every int(logEvery) steps.

#### saveEvery
`Default: 100`
How often the .ckpt file should be saved. Saves every int(saveEvery) steps.

### Database file
Stored as a hdf5-container. Should have the following structure: First level has two groups /train and /validation, they contain the groups that contain the datasets 'fzvals', 'solution' and 'atomPosition'. E.g. `/train/molXXXXXortnXXXXX/fzvals`. The molecule groups should have attributes containing the atomNameString etc.

The code in databaseCode creates such a file. See the README there.
  
  
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
sbatch run_minimalCNN_gpu.slrm runName parameterfile.in
```

### start_job.sh
Template script to start multiple jobs. Replace the respective value in parameters.in with a dummy string and sed will replace it with a special value.

### utils.py
Contains auxiliary functions.

### view_results.py
Code snippets and functions to visualize arrays saved in .npy or .hdf5 files.











