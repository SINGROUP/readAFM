from __future__ import print_function
import sys
import os, os.path
import errno
import time
import h5py
import numpy as np


def is_float(s):
    """ Tests if a string (or any object) represents a float number.
    
    This is tested by trying to convert to a float number with float(s).
    
    Args:
        s: can be a string (intended use) or any other object.
    
    Return:
        True if s can be interpreted as float (would also be true for an integer!)
        False if s is not a float (i.e. a string containing characters,...)
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def parseInputFile(inputFile):
    """ Parses an input file.
    
    Supply a path to a file with parameters and values separated by colons : returns them in a dictionary.
    Use exactly one colon, the type of the object after the colon should be recognized automatically. Don't use quotation marks for strings.     
    Intended to use for the 'parameters.in' file. A # at the beginning of the line comments it out
    
    Args:
        imputFile: path to inputfile
    Returns:
        dict of the lines not starting with a #, splitting the lines at the :
    
    """
    f = open(inputFile, 'r')
    data = f.readlines()
    parameter = {}
    for line in data:
        # parse input, assign values to variables
        if line[0] == '#':
            continue
        key, value = line.split(":")
        v = value.strip()
        if v.isdigit() is True:
            parameter[key.strip()] = int(v)
        elif is_float(v) is True:
            parameter[key.strip()] = float(v)
        elif v[0] == '[' or v[0] == '(':
            try:
                parameter[key.strip()] = [int(item) for item in v.strip('[]').split(',')]
            except ValueError:
                parameter[key.strip()] = [float(item) for item in v.strip('[]').split(',')]
        elif v == 'None':
            parameter[key.strip()] = None
        elif v == 'False':
            parameter[key.strip()] = False
        elif v == 'True':
            parameter[key.strip()] = True
        else:
            parameter[key.strip()] = v            
    f.close()
    return parameter


# Taken from https://stackoverflow.com/a/600612/119527
def mkdir_p(path):
    """ Short def for the os-command 'mkdir -p path', the mkdir command also creating the parent dirs.
    
    Taken from https://stackoverflow.com/a/600612/119527
    
    Args:
        path: path that should be createt if it doesnt exist
        
    Returns:
        no return statement
    """
    
    
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise


def safe_open(path, mode, buffersize=None):
    """ Open "path" for writing, creating any parent directories as needed.
    
    Taken from https://stackoverflow.com/a/600612/119527
    
    Args:
        path: path to be created
        mode: mode for the builtin 'open' function
        buffersize: if None or not specified, the os default is used, otherwise passed to the builtin open function as buffersize param.
    
    """
    mkdir_p(os.path.dirname(path))
    if buffersize is None:
        return open(path, mode)
    else:
        return open(path, mode, buffersize)


def eprint(*args, **kwargs):
    """ Short definition for print to stderr. """
    print(*args, file=sys.stderr, **kwargs)


def progressmod4(i):
    """ Rotating line in stderr. """
    if i%4==0:
        eprint('\r -', end='')
    elif i%4==1:
        eprint('\r /', end='')
    elif i%4==2:
        eprint('\r |', end='')
    elif i%4==3:
        eprint('\r \\', end='')


def progresspercent(i, imax):
    """ Prints the progress in percent i/imax * 100% """
    eprint('\r Running ... {: 5.1f} %'.format(float(i)/(imax)*100.), end='')


def make_viewfile(parameters, testaccuracy, predictions, labels, atomPosition):
    """ Create a viewfile containing information about the NN.
    
    Creates a hdf5-file with the name as specified in the parameters dict, contains the predictions, the solutions, the final testaccuracy, the parameters dict and the atomPositions.
    
    Args:
        parameters: The parameters dict.
        testaccuracy: The testaccuracy to be written into the file. (usually float, could be any type)
        predictions: A np.array containing the predictions (or a tf-tensor)
        labels: A np.array containing the solutions(=labels) (or a tf-tensor)
        atomPosition: The atomPosition list as returned by the AFMdata.batch() method(s) in readAFMHDF5.py
        
    """
    
    print('Opening viewfile')
    viewfile = h5py.File(parameters['viewPath'], 'w')
    print('start writing attrs')
    viewfile.attrs['testaccuracy']=testaccuracy
    for key in parameters.keys():
        print(key, parameters[key])
        try:
            viewfile.attrs[key]=parameters[key]
        except TypeError:
            viewfile.attrs[key]=False
    print('create dataset: predictions')
    viewfile.create_dataset('predictions', data=predictions)
    print('create dataset: solutions')
    viewfile.create_dataset('solutions', data=labels)
    print('Add attr: AtomPosition')
    viewfile.attrs['AtomPosition'] = atomPosition[0][1]
    viewfile.close()

def stripnumbers(string):
    """ Strips a string from all the trailing numbers. 
    
    E.g. stripnumbers('example001') -> 'example'
    """ 
    if string[-1].isdigit():
        string=string[:-1]
        string = stripnumbers(string)
        return string
    else:
        return string

if __name__ == '__main__':
    print(parseInputFile('../scratch/testparseparams.in'))
#     for i in range(100):
#         time.sleep(0.1)
#         progresspercent(i, 99)
#     with safe_open_w('/Users/bill/output/output-text.txt') as f:
#         f.write(...)
