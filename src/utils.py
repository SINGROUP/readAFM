import os, os.path
import errno

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def parseInputFile(inputFile):
    """ Supply an InputFile(path!) with a parameters file with parameters and values separated by colons :,
    returns a dictionary. """
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
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def safe_open_w(path):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    mkdir_p(os.path.dirname(path))
    return open(path, 'w')



if __name__ == '__main__':
    print(parseInputFile('./parameters.in'))
#     with safe_open_w('/Users/bill/output/output-text.txt') as f:
#         f.write(...)
