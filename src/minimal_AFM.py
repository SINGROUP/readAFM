import tensorflow as tf
import argparse
from utils import * 
from model import define_model
from train_model import train_model
from eval_model import eval_model


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Name given to all output files.")
    parser.add_argument("-i", "--input_file", default="parameters.in", help="the path to the input file (default: %(default)s)")
    # parser.add_argument("-o", "--output", default="output/", help="produced files and folders will be saved here (default: %(default)s)")
    args=parser.parse_args()
    
    # parser.add_argument("-v", "--verbosity", action="count", default=0)
    
    print('Parsed name argument {} of type {}.'.format(args.name, type(args.name)))
    print('Parsing parameters from {}'.format(args.input_file))
    
    parsedParameters = parseInputFile(args.input_file)
    
    # These are the default Parameters!
    parameters = {'train': True,
                  'restorePath': None, # Typically: "./CNN_minimal_TR1_{}.ckpt"
                  'saveName': "CNN_minimal_TR1_{}.ckpt".format(args.name),         # Typically: "CNN_minimal_TR1_{}.ckpt".format(args.name)
                  'DBPath': '../AFMDB_version_01.hdf5',
                  'DBShape': [41,41,41,1],
                  'outChannels': 1,
                  'logdir': './save{}/'.format(args.name),
                  'viewPath': './viewfile_{}.hdf5'.format(args.name),
                  'logPath': './out_minimal_{}.log'.format(args.name),
                  'trainstepsNumber': 1,
                  'trainbatchSize':1,
                  'testbatchSize': 1,
                  'LearningRate': 0.001,
                  'costWeight': 1.0,
                  'useRuntimeSolution': False,
                  'RuntimeSol.method': 'xymap_collapsed', 
                  'RuntimeSol.COMposition': [0.,0.,0.], 
                  'RuntimeSol.sigmabasexy': 1.0,
                  'RuntimeSol.sigmabasez': 17.0, 
                  'RuntimeSol.amplificationFactor': 1.0,
                  'numberTBImages': 5,
                  'logEvery': 100,
                  'saveEvery': 100
                  }
    
    parameters.update(parsedParameters)
    
    DBShape = parameters['DBShape']
    outChannels = parameters['outChannels']

    LOGDIR = parameters['logdir']    
    if LOGDIR[-1] != '/':
        LOGDIR = LOGDIR + '/'
    parameters['logdir'] = LOGDIR

    # Output has to have the same xy dimensions as input (DBShape)
    
    logfile=safe_open(parameters['logPath'], 'w', 0)

    logfile.write('define the  placeholders \n')
    Fz_xyz = tf.placeholder(tf.float32, [None,]+DBShape)
    tf.summary.image('Fzinput_0', Fz_xyz[:,:,:,0,:], parameters['numberTBImages'])
    tf.summary.image('Fzinput_half', Fz_xyz[:,:,:,int(Fz_xyz.shape[2]/2),:], parameters['numberTBImages'])
    tf.summary.image('Fzinput_last', Fz_xyz[:,:,:,-1,:], parameters['numberTBImages'])
    print(int(Fz_xyz.shape[2]/2))
    
    solution = tf.placeholder(tf.float32, [None,]+DBShape[:2]+[outChannels])
    tf.summary.image('solutions', solution, parameters['numberTBImages'])

    posxyz = tf.placeholder(tf.string, [None, 2])
    posxyzcropped = posxyz[:parameters['numberTBImages'], :]
    tf.summary.text('posxyz', posxyzcropped)


    keep_prob = tf.placeholder(tf.float32)
    
    print(parameters['train'])

    if parameters['train']:
        print('Start training:')
        train_model(define_model, Fz_xyz, solution, keep_prob, posxyz, parameters, logfile)
    else:
        print('Start evaluation:')
        eval_model(define_model, Fz_xyz, solution, keep_prob, posxyz, parameters, logfile)

        
    logfile.write('finished! \n')
    logfile.close()
    print 'Finished!'
