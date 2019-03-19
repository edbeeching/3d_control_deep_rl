#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 11:22:32 2019

@author: edward
"""

import os, logging, datetime

def initialize_logging(args):
    # two options
    # 2. loading saved model so use old logfile
    if args.reload_model:
        results_dir = args.reload_model.split(',')[0]
        log_name = results_dir + '_log.txt'
        start_iter = args.reload_model.split(',')[1]
        output_dir = '{}tmp/results/{}_rl/{}/'.format(args.out_dir, args.simulator, results_dir)
        assert os.path.exists(output_dir), 'Trying to reload model but base output dir does not exist' 
        assert os.path.exists(os.path.join(output_dir, 'models/')), 'Trying to reload model but model output dir does not exist' 
        assert os.path.exists(os.path.join(output_dir, 'evaluations/')),  'Trying to reload model but eval output dir does not exist'  
        logging.basicConfig(filename=os.path.join(output_dir, log_name),
                            format='%(asctime)s %(message)s', 
                            datefmt='%Y/%m/%D %I:%M:%S %p',
                            level=logging.INFO) 
        logging.getLogger().addHandler(logging.StreamHandler())

        # Print to log file that training has resumed
        logging.info('========== Resuming training iter: {} ========== \n'.format(int(start_iter)))
    # 1. new logfile
    else:   
        now = datetime.datetime.now()
        results_dir = '{}_{}_{:02}_{:02}_{:02}_{:02}_{:02}'.format(args.job_id, 
                                                          args.test_name,
                                                          now.year,
                                                          now.month, 
                                                          now.day, 
                                                          now.hour, 
                                                          now.minute)        
        log_name = results_dir + '_log.txt'
        output_dir = '{}tmp/results/{}_rl/{}/'.format(args.out_dir, args.simulator, results_dir)
        assert not os.path.exists(output_dir), 'The output directory already exists'
        if not os.path.exists(output_dir):            
            os.makedirs(output_dir)    
            for directory in ['models/', 'evaluations/']:
                os.makedirs(os.path.join(output_dir, directory))    
            
        print('Created log output directory {}'.format(output_dir))
        
         # Write args to a file
        with open(os.path.join(output_dir,'log_args.txt'), 'w') as args_log:
            for param, val in sorted(vars(args).items()):
                args_log.write('{} : {} \n'.format(param, val))
        
        logging.basicConfig(filename=os.path.join(output_dir, log_name),
                            format='%(asctime)s %(message)s', 
                            datefmt='%Y/%m/%D %I:%M:%S %p',
                            level=logging.INFO) 

        logging.getLogger().addHandler(logging.StreamHandler())
        logging.info('========== Training Log file ==========')
        
        
        
    return output_dir

