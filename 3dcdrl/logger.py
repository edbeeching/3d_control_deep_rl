#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 10:00:32 2019

@author: edward
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:04:41 2018

@author: anonymous
"""

import os
import datetime

class Logger(object):
    def __init__(self, args):
        self.args = args
        assert os.path.exists(args.scenario_dir)
        #assert os.path.exists(args.scenario_dir + args.scenario)
        if args.reload_model:
            results_dir = args.reload_model.split(',')[0]
            self.log_name = results_dir + '_log.txt'
            start_iter = args.reload_model.split(',')[1]
            args.output_dir = '{}tmp/results/{}_rl/{}/'.format(args.out_dir, args.simulator, results_dir)
            assert os.path.exists(args.output_dir), 'Trying to reload model but base output dir does not exist' 
            assert os.path.exists(args.output_dir + 'models/'), 'Trying to reload model but model output dir does not exist' 
            assert os.path.exists(args.output_dir + 'evaluations/'), 'Trying to reload model but eval output dir does not exist' 
            print(args.output_dir)
            self.output_dir = args.output_dir
            # Print to log file that training has resumed
            with open(self.output_dir + 'log.txt', 'a') as log:
                log.write('========== Resuming training iter: {} ========== \n'.format(int(start_iter)))
            
        else:
            now = datetime.datetime.now()
            print(now)    
        
            results_dir = '{}_{}_{:02}_{:02}_{:02}_{:02}_{:02}'.format(args.job_id, 
                                                              args.test_name,
                                                              now.year,
                                                              now.month, 
                                                              now.day, 
                                                              now.hour, 
                                                              now.minute)        
            self.log_name = results_dir + '_log.txt'
            args.output_dir = '{}tmp/results/{}_rl/{}/'.format(args.out_dir, args.simulator, results_dir)
            assert not os.path.exists(args.output_dir), 'The output directory already exists'
            self.output_dir = args.output_dir
            
            # create output directories  
            self.ensure_dir(self.output_dir)
            print('Created log output directory {}'.format(self.output_dir))
            
            #create model and eval dirs
            output_directories = ['models/', 'evaluations/']
            for directory in output_directories:
                self.ensure_dir(self.output_dir + directory)
             # Write args to a file
            with open(self.output_dir + 'log_args.txt', 'w') as args_log:
                for param, val in sorted(vars(args).items()):
                    args_log.write('{} : {} \n'.format(param, val))
            
            # Create log file
            with open(self.output_dir + self.log_name, 'w') as log:
                log.write('========== Training Log file ========== \n')
                
    def ensure_dir(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):            
            os.makedirs(directory)  
            
    def write(self, text):
        now = datetime.datetime.now()
        print(now, ':', text)
        with open(self.output_dir + self.log_name, 'a') as log:
            log.write('{} :  {}\n'.format(now, text))
            
    def get_eval_output(self):
        return self.output_dir + 'evaluations/'
        
        
if __name__ =='__main__':    
    pass
            
        
        