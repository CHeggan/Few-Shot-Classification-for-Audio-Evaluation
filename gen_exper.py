"""
Script deals with creation, submission and storage of argparse based shell 
    scripts for the mlssp server. 

IN particular the code does the following:
    -> Takes in a csv file with predefined argument titles and values (also contains
        information for what size of gpu to use - this is a proprietary setting for 
        our servers)
    -> For each row (experiment) we generate a cmd line to run our python script, 
        this includes all the relevant inclusions for the arguments, one 
        specific inclusion here is the keyword ['gpu'] for which we specify
        big or small for. Big is set so that the experiment is only loaded onto 
        the 3090s included in the server. Small allows to run on all available.
        This is done based on specific addresses and so is not generalisable! 
    -> This line along with the others required (bash initialisation etc) are 
        loaded into a .sh file, which we create a shorthand name for
    -> The .sh file is copied into the 'submitted_sh_files' folder and submitted 
        to slurm
"""
##############################################################################
# IMPORTS
##############################################################################
import os
import sys
import time
import shutil
import pandas as pd

##############################################################################
# HELPER FUNCTIONS
##############################################################################

##################################
# GENERATE PYTHON EXECUTABLE LINE
##################################
def gen_python_line(call_file_name, row_dict):
    exec_str = 'python' + ' ' + call_file_name + '.py'
    for key, value in row_dict.items():

        # Space + -- + key + space + arg value
        arg_add = ' ' + '--' + str(key) + ' ' + str(value)

        exec_str += arg_add
    
    return exec_str

##################################
# GENERATE FILE NAME
##################################
def gen_file_name(row_dict):
    file_str = ''
    for key, value in row_dict.items():

        # key + _ value + _
        #arg_add = str(key) + '_' + str(value) + '_'
        arg_add = str(value) + '__'

        file_str += arg_add
    
    return file_str

##################################
# CREATE SHELL FILE
##################################
def gen_shell_file(env_name, python_line, file_name):
    lines = []

    lines.append('#!/bin/bash')
    lines.append('\nsource ~/.bashrc')
    lines.append('\nconda activate ' + env_name)
    lines.append('\n' + python_line)

    with open(file_name + '.sh', 'w') as sh_file:
        sh_file.writelines(lines)
    
    return file_name + '.sh'


##################################
# CACHE/STORE SHELL FILE
##################################
def copy_shell_file(file_name, copy_folder):
    shutil.copyfile(file_name, os.path.join(copy_folder, file_name))
    os.remove(file_name)
    

##################################
# GEN SLURM SHELL CALL
##################################
def gen_slurm_call(file_name, gpu_part):
    # slurm main sbatch + gpu part (exclude or nothing) + file name
    slurm_line = 'sbatch --gres=gpu:1' + ' ' + gpu_part + ' ' + file_name

    return slurm_line



##############################################################################
# MAIN
##############################################################################
df = pd.read_csv('eval_models_validation.csv')
gpu_col = df['gpu'].values
df = df.drop(['gpu'], axis=1)

run = True

df_list = df.to_dict('records')

for i in range(len(df_list)):
    # Generates the actual ython liine to be embedded into shell file
    line = gen_python_line('main', df_list[i])
    # Generates a basic file name for tracking
    file_name = gen_file_name(df_list[i])
    file_name = file_name.replace('/', '_') 
    file_name = file_name.replace(' ', '_')
    # Generates the shell file and saves it using file name in base dir
    full_name = gen_shell_file('cont', line, file_name)

    # Decides which gpu add on to include
    if gpu_col[i] == 'small':
        gpu_part = '--exclude=mlssp[02],w[7128,7560,8068]'
    elif gpu_col[i] == 'big':
        gpu_part = '--exclude=mlssp[02-04],w[7128,7560,8068]'

    # Generates the slurm line call
    slurm_line = gen_slurm_call(full_name, gpu_part)
    print(slurm_line)

    if run:
        # Runs the actual line 
        os.system(slurm_line)

    # Copy the shell file to cache and delete it from base dir
    copy_shell_file(full_name, 'submitted_sh_files')

    time.sleep(3)


# print(df)
# print(df_list[0])