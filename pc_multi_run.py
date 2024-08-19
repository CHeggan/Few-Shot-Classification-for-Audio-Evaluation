import os
import sys
import pandas as pd


def gen_python_line(call_file_name, row_dict):
    exec_str = 'python' + ' ' + call_file_name + '.py'
    for key, value in row_dict.items():

        # Space + -- + key + space + arg value
        if str(value) == 'nan':
            value = ''
        arg_add = ' ' + '--' + str(key) + ' ' + str(value)

        exec_str += arg_add
    
    return exec_str

all_rows = pd.read_csv('supervised_1_shot.csv')
all_rows = all_rows.drop('gpu', axis=1)

df_list = all_rows.to_dict('records')

for i in range(len(df_list)):
    # Generates the actual ython liine to be embedded into shell file
    line = gen_python_line('imagenet_main', df_list[i])
    
    print(f'Running Line: {line}')
    os.system(line)