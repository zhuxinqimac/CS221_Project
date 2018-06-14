import numpy as np
import string
import os


class_index_file = "./class_index.npz"

class_index_file_loaded = np.load(class_index_file)
class_index = class_index_file_loaded['class_index'][()]
index_class = class_index_file_loaded['index_class'][()]

files = os.listdir('/home/xinqizhu/ucfTrainTestlist')
test_files = [f for f in files if f[:4] == 'test']

for test_f in test_files:
    with open(os.path.join(
        '/home/xinqizhu/ucfTrainTestlist', test_f), 'r') as f:
        content = f.readlines()
    with open(os.path.join(
        '/home/xinqizhu/ucfTrainTestlist', 
        test_f.split('.')[0]+'_cl.txt'), 'w') as f:
        for line in content:
            class_name = string.lower(line.split('/')[0])
            class_id = class_index[class_name]
            new_line = line.strip()+' '+str(class_id)+'\n'
            f.write(new_line)

