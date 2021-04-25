import json as js
import numpy as np

f_name = 'base_linux.json'  # 26 category, 57792 images
# len =  26
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25]




f_name = 'val_linux.json'   # 8 category, 18861 images
# len =  8
# [26 27 28 29 30 31 32 33]

f_name = 'novel_linux.json' # 12 category, 22759 images
# len =  12
# [34 35 36 37 38 39 40 41 42 43 44 45]


def display_file(f_name):
    with open(f_name, 'r') as f:
        s1 = js.load(f)
        
        for key, val in s1.items():
            # if key=='image_labels':
            #     labels = np.unique(np.array(val))
            #     print('len = ', len(labels))
            #     print(labels)


            print(key, end='   ')
            print(val[:3], end='   ')
            print(val[-3:], end = '   ')
            print(len(val))

        

display_file(f_name)