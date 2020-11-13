import json as js
import numpy as np

# f_name = 'base.json'  
# len =  11
# [ 0  2  4  6  8 10 12 14 16 18 20]




# f_name = 'val.json'   # 50 category, 2951 images
# len =  5
# [ 1  5  9 13 17]

f_name = 'novel.json'
# len =  5
# [ 3  7 11 15 19]


def display_file(f_name):
    with open(f_name, 'r') as f:
        s1 = js.load(f)
        
        for key, val in s1.items():
            if key=='image_labels':
                labels = np.unique(np.array(val))
                print('len = ', len(labels))
                print(labels)


            # print(key, end='   ')
            # print(val[:3], end='   ')
            # print(val[-3:], end = '   ')
            # print(len(val))

        

display_file(f_name)