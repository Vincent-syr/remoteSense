import json as js
import numpy as np

# f_name = 'base.json'  # 100 category, 5885 images
# len =  23
# [ 0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44]




# f_name = 'val.json'   # 50 category, 2951 images
# len =  11
# [ 1  5  9 13 17 21 25 29 33 37 41]

f_name = 'novel.json'
# len =  11
# [ 3  7 11 15 19 23 27 31 35 39 43]


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