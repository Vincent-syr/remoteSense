import json as js
import numpy as np
import os


file_list = ["base_linux.json", "val_linux.json", "novel_linux.json"]

# 删除不合格文件
for f_name in file_list:
    with open(f_name, 'r') as f:
        s1 = js.load(f)
        
        for key, val in s1.items():
            if key=='image_names':
                for img in val:
                    if img[-4:] != ".png":
                        print(img)
                        # os.remove(img)



            # print(key, end='   ')
            # print(val[:3], end='   ')
            # print(val[-3:], end = '   ')
            # print(len(val))
    # break