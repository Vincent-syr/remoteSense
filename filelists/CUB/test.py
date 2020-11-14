import json as js
import numpy as np
# base + val + novel: totally 11789 images

f_name = './base.json'  # 100 category, 5885 images
'''
label_names   ['001.Black_footed_Albatross', '002.Laysan_Albatross', '003.Sooty_Albatross']   200
image_names   ['/test/0Dataset_others/Dataset/Caltech-UCSD-Birds-200-2011/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0064_796101.jpg', ..] 5885
image_labels   [0, 0, 0]   5885
len =  100
[  0   2   4   6   8  10  12  14  16  18  20  22  24  26  28  30  32  34
  36  38  40  42  44  46  48  50  52  54  56  58  60  62  64  66  68  70
  72  74  76  78  80  82  84  86  88  90  92  94  96  98 100 102 104 106
 108 110 112 114 116 118 120 122 124 126 128 130 132 134 136 138 140 142
 144 146 148 150 152 154 156 158 160 162 164 166 168 170 172 174 176 178
 180 182 184 186 188 190 192 194 196 198]


'''
f_name = 'val.json'   # 50 category, 2951 images
'''
label_names   ['001.Black_footed_Albatross', '002.Laysan_Albatross', '003.Sooty_Albatross']   200
image_names    ...  2950
image_labels   [1, 1, 1]   [197, 197, 197]   2950
len =  50
[  1   5   9  13  17  21  25  29  33  37  41  45  49  53  57  61  65  69
  73  77  81  85  89  93  97 101 105 109 113 117 121 125 129 133 137 141
 145 149 153 157 161 165 169 173 177 181 185 189 193 197]


'''


# f_name = 'novel.json'  # 50 category, 2053 images

'''
label_names   ['001.Black_footed_Albatross', '002.Laysan_Albatross', '003.Sooty_Albatross']   200
image_names    ...  2953
image_labels   [3, 3, 3]   2953
len =  50
[  3   7  11  15  19  23  27  31  35  39  43  47  51  55  59  63  67  71
  75  79  83  87  91  95  99 103 107 111 115 119 123 127 131 135 139 143
 147 151 155 159 163 167 171 175 179 183 187 191 195 199]

'''



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
