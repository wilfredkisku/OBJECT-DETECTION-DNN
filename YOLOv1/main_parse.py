import os
import pandas as pd

images = os.listdir('images/')
labels = os.listdir('labels/')

csv_df = pd.read_csv(r'train.csv')

images.sort()
labels.sort()

print(len(images))
print(len(labels))
print(len(csv_df))

img_new = [img[:-4] for img in images]
lab_new = [lab[:-4] for lab in labels]

not_in_img = [img for img in img_new if img not in lab_new]
not_in_lab = [lab for lab in lab_new if lab not in img_new]

print(len(not_in_img))
print(len(not_in_lab))

for files in labels:
    with open('labels/'+files) as f:
        lines = f.readlines()
        print(lines)

        for lin in lines:
            print(lin.split())
            print([int(val) if idx==0 else float(val) for idx, val in enumerate(lin.split())])
            if int(lin.split()[0]) != 0:
                with open('labels_persons/'+files, 'a+') as ff:
                    ff.write(lin)
                ff.close()

    print('*****')
