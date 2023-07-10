import os
import pandas as pd

labels_classes = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','potted','plant','sheep','sofa','train','tvmonitor'] 

#def split(df, train_pre):
#    return df[:int(len(df)*train_per)], df[int(len(df)*train_per):]

def create(lines):
    clean = []
    for l in lines:
        lin = l.split()
        if int(lin[0]) == 14:
            lin[0] = str(0)
            #print(l)
            clean.append(' '.join([elem for elem in lin]))
    return clean

def parse(path):
    label_files = os.listdir(path)
    for lab_file in label_files:
        with open('labels/'+lab_file) as f:
            lines = f.readlines()
            clean = create(lines)
            count = len(clean)
            for c in clean:
                count -= 1
                print(c)
                if count != 0:
                    str_ = c+'\n'
                else:
                    str_ = c

                with open('labels_persons/'+lab_file, 'a+') as ff:
                    ff.write(str_)
                ff.close()
    print("completed...")
    return 

'''
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
'''
if __name__ == "__main__":

    parse('labels/')
