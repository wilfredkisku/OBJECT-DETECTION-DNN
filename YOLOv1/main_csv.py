import os
import csv
import pandas as pd

def create_csv(path):
    
    lst_file = os.listdir(path)
    my_dict = {}
    lst_images = [val[:-3]+'jgp' for val in lst_file]
    my_dict['image'] = lst_images
    my_dict['label'] = lst_file
    #print(my_dict)
    my_df = pd.DataFrame.from_dict(my_dict)
    '''
    with open('mycsvfile.csv','w') as f:
        w = csv.writer(f)
        w.writerow(my_dict.keys())
        w.writerow(my_dict.values())
    '''
    print(my_df.head())
    my_df.to_csv('mycsvfile_train.csv',index=False, header=False)
    #my_df.to_csv('mycsvfile_train.csv',index=False)
    return None

if __name__ == "__main__":

    create_csv('labels_persons/')
