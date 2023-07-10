import os
import csv
import pandas as pd

def split(df, train_pre):
    return df[:int(len(df)*train_pre)], df[int(len(df)*train_pre):]

def create_csv(path):
    
    assert len(os.listdir(path)) != 0, "LABEL text files should be generated."

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
    return my_df

if __name__ == "__main__":

    my_df = create_csv('labels_persons/')
    my_df_train, my_df_test = split(my_df, 0.7)

    print(len(my_df_train), len(my_df_test))
    my_df_train.to_csv('mycsvfile_train_split.csv',index=False, header=False)
    my_df_test.to_csv('mycsvfile_test_split.csv',index=False, header=False)
