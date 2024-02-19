import os
import csv
import pandas as pd

def split(df, train_pre):
    return df[:int(len(df)*train_pre)], df[int(len(df)*train_pre):]

def create_csv(t_l_p, v_l_p):

    #print("COUNT :: "+str(len(os.listdir(data_path))), "COUNT LABEL :: "+str(len(os.listdir(label_path))))
    
    assert len(os.listdir(t_l_p)) != 0, "TRAIN LABEL TEXT FILES PRESENT"
    assert len(os.listdir(v_l_p)) != 0, "TEST LABEL TEST FILES PRESENT"

    #TRAIN
    lst_file = os.listdir(t_l_p)
    my_dict = {}
    lst_images = [val[:-3]+'jpg' for val in lst_file]

    my_dict['image'] = lst_images
    my_dict['label'] = lst_file

    #TEST
    v_lst_file = os.listdir(v_l_p)
    v_my_dict = {}
    v_lst_images = [val[:-3]+'jpg' for val in v_lst_file]

    v_my_dict['image'] = v_lst_images
    v_my_dict['label'] = v_lst_file

    my_df = pd.DataFrame.from_dict(my_dict)
    v_my_df = pd.DataFrame.from_dict(v_my_dict)

    print(my_df.head())
    print(v_my_df.head())

    my_df.to_csv('./data/mycsvfile_train.csv',index=False, header=False)
    v_my_df.to_csv('./data/mycsvfile_test.csv',index=False, header=False)

    return my_df, v_my_df

if __name__ == "__main__":

    t_data_path = "./data/TrainImageFolder/images"
    t_label_path = "./data/train_labels_persons"

    v_data_path = "./data/ValImageFolder/images"
    v_label_path = "./data/val_labels_persons"

    my_df, v_my_df = create_csv(t_label_path, v_label_path)

    print("CONVERSION COMPLETED ...")
