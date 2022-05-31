import sys
import pickle
import random
from tqdm import tqdm

data_dir = './pre_processing_data/'

train_name = './train_data.pkl'
val_name = './val_data.pkl'
test_name = './test_data.pkl'

names = ["a01", "a02", "a03", "a04", "a05", "a06", "a07", "a08", "a09", "a10",
         "a11", "a12", "a13", "a14", "a15", "a16", "a17", "a18", "a19", "a20",
         "b01", "b02", "b03", "b04", "b05",
         "c01", "c02", "c03", "c04", "c05", "c06", "c07", "c08", "c09", "c10",

         "x01", "x02", "x03", "x04", "x05", "x06", "x07", "x08", "x09", "x10",
         "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20",
         "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29", "x30",
         "x31", "x32", "x33", "x34", "x35"]


def merging(data_list, save_name):

    o_data = []
    y_data = []
    print('On Going...')

    for num in tqdm(data_list, file=sys.stdout):
        with open(data_dir + f'{num}.pkl', 'rb') as f:
            data = pickle.load(f)
        x, y, groups = data
        o_data.extend(x)
        y_data.extend(y)

    apnea_ecg = dict(o_data=o_data, y_data=y_data)
    with open(save_name, 'wb') as f:
        pickle.dump(apnea_ecg, f, protocol=2)

    print('\nok!')


if __name__ == "__main__":
    train_list = random.sample(names, int(len(names) * 0.7))
    residue = [i for i in names if i not in train_list]
    val_list = random.sample(residue, int(len(residue) * 0.5))
    test_list = [i for i in residue if i not in val_list]

    merging(train_list, train_name)
    merging(val_list, val_name)
    merging(test_list, test_name)
