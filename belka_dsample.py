from tqdm import tqdm
import random


DATA_CSV = 'data/BELKA/train.csv'
OUTPUT_DATA_CSV = 'data/BELKA/data_dsampled.csv'
OUTPUT_VAL_CSV = 'data/BELKA/val_dsampled.csv'
OUTPUT_TRAIN_CSV = 'data/BELKA/train_dsampled.csv'


def dsample_train():
    with open(OUTPUT_DATA_CSV, 'w') as fo:
        with open(DATA_CSV, 'r') as f:
            fo.write(next(f))
            for line in tqdm(f, total=300000000):
                if line.strip().endswith('0'):
                    if random.random() > 0.995:
                        fo.write(line)
                else:
                    fo.write(line)

def create_validation(data_csv, ratio=0.02):
    fval = open(OUTPUT_VAL_CSV, 'w')
    ftr = open(OUTPUT_TRAIN_CSV, 'w')
    with open(data_csv, 'r') as f:
        head = next(f)
        fval.write(head)
        ftr.write(head)
        for line in tqdm(f, total=3058455):
            if random.random() > ratio:
                ftr.write(line)
            else:
                fval.write(line)



if __name__ == '__main__':
    #dsample_train()
    create_validation(OUTPUT_DATA_CSV, 0.002)
                    
            
    

