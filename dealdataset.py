import numpy as np
import os
import h5py

datadir = "./dataset"
classes = os.listdir(datadir)
i = 0
NUM_POINT = 128
raw_datas = []
raw_labels = []
train_data = []
train_lebel = []
test_data = []
test_label = []


for classname in classes:
    txtfiles = os.listdir(datadir+"/"+classname)
    class_label = i
    for txtfile in txtfiles:
        sample_data = []
        with open(datadir+"/"+classname+"/"+txtfile, 'r') as f:
            raw_data = []
            sample_label = class_label
            for line in f:
                arr = line.split(',')
                raw_data.append([float(arr[0]), float(arr[1]), float(arr[2])])
            index = [i for i in range(0, len(raw_data))]
            if len(raw_data) <= 128:
                new_index = np.random.choice(index, NUM_POINT, replace=True)
            else:
                new_index = np.random.choice(index, NUM_POINT, replace=False)
            sample_data = [raw_data[i] for i in new_index]
            raw_datas.append(sample_data)
            raw_labels.append(sample_label)
    i = i+1


print(np.array(raw_datas).shape)
print(np.array(raw_labels).shape)

index = [i for i in range(0, len(raw_datas))]

train_index = np.random.choice(index, int(len(raw_datas) * 0.7), replace=False)
train_data = [raw_datas[i] for i in train_index]
train_lebel = [raw_labels[i] for i in train_index]

test_index = list(set(index)-set(train_index))
test_data = [raw_datas[i] for i in test_index]
test_lebel = [raw_labels[i] for i in test_index]

print(len(train_data))
print(len(test_data))

train_data=np.array(train_data)
train_lebel=np.array(train_lebel)
test_data=np.array(test_data)
test_lebel=np.array(test_lebel)

f = h5py.File('./train_data.h5','w')
f['data'] = train_data
f['labels'] = train_lebel
f.close()

f = h5py.File('./test_data.h5','w')
f['data'] = test_data
f['labels'] = test_lebel
f.close()