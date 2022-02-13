import random
from tqdm import tqdm
nums = 9346
trainnums = int(nums*0.7)
testnums = int(nums*0.2)
devnums = nums - trainnums - testnums

# with open('policymultilabel1.txt', encoding='utf-8') as f1:
#     train = open('train1.txt', 'r+', encoding='utf-8')
#     test = open('test1.txt', 'r+', encoding='utf-8')
#     dev = open('dev1.txt', 'r+', encoding='utf-8')
#     policynums = 0
#     while True:
#         line = f1.readline()
#         if line:
#             if policynums < trainnums:
#                 train.write(line)
#             elif policynums >= trainnums and policynums < trainnums+testnums:
#                 test.write(line)
#             else:
#                 dev.write(line)
#             policynums+=1
#         else:
#             break
#     train.close()
#     test.close()
#     dev.close()


#需要对数据进行采样

random.seed(1)

labels = []

with open('class.txt', encoding='utf-8') as f1:
    for item in f1:
        labels.append(item.strip('\n'))

label2ids = {label:id for id,label in enumerate(labels) }
id2labels = {id:label for id,label in enumerate(labels)}

with open('policymultilabel1.txt', encoding='utf-8') as f1:
    num_classes = len(list(set(labels)))
    train, test, dev = [], [], []
    bucket = [[] for _ in range(num_classes)]
    language_nums = 0
    for step, item in enumerate(f1):
        language_nums += 1
        item = item.strip('\n')
        items = item.split('\t')
        for i in items[1].split(','):
            bucket[int(i)].append(item)

    #random.seed(1)
    #random.shuffle(bucket)

    labnum = 0
    for bt in tqdm(bucket, desc='split'):
        N = len(bt)
        print('\n'+str(0)+':'+str(N))
        labnum+=1
        if N == 0:
            continue
        train_size = int(N * 0.7)
        test_size = int(N*0.2)
        val_size = N - train_size - test_size

        random.seed(1)
        random.shuffle(bt)
        train.extend(bt[:train_size])
        test.extend(bt[train_size:train_size+test_size])
        dev.extend(bt[train_size+test_size:])

    with open('train.txt', 'w+', encoding='utf-8') as f2, open('test.txt', 'w+', encoding='utf-8') as f3, open('dev.txt', 'w+', encoding='utf-8') as f4:
        random.seed(1)
        random.shuffle(train)
        for k in train:
            f2.write(k+'\n')
        random.shuffle(test)
        for k in test:
            f3.write(k + '\n')
        random.shuffle(dev)
        for k in dev:
            f4.write(k + '\n')






