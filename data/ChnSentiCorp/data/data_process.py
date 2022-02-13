import os
import time
import fileinput

label_list = ["政府采购", "科技基础设施建设", "市场监管", "公共服务", "科技成果转移转化",
                "科创基地与平台", "金融支持", "教育和科普", "人才队伍", "贸易协定", "税收激励",
                "创造和知识产权保护", "项目计划", "财政支持", "技术研发"
                ]

label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

labelsMap = {}

def label2str(labellist):
    labelstr = ''
    for i,value in enumerate(labellist):
        if i == len(labellist)-1:
            labelstr += str(value)
        else:
            labelstr += str(value)+','
    return labelstr

def str2label(labelstr):
    return [int(item) for item in labelstr.split(',')]



# with open('policy.txt', encoding='utf-8') as f1:
#     while True:
#         line = f1.readline()
#         if line:
#             lines = line.split('\t')
#             key = lines[1]
#             value = lines[0]
#             f2 = open('policymultilabel.txt','r+', encoding='utf-8')
#             f3 = open('policymultilabel1.txt', 'w+', encoding='utf-8')
#             getKey = False
#             isFirst = True
#             while True:
#                 #alltext = f2.readlines()
#                 line2 = f2.readline()
#                 if line2 == '\n':
#                     continue
#                 if line2:
#                     isFirst = False
#                     lines2 = line2.split('\t')
#                     line2value = lines2[0]
#                     line2key = lines2[1]
#                     if line2key == key:
#                         templabel = str2label(line2value)
#                         templabel[label2id[value]] = 1
#                         templabelstr = label2str(templabel)
#                         f3.write(templabelstr + '\t' + key)
#                         #f3.write('\n')
#                         #time.sleep(1)
#                         getKey = True
#                     else:
#                         f3.write(line2)
#                         #f3.write('\n')
#                         #time.sleep(1)
#                 elif isFirst or not getKey:
#                     label3list = [0] * len(label_list)
#                     label3list[label2id[value]] = 1
#                     f3.write(label2str(label3list) + '\t' + key)
#                     #f3.write('\n')
#                     #time.sleep(1)
#                     break
#                 else:
#                     break
#             f2.close()
#             f3.close()
#             os.remove("policymultilabel.txt")
#             os.rename("policymultilabel1.txt", "policymultilabel.txt")
#             #time.sleep(1)
#         else:
#             break
        #break
# with open('policymultilabel.txt','r+', encoding='utf-8') as f2:
#     line1label = f2.readline()
#     print(line1label)

# with open('policymultilabel.txt', encoding='utf-8') as f1:
#     f2 = open('policymultilabel1.txt', 'r+', encoding='utf-8')
#     while True:
#         line = f1.readline()
#         if line:
#             lines = line.split('\t')
#             key = lines[1].strip('\n')
#             label = lines[0]
#             labellist = str2label(label)
#             labelstr = ''
#             for i,item in enumerate(labellist):
#                 if item == 1:
#                     labelstr+= str(i)+','
#             labelstr = labelstr[:-1]
#             f2.write(key + '\t' + labelstr + '\n')
#         else:
#             break
#     f1.close()
#     f2.close()


#计算多类别标签
with open('policymultilabel1.txt', encoding='utf-8') as f1:
    multilabelnums = 0
    while True:
        line = f1.readline()
        if line:
            lines = line.split('\t')
            key = lines[0]
            label = lines[1].strip('\n')
            if len(label.split(',')) > 1:
                multilabelnums+=1
        else:
            break
    print(multilabelnums)
    f1.close()
