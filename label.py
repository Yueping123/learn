import csv
def read_csv(data_path):
    file_reader = csv.reader(open(data_path, "rt", errors="ignore", encoding="utf-8"), delimiter=',')
    data_train = []
    label=[]
    for row in file_reader:
        sent = row[0]
        score = row[1]
        data_train.append(sent)
        label.append(score)
    return data_train,label


data_train=[]
label_list=[]
data_list,label_list= read_csv("test_results.tsv")
print(label_list)
print(len(label_list))
label_list1=[]
for label in label_list:
    if(float(label)>0.9):
        label=1
    else:
        label=0
    print(label)
    label_list1.append(label)
print(label_list1)






