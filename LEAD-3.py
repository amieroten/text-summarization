'''
Module to compute and run evaluation on
LEAD-3 baseline.
'''
from nltk.tokenize import sent_tokenize

from dataloader import load_cnn_dailymail_lead

def predict(test):
    gold = []
    model = []
    for x,y in test.as_numpy_iterator():
        y = y[0].decode('utf-8').replace(" .", ". ")
        y = sent_tokenize(y, language="english")
        x = sent_tokenize(x[0].decode('utf-8'), language="english")[:3]
        gold.append(y)
        model.append(x)
    return gold, model

def save(gold, model):
    for index in range(len(model)):
        with open("output/LEAD-3_val/gold/"+str(index)+".txt","w") as f:
            for line in gold[index]:
                f.write(line+"\n")
        with open("output/LEAD-3_val/model/"+str(index)+".txt","w") as f:
            for line in model[index]:
                f.write(line+"\n")

def save_csv(gold, model):
    print("pause")


if __name__ == "__main__":
    val = load_cnn_dailymail_lead(batch_size=1)
    gold, model = predict(val)
    #save(gold, model)
    save_csv(gold, model)


