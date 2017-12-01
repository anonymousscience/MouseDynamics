import pandas as pd

# usertestscores.csv
def user_accuracies( filename ):
    dataset = pd.read_csv(filename,  names=['userid','label','score'])
    numCols = int(dataset.shape[1])
    numRows = int(dataset.shape[0])
    userid = dataset['userid']
    label = dataset['label']
    score = dataset['score']

    userid = [int(e) for e in userid]
    label  = [int(e) for e in label]
    score = [float(e) for e in score]

    fr_samples = {'7': 0, '9': 0, '12': 0, '15': 0, '16': 0, '20': 0, '21': 0, '23': 0, '29': 0, '35': 0}
    fr_correct_samples = {'7': 0, '9': 0, '12': 0, '15': 0, '16': 0, '20': 0, '21': 0, '23': 0, '29': 0, '35': 0}
    for i in range(0,numRows):
        fr_samples[str(userid[i])]+= 1
        if (label[i] == 0 and score[i] <0.5) or (label[i] == 1 and score[i] >= 0.5):
            fr_correct_samples[str(userid[i])] += 1

    print( fr_samples )
    print( fr_correct_samples)
    for i in fr_samples:
        print( i+": "+str(fr_correct_samples[i]/fr_samples[i]))
    return
