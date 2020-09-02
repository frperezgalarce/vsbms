import numpy as np
from warnings import filterwarnings
import sys
filterwarnings('ignore')
np.random.seed(1)
sys.path.insert(0, './src')
import utilFunctions as ut
import samplebiasselection as sbs

Data, ID, Class_col, Classes = ut.initialize_data(survey='OGLE')
Classes = ['rrlyr']
for i in Classes:
    if i != 'NonVar':
        Data, ID, Class_col, Classes = ut.initialize_data(survey='OGLE')
        Data = Data[Data[Class_col] != 'NonVar']
        print(Data[Class_col].value_counts())
        classA = [i]
        label1 = 'class_a'
        label2 = 'ClassB'
        Data = ut.joint_classes(classA, Data, Class_col, label1)
        Data = ut.jointComplementClasses(classA, Classes, Data, Class_col, label2)
        print('Class A: obs: ', Data[Data[Class_col] == label1].count()[0])
        print('Class B: obs: ', Data[Data[Class_col] == label2].count()[0])
        kwargs = ut.read_kwargs('experimentParameters/globalVariables.txt')
        Data = Data.dropna()
        Data_train, Data_test, label_train, label_test = sbs.samplebiasselection(Data, name=i, **kwargs)
        train = Data_train.Pred
        test = Data_test.Pred
        sbs.plot(train, test, title=classA[0] + ' in OGLE')
        Data_train['label'] = np.asanyarray(label_train)
        Data_test['label'] = np.asanyarray(label_test)
        sbs.export(Data_train, Data_test, name=classA[0])
