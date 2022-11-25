#PLEASE WRITE THE GITHUB URL BELOW!
#https://github.com/hoorock/sklearn1.git

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassfier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassfier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import sys


def load_dataset(dataset_path):
	#To-Do: Implement this function
	df =pd.read_csv(dataset_path)
	return df

def dataset_stat(dataset_df):	
	#To-Do: Implement this function
	class0 = dataset_df.groupby('target').size()[0]
	class1 = dataset_df.groupby('target').size()[1]
	return len(dataset_df.columns)-1, class0, class1
    
def split_dataset(dataset_df, testset_size):
	#To-Do: Implement this function
	x = dataset_df.drop(columns-'target', axis=1)
	y = dataset_df['target']
	return train_test_split(x, y, test_size=testset_size)


def decision_tree_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	dt_cls = DecisionTreeClassfier()
	dt_cls.fit(x_train, y_train)
	acc = accuracy_score(y_test, dt_cls.predict(x_test))
	prec = precision_score(y_test, dt_cls.predict(x_test))
	rec = recall_score(y_tet, dt_clas.predict(x_test))
	return acc, pre, rec

def random_forest_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	rf_cls= RandomForestClassfier()
	rf_cls.fit(x_train, y_train)
	acc = accuracy_score(y_test, rf_cls.predict(x_test))
	prec = precision_score(y_test, rf_cls.predict(x_test))
	rec = recall_score(y_test, rf_cls.predict(x_test))

	return acc, prec, rec

def svm_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	svm_cls =SVC()
	svm_cls.fit(x_train, y_train)
	acc = accuracy_score(y_test, svm_cls.predict(x_test))
	prec = precision_score(y_test, svm_cls.predict(x_test))
	rec = recall_score(y_test, svm_cls.predict(x_test))
	return acc, prec, rec

def print_performances(acc, prec, recall):

	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)