import pandas as pd
import numpy as np
import tqdm
from sklearn.utils import resample
import scipy.stats as sstats
import DecisionTree

class RandomForest:
    def __init__(self, train, test, trees_no, criterion, nattrs):
        self.train = train
        self.test = test
        self.criterion = criterion
        self.nattrs = nattrs
        self.trees_no = trees_no
        self.forest = []
        self.errors = [] # (error_rate, OOB_error, forest_error)
        self.oob_classification = {}
        self.create_forest()
      
        
        
    def print_forest(self):
        for i, T in enumerate(self.forest):
            errors = self.errors[i]
            print(f'Tree: {i}  --> Forest error: {self.errors[i][2] * 100:.3f}%, \t \
                  Tree error rate: {self.errors[i][0] * 100:.3f}%, \t \
                  OOB error: {self.errors[i][1] * 100:.3f}%')
            
            
    def create_forest(self):
        for i in tqdm.tqdm(range(self.trees_no), desc='Creating forest', position=0, leave=True):
            T, out_of_bag = self.create_tree()
            self.forest.append(T)
            
            oob_err = self.get_tree_err(T, out_of_bag)
            tree_err = self.get_tree_err(T, self.test)
            forest_err = self.get_forest_err(self.test)
            self.errors.append((tree_err, oob_err, forest_err))
            self.forest_OOB_err(out_of_bag, T)
    
    def create_tree(self):
        bootstrap, out_of_bag = self.bootstrap_data()
        T = DecisionTree.Tree(bootstrap, criterion=self.criterion, nattrs=self.nattrs)
        return T, out_of_bag
       
        
    def bootstrap_data(self):
        bootstrap = resample(self.train, n_samples=len(self.train))
        out_of_bag = pd.DataFrame(self.train.loc[i] for i in self.train.index 
                                  if i not in bootstrap.index)
        return bootstrap, out_of_bag
        
        
    def classify_test_set(self, tree, test):
        return [tree.classify(test.iloc[i]) for i in range(test.shape[0])]
        
        
    def get_tree_err(self, tree, test):
        res_targets = self.classify_test_set(tree, test)
        classification = list((np.array(test['target']) == np.array(res_targets)))
        return classification.count(False) / test.shape[0]
    
    
    def get_forest_err(self, test):
        targets_no = len(test)
        res_targets = np.array([self.classify_test_set(tree, test) for tree in self.forest])
        
        results_after_majority_voting = []
        for i in range(len(test)):
            trees_decision = res_targets[:, i]
            best = sstats.mode(trees_decision)[0][0]
            results_after_majority_voting.append(best)
     
        classification = list((np.array(test['target']) == np.array(results_after_majority_voting)))
     
        return classification.count(False) / len(test)
    
    
    def mean_tree_errors(self):
        mean_errors = np.array(self.errors).mean(axis=0) 
        return mean_errors
    
    
    def forest_mean_agreement(self):
        trees_targets = np.array([self.classify_test_set(tree, self.test) for tree in self.forest])
        no = len(self.forest)
        ans = 0
        for i in range(no):
            for j in range(i + 1, no):
                a = list(trees_targets[i] == trees_targets[j])
                ans += (a.count(True) / len(self.test))
      
        if no == 1:
            return 0
        return ans / (no * (no - 1) / 2)
    
    
    def forest_OOB_err(self, out_of_bag, tree):
        for i in out_of_bag.index:
            c = tree.classify(out_of_bag.loc[i])
            if i in self.oob_classification:
                if c in self.oob_classification[i]:
                    self.oob_classification[i][c] += 1
                else:
                    self.oob_classification[i][c] = 1
            else:
                self.oob_classification[i] = {}
                self.oob_classification[i][c] = 1
         
    
    def get_forest_OOB_err(self):
        bad = 0
        for elem, classification in self.oob_classification.items():
            correct_target = self.train.loc[elem].target
            majority_voting =  max(classification.items(), key=lambda x: x[1])[0]
            if majority_voting != correct_target:
                bad += 1
                
        return bad / len(self.train)