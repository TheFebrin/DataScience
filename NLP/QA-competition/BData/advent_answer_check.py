import editdistance
import sys

rn = ['ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi', 'xii', 'xiii',
                 'xiv', 'xv', 'xvi', 'xvii', 'xviii', 'xix', 'xx', 'xxi', 'xxii']

rome_numbers = dict(zip(rn, range(2, 23)))

def numbers_from(s):
    res = set()
    for w in s.split():
        w = w.lower()
        if w.isdecimal():
            res.add(w)
        if w in rome_numbers:
            res.add(rome_numbers[w])
    return res                 
                 
                 
def is_number(s):
    return lower(s) in rome_numbers or s.isdecimal()
                     
def scaled_editdist(ans, cor):
    ans = ans.lower()
    cor = cor.lower()
    
    return editdistance.eval(ans, cor) / len(cor)
    
def single_match(a, c):
    numbers_c = numbers_from(c)
    numbers_a = numbers_from(a)
        
    return numbers_a == numbers_c and scaled_editdist(a, c) < 0.5
        
def match(ans, cor):
    return any(single_match(ans, c) for c in cor)
        
found_answers = []
correct_answers = []

for x in open('correct_answers.txt'):
    x = x.strip()
    correct_answers.append(x.lower().split('\t'))
    
for x in open('found_answers.txt'):    
    x = x.strip()
    found_answers.append(x.lower())
    
N = len(correct_answers)
score = 0.0

for ans, cor in zip(found_answers, correct_answers):    
    if match(ans, cor):
        score += 1
        
print ('TOTAL SCORE:', score / len(correct_answers))        
