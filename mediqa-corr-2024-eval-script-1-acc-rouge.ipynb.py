#!/usr/bin/env python
# coding: utf-8

# ## **Evaluation Script for MEDIQA-CORR 2024**

# In[1]:


#!pip install rouge


# In[8]:


import re
import pandas as pd
from rouge import Rouge
import scipy.stats as stats
import numpy as np
import math
import string


# 

# In[9]:


#################
# Parsing Funcs #
#################


def parse_reference_file(filepath):
    """Parsing reference file path.

    Returns:
        reference_corrections (dict) {text_id: "reference correction"}
        reference_flags (dict) {text_id: "1 or 0 error flag"}
        reference_sent_id (dict) {text_id: "error sentence id or -1"}
    
    """

    reference_corrections = {}
    reference_flags = {}
    reference_sent_id = {}

    df = pd.read_csv(filepath)
    
    for index, row in df.iterrows():
        text_id = row['Text ID']
        corrected_sentence = row['Corrected Sentence']
        
        if not isinstance(corrected_sentence, str):
            if math.isnan(corrected_sentence):
                corrected_sentence = "NA"
            else:
                corrected_sentence = str(corrected_sentence)
                corrected_sentence = corrected_sentence.replace("\n", " ") \
                  .replace("\r", " ").strip()
                  
        reference_corrections[text_id] = corrected_sentence
        reference_flags[text_id] = str(row['Error Flag'])
        reference_sent_id[text_id] = str(row['Error Sentence ID'])

    return reference_corrections, reference_flags, reference_sent_id


def parse_run_submission_file(filepath):
    
    file = open(filepath, 'r')

    candidate_corrections = {}
    predicted_flags = {}
    candidate_sent_id = {}
    
    lines = file.readlines()
    
    for line in lines:
        line = line.strip()
        
        if len(line) == 0:
            continue
            
        if not re.fullmatch('[a-z0-9\-]+\s[0-9]+\s\-?[0-9]+\s.+', line):
            print("Invalid line: ", line)
            continue
            
        # replacing consecutive spaces
        line = re.sub('\s+', line, ' ')
        
        # parsing
        items = line.split()
        text_id = items[0]
        error_flag = items[1]
        sentence_id = items[2]
        corrected_sentence = ' '.join(items[3:]).strip()
        
        # debug - parsing check
        # print("{} -- {} -- {} -- {}".format(text_id, error_flag, sentence_id, corrected_sentence))

        predicted_flags[text_id] = error_flag
        candidate_sent_id[text_id] = sentence_id

        # processing candidate corrections
        # removing quotes

        while corrected_sentence.startswith('"') and len(corrected_sentence) > 1:
            corrected_sentence = corrected_sentence[1:]
            
        while corrected_sentence.endswith('"') and len(corrected_sentence) > 1:
            corrected_sentence = corrected_sentence[:-1]
                   
        if error_flag == '0':
            # enforcing "NA" in predicted non-errors (used for consistent/reliable eval)
            candidate_corrections[text_id] = "NA"
        else:
            candidate_corrections[text_id] = corrected_sentence

    return candidate_corrections, predicted_flags, candidate_sent_id

            


# In[17]:


##############
# Eval Funcs #
##############

def compute_accuracy(reference_flags, reference_sent_id, predicted_flags, candidate_sent_id):
    # Error Flags Accuracy (missing predictions are counted as false)
    matching_flags_nb = 0
    
    for text_id in reference_flags:
        if text_id in predicted_flags and reference_flags[text_id] == predicted_flags[text_id]:
            matching_flags_nb += 1
            
    flags_accuracy = matching_flags_nb / len(reference_flags)
    
    # Error Sentence Detection Accuracy (missing predictions are counted as false)
    matching_sentence_nb = 0
    
    for text_id in reference_sent_id:
        if text_id in candidate_sent_id and candidate_sent_id[text_id] == reference_sent_id[text_id]:
            matching_sentence_nb += 1
            
    sent_accuracy = matching_sentence_nb / len(reference_sent_id)

    return {
        "Error Flags Accuracy": flags_accuracy,
        "Error Sentence Detection Accuracy": sent_accuracy
    }

def increment_counter(counters, counter_name):
    counters[counter_name] = counters[counter_name] + 1

class NLGMetrics(object):

    def __init__(self, metrics = ['ROUGE']):
        self.metrics = metrics
    
    def compute(self, references, predictions, counters):
        results = {}
        
        if 'ROUGE' in self.metrics:
            rouge = Rouge() 
            rouge_scores = rouge.get_scores(predictions, references)
                            
            rouge1f_scores = []
            rouge2f_scores = []
            rougeLf_scores = []
            
            for i in range(len(references)):
                r1f = rouge_scores[i]["rouge-1"]["f"]
                r2f = rouge_scores[i]["rouge-2"]["f"]
                rlf = rouge_scores[i]["rouge-l"]["f"]
                
                rouge1f_scores.append(r1f)	
                rouge2f_scores.append(r2f)
                rougeLf_scores.append(rlf)
                
            # for checking comparison with composite
            rouge1check = np.array(rouge1f_scores).mean()
            rouge2check = np.array(rouge2f_scores).mean()
            rougeLcheck = np.array(rougeLf_scores).mean()

            results['R1F_subset_check'] = rouge1check
            results['R2F_subset_check'] = rouge2check
            results['RLF_subset_check'] = rougeLcheck
            
            ###############################
            # Composite score computation #
            ###############################
            
            """
            NLG METRIC on sentence vs. sentence cases + ones or zeros 
            when either the reference or the candidate correction is NA
            """
            
            rouge1score = np.array(rouge1f_scores).sum()
            rouge2score = np.array(rouge2f_scores).sum()
            rougeLscore = np.array(rougeLf_scores).sum()
            
            composite_score_rouge1 = (rouge1score + counters["system_provided_correct_na"]) / counters["total_texts"]
            composite_score_rouge2 = (rouge2score + counters["system_provided_correct_na"]) / counters["total_texts"]
            composite_score_rougeL = (rougeLscore + counters["system_provided_correct_na"]) / counters["total_texts"]

            results['R1FC'] = composite_score_rouge1
            results['R2FC'] = composite_score_rouge2
            results['RLFC'] = composite_score_rougeL

        
        return results


def get_nlg_eval_data(reference_corrections, candidate_corrections, remove_nonprint = False):
    references = []
    predictions = []
    
    counters = {
        "total_texts": 0,
        "reference_na": 0,
        "total_system_texts": 0,
        "system_provided_na": 0,
        "system_provided_correct_na": 0,
    }
    
    for text_id in reference_corrections:
        increment_counter(counters, "total_texts")
        
        # removing non ascii chars
        reference_correction = reference_corrections[text_id]
        
        if remove_nonprint:
            reference_correction = ''.join(filter(lambda x: x in string.printable, str(reference_correction)))
            
        if reference_correction == "NA":
            increment_counter(counters, "reference_na")
            
        if text_id in candidate_corrections:
            increment_counter(counters, "total_system_texts")
            candidate = candidate_corrections[text_id]
            
            if remove_nonprint:
                candidate = ''.join(filter(lambda x: x in string.printable, candidate))
                
            if candidate == "NA":
                increment_counter(counters, "system_provided_na")
                
            # matching NA counts as 1
            if reference_correction == "NA" and candidate == "NA":
                increment_counter(counters, "system_provided_correct_na")
                continue
                
            # Run provided "NA" when a correction was required (=> 0)
            # or Run provided a correction when "NA" was required (=> 0)
            if candidate == "NA" or reference_correction == "NA":
                continue
                
            # remaining case is both reference and candidate are not "NA"
            # both are inserted/added for ROUGE/BLEURT/etc. computation
            references.append(reference_correction)
            predictions.append(candidate)
    

    
    return references, predictions, counters


# In[18]:


submission_file = "data/baseline-run-1.txt"
reference_csv_file = "data/MEDIQA-CORR-2024-MS-ValidationSet-1.csv"


reference_corrections, reference_flags, reference_sent_id = parse_reference_file(reference_csv_file)
candidate_corrections, candidate_flags, candidate_sent_id = parse_run_submission_file(submission_file)

# Accuracy
accuracy_results = compute_accuracy(reference_flags, reference_sent_id, candidate_flags, candidate_sent_id)
print("Accuracy Results:\n", accuracy_results)
print()

# NLG Eval for corrections
references, predictions, counters = get_nlg_eval_data(reference_corrections, candidate_corrections)
metrics = NLGMetrics()
nlg_eval_results = metrics.compute(references, predictions, counters) 


print("NLG Eval Results:\n", nlg_eval_results) 
print()

# debug check
print(counters)
print()

