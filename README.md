# MEDIQA-CORR-2024

* Website: <https://sites.google.com/view/mediqa2024/mediqa-corr>

* MEDIQA-CORR@Codabench: <https://www.codabench.org/competitions/1900>

# Task Description

This task addresses the problem of identifying and correcting (common sense) medical errors in clinical notes. 
From a human perspective, these errors require medical expertise and knowledge to be both identified and corrected. 

Each clinical text is either correct or contains one error. The task consists in:
(a) predicting the error flag (1: the text contains an error, 0: the text has no errors),
and for flagged texts (with error):
(b) extracting the sentence that contains the error, and
(c) generating a corrected sentence.

# Data Description
* The MS Training Set contains 2,189 clinical texts.
* The MS Validation Set (#1) contains 574 clinical texts. 
* The UW Validation Set (#2) contains 160 clinical texts. 
* The test will include clinical texts from the MS and UW collections.
* The test set will contain the following 3 items: Text ID | Text | Sentences

# Run Submission
The submission format should follow the data format and consists of:

[Text ID] [Error Flag] [Error sentence ID or -1 for texts without errors] [Corrected sentence or NA for texts without errors]

  E.g.:
  * text-id-1 0 -1 NA
  * text-id-2 1 8 "correction of sentence 8..."
  * text-id-3 1 6 "correction of sentence 6..."
  * text-id-4 0 -1 NA
  * text-id-5 1 15 "correction of sentence 15..."

# Evaluation Metrics & Scripts 

---
**We use the following evaluation metrics:**
-  **Accuracy** for Error Flag Prediction _(subtask A)_ and Error Sentence Detection _(subtask B)_
-  **NLG metrics**: ROUGE, BERTScore, BLEURT, their **Aggregate-Score** (Mean of ROUGE-1-F, BERTScore, BLEURT-20), and their **Composite Scores** for the evaluation of Sentence Correction _(subtask C)_.
    -  The Composite score is the mean of individual scores computed as follows for each text: ​
       - 1 point if both the system correction and the reference correction are "NA"​
       -  0 point if only one of the system or the reference is "NA"​
       -  NLG metrics value in [0, 1] range (e.g., ROUGE, BERTScore, BLEURT or Aggregate-Score) if both the system correction and reference correction are non-"NA" sentences.
    -  <ins>**Aggregate-Composite score**</ins> is the main evaluation score to rank the participating systems. 

---

**Evaluation scripts:** <https://github.com/abachaa/MEDIQA-CORR-2024/tree/main/evaluation>

1. The [first evaluation script](https://github.com/abachaa/MEDIQA-CORR-2024/blob/main/evaluation/mediqa-corr-2024-eval-script-1-acc-rouge.ipynb.py) computes Accuracy, ROUGE, and ROUGE-Composite scores:
   - Error Flag Accuracy _(subtask A)_
   - Error Sentence Detection Accuracy _(subtask B)_
   - NLG Metrics _(subtask C)_: ROUGE-1-F, ROUGE-2-F, ROUGE-L-F, ROUGE-1-F-Composite, ROUGE-2-F-Composite, ROUGE-L-F-Composite.
   - **Composite score computation**: ROUGE on sentence vs. sentence cases + ones or zeros when either the reference or the candidate correction is NA. 

​2. The [second evaluation script](https://github.com/abachaa/MEDIQA-CORR-2024/blob/main/evaluation/mediqa-corr-2024-eval-script-2-all-metrics.py) computes Accuracy, ROUGE/BERTScore/BLEURT, Aggregate-Score and their Composite scores: 
   - Error Flag Accuracy _(subtask A)_
   - Error Sentence Detection Accuracy _(subtask B)_
   - NLG Metrics _(subtask C)_:
       - ROUGE-1-F, ROUGE-2-F, ROUGE-L-F, ​BERTScore (microsoft/deberta-xlarge-mnli), BLEURT-20
       - ROUGE-1-F-Composite, ROUGE-2-F-Composite, ROUGE-L-F​-Composite, BERTScore-Composite, BLEURT-20​-Composite
       - Aggregate-Score: Mean of ROUGE-1-F, BERTScore, BLEURT-20 (correlates better with human judgments; cf. our ACL Findings 2023 paper on evaluation metrics <https://aclanthology.org/2023.findings-acl.161.pdf>
       - **Aggregate-Composite**​ (main score to rank the participating systems) 

​
# Contact
 MEDIQA-NLP mailing list: https://groups.google.com/g/mediqa-nlp 
 Email: mediqa.organizers@gmail.com 

# Organizers   
* Asma Ben Abacha, Microsoft, USA
* Wen-wai Yim, Microsoft, USA
* Meliha Yetisgen, University of Washington, USA
* Fei Xia, University of Washington, USA
