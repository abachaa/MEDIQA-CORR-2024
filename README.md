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
* The UW Validation Set (#2) will be released on February 2nd.
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

# Evaluation Script 
[Will be added soon]

# Contact
 MEDIQA-NLP mailing group: https://groups.google.com/g/mediqa-nlp 

# Organizers   
* Asma Ben Abacha, Microsoft, USA
* Wen-wai Yim, Microsoft, USA
* Meliha Yetisgen, University of Washington, USA
* Fei Xia, University of Washington, USA
