# Tag Categorization Project
Project guidlines can be found here: http://faculty.cooper.edu/sable2/courses/spring2021/ece467/NLP_Spring2021_Project1.docx

Corpora link: http://faculty.cooper.edu/sable2/courses/spring2021/ece467/TC_provided.tar.gz

## TLDR:
- Create a text categorization system (3 ml methods covered in class are recommended)
- Allowed to use NLP resources not related to what we learned (like tokenizer or POS tagger)
- Can use NLTK 
- CANNOT use pre-existing routines for word stats, text categorization, or ML
- Need to read input files 
- Train file with rows that have paths, filenames to test docs, and labels
- Test which has all of that but labels
- Can separate this into training and testing programs that run separately
- If so, make sure to save the training output
- Each corpus has different labels
- We get 3 corpora, 1 with both train and test & 2 with only training sets
- ⅓ of the corpus is used as a test set
- Get a Perl script to analyze predictions
- May hardcode categories in doc if we want but don’t have the user tell you which are used (find them using the training set)
- For the first corpus, can’t hardcode info from the test set!!!!
- Needs a writeup
