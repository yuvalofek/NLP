# Parser
Project guidlines: http://faculty.cooper.edu/sable2/courses/spring2021/ece467/NLP_Spring2021_Project2.docx
Sample grammar file: http://faculty.cooper.edu/sable2/courses/spring2021/ece467/sampleGrammar.cnf

## TLDR:
* Grammar rules are in Chonkky normal form
* Build a parser based on the Cocke Kasami Younger (CKY) algorithm 
* If there are no valid parser output "NO VALID PARSES"
* If there are parses output the number of valid parses
* Grammar file is in the form of A --> B
* All non-terminals will start with a capital letter or an underscore
* All other characters are alphanumeric (lowercase letters of digits)
* Provided a program to convert context free grammar (CFG) to CNF (all test inputs are already in CNF)
* Assume all test cases are valid inputs (so there is no need to reformat input strings
* An input of 'quit' should end the program
* Output parses should be in bracketed notation
* User should be given the option of displaying output in one-line brackets or as textual parse trees

### CKY algorithm
![alt CKY algorithm](https://github.com/yuvalofek/NLP/blob/main/Parser/CKY.JPG)

### Sample Desired Outputs
```
Loading grammar...
Do you want textual parse trees to be displayed (y/n)?: n
Enter a sentence: i book the flight through houston
VALID SENTENCE

Valid parse #1:
[S [NP i] [VP [Verb book] [NP [Det the] [Nominal [Nominal flight] [PP [Preposition through] [NP houston]]]]]]
Valid parse #2:
[S [NP i] [VP [VP [Verb book] [NP [Det the] [Nominal flight]]] [PP [Preposition through] [NP houston]]]]
Valid parse #3:
[S [NP i] [VP [_Dummy2 [Verb book] [NP [Det the] [Nominal flight]]] [PP [Preposition through] [NP houston]]]]

Number of valid parses: 3

Enter a sentence: quit
Goodbye!

```
```
Loading grammar...
Do you want textual parse trees to be displayed (y/n)?: y
Enter a sentence: does the flight fly
VALID SENTENCE

Valid parse #1:
[S [_Dummy3 [Aux does] [NP [Det the] [Nominal flight]]] [VP fly]]

[S
  [_Dummy3
    [Aux does]
    [NP
      [Det the]
      [Nominal flight]
    ]
  ]
  [VP fly]
]

Number of valid parses: 1

Enter a sentence: quit
Goodbye!

```
