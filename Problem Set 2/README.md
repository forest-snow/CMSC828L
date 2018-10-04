# Due October 9th

# Using keras, write seperate neural networks that

* Classify the data shown in the MNIST data

* Classify the kind of clothing item (pant, shirt, etc.) shown in the Fashion MNISt dataset

* Classify the breast cancer measurements as benign or malignant

* Do this using the best practices discussed in class (i.e. one hot encoding, ReLU, CNNS when you should, etc.)

* Have reasonable hyperparameters


# You will submit

* 3 .py files and .pdf, or 3 jupyter notebook files (*using the most recent version of jupyter*)

* Each much be submitted to the correct assignment for each dataset on ELMNS

# Your submission for each data set must include

* Training and testing accuracy vs epoch plots

* Some sort of reasonable print outs representative of the weight and biases (*not the entire things*)
  * The goal of this is to ensure that, for example, all you weights are not 0 (or something akin to that), do what you feel makes sense
  
* Examples of datapoints that fail in your models

* An explanation of why what you've included for all those things make sense

* An explanation of how/why your network's shape and layer choices, your loss function, and your activation function are correct

* Anything else you did that would convince us that you know what you're doing

* Do not submit the data after you split it etc., or a saved version of your model. 

* Do not include trash files in your submission (_MACOSX directories, .git file, .trash files, etc.)

* Please only zip your submission file if there's a bunch (i.e. multiple .py files)
  * If you're only submitting one file, do *not* put in in a .zip, .tar, .etc file
  * If you're only submitting a .py and a .pdf file, please submit them individually as well

# Allowed dependencies

* Keras 

  * You must use TensorFlow as your backend; this will impact your data shape for CNNS

* Tensorflow

* TensorBoard

* NumPy

* SciPy

* matplotlib

* SK-Learn 
  * You can only use the preprocessing tools

* seaborn 

  * Wrapper for matplotlib that makes it not-terrible to use and easier to read/grade, I encourage you to use it

* Other PyData ecosystem tools may be added upon request with a good reason, misc tools (i.e. a custom progress bar package) will not be allowed for the sanity of the TAs

* You may *not* depend on python-mnist (used in some of the preprocessing scripts)


# Report Guidelines

* The code you give us should be a formality, everything should be laid out in the pdf or clearly explained in Jupyter notebook (in English and without a bunch of extra outputs)

* Making submissions shorter faster/easier to grade is better for both of us (please don't be unnescesarily verbose)
  * 6 total pages of pdfs is the absolute maximum you should be submitting (or the equivalent for jupyter notebooks)

* Including all relevant figures, data displays, etc. in an immediately obvious form is the simplest thing you can do to improve your grade. These all must be graded very fast, and if it isn't obvious you have something, then your grade will reflect you not having it.

* Please make sure that the write up isn't in broken or incoherent English (please don't do it after being awake for 30 hours straight, etc.)

* You're allowed to do some models in Jupyter and some in .py+.pdf, but please dont; making things easier for the TAs is better for you too

* If you're on the fence about .py and .pdf vs a jupyter notebook, please use .py and .pdf (it's easier for us to grade)

* No submitting writeup files with an unreasonably large amount of white space

* *Do not include large amounts of code in your writeups*

* Figures are always better than words (this is something that will always be true for as long as you work in CS)

* Use a font typically found to represent words, not a monospace font typically found in text editors and IDEs

* Don't do weird things with colors (though colored figures are much better; that's also true for as long as you work in CS)

* If your report is terrible, even if your code is perfect, you will not recieve full credit

# Code Guidelines

* You must use Python 3.5 or greater

* You must start with the datasets as included from this repo, using the processed forms provided in the git repo; when your code is run during grading, this is all that will be available

* You can not specify initial conditions (i.e. via a fixed random seed) in the file you submit

* You aren't allowed to do data augmentation, to pad the data, or do anything similar to those
  * In the most general form possible, you aren't allowed to alter the data in any meaningful way before putting it into the model

* If you implement a grid or genetic algorithm search for optimizing hyperparameters, submit your code in a way so that when we run your code *without* having to rerun that portion, that portion won't run by default (but is trivial to enable if needed)

* You cannot .py and jupyter files for one dataset

* Your writeup *must* be in the form of a .pdf (not .docx etc.) or we will not read it

* The data processing files can not be called by your code

* Your code can not save things to disk when run by default, or edit files on disk in any way
  * Editing local files is tantamount to cheating

* Your code can't use more than 6GB of ram
  * Please don't screw this one up in particular, it'll make grading a mess and will result in a major deduction in points

* Make sure you split the data into testing and training sets in your code where appropriate

* All computations/machine learning must be done in Keras. Don't see that NumPy, etc. are allowed dependencies and do something weird.

* Make sure to scale your data from 0 to 1 and convert it to float32 before using it

  * You *must* use single precision for all computations for this task
  
* No including more than ~5 files per dataset. Remember we have to actually read these things.

* Mixing Jupyter noteboks with the report in a .pdf *is* allowed, and is preferred to including the report only in the jupyter notebook

# Other Notes

* Each folder includes the original data, and my processing code (which contains the original data source)

* If you can find a meaningful bug in the data processing code, some sort of extra credit will be awarded

* The best way to get logistical/administrative questions about this assignment answered is to come to Justin's office hours (MW 5-6 in AVW 4101/4103)

* That's also the best way to get weird Linux problems answered 

  * Ask Chen about weird macOS problems

* If you're doing this on a Windows system and have any issues beyond your Python code (i.e. "TensorFlow won't work/install right"), we can't help you

  * None of these tools were designed for Windows and using Windows with them a terrible idea; even Microsoft's data science and ML people generally don't anymore
  
  * Please dual boot Linux, I can show you the basics during office hours if needed
  
  * Yes, macOS should be fine
  
* You're *highly* encouraged to use a GPU and/or the AVX instructions for training your code when possible

* Any malicious pieces of code caused to be during grading will be considered accademic dishonesty
