# Language-Identification-NLP
Given a text, the task is to assign a language ID - eng (English), deu (German), fra (French), ita (Italian), or spa (Spanish)

Using a neural network model with bigram embeddings, a hidden layer, and an output layer, using PyTorch version 1.9.

Given a training set of sentences (x_train.txt) and the respective language IDs (y_train.txt). 

Command to train your neural network using this training set. The command for training the model is:

```
python3.8 a2part2.py --train --text_path x_train.txt --label_path y_train.txt --model_path model.pt
```

The file model.pt is the output of the training process that contains the model weight from training as well as other information that you need to save for testing.
You are also supplied with a test set (x_test.txt). The command to test on this test file and generate an output file out.txt is:
```
python3.8 a2part2.py --test --text_path x_test.txt --model_path model.pt --output_path out.txt
```
Your output file out.txt must contain the language ID in the same format as the y_train.txt file.
The following command calculates the accuracy of your language identification model, where out.txt is the output file of your code and y_test.txt is the actual language id of x_test.txt
```
python3.8 eval.py out.txt y_test.txt You are provided with the scoring code eval.py.
```
