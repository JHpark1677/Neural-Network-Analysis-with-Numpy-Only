# Simple-Neural-Network-Analysis-with-Numpy-Only

Numpy implementation without open source deep learning frameworks such as pytorch, tensorflow, keras, etc. 



# Dependencies

Python >= 3.6

numpy, matplotlib


# Data Preparation

The main task in this project is logistic regression of 'make_moons' data from sklearn datasets. 
You can select the arguments describing noise level, or sample number by executing "generate_dataset.py". 
The datasets with default arguments are already set with 0.1 / 0.3 / 0.5 / 0.7 / 0.9 noise level. 

Put downloaded data into the following directory structure:

```
  - two_moon_0.1/
    - test.txt
    - train.txt
    - val.txt
      ...
  - two_moon_0.9/
    - test.txt
    - train.txt
    - val.txt
    
  - 1_hidden_layer.ipynb
  - 1_hidden_layer.py
  - 2_hidden_layer.ipynb
  - 2_hidden_layer.py
  - generate_dataset.py
```
   
   
   
# Training & Testing
  
The general training, scoring and plotting command of 1 hidden layer neural network
```
python 1_hidden_layer.py
   --number_node    # first hidden layer's number of nodes
   --noise          # exploited dataset's noise level for train / val/ test
   --lr             # learning rate
   --epoch          # total training epochs
   --print_by       # print train loss by "print_by" epochs
```

The general training, scoring and plotting command of 2 hidden layer neural network
```
python 2_hidden_layer.py
 --number_node1    # first hidden layer's number of nodes
 --number_node2    # second hidden layer's number of nodes
 --noise
 --lr
 --epoch
 --print_by
```

- Examples
  - 'python 2_hidden_layer.py --noise 0.5 --lr 0.1 --epoch 10000 --number_node1 20 --number_node2 10 --print_by 100
  

You can also use jupyter notebook with "1_hidden_layer.ipynb" and "2_hidden_layer.ipynb".
Don't require any input arguments but should be modified to experiment corresponding hyperparameters in source code.
  
# Contact
Please email ltbljb1677@postech.ac.kr for further questions
