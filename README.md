## Entity Embedding Encoder

A tool to numerically encode categorical variables in datasets, based on the work 
'Entity Embeddings of Categorical Variables' by C. Guo and F. Berkhahr, 2016.


## Description and usage
Transforms a dataset with categorical and numerical variables into a purely numerical 
dataset. **The dataset is required to have a dependent variable**. 

The encoding is performed as follows:
1. Numerical variables, if any, are scaled to [0,1]
2. Each categorical variable is encoded with the LabelEncoder from Scikit-Learn
3. A neural network N is built: \
   a) for each categorical variable a Keras Embedding layer is added, which we 
   call an EE-layer. \
   b) all the numerical variables are concatenated to the outputs of all the EE-layers 
   into a single numerical vector, which is then fed to a dense layer. \
   c) the output from b) is then passed to a smaller dense layer.\
   d) the output from c) is fed to a final layer appropriate for the dependent
   variable.
5. N is trained with 70% of the data and validated with the remaining 30%. 
6. Once N is trained, the encoding of the dataset are the outputs from all the 
   EE-layers with the originally numerical variables appended.

 
**Initialisation:** 

encoder = EntityEmbeddingEncoder(epochs=100, dense_layers_sizes=(1000,500), dropout=True)


**Main fit method:**

encoder.fit(X, y, cat_cols=[], ee_sizes={}, verbose=True, test=None)\
encoder.fit_transform(X, y, cat_cols=[], ee_sizes={}, verbose=True, test=None)

Where:

- X is a pandas dataframe with all the independet  variables
- y is the dependent variable
- cat_cols is a list with the column numbers of all the categorical variables to encode. If
empty, all categorical variables detected will be encoded
- ee_sizes is a dictionary with the sizes for EE-layers. If empty, a categorical variable with
k distinct values will be assigned an EE-layer of size min(30, k/3)
- verbose is a boolean for whether to print details on the training of the network
- test is a pandas dataframe containing all the independet variables of the test data 
(use in case there is a suspicion that the test set contains categories not present in X) 

For further theoretical details see Section 2.3.6 of *'On the encoding of categorical
variables for machine learning applications'.*

## Requirements (developed and tested under)
- Python 3.6
- Tensorflow 2.0
- Pandas 1.1.4
- Scikit-Learn 0.22.2
- Category Encoders 2.1.0 ([site](https://contrib.scikit-learn.org/categorical-encoding/))


## Authors

* **Erick GR** - [erickgrm](https://github.com/erickgrm)

## License

This project is licensed under the GNU GPLv3 License - see the [LICENSE](LICENSE) file for details
