''' Entity Embedding Encoder

    author: github.com/erickgrm
'''
# Required libraries
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda, \
                                    Embedding, Reshape, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from category_encoders import OneHotEncoder

# Clerical
from .utilities import *
import os

class EntityEmbeddingEncoder(): 
    
    def __init__(self, epochs=100, dense_layers_sizes=(1000,500), dropout=True):
        super(EntityEmbeddingEncoder,self).__init__() 
        self.ee_layers_model = keras.Model()
        self.model = keras.Model()
        self.epochs = epochs
        self.batch_size = 256
        self.categorical_var_list = []
        self.ohencoders = {}
        self.ee_sizes = {}
        self.df = None
        self.dense_layers_sizes = dense_layers_sizes
        self.dropout = dropout

        # Delete old model_ee.h5 file from previous instances
        if os.path.isfile('model_ee.h5'):
            try:
                os.remove('model_ee.h5')
            except OSError as e:  
                print ("Error: %s - %s." % (e.filename, e.strerror))
        
    def fit(self, df, y, cat_cols=[], ee_sizes={}, verbose=True, test=pd.DataFrame([])):
        ''' Retrieves the model architecture and fits the encoding model
        '''
        # If vars in categorical_var_list are not categorical yet, make them
        # Scale numerical variables to [0,1]
        df = set_categories(df.copy(), cat_cols)
        if not(test.shape[0] == 0):
            test = set_categories(test.copy(), cat_cols)

        # Set which variables will be encoded and fit one-hot encoders for each 
        self.categorical_var_list, self.ohencoders = var_types(df, test)

        # Scale numerical vars to [0,1]
        self.df = scale_df(df)

        # Define architecture
        model, y = self.architecture(y,  ee_sizes, test) # creates a keras model
        
        # Training dataset
        X_train = self.burst_and_ohencode(self.df)
        y_train = np.array(y)
        
        # Before training, try to free up memory
        del self.df, y, test
        
        # Callbacks for early stopping and saving the best model
        early_stopping = keras.callbacks.EarlyStopping(patience=20)
        model_checkpoint = keras.callbacks.ModelCheckpoint('model_ee.h5', save_best_only=True)

        # Fit the tensorflow model
        model.fit(X_train,y_train, epochs=self.epochs, verbose=verbose, 
                  batch_size=self.batch_size, validation_split = 0.25, 
                  callbacks=[model_checkpoint, early_stopping])

        del model, X_train, y_train

        # Load best model, set encoding layers 'EE_layers'
        self.model = keras.models.load_model('model_ee.h5')
        self.ee_layers_model = keras.Model(inputs=self.model.input, 
                                           outputs=self.model.get_layer('C').output)

        return self
        
    def transform(self, df):
        df = scale_df(set_categories(df.copy(), self.categorical_var_list))
        X = self.burst_and_ohencode(df)
        return pd.DataFrame(self.ee_layers_model.predict(X))
    

    def fit_transform(self, df, y, cat_cols=[], ee_sizes={}, verbose=True, test=pd.DataFrame([])):
        self.fit(df, y, cat_cols, ee_sizes, verbose, test)
        return self.transform(df)
    

    def architecture(self, y, ee_sizes={}, test=pd.DataFrame([])):
        ''' Sets the architecture of the encoding model,
            depending on the dataset structure
        '''
        inputs = []
        ee_layers = []
        for x in self.df.columns:
            if x in self.categorical_var_list:
                # number of different categories in column
                if not(test.shape[0] == 0):
                    k = len(np.union1d(np.unique(self.df[x]), np.unique(test[x])))
                else: 
                    k = len(np.unique(self.df[x]))

                # Set the number of neurons for EE layer
                if x in ee_sizes: 
                    ee_dim = ee_sizes[x]
                else: 
                    # About the choice below, see Section A.3. of "On the encoding of 
                    # categorical variables for machine learning applications"
                    ee_dim = min(30, int(np.ceil((k+1)/3)))

                # Add EE Layer
                input = Input(shape=(1,), name='Var_'+str(x))
                output = Embedding(k+1, ee_dim,
                                   name='EEL'+str(x)+'_size_'+str(ee_dim))(input) 
                output = Reshape(target_shape=(ee_dim, ))(output)

                inputs.append(input)
                ee_layers.append(output)
            else:
                # Add identity layer when the column is numerical
                input = Input(shape=(1,),name='Var_'+str(x))
                output = Lambda(lambda x: x, name='Identity' + str(x))(input)
                inputs.append(input)
                ee_layers.append(output)
                
        # Append las two dense layers
        last_layers = Concatenate(name='C')(ee_layers)
        last_layers = Dense(self.dense_layers_sizes[0], activation='relu', 
                            name='Dense_size_'+str(self.dense_layers_sizes[0]))(last_layers)
        if self.dropout:
            last_layers = Dropout(0.3)(last_layers)
        last_layers = Dense(self.dense_layers_sizes[1], activation='relu', 
                            name='Dense_size_'+str(self.dense_layers_sizes[1]))(last_layers)
        if self.dropout:
            last_layers = Dropout(0.2)(last_layers)
        
        # Define output layer according to target variable
        if len(np.unique(y)) == 2: # binary classification

            last_layers = Dense(2, activation='softmax', name='Softmax')(last_layers)
            
            model = keras.Model(inputs=inputs, outputs=last_layers)
            model.compile(optimizer='adam', loss='binary_crossentropy', 
                          metrics=[keras.metrics.AUC()])
            
            y = OneHotEncoder().fit_transform(keras.utils.to_categorical(y))

        elif 2 < len(np.unique(y)) < 20: # if y has only a few values, treat them as classes

            last_layers = Dense(len(np.unique(y)), activation='softmax', 
                                name='Softmax')(last_layers)
            
            model = keras.Model(inputs=inputs, outputs=last_layers)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
             
            y = LabelEncoder().fit_transform(y)

        else:   # Otherwise, assume y is continuous
            last_layers = Dense(1, activation='sigmoid', name='Sigmoid')(last_layers)
            model = keras.Model(inputs=inputs, outputs=last_layers)
            model.compile(optimizer='adam', loss='mean_squared_error',
                          metrics=[keras.metrics.MeanSquaredError()])
            
            y = MinMaxScaler().fit_transform(y)

        return model, np.array(y)


    def plot_model(self):
        """ Plots the model used by the encoder
        """
        try:
            import graphviz
            import pydot
        except:# ImportError as e:
            print('graphviz or pydot module missing')

        keras.utils.plot_model(self.model)
        print('Plot saved as model.png.')
        print('To display it in a Jupyter notebook, type:')
        print('from IPython.display import Image\nImage(\'model.png\')')


    def burst_and_ohencode(self, df):
        """ Splits df into single columns, and applies the corresponding OHE
            to those that are categorical
        """
        X = []
        for x in df.columns:
            if x in self.categorical_var_list:
                X.append(self.ohencoders[x].transform(df[x]))
            else: 
                X.append(df[x].values)
        return X
