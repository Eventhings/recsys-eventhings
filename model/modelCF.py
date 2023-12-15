import pickle
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Multiply, Dropout, Dense, BatchNormalization, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L1, L2

from utils import *

#------------------------------------------------------------------------------------------------------------------------------#


def model(users, items):
    # HYPERPARAMS
    latent_features = 20
    learning_rate = 0.005

    # TENSORFLOW GRAPH
    # Using the functional API

    # Define input layers for user, item, and label.
    user_input = Input(shape = (1,), dtype = tf.int32, name = 'user')
    item_input = Input(shape = (1,), dtype = tf.int32, name = 'item')
    label_input = Input(shape = (1,), dtype = tf.int32, name = 'label')

    # User embedding for MLP
    mlp_user_embedding = Embedding(input_dim = len(users), 
                                output_dim = latent_features,
                                embeddings_initializer = 'random_normal',
                                embeddings_regularizer = L1(0.01),
                                input_length = 1, 
                                name = 'mlp_user_embedding')(user_input)

    # Item embedding for MLP
    mlp_item_embedding = Embedding(input_dim = len(items), 
                                output_dim = latent_features,
                                embeddings_initializer = 'random_normal',
                                embeddings_regularizer = L1(0.01),
                                input_length = 1, 
                                name = 'mlp_item_embedding')(item_input)

    # User embedding for GMF
    gmf_user_embedding = Embedding(input_dim = len(users), 
                                output_dim = latent_features,
                                embeddings_initializer = 'random_normal',
                                embeddings_regularizer = L1(0.01),
                                input_length = 1, 
                                name = 'gmf_user_embedding')(user_input)

    # Item embedding for GMF
    gmf_item_embedding = Embedding(input_dim = len(items), 
                                output_dim = latent_features,
                                embeddings_initializer = 'random_normal',
                                embeddings_regularizer = L1(0.01),
                                input_length = 1, 
                                name = 'gmf_item_embedding')(item_input)

    # GMF layers
    gmf_user_flat = Flatten()(gmf_user_embedding)
    gmf_item_flat = Flatten()(gmf_item_embedding)
    gmf_matrix = Multiply()([gmf_user_flat, gmf_item_flat])

    # MLP layers
    mlp_user_flat = Flatten()(mlp_user_embedding)
    mlp_item_flat = Flatten()(mlp_item_embedding)
    mlp_concat = Concatenate()([mlp_user_flat, mlp_item_flat])

    mlp_dropout = Dropout(0.1)(mlp_concat)

    mlp_layer_1 = Dense(64, 
                        activation = 'relu', 
                        name = 'mlp_layer1')(mlp_dropout)
    mlp_batch_norm1 = BatchNormalization(name = 'mlp_batch_norm1')(mlp_layer_1)
    mlp_dropout1 = Dropout(0.1, 
                        name = 'mlp_dropout1')(mlp_batch_norm1)

    mlp_layer_2 = Dense(32, 
                        activation = 'relu', 
                        name = 'mlp_layer2')(mlp_dropout1)
    mlp_batch_norm2 = BatchNormalization(name = 'mlp_batch_norm2')(mlp_layer_2)
    mlp_dropout2 = Dropout(0.1, 
                        name = 'mlp_dropout2')(mlp_batch_norm2)

    mlp_layer_3 = Dense(16, 
                        activation = 'relu', 
                        kernel_regularizer = L2(0.01),
                        name = 'mlp_layer3')(mlp_dropout2)
    mlp_layer_4 = Dense(8, 
                        activation = 'relu', 
                        activity_regularizer = L2(0.01),
                        name = 'mlp_layer4')(mlp_layer_3)

    # Merge the two networks
    merged_vector = Concatenate()([gmf_matrix, mlp_layer_4])

    # Output layer
    output_layer = Dense(1, 
                        activation = 'sigmoid',
                        kernel_initializer = 'lecun_uniform',
                        name = 'output_layer')(merged_vector)

    # Define the model
    modelCF = Model(inputs = [user_input, item_input], outputs = output_layer)

    # Compile the model with binary cross entropy loss and Adam optimizer
    optimizer = Adam(learning_rate = learning_rate)
    modelCF.compile(optimizer = optimizer,
                    loss = 'binary_crossentropy',
                    metrics = ['accuracy'])
    
    return modelCF


def train_model_CF():
    # Load dataset
    uids, iids, df_train, df_test, df_neg, users, items = load_dataset()
    
    # HYPERPARAMS
    num_neg = 4
    epochs = 4
    batch_size = 1024

    prev_modelCF = load_model('model/modelCF.h5')
    try:
        with open('model/performance/hitrates_avg_CF.pkl', 'rb') as f:
            best_hr = pickle.load(f)
        with open('model/performance/ndcgs_avg_CF.pkl', 'rb') as f:
            best_ndcgs = pickle.load(f)
    except:
        best_hr = 0
        best_ndcgs = 0

    curr_modelCF = model(users, items)
    for epoch in range(epochs):
        # Get our training input.
        user_input, item_input, labels = get_train_instances(uids, iids, items, num_neg)
    
        # Training        
        hist = curr_modelCF.fit([np.array(user_input), np.array(item_input)], #input
                                np.array(labels), # labels 
                                batch_size = batch_size, 
                                verbose = 0, 
                                shuffle = True)

        # Evaluation
        (hitrates, ndcgs) = evaluate(curr_modelCF, df_test, df_neg)
        hitrates_avg, ndcgs_avg, loss = np.array(hitrates).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
        if hitrates_avg > best_hr:
            curr_modelCF.save('model/modelCF.h5')
            
            with open('model/performance/hitrates_avg_CF.pkl', 'wb') as f:
                pickle.dump(hitrates_avg, f)

            with open('model/performance/ndcgs_avg_CF.pkl', 'wb') as f:
                pickle.dump(ndcgs_avg, f)

            print("Model has been updated.")