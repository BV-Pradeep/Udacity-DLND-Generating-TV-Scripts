### Generating TV Scripts

This project uses Recurrent Neural Networks(RNN) and Long - Short Term Networks (LSTM) to generated TV Scripts.
This is based on Pytorch Library.
I built a Recurrent Neural Network (i.e. RNN) that can be used to generate new TV scripts for the Seinfeld show.
My dataset consists of a subset of the Seinfeld dataset of scripts from 9 seasons.

The RNN is built on Pytorch, written in Python 3 and is presented via Jupyter Notebook. 
The RNN was trained on a cloud-based GPU.

Note: the generated TV script output content is still fairly nonsensical.

The following are some of the steps I took to build this RNN:

##Preprocessing

Created a Lookup Table with two dictionaries (Word to ID and ID to Word) used for word embeddings
Split scripts into word arrays and implemented a function for tokenizing punctuation. The punctuation becomes like another word in the word array. This makes it easier for the RNN to predict the next word.

##Build the Neural Network

Implemented the following functions as core components for building the RNN
get_inputs: creates TF Placeholders for inputs, targets, and learning rate in the Neural Network
get_init_cell: build RNN cell and initialize; Stacked multiple LSTM layers.
get_embed: Applied word embedding to input_data, Return the embedded sequence.
build_rnn: Build the RNN using nn.rnn()
build_nn: Build the NN by calling functions get_embed, build_rnn. Apply FC layer with linear activation. Return logits, final_state.
get_batches: Create batches of input and targets as a Numpy array with shape (num_batches, 2, batch_size, seq_length)
Training the network

##Hyperparameters

epochs, batch size, rnn size, sequence length, learning rate

Training: Trained the neural network on the preprocessed data. 



