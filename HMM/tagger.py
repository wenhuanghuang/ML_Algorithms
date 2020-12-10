import numpy as np
from hmm import HMM

def model_training(train_data, tags):
    """
    Train an HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class 
            defined in data_process.py (read the file to see what attributes this class has)
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - model: an object of HMM class initialized with parameters (pi, A, B, obs_dict, state_dict) calculated 
            based on the training dataset
    """

    # unique_words.keys() contains all unique words
    unique_words = get_unique_words(train_data)
    
    word2idx = {}
    tag2idx = dict()
    S = len(tags)
    ###################################################
    #  build two dictionaries
    #   - from a word to its index 
    #   - from a tag to its index 
    # The order you index the word/tag does not matter, 
    # as long as the indices are 0, 1, 2, ...
    ###################################################
    for index, key in enumerate(unique_words):
        word2idx[key] = index
    for ind, tag in enumerate(tags):
        tag2idx[tag] = ind
    
    pi = np.zeros(S)
    A = np.zeros((S, S))
    B = np.zeros((S, len(unique_words)))
    ###################################################
    #  estimate pi, A, B from the training data.
    #   When estimating the entries of A and B, if  
    #   "divided by zero" is encountered, set the entry 
    #   to be zero.
    ###################################################
    
    for line in train_data:
        words = line.words
        tags = line.tags
        length = line.length
        pi[tag2idx[tags[0]]]+=1
        for i in range(length):   
            curr_word = words[i]
            curr_tag = tags[i]
            curr_w_index = word2idx[curr_word]
            curr_t_index = tag2idx[curr_tag] 
            B[curr_t_index, curr_w_index]+=1
            if i != length - 1:
                next_tag = tags[i+1]
                next_t_index = tag2idx[next_tag]
                A[curr_t_index, next_t_index]+=1
    pi = pi / np.sum(pi)
    A = A / (np.sum(A, axis=1).reshape(S, -1))
    B = B / np.sum(B, axis=1).reshape(S, -1)
    # print(A, B.shape)
    # DO NOT MODIFY BELOW
    model = HMM(pi, A, B, word2idx, tag2idx)
    return model
 	###################################################
	# Edit here
	###################################################

def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class
    - model: an object of the HMM class
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    ######################################################################
    # for each sentence, find its tagging using Viterbi algorithm.
    #    Note that when encountering an unseen word not in the HMM model,
    #    you need to add this word to model.obs_dict, and expand model.B
    #    accordingly with value 1e-6.
    ######################################################################
    for line in test_data:
        for word in line.words:
            if word not in model.obs_dict:
                length = len(model.obs_dict)
                model.obs_dict[word] = length 
                model.B = np.concatenate((model.B, np.zeros((model.B.shape[0], 1)) + 1e-6), axis=1)
        words = np.asarray(line.words)
        sequence= model.viterbi(words)
        tagging.append(sequence)
    return tagging

# DO NOT MODIFY BELOW
def get_unique_words(data):

    unique_words = {}

    for line in data:
        for word in line.words:
            freq = unique_words.get(word, 0)
            freq += 1
            unique_words[word] = freq

    return unique_words
