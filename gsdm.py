import pandas as pd
from gsdmm import MovieGroupProcess

def gsdmm_train(text_list,alpha,beta,iterations,number_of_topics=15):
    """funtion create an instance of mgp using MovieGroupProcess
    in: text_list and hyperparameters
    out: mgp model
    """
    vocab = set(x for doc in text_list for x in doc)
    n_terms = len(vocab)

    mgp = MovieGroupProcess(K=number_of_topics,
                            alpha=alpha,
                            beta=beta,
                            n_iters=iterations)

    y = mgp.fit(text_list, n_terms)

    
    return mgp

