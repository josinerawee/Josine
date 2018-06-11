import numpy as np
import utils
import sys

def mk_training_matrices(pairs, en_dimension, cat_dimension, semanticspace, catalan_space):
    en_mat = np.zeros((len(pairs),en_dimension)) 
    cat_mat = np.zeros((len(pairs),cat_dimension))
    c = 0
    for p in pairs:
        en_word,cat_word = p.split()
        en_mat[c] = semanticspace[en_word]   
        cat_mat[c] = catalan_space[cat_word]   
        c+=1
    return en_mat,cat_mat



if len(sys.argv) == 4:
    space=sys.argv[1]
    if space=='reducedcolors':
        semanticspace=utils.readDM("data/reducedcolors.dm")s
    if space =='fullcolors':
        semanticspace=utils.readDM("data/full.dm")
    word = sys.argv[2]
    num_neighbours = int(sys.argv[3])
    print(utils.neighbours(semanticspace, semanticspace[word],num_neighbours))
    english_neighbours = utils.neighbours(semanticspace,semanticspace[word],num_neighbours)
    utils.run_PCAneighbours(semanticspace,[word]+english_neighbours,"english_neighbours"+word+".png")