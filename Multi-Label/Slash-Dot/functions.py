import numpy as np
from scipy import sparse
import arff

def load_from_arff(filename, labelcount, endian="big", input_feature_type='float', encode_nominal=True, load_sparse=False, return_attribute_definitions=False):
    matrix = None
    if not load_sparse:
        arff_frame = arff.load(
            open(filename, 'r'), encode_nominal=encode_nominal, return_type=arff.DENSE)
        matrix = sparse.csr_matrix(
            arff_frame['data'], dtype=input_feature_type)
    else:
        arff_frame = arff.load(
            open(filename, 'r'), encode_nominal=encode_nominal, return_type=arff.COO)
        data = arff_frame['data'][0]
        row = arff_frame['data'][1]
        col = arff_frame['data'][2]
        matrix = sparse.coo_matrix(
            (data, (row, col)), shape=(max(row) + 1, max(col) + 1))

    X, y = None, None

    if endian == "big":
        X, y = matrix.tocsc()[:, labelcount:].tolil(), matrix.tocsc()[
            :, :labelcount].astype(int).tolil()
    elif endian == "little":
        X, y = matrix.tocsc()[
            :, :-labelcount].tolil(), matrix.tocsc()[:, -labelcount:].astype(int).tolil()
    else:
        # unknown endian
        return None

    if return_attribute_definitions:
        return X, y, arff_frame['attributes']
    else:
        return X, y

def fill_bool_array(bool_array, label_list, y_train):
    for i in range(1,3):
        for j in label_list[i]:
            bool_array[i] = np.logical_or(np.array(bool_array[i]), np.array(y_train[:,j] == 1).flatten())
            # Flatten is being used to convert multi-dimensional array to single dimension
    return bool_array

def fit_classifiers(classifiers, bool_array, label_list, X_train, y_train):
    temp_y_train = np.zeros(shape=(y_train.shape[0], 4))
    for i in range(y_train.shape[0]):
        temp_y_train[i,3] = y_train[i,20]
        temp_y_train[i,2] = y_train[i,8]
        answer=False
        for j in label_list[2]:
            answer = answer or y_train[i,j]
        temp_y_train[i,1] = int(answer)

        answer=False
        for j in label_list[1]:
            answer = answer or y_train[i,j]
        temp_y_train[i,0] = int(answer)
    classifiers[0].fit(X_train[bool_array[0],:], temp_y_train[bool_array[0],:])

    for i in range(1,3):
        temp_y_train = y_train[:,label_list[i]]
        temp_y_train = temp_y_train[bool_array[i],:]
        # classifiers[i].fit(X_train[bool_array[i],:], y_train[bool_array[i], label_list[i]])
        classifiers[i].fit(X_train[bool_array[i],:], temp_y_train)
    return classifiers