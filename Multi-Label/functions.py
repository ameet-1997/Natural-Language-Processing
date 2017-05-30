from scipy.io.arff import loadarff
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
