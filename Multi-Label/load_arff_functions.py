from scipy import sparse



def load_from_arff(filename, labelcount, endian="big", input_feature_type='float', encode_nominal=True, load_sparse=False, return_attribute_definitions=False):
    """Method for loading ARFF files as numpy array
    Parameters
    ----------
    filename : string
        Path to ARFF file
    labelcount: integer
        Number of labels in the ARFF file
    endian: string{"big", "little"}
        Whether the ARFF file contains labels at the beginning of the attributes list ("big" endianness, MEKA format)
        or at the end ("little" endianness, MULAN format)
    input_feature_type: numpy.type as string
        The desire type of the contents of the return 'X' array-likes, default 'i8',
        should be a numpy type, see http://docs.scipy.org/doc/numpy/user/basics.types.html
    encode_nominal: boolean
        Whether convert categorical data into numeric factors - required for some scikit classifiers that can't handle non-numeric input featuers.
    load_sparse: boolean
        Whether to read arff file as a sparse file format, liac-arff breaks if sparse reading is enabled for non-sparse ARFFs.
    Returns
    -------
    X: scipy sparse matrix with ``input_feature_type`` elements,
    y: scipy sparse matrix of binary label indicator matrix
    """
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
