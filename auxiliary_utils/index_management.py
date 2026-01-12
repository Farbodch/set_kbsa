from numpy import binary_repr as np_binary_repr

# unit_testing -> TRANSFER!.
def flip_str(str: str) -> str:
    input_string = str

    #----------- error check(s) -----------
    if (type(input_string) is not type('')):
        raise ValueError(f'Cannot parse a {type(input_string)}. String input required.')
    #--------------------------------------

    tmpStr = ''
    for idx in range(len(input_string)-1,-1,-1):
        tmpStr += input_string[idx]
    return tmpStr

# unit_testing -> TRANSFER!.
def direct_binstr_sum(str: str) -> int:
    """...direct sum of a binary string..."""
    input_string = str

    #----------- error check(s) -----------
    if (type(input_string) is not type('')):
        raise ValueError(f'Cannot parse a {type(input_string)}. String input required.')
    #--------------------------------------

    sum = 0
    for s in input_string:
        if s not in ['0', '1']:
            raise ValueError(f'Input needs to be in a binary-string format. Received {input_string}, where {s} is non-binary.')
        sum += int(s)
    return sum

# unit_testing -> TRANSFER!.
def get_index_superset(vectSize: int, 
                    include_zero: bool = False, 
                    only_singletons: bool = False,
                    higher_order: bool = True) -> list:
        """Generate the index superset for Pick-and-Freeze Sobol sensitivity indices."""

        #----------- error check(s) -----------
        if type(vectSize) is not int:
            raise ValueError(f'vectSize must be an integer type. {type(vectSize)} was passed in.')
        if vectSize <= 0:
            raise ValueError(f'vectSize must be >=1. {vectSize} was passed in.')
        if type(include_zero) is not bool:
            raise ValueError(f'include_zero must be a bool type. {type(include_zero)} was passed in.')
        if type(only_singletons) is not bool:
            raise ValueError(f'only_singletons must be a bool type. {type(only_singletons)} was passed in.')
        #--------------------------------------

        supset = []
        cardinOfSupset = 2 ** vectSize
        for i in range(cardinOfSupset):
            if i == 0 and not include_zero:
                continue
            oneHotIdxSet = np_binary_repr(i, vectSize)
            if only_singletons:
                if direct_binstr_sum(oneHotIdxSet) > 1:
                    continue
            if not higher_order:
                if (direct_binstr_sum(oneHotIdxSet) > 1) and (direct_binstr_sum(oneHotIdxSet) != vectSize):
                    continue
            supset.append(oneHotIdxSet)
        return supset

# unit_testing -> TRANSFER!.
def get_index_subsets(indexStr: str) -> list:
    """Generate all the unordered 1-hot-encoded list of subsets of the 1-hot-encoded indexStr. E.g. if indexStr = '101' then getIndexSubets returns the list ['000', '100', '001', '101']"""
    
    #----------- error check(s) -----------
    if (type(indexStr) is not type('')):
        raise ValueError(f'Cannot parse a {type(indexStr)}. String input required.')
    for s in indexStr:
        if s not in ['0', '1']:
            raise ValueError(f'Input needs to be in a binary-string format. Received {indexStr}, where {s} is non-binary.')
    #--------------------------------------

    ones_idx = [i for i, singleBit in enumerate(indexStr) if singleBit == '1']
    k = len(ones_idx)
    subsets = []
    # loop over all 2^k "masks", using the integer mask as a collection of k binary flags, one for each position where the idxA string had a '1'.
    for mask in range(2**k):
        # print(bin(mask))
        # build a fresh list of '0's
        t = ['0'] * len(indexStr)
        # loop over indices of the ones_idx list
        for j in range(k):
            # for each bit in mask, if it's 1 we save the '1' in t:
            #   Let a,b,x,y be non-negative integers:
            #   a >> b: right-shift the bits in the binary-representation of a by b digits (eg 1010 >> 1 == 101). 
            #   x & y operator: bitwise compare binary-representation of x and y (eg 1011 & 1101 == 1001; 101 & 1 == 1; 001 & 1 == 0; 110 & 1 == 0).
            # This means we're checking the least-significant bit of the binary-representation of the j-shifted mask integer.
            shiftMaskBinaryBy = j
            if (mask >> shiftMaskBinaryBy) & 1:
                t[ones_idx[j]] = '1'
        subsets.append(''.join(t))
    return subsets

# unit_testing -> TRANSFER!.
def get_index_complement(idxBinString: str) -> str:
        """Get the complement of an index binary string."""

        #----------- error check(s) -----------
        if (type(idxBinString) is not type('')):
            raise ValueError(f'Cannot parse a {type(idxBinString)}. String input required.')
        for s in idxBinString:
            if s not in ['0', '1']:
                raise ValueError(f'Input needs to be in a binary-string format. Received {idxBinString}, where {s} is non-binary.')
        #--------------------------------------

        return ''.join('1' if bit == '0' else '0' for bit in idxBinString)

# unit_testing -> TRANSFER!.
def get_singleton_index_as_int(idxBinString: str) -> int:
    """Convert the one-hot-encoding format of the idxBinString into an integer index value, read from left to right."""

    #----------- error check(s) -----------
    if (type(idxBinString) is not type('')):
            raise ValueError(f'Cannot parse a {type(idxBinString)}. String input required.')
    for s in idxBinString:
        if s not in ['0', '1']:
            raise ValueError(f'Input needs to be in a binary-string format. Received {idxBinString}, where {s} is non-binary.')
    if direct_binstr_sum(idxBinString) != 1:
        raise ValueError(f'One and only one index in the one-hot-encoded input, corresponding to the index-integer of interest, needs to be 1 (e.g. 100 -> 0, 0001 -> 3). {idxBinString} was passed in.')
    #--------------------------------------

    # assert direct_binstr_sum(idxBinString) == 1 ### DEPRECATED
    for i in range(len(idxBinString)):
        if idxBinString[i] == "1":
            return i
    return -1

# unit_testing -> ???!.
def get_u_index_superset_onehot(dim_of_U, higher_order=False):
    return [flip_str(i) for i in sorted(get_index_superset(dim_of_U, higher_order=higher_order), key=lambda x: direct_binstr_sum(x))]

# unit_testing -> ???!.
#adapted from "Compute the lexicographically next bit permutation"
#Bit Twiddling Hacks By Sean Eron Anderson
#https://graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation, last accessed Jan 7, 2026
def generator_order_r_idcs_as_onehot(r: int, d: int):
    """creates a generator generator that one contains all of the one-hot-encoded d-dimensional indices of order r, where one can iterate through. 
    E.g., generator(r=1,d=3) will contain the strings '001', '010', '100'.

    Args:
        r (int): order of indices.
        d (int): dimension of indices.

    Yields:
        generator: generator that contains all of the one-hot-encoded d-dimensional indices of order r.
    """
    #lowest r bits set
    x = (1 << r) - 1
    limit = 1 << d
    while x < limit:
        yield format(x, f'0{d}b')
        c = x & -x
        r_ = x + c
        x = (((r_ ^ x) >> 2) // c) | r_