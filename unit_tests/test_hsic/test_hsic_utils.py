import pytest
import hsic.hsic_utils as hu
import numpy as np

#-------------------------------
# fixtures
#-------------------------------
# @pytest.fixture
# def some_fen(): return 0

#-------------------------------
# test_hu_approximate_set_lebesgue
#-------------------------------
def test__get_direct_sums_1():
    computed = hu._get_direct_sums(data_array=np.array([[1,0]]))
    expected = 1
    assert computed == expected
def test__get_direct_sums_2():
    computed = hu._get_direct_sums(data_array=np.array([[1,0,1,1]]))
    expected = 3
    assert computed == expected
def test__get_direct_sums_3():
    computed = hu._get_direct_sums(data_array=np.array([[0,0]]))
    expected = 0
    assert computed == expected
def test__get_direct_sums_4():
    computed = hu._get_direct_sums(data_array=np.array([[0,1],[1,0]]))
    expected = np.array([1,1])
    assert np.array_equal(computed, expected)
def test__get_direct_sums_5():
    computed = hu._get_direct_sums(data_array=np.array([[1,1,0],[1,0,0],[1,1,1]]))
    expected = np.array([2,1,3])
    assert np.array_equal(computed, expected)
#-------------------------------
def test__get_gram_matrix_1():
    computed = hu._get_gram_matrix(binary_data_array=np.array([[1,0]]))
    expected = np.array([[1]])
    assert np.array_equal(computed, expected)
def test__get_gram_matrix_2():
    computed = hu._get_gram_matrix(binary_data_array=np.array([[1,1]]))
    expected = np.array([[2]])
    assert np.array_equal(computed, expected)
def test__get_gram_matrix_3():
    computed = hu._get_gram_matrix(binary_data_array=np.array([[0,0]]))
    expected = np.array([[0]])
    assert np.array_equal(computed, expected)
def test__get_gram_matrix_4():
    computed = hu._get_gram_matrix(binary_data_array=np.array([[1]]))
    expected = np.array([[1]])
    assert np.array_equal(computed, expected)
def test__get_gram_matrix_5():
    computed = hu._get_gram_matrix(binary_data_array=np.array([[1,0,1]]))
    expected = np.array([[2]])
    assert np.array_equal(computed, expected)
def test__get_gram_matrix_6():
    computed = hu._get_gram_matrix(binary_data_array=np.array([[0,1,1]]))
    expected = np.array([[2]])
    assert np.array_equal(computed, expected)
#-------------------------------
def test__get_XOR_count_1():
    M = np.array([[1,0],[1,1]])
    computed = hu._get_XOR_count(binary_direct_sums=hu._get_direct_sums(M),
                                 pairwise_AND_count=hu._get_gram_matrix(M))
    expected = np.array([[0,1],[1,0]])
    assert np.array_equal(computed, expected)
def test__get_XOR_count_2():
    M = np.array([[1,0,0,0,1,0],[1,1,1,1,0,0]])
    computed = hu._get_XOR_count(binary_direct_sums=hu._get_direct_sums(M),
                                pairwise_AND_count=hu._get_gram_matrix(M))
    expected = np.array([[0,4],[4,0]])
    assert np.array_equal(computed, expected)
def test__get_XOR_count_3():
    M = np.array([[1,0,0],[1,1,1],[0,1,0]])
    computed = hu._get_XOR_count(binary_direct_sums=hu._get_direct_sums(M),
                                 pairwise_AND_count=hu._get_gram_matrix(M))
    expected = np.array([[0,2,2],[2,0,2],[2,2,0]])
    assert np.array_equal(computed, expected)
#-------------------------------
def test_approximate_set_lebesgue_1():
    M = np.array([[1,0],[1,1]])
    lambda_X = 1
    computed = hu.approximate_set_lebesgue(binary_system_output_data=M, lambda_X=lambda_X)
    expected = lambda_X*(np.array([[0,1],[1,0]])/2)
    assert np.allclose(computed, expected)
def test_approximate_set_lebesgue_2():
    M = np.array([[1,0],[1,1]])
    lambda_X = 7
    computed = hu.approximate_set_lebesgue(binary_system_output_data=M, lambda_X=lambda_X)
    expected = lambda_X*(np.array([[0,1],[1,0]])/2)
    assert np.allclose(computed, expected)
def test_approximate_set_lebesgue_3():
    M = np.array([[1,0,0,0,1,0],[1,1,1,1,0,0]])
    lambda_X = 13
    computed = hu.approximate_set_lebesgue(binary_system_output_data=M, lambda_X=lambda_X)
    expected = lambda_X*(np.array([[0,4],[4,0]])/6)
    assert np.allclose(computed, expected)
def test_approximate_set_lebesgue_4():
    M = np.array([[1,0,0],[1,1,1],[0,1,0]])
    lambda_X = 11
    computed = hu.approximate_set_lebesgue(binary_system_output_data=M, lambda_X=lambda_X)
    expected = lambda_X*(np.array([[0,2,2],[2,0,2],[2,2,0]])/3)
    assert np.allclose(computed, expected)
def test_approximate_set_lebesgue_5():
    M = np.array([[0,0,0],[0,0,0],[0,0,0]])
    lambda_X = 11
    computed = hu.approximate_set_lebesgue(binary_system_output_data=M, lambda_X=lambda_X)
    expected = lambda_X*(np.array([[0,0,0],[0,0,0],[0,0,0]])/3)
    assert np.allclose(computed, expected)
def test_approximate_set_lebesgue_6():
    M = np.array([[0]])
    lambda_X = 11
    computed = hu.approximate_set_lebesgue(binary_system_output_data=M, lambda_X=lambda_X)
    expected = lambda_X*(np.array([[0]])/1)
    assert np.allclose(computed, expected)

#-------------------------------
# test_get_K_gamma
#-------------------------------
