import pytest
import hsic.hsic_utils as hu
import numpy as np
from scipy import stats

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
def test_get_K_gamma_1():
    binary_system_output_data = np.array([[1,0],[1,1]])
    test_domain = np.array([[0,1]])
    expected = np.array([[0.0, np.exp(-1)],
                        [np.exp(-1), 0.0]])
    computed = hu.get_K_gamma(binary_system_output_data=binary_system_output_data,
                              test_domain=test_domain)
    assert np.allclose(computed, expected)
def test_get_K_gamma_2():
    binary_system_output_data = np.array([[1,0],[1,1]])
    test_domain = np.array([[0,7]])
    expected = np.array([[0.0, np.exp(-1)],
                        [np.exp(-1), 0.0]])
    computed = hu.get_K_gamma(binary_system_output_data=binary_system_output_data,
                              test_domain=test_domain)
    assert np.allclose(computed, expected)
def test_get_K_gamma_3():
    binary_system_output_data = np.array([[1,0,0,0,1,0],[1,1,1,1,0,0]])
    test_domain = np.array([[0,1]])
    expected = np.array([[0.0, np.exp(-1)],
                        [np.exp(-1), 0.0]])
    computed = hu.get_K_gamma(binary_system_output_data=binary_system_output_data,
                              test_domain=test_domain)
    assert np.allclose(computed, expected)
def test_get_K_gamma_4():
    binary_system_output_data = np.array([[1,0,0,0,1,0],[1,1,1,1,0,0]])
    test_domain = np.array([[-9,13]])
    expected = np.array([[0.0, np.exp(-1)],
                        [np.exp(-1), 0.0]])
    computed = hu.get_K_gamma(binary_system_output_data=binary_system_output_data,
                              test_domain=test_domain)
    assert np.allclose(computed, expected)
def test_get_K_gamma_5():
    binary_system_output_data = np.array([[1,0,0],[1,1,1],[0,1,0]])
    test_domain = np.array([[0,1]])
    expected = np.array([[0.0, np.exp(-3/4), np.exp(-3/4)],
                         [np.exp(-3/4), 0.0, np.exp(-3/4)],
                         [np.exp(-3/4), np.exp(-3/4), 0.0]])
    computed = hu.get_K_gamma(binary_system_output_data=binary_system_output_data,
                              test_domain=test_domain)
    assert np.allclose(computed, expected)
def test_get_K_gamma_6():
    binary_system_output_data = np.array([[1,0,0],[1,1,1],[0,1,0]])
    test_domain = np.array([[-13,17]])
    expected = np.array([[0.0, np.exp(-3/4), np.exp(-3/4)],
                         [np.exp(-3/4), 0.0, np.exp(-3/4)],
                         [np.exp(-3/4), np.exp(-3/4), 0.0]])
    computed = hu.get_K_gamma(binary_system_output_data=binary_system_output_data,
                              test_domain=test_domain)
    assert np.allclose(computed, expected)

#-------------------------------
# test_sobolev_kernel_univar
#-------------------------------
def test_sobolev_kernel_univar_1():
    computed = hu.sobolev_kernel_univar(0,0)
    expected = 1.0+0.25+(1.0/12.0)
    assert computed == expected
def test_sobolev_kernel_univar_2():
    computed = hu.sobolev_kernel_univar(0.5, 0.5)
    expected = 1.0+(1.0/12.0)
    assert computed == expected
def test_sobolev_kernel_univar_3():
    computed = hu.sobolev_kernel_univar(-1.0, -1.0)
    expected = 1.0+2.25+(1.0/12.0)
    assert computed == expected
def test_sobolev_kernel_univar_4():
    computed = hu.sobolev_kernel_univar(-1.0, 1.0)
    expected = 1.0-0.75+(2.0-1.0+1.0/12.0)
    assert computed == expected

#-------------------------------
# test_consistency_between_looped_and_vectorized_get_K_U_sobolev
#-------------------------------
def test_get_K_U_sobolev_looped_vs_vectorized_1():
    n = 3
    u_cardinality = 2
    input_data = np.random.uniform(0, 1, size=(n, u_cardinality))
    ku_looped = hu.get_K_U_sobolev_looped(n=n, num_of_inputs=u_cardinality, input_data=input_data, which_input_one_hot='01')
    ku_vectorized = hu.get_K_U_sobolev_vectorized(n=n, num_of_inputs=u_cardinality, input_data=input_data, which_input_one_hot='01')
    assert np.allclose(ku_looped, ku_vectorized)
def test_get_K_U_sobolev_looped_vs_vectorized_2():
    n = 10
    u_cardinality = 5
    input_data = np.random.uniform(0, 1, size=(n, u_cardinality))
    ku_looped = hu.get_K_U_sobolev_looped(n=n, num_of_inputs=u_cardinality, input_data=input_data, which_input_one_hot='10000')
    ku_vectorized = hu.get_K_U_sobolev_vectorized(n=n, num_of_inputs=u_cardinality, input_data=input_data, which_input_one_hot='10000')
    assert np.allclose(ku_looped, ku_vectorized)
def test_get_K_U_sobolev_looped_vs_vectorized_3():
    n = 10
    u_cardinality = 5
    input_data = np.random.uniform(0, 1, size=(n, u_cardinality))
    ku_looped = hu.get_K_U_sobolev_looped(n=n, num_of_inputs=u_cardinality, input_data=input_data, which_input_one_hot='00100')
    ku_vectorized = hu.get_K_U_sobolev_vectorized(chunk_size=5, n=n, num_of_inputs=u_cardinality, input_data=input_data, which_input_one_hot='00100')
    assert np.allclose(ku_looped, ku_vectorized)
def test_get_K_U_sobolev_looped_vs_vectorized_4():
    n = 30
    u_cardinality = 5
    input_data = np.random.uniform(0, 1, size=(n, u_cardinality))
    ku_looped = hu.get_K_U_sobolev_looped(n=n, num_of_inputs=u_cardinality, input_data=input_data, which_input_one_hot='00001')
    ku_vectorized = hu.get_K_U_sobolev_vectorized(chunk_size=5, n=n, num_of_inputs=u_cardinality, input_data=input_data, which_input_one_hot='00001')
    assert np.allclose(ku_looped, ku_vectorized)
#-------------------------------
# test_consistency_between_looped_and_vectorized_calculate_hsic
#-------------------------------
def test_calculate_hsic_looped_vs_vectorized_1():
    n = 30
    m = 5
    u_cardinality = 5
    test_domain = np.array([[0,1]])
    input_data = np.random.uniform(0, 1, size=(n, u_cardinality))
    binary_system_output_data = np.random.choice([0,1], size=(n, m))
    ku_vectorized = hu.get_K_U_sobolev_vectorized(chunk_size=10, n=n, num_of_inputs=u_cardinality, input_data=input_data, which_input_one_hot='00001')
    k_gamma = hu.get_K_gamma(binary_system_output_data=binary_system_output_data, test_domain=test_domain)
    hsic_looped = hu.calculate_hsic_looped(K_U=ku_vectorized, K_gamma=k_gamma)
    hsic_vectorized = hu.calculate_hsic_vectorized(K_U=ku_vectorized, K_gamma=k_gamma)
    assert np.allclose(hsic_looped, hsic_vectorized)
def test_calculate_hsic_looped_vs_vectorized_2():
    n = 13
    m = 7
    u_cardinality = 3
    test_domain = np.array([[-19,17]])
    input_data = np.random.uniform(0, 1, size=(n, u_cardinality))
    binary_system_output_data = np.random.choice([0,1], size=(n, m))
    ku_vectorized = hu.get_K_U_sobolev_vectorized(chunk_size=10, n=n, num_of_inputs=u_cardinality, input_data=input_data, which_input_one_hot='010')
    k_gamma = hu.get_K_gamma(binary_system_output_data=binary_system_output_data, test_domain=test_domain)
    hsic_looped = hu.calculate_hsic_looped(K_U=ku_vectorized, K_gamma=k_gamma)
    hsic_vectorized = hu.calculate_hsic_vectorized(K_U=ku_vectorized, K_gamma=k_gamma)
    assert np.allclose(hsic_looped, hsic_vectorized)
def test_calculate_hsic_looped_vs_vectorized_3():
    n = 43
    m = 43
    u_cardinality = 3
    test_domain = np.array([[-19,17]])
    input_data = np.random.uniform(0, 1, size=(n, u_cardinality))
    binary_system_output_data = np.random.choice([0,1], size=(n, m))
    ku_vectorized = hu.get_K_U_sobolev_vectorized(chunk_size=10, n=n, num_of_inputs=u_cardinality, input_data=input_data, which_input_one_hot='010')
    k_gamma = hu.get_K_gamma(binary_system_output_data=binary_system_output_data, test_domain=test_domain)
    hsic_looped = hu.calculate_hsic_looped(K_U=ku_vectorized, K_gamma=k_gamma)
    hsic_vectorized = hu.calculate_hsic_vectorized(K_U=ku_vectorized, K_gamma=k_gamma)
    assert np.allclose(hsic_looped, hsic_vectorized)
def test_calculate_hsic_looped_vs_vectorized_4():
    n = 111
    m = 222
    u_cardinality = 3
    test_domain = np.array([[-19,17]])
    input_data = np.random.uniform(0, 1, size=(n, u_cardinality))
    binary_system_output_data = np.random.choice([0,1], size=(n, m))
    ku_vectorized = hu.get_K_U_sobolev_vectorized(chunk_size=30, n=n, num_of_inputs=u_cardinality, input_data=input_data, which_input_one_hot='010')
    k_gamma = hu.get_K_gamma(binary_system_output_data=binary_system_output_data, test_domain=test_domain)
    hsic_looped = hu.calculate_hsic_looped(K_U=ku_vectorized, K_gamma=k_gamma)
    hsic_vectorized = hu.calculate_hsic_vectorized(K_U=ku_vectorized, K_gamma=k_gamma)
    assert np.allclose(hsic_looped, hsic_vectorized)
#-------------------------------
# test_transform_logUnif_to_unitUnif
#-------------------------------
def test_transform_logUnif_to_unitUnif_1():
    min_u = 1e-3
    max_u = 1e3
    samples = np.array([min_u, max_u])
    computed = hu.transform_logUnif_to_unitUnif(min_u, max_u, samples)
    assert np.allclose(computed, [0.0, 1.0])
def test_transform_logUnif_to_unitUnif_2():
    min_u = 1e-2
    max_u = 1e2
    samples = np.array([np.sqrt(min_u * max_u)]) #geometric midpoint
    computed = hu.transform_logUnif_to_unitUnif(min_u, max_u, samples)
    assert np.allclose(computed, [0.5], atol=1e-12)
def test_transform_logUnif_to_unitUnif_3():
    min_u = 1e-3
    max_u = 1e3
    rng = np.random.default_rng(0)
    samples = np.exp(rng.uniform(np.log(min_u), np.log(max_u), size=100_000))
    computed = hu.transform_logUnif_to_unitUnif(min_u, max_u, samples)
    #mean and var tests
    assert abs(computed.mean() - 0.5) < 5e-3
    assert abs(computed.var() - 1.0 / 12.0) < 5e-3
    #kolmogorov–smirnov test against unif distrib in (0,1)
    _, p = stats.kstest(computed, "uniform")
    assert p > 0.05
def test_transform_logUnif_to_unitUnif_4():
    min_u = 1e-6
    max_u = 1e6
    eps = 1e-15
    samples = np.array([min_u*(1+eps), max_u*(1-eps)])
    computed = hu.transform_logUnif_to_unitUnif(min_u, max_u, samples)
    assert np.all(computed >= 0.0)
    assert np.all(computed <= 1.0)
def test_transform_logUnif_to_unitUnif_5():
    min_u = 1e-300
    max_u = 1e300
    rng = np.random.default_rng(2)
    samples = np.exp(rng.uniform(np.log(min_u), np.log(max_u), size=1000))
    computed = hu.transform_logUnif_to_unitUnif(min_u, max_u, samples)
    assert np.all(np.isfinite(computed))
    assert np.all(computed >= 0.0)
    assert np.all(computed <= 1.0)
def test_transform_logUnif_to_unitUnif_6():
    min_u = 1e-2
    max_u = 1e2
    samples = np.array([min_u*0.9, max_u*1.1])
    computed = hu.transform_logUnif_to_unitUnif(min_u, max_u, samples)
    assert np.all(computed >= 0.0)
    assert np.all(computed <= 1.0)
def test_transform_logUnif_to_unitUnif_7():
    min_u = 1e-3
    max_u = 1e3
    samples = np.logspace(np.log10(min_u), np.log10(max_u), 1000)
    computed = hu.transform_logUnif_to_unitUnif(min_u, max_u, samples)
    assert np.all(np.diff(computed) >= 0)
#-------------------------------
# test_transform_unif_to_unitUnif
#-------------------------------
def test_transform_unif_to_unitUnif_1():
    min_u = -5.0
    max_u = 5.0
    samples = np.array([min_u, max_u])
    computed = hu.transform_unif_to_unitUnif(min_u, max_u, samples)
    assert np.allclose(computed, [0.0, 1.0])
def test_transform_unif_to_unitUnif_2():
    min_u = 0.0
    max_u = 10.0
    samples = np.array([5.0]) #midpoint
    computed = hu.transform_unif_to_unitUnif(min_u, max_u, samples)
    assert np.allclose(computed, [0.5])
def test_transform_unif_to_unitUnif_3(): 
    min_u = -10.0
    max_u = 10.0
    rng = np.random.default_rng(1)
    samples = rng.uniform(min_u, max_u, size=100_000)
    computed = hu.transform_unif_to_unitUnif(min_u, max_u, samples)
    #mean and var tests
    assert abs(computed.mean() - 0.5) < 5e-3
    assert abs(computed.var() - 1.0 / 12.0) < 5e-3
    #kolmogorov–smirnov test against unif distrib in (0,1)
    _, p = stats.kstest(computed, "uniform")
    assert p > 0.05
def test_transform_unif_to_unitUnif_4(): 
    min_u = 0.0
    max_u = 1.0
    samples = np.array([-10.0, 10.0])
    computed = hu.transform_unif_to_unitUnif(min_u, max_u, samples)
    assert np.allclose(computed, [0.0, 1.0])
def test_transform_unif_to_unitUnif_5():
    min_u = 0.0
    max_u = 10.0
    samples = np.linspace(min_u, max_u, 1000)
    computed = hu.transform_unif_to_unitUnif(min_u, max_u, samples)
    assert computed.shape == samples.shape
def test_transform_unif_to_unitUnif_6():
    min_u = -5.0
    max_u = 5.0
    samples = np.linspace(min_u, max_u, 1000)
    computed = hu.transform_unif_to_unitUnif(min_u, max_u, samples)
    assert np.all(np.diff(computed) >= 0)
#-------------------------------
# test_transform_all_u_inputs
#-------------------------------
def test_transform_all_u_inputs_1():
    n = 100
    u_cardinality = 3
    u_arr = np.random.rand(n, u_cardinality)
    specs = [{'min': 1e-3, 'max': 1e3, 'distribution_type': 'log_uniform'},
            {'min': -1.0, 'max': 1.0, 'distribution_type': 'uniform'},
            {'min': 0.0, 'max': 10.0, 'distribution_type': 'uniform'}]
    computed = hu.transform_all_u_inputs(u_arr, specs)
    assert computed.shape == (n, u_cardinality)
    assert np.all(computed >= 0.0)
    assert np.all(computed <= 1.0)
def test_transform_all_u_inputs_2():
    n = 500
    rng = np.random.default_rng(3)
    u_arr = np.column_stack([np.exp(rng.uniform(np.log(1e-3), np.log(1e3), size=n)),
                            rng.uniform(-10, 10, size=n)])
    specs = [{'min': 1e-3, 'max': 1e3, 'distribution_type': 'log_uniform'},
            {'min': -10, 'max': 10, 'distribution_type': 'uniform'}]
    computed = hu.transform_all_u_inputs(u_arr, specs)
    assert np.all(computed >= 0.0)
    assert np.all(computed <= 1.0)
def test_transform_all_u_inputs_3():
    u_arr = np.random.rand(10, 2)
    specs = [{'min': 0, 'max': 1, 'distribution_type': 'uniform'}]
    with pytest.raises(AssertionError):
        hu.transform_all_u_inputs(u_arr, specs)
def test_transform_all_u_inputs_4():
    u_arr = np.random.rand(10, 1)
    specs = [{'min': 0, 'max': 1, 'distribution_type': 'gaussian'}]
    with pytest.raises(ValueError):
        hu.transform_all_u_inputs(u_arr, specs)
# def test_transform_all_u_inputs_5():
    