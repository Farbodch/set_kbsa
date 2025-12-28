import pytest
import hsic.hsic_utils as hu
import numpy as np
from scipy import stats
from numeric_models.analytic_models import (ishigami_vectorized_generator as gen_ishigami, fellmann_function_generator as gen_fellmann)
from types import FunctionType

def fixed_uniform_midpoint(low, high, size):
    low = np.asarray(low)
    high = np.asarray(high)
    midpoint = (low + high) / 2
    return np.broadcast_to(midpoint, size)
def fixed_uniform_linspace(low, high, size):
        low = np.asarray(low)
        high = np.asarray(high)
        linspaces_grid = [np.linspace(low[j], high[j], size[0]) for j in range(size[1])]
        return np.column_stack(linspaces_grid)
#-------------------------------
# fixtures
#-------------------------------
@pytest.fixture
def fix_np_unif_rng_to_midpoint(monkeypatch):
    #force np_unif in hsic.hsic_utils to be fixed to midpoint of the domain for repeatable tests.
    def _fixed_uniform(low, high, size):
        return fixed_uniform_midpoint(low, high, size)
    monkeypatch.setattr("hsic.hsic_utils.np_unif", _fixed_uniform)
@pytest.fixture
def fix_np_unif_rng_to_linspace(monkeypatch):
    #force np_unif in hsic.hsic_utils to be fixed to linearly spaced points on the domain for repeatable tests.
    def _fixed_uniform(low, high, size):
        return fixed_uniform_linspace(low, high, size)
    monkeypatch.setattr("hsic.hsic_utils.np_unif", _fixed_uniform)
@pytest.fixture
def linear_process_fixture():
    #f(x) = u1 + u2 + x
    def gen_f(u):
        u1, u2 = u
        def f(x):
            return u1 + u2 + x
        return f
    return gen_f
@pytest.fixture
def quadratic_process_fixture():
    #f(x) = u1 * ||x||^2 + u2
    def gen_f(u):
        u1, u2 = u
        def f(x):
            return u1 * np.sum(x**2) + u2
        return f
    return gen_f
@pytest.fixture
def fixed_test_domain_1d():
    return np.array([[0.0, 2.0]])
@pytest.fixture
def fixed_test_domain_2d():
    return np.array([[0.0, 1.0], [-1.0, 1.0]])
@pytest.fixture
def fixed_u_2d():
    return np.array([1.0, 2.0])
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
#-------------------------------
# test_sample_analytical_function!!!!! CHANGE LINEAR_PROCESS_FIXTURE AND FIX THESE TESTCASES!!
#-------------------------------
# def test_sample_analytical_function_1(linear_process_fixture, fixed_test_domain_1d, fixed_u_2d):
#     m = 10
#     computed = hu.sample_analytical_function(u=fixed_u_2d,
#                                         process_generator=linear_process_fixture,
#                                         test_domain=fixed_test_domain_1d,
#                                         num_of_spatial_sampling_m=m)
#     assert isinstance(computed, np.ndarray),  f"{computed}"
#     assert computed.shape == (m,)
# def test_sample_analytical_function_2(linear_process_fixture, fixed_test_domain_2d, fixed_u_2d):
#     m = 5
#     computed_1 = hu.sample_analytical_function(u=fixed_u_2d,
#                                             process_generator=linear_process_fixture,
#                                             test_domain=fixed_test_domain_2d,
#                                             num_of_spatial_sampling_m=m,
#                                             rng=fixed_uniform_midpoint)
#     computed_2 = hu.sample_analytical_function(u=fixed_u_2d,
#                                             process_generator=linear_process_fixture,
#                                             test_domain=fixed_test_domain_2d,
#                                             num_of_spatial_sampling_m=m,
#                                             rng=fixed_uniform_midpoint)
#     assert np.array_equal(computed_1, computed_2)
# def test_sample_analytical_function_3(linear_process_fixture, fixed_test_domain_1d, fixed_u_2d):
#     g_constraint = 5.0
#     m = 20
#     computed = hu.sample_analytical_function(u=fixed_u_2d,
#                                         process_generator=linear_process_fixture,
#                                         test_domain=fixed_test_domain_1d,
#                                         num_of_spatial_sampling_m=m,
#                                         g_constraint=g_constraint,
#                                         rng=fixed_uniform_midpoint)

#     assert computed.dtype == np.uint8
#     assert set(np.unique(computed)).issubset({0, 1})
# def test_sample_analytical_function_4(linear_process_fixture, fixed_test_domain_1d):
#     u = np.array([1.0, 1.0])
#     g_constraint = 10.0
#     m = 10
#     computed = hu.sample_analytical_function(u=u,
#                                         process_generator=linear_process_fixture,
#                                         test_domain=fixed_test_domain_1d,
#                                         num_of_spatial_sampling_m=m,
#                                         g_constraint=g_constraint,
#                                         rng=fixed_uniform_midpoint)
#     computed_1 = hu.sample_analytical_function(u=u,
#                                         process_generator=linear_process_fixture,
#                                         test_domain=fixed_test_domain_1d,
#                                         num_of_spatial_sampling_m=m,
#                                         rng=fixed_uniform_midpoint)
#     assert np.all(computed == 1), f"w_g:{computed}, no_g:{computed_1}"
# def test_sample_analytical_function_5(quadratic_process_fixture, fixed_test_domain_2d, fixed_u_2d):
#     m = 10
#     computed = hu.sample_analytical_function(u=fixed_u_2d,
#                                         process_generator=quadratic_process_fixture,
#                                         test_domain=fixed_test_domain_2d,
#                                         num_of_spatial_sampling_m=m,
#                                         rng=fixed_uniform_midpoint)
#     assert np.all(computed >= fixed_u_2d[1])

# @pytest.mark.parametrize("domain", [np.array([[0.0, 1.0]]), np.array([[0.0, 1.0], [0.0, 1.0]]), np.array([[0.0, 1.0], [-1.0, 1.0], [2.0, 3.0]]),])
# def test_sample_analytical_function_6(linear_process_fixture, domain):
#     u = np.array([1.0, 2.0])
#     computed = hu.sample_analytical_function(u=u,
#                                     process_generator=linear_process_fixture,
#                                     test_domain=domain,
#                                     num_of_spatial_sampling_m=5,
#                                     rng=fixed_uniform_midpoint)
#     assert computed.shape == (5,)
# def test_sample_analytical_function_7():
#     with pytest.raises(TypeError):
#         hu.sample_analytical_function(u=np.array([1.0, 2.0]), 
#                                     process_generator=None)
# def test_sample_analytical_function_8(linear_process_fixture, fixed_test_domain_1d):
#     u = np.array([1.0, 2.0])
#     m=0
#     computed = hu.sample_analytical_function(u=u,
#                                     process_generator=linear_process_fixture,
#                                     test_domain=fixed_test_domain_1d,
#                                     num_of_spatial_sampling_m=m)
#     assert computed.size == 0
#-------------------------------
# test_sample_analytical_function_ishigami
#-------------------------------
def test_sample_analytical_function_ishigami_1():
    u = np.array([1, 1])
    test_domain = np.array([[-1,1]])
    m = 3
    computed = hu.sample_analytical_function(u=u,
                                    process_generator=gen_ishigami,
                                    test_domain=test_domain,
                                    num_of_spatial_sampling_m=m,
                                    rng=fixed_uniform_linspace)
    expected = np.array([5.882132011203685, 5.797984912722895, 5.882132011203685])
    assert np.allclose(computed, expected)
def test_sample_analytical_function_ishigami_2():
    u = np.array([-2, 2])
    test_domain = np.array([[-3,3]])
    m = 9
    computed = hu.sample_analytical_function(u=u,
                                    process_generator=gen_ishigami,
                                    test_domain=test_domain,
                                    num_of_spatial_sampling_m=m,
                                    rng=fixed_uniform_linspace)
    expected = np.array([-2.486853911091064, 2.548025395648797, 4.418123423866458,
                        4.849684507301303, 4.8784552461969595, 4.849684507301303,
                        4.418123423866458, 2.548025395648797, -2.486853911091064])
    assert np.allclose(computed, expected)
def test_sample_analytical_function_ishigami_3():
    u = np.array([1, 1])
    test_domain = np.array([[-1,1]])
    g_constraint = 10
    m = 3
    computed = hu.sample_analytical_function(u=u,
                                    process_generator=gen_ishigami,
                                    test_domain=test_domain,
                                    num_of_spatial_sampling_m=m,
                                    g_constraint=g_constraint,
                                    rng=fixed_uniform_linspace)
    expected = np.array([1, 1, 1])
    assert np.array_equal(computed, expected)
def test_sample_analytical_function_ishigami_4():
    u = np.array([1, 1])
    test_domain = np.array([[-1,1]])
    g_constraint = 0
    m = 3
    computed = hu.sample_analytical_function(u=u,
                                    process_generator=gen_ishigami,
                                    test_domain=test_domain,
                                    num_of_spatial_sampling_m=m,
                                    g_constraint=g_constraint,
                                    rng=fixed_uniform_linspace)
    expected = np.array([0, 0, 0])
    assert np.array_equal(computed, expected)
def test_sample_analytical_function_ishigami_5():
    u = np.array([-2, 2])
    test_domain = np.array([[-3,3]])
    g_constraint = 0
    m = 9
    computed = hu.sample_analytical_function(u=u,
                                    process_generator=gen_ishigami,
                                    test_domain=test_domain,
                                    num_of_spatial_sampling_m=m,
                                    g_constraint=g_constraint,
                                    rng=fixed_uniform_linspace)
    expected = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1])
    assert np.array_equal(computed, expected)
#-------------------------------
# test_sample_analytical_function_fellmann
#-------------------------------
def test_sample_analytical_function_fellmann_1():
    u = np.array([-2, 2])
    test_domain = np.array([[-3,3], [2, 2]])
    m = 5
    computed = hu.sample_analytical_function(u=u,
                                    process_generator=gen_fellmann,
                                    test_domain=test_domain,
                                    num_of_spatial_sampling_m=m,
                                    rng=fixed_uniform_linspace)
    expected = np.array([6.0, 12.75, 15.0, 12.75, 6.0])
    assert np.allclose(computed, expected)
def test_sample_analytical_function_fellmann_2():
    u = np.array([-4, 3])
    test_domain = np.array([[3, 3], [-5, 5]])
    m = 20
    computed = hu.sample_analytical_function(u=u,
                                    process_generator=gen_fellmann,
                                    test_domain=test_domain,
                                    num_of_spatial_sampling_m=m,
                                    rng=fixed_uniform_linspace)
    expected = np.array([-22.0, -19.36842105263158, -16.736842105263158, -14.105263157894736,
            -11.473684210526315, -8.842105263157897, -6.210526315789473, -3.578947368421053,
            -0.9473684210526336, 1.6842105263157876, 4.315789473684208, 6.947368421052628,
            9.578947368421053, 12.210526315789473, 14.842105263157894, 17.473684210526315,
            20.105263157894733, 22.736842105263158, 25.368421052631575, 28.0])
    assert np.allclose(computed, expected)
def test_sample_analytical_function_fellmann_3():
    u = np.array([-2, 2])
    g_constraint = 20
    test_domain = np.array([[-3,3], [2, 2]])
    m = 5
    computed = hu.sample_analytical_function(u=u,
                                    process_generator=gen_fellmann,
                                    test_domain=test_domain,
                                    num_of_spatial_sampling_m=m,
                                    g_constraint=g_constraint,
                                    rng=fixed_uniform_linspace)
    expected = np.array([1, 1, 1, 1, 1])
    assert np.array_equal(computed, expected)
def test_sample_analytical_function_fellmann_4():
    u = np.array([-2, 2])
    g_constraint = 0
    test_domain = np.array([[-3,3], [2, 2]])
    m = 5
    computed = hu.sample_analytical_function(u=u,
                                    process_generator=gen_fellmann,
                                    test_domain=test_domain,
                                    num_of_spatial_sampling_m=m,
                                    g_constraint=g_constraint,
                                    rng=fixed_uniform_linspace)
    expected = np.array([0, 0, 0, 0, 0])
    assert np.array_equal(computed, expected)