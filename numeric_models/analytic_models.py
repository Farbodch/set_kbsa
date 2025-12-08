from numpy import (array as np_array, sin as np_sin)

def ishigami(input_data, a=7, b=0.1):
        """Define the Ishigami function."""
        u_1, u_2, u_3 = input_data
        return np_sin(u_1) + a * np_sin(u_2)*np_sin(u_2) + b * u_3*u_3*u_3*u_3 * np_sin(u_1)

def ishigami_vect_generator(rand_inputs_realization: list, a=7, b=0.1) -> float:
    u_1, u_2 = np_array(rand_inputs_realization, dtype=float)
    def ishigami_vect_fen(x):
        input_data = [u_1, u_2, x]
        return ishigami(a=a, b=b, input_data=input_data)
    return ishigami_vect_fen

def fellmann_function_generator(rand_inputs_realization: list):
    u_1, u_2 = rand_inputs_realization
    def fellmann_fen(x):
        x_1, x_2 = x
        return -x_1*x_1+5*x_2-u_1+u_2*u_2-1
    return fellmann_fen
