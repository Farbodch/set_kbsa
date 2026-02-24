import numpy as np
from dolfin import conditional, Constant, le, FunctionSpace, project
from numeric_models.numeric_models_utils import model as nm_model


def gen_1D_diff_explicit(x_arr, params):   
    def _get_at_x(u):
        model = nm_model(model_type="diffusion_1d_explicit")
        model.P = params['P']
        model.mean = params['mu']
        model.std = params['std']
        model_output = model.diffuFen_expl(coefficients=u,scalarDiffuIdx=x_arr[0])
        if params['auto_mean']:
            params['g_ineq_c'] = np.mean(model_output)
        
        return int(model_output<=params['g_ineq_c'])
    return _get_at_x

def get_1D_diff_FEM(params, comm=None, local_uid=None):
    def _get_at_all_x(u):
        model = nm_model(model_type="diffusion_1d",
                        output_paraview=False,
                        P=params['P'], 
                        mean=params['mu'], 
                        std=params['std'],
                        mesh_directory=params['mesh_directory'],
                        mesh_num_of_steps=params['num_of_mesh_steps'],
                        comm=comm,
                        local_uid=local_uid)

        model.diffuFen(coefficients=u, reset=True)
        model_output = model.diffuFen_curr_u
        # if params['auto_mean']:
        #     params['g_ineq_c'] = np.mean(model_output)
        # if params['return_bool']:
        #     return (model_output<=params['g_ineq_c']).astype(int)
        # else:
        return model_output
    return _get_at_all_x

def get_CDR(params, comm=None, local_uid=None):
    def _get_output_fens(u):
        model = nm_model(model_type='cdr',
                        output_paraview=False,
                        mesh_directory=params['mesh_directory'],
                        mesh_steps=params['mesh_steps'],
                        t_end_cdr=params['t_end'],
                        num_steps_cdr=params['num_steps'],
                        comm=comm,
                        local_uid=local_uid)
        model.get_cdr(coefficients=u, reset=True)

        if params['return_bool']:
            fuel_bool = project(conditional(le(model.fuel_field_t_now,params['g_ineq_c']['fuel']), Constant(1), Constant(0)), FunctionSpace(model.dolf_mesh, 'CG', 1))
            oxygen_bool = project(conditional(le(model.oxyxen_field_t_now,params['g_ineq_c']['oxygen']), Constant(1), Constant(0)), FunctionSpace(model.dolf_mesh, 'CG', 1))
            product_bool = project(conditional(le(model.product_field_t_now,params['g_ineq_c']['product']), Constant(1), Constant(0)), FunctionSpace(model.dolf_mesh, 'CG', 1))
            temp_bool = project(conditional(le(model.temp_field_t_now,params['g_ineq_c']['temp']), Constant(1), Constant(0)), FunctionSpace(model.dolf_mesh, 'CG', 1))
            return [fuel_bool, oxygen_bool, product_bool, temp_bool]
        else:
            fuel = model.fuel_field_t_now
            oxygen = model.oxyxen_field_t_now
            product = model.product_field_t_now
            temp = model.temp_field_t_now
            return [fuel, oxygen, product, temp]
    return _get_output_fens