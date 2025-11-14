import argparse, time
# from kbsa.kbsa_main import run_kbsa, get_kbsa_datafile_name
# import numpy as np
from utils import numeric_models as nm

def main():

    print("Creating CDR Model")
    cdr_model = nm.model(model_type='cdr', 
                    t_end_cdr=0.2, 
                    num_steps_cdr=2, 
                    output_paraview=False,
                    mesh_2D_dir='data/CDR/mesh_save_dir/rectangle.xdmf', 
                    mesh_steps=0.025)
    cdr_model.output_paraview = True

    print("Beginning Simulation.")
    t_start = time.time()
    cdr_model.get_cdr(reset=True, t_end=0.2, num_steps=1000)
    t_end = time.time()
    print("Simulation Done.")
    print(f"Execution time: {t_end - t_start:.6f} seconds")
    # parser = argparse.ArgumentParser(description="...")
    # parser.add_argument("--variable_name", default="default_value", help="tooltip-here")
    # args = parser.parse_args()
    # print(f"{args.variable_name}")
    return 0

if __name__ == "__main__":
    main()