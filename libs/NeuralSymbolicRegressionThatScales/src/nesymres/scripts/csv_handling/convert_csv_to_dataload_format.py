import pandas as pd 
import numpy as np
from torch.multiprocessing import Manager
import click
import warnings
from tqdm import tqdm
import json
import os
from nesymres.dataset import generator
import time
import signal
from nesymres import dclasses
from pathlib import Path
import pickle
from sympy import lambdify
from nesymres.utils import create_env, H5FilesCreator
from nesymres.utils import code_unpickler, code_pickler
import copyreg
import types
from itertools import chain
import traceback
import sympy as sp
from nesymres.dataset.sympy_utils import add_multiplicative_constants, add_additive_constants
import h5py


class Pipepile:
    def __init__(self, env: generator.Generator, is_timer=False):
        self.env = env
        manager = Manager()
        self.cnt = manager.list()
        self.is_timer = is_timer
        self.fun_args = ",".join(chain(list(env.variables),env.coefficients))

    def handler(self,signum, frame):
        raise TimeoutError

    def return_training_set(self, i):
        np.random.seed(i)
        while True:
            try:
                res = self.create_lambda(np.random.randint(2**32-1))
                assert type(res) == dclasses.Equation
                return res
            except TimeoutError:
                signal.alarm(0)
                continue
            except generator.NotCorrectIndependentVariables:
                signal.alarm(0)
                continue
            except generator.UnknownSymPyOperator:
                signal.alarm(0)
                continue
            except generator.ValueErrorExpression:
                signal.alarm(0)
                continue
            except generator.ImAccomulationBounds:
                signal.alarm(0)
                continue

        

    def convert_lambda(self, i, variables, support) -> dclasses.Equation:
        sym = self.env.infix_to_sympy(i, self.env.variables, self.env.rewrite_functions)
        placeholder = {x:sp.Symbol(x, real=True,nonzero=True) for x in ["cm","ca"]}
        constants_expression = sym
        consts_elemns = {}
        infix = str(constants_expression)
        eq = lambdify(self.fun_args,constants_expression,modules=["numpy"])
        res = dclasses.Equation(expr=infix, code=eq.__code__, coeff_dict=consts_elemns, variables=variables)
        return res

@click.command()
@click.option("--folder_csv", default="data/benchmark")
def converter(folder_csv):
    csv_availables = ["nc_old.csv"]
    for file_csv in csv_availables:
        path_csv = os.path.join(folder_csv,file_csv)
        validation = pd.read_csv(path_csv)
        copyreg.pickle(types.CodeType, code_pickler, code_unpickler) #Needed for serializing code objects
        env, param, config_dict = create_env("config.json")
        folder_path = Path("data/validation")
        folder_path.mkdir(parents=True, exist_ok=True)
        h5_creator = H5FilesCreator(target_path=folder_path)
        env_pip = Pipepile(env, is_timer=False)

        res = []
        for idx in range(len(validation)):
            gt_expr = validation.iloc[idx]["eq"]
            gt_expr = gt_expr.replace("pow","Pow")
            variables = list(eval(validation.iloc[idx]["support"]).keys())
            support = validation.iloc[idx]["support"]
            curr = env_pip.convert_lambda(gt_expr,variables,support) 
            res.append(curr)
    
    print("Finishing generating set")
    h5_creator.create_single_hd5_from_eqs(("0", res))
    dataset = dclasses.DatasetDetails(
                            config=config_dict, 
                            total_coefficients=env.coefficients, 
                            total_variables=list(env.variables), 
                            word2id=env.word2id, 
                            id2word=env.id2word,
                            una_ops=env.una_ops,
                            bin_ops=env.una_ops,
                            rewrite_functions=env.rewrite_functions,
                            total_number_of_eqs=len(res),
                            eqs_per_hdf=len(res),
                            generator_details=param)

    t_hf = h5py.File(os.path.join(folder_path, "metadata.h5") , 'w')
    t_hf.create_dataset("other", data=np.void(pickle.dumps(dataset)))
    t_hf.close()
    


if __name__ == "__main__":
    converter()