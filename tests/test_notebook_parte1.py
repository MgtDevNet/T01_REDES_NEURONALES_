import unittest
import importlib.util
import sys
from pathlib import Path
import numpy as np

ruta=Path(__file__).resolve().parents[1]/'punto_1/funcion1.py'
spec=importlib.util.spec_from_file_location('parte1_notebook_test',ruta)
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)


class NotebookGDTests(unittest.TestCase):
    def test_x0_en_ambos_problemas_y_dimensiones(self):
        for problema in ['rosenbrock','rastrigin']:
            for d,punto in [(2,[-1.2,1]),(3,[-1.2,1,.5])]:
                with self.subTest(problema=problema,dim=d):
                    r=m.optimize(problem=problema,dim=d,x0=punto,record=True,max_evals=100)
                    np.testing.assert_array_equal(r.frames[0],punto)
                    self.assertLessEqual(r.n_equiv,100)
                    self.assertEqual(r.history[0],m.PROBLEMS[problema]['func'](np.array(punto)))

    def test_x0_invalido_no_se_recorta(self):
        for punto in [[1],[99,0],[float('nan'),0]]:
            with self.assertRaises(ValueError):m.optimize(x0=punto)


if __name__=='__main__':unittest.main()
