import numpy as np
from api import make_json_safe

def test_make_json_safe_for_int():
    resultat = make_json_safe('15')
    assert resultat == '15'

def test_make_json_safe_for_float():
    resultat = make_json_safe('1.5')
    assert resultat == '1.5'

def test_make_json_safe_for_Na():
    resultat = make_json_safe(np.nan)
    assert resultat == None

def test_make_json_safe_for_Inf():
    resultat = make_json_safe(np.inf)
    assert resultat == None