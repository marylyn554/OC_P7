from api import make_json_safe

def test_make_json_safe_for_int():
    resultat = make_json_safe("15")
    assert result == 15

def test_make_json_safe_for_float():
    resultat = make_json_safe("1.5")
    assert result == 1.5

def test_make_json_safe_for_Na():
    resultat = make_json_safe(np.nan)
    assert result == None