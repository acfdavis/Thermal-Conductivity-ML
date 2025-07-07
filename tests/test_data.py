import os, sys
import pytest
import numpy as np
from src.data import parse_temperature

def test_parse_celsius():
    assert parse_temperature("100 C") == pytest.approx(373.15)

def test_parse_kelvin():
    assert parse_temperature("250K") == pytest.approx(250)

def test_parse_plain_number():
    assert parse_temperature("20") == pytest.approx(20)

def test_parse_fahrenheit():
    assert parse_temperature("32 F") == pytest.approx(273.15)
    
def test_parse_room_temperature():
    assert parse_temperature("room temperature") == pytest.approx(298.15)

def test_parse_none():
    assert np.isnan(parse_temperature(None))
