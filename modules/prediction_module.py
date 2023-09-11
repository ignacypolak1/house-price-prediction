import pandas as pd

from scipy.sparse import csr_matrix

from sklearn.model_selection import GridSearchCV
from typing import Dict

import json
import os

from joblib import load
from typing import Union

# Stałe obiekty
KEYS = ['bedrooms', 'bathrooms', 'm2_living',
        'm2_lot', 'floors', 'waterfront',
        'view', 'condition', 'grade',
        'm2_basement', 'zipcode', 'lat',
        'long', 'm2_living15', 'm2_lot15', 'age']

ATTRIBUTES = ['Ilość sypialni', 'Ilość łazienek', 'Powierzchnia mieszkalna (m\N{SUPERSCRIPT TWO})',
        'Powierzchnia działki (m\N{SUPERSCRIPT TWO})', 'Ilość pięter', 'Czy nad nabrzeżem',
        'Ocena widoku (0-4)', 'Ocena kondycji (1-5)', 'Ocena konstrukcji (1-13)',
        'Powierzchnia pod ziemią (m\N{SUPERSCRIPT TWO})', 'Kod pocztowy', 'Szerokość geograficzna',
        'Długość geograficzna', 'Powierzchnia mieszkalna piętnastu sąsiadów (m\N{SUPERSCRIPT TWO})',
        'Powierzchnia działek piętnastu sąsiadów (m\N{SUPERSCRIPT TWO})', 'Wiek nieruchomości']

COLS_TO_STANDARIZE = ['bedrooms', 'bathrooms', 'm2_living',
                      'm2_lot', 'floors', 'view',
                      'condition', 'grade', 'm2_basement',
                      'lat', 'long', 'm2_living15',
                      'm2_lot15', 'age']

MODEL = load('saves/GBR.model')
SCALER = load('saves/Scaler.model')
ENCODER = load('saves/Encoder.model')

ENCODER_VALID_VALUES = [element.replace('zipcode_', '') for element in ENCODER.get_feature_names_out(['zipcode'])]

def validate_numeric(dictionary: dict,
                     key: str,
                     nonnegative: bool = True,
                     num_type: str = 'int',
                     boolean: bool = False) -> Union[int, float]:
    """Funkcja walidująca wartości numeryczne podane przez użytkownika w polach tekstowych

    Args:
        dictionary (dict): Słownik zawierający wartości podane przez użytkownika.
        key (str): Klucz w słowniku, pod którym znajduje się dana wartość.
        nonnegative (bool, optional): Czy liczba musi być dodatnia. Domyślnie True.
        num_type (str, optional): Typ, w którym zostanie zwrócona liczba - "int" albo "float". Domyślnie 'int'.
        boolean (bool, optional): Czy liczba może mieć tylko wartości 0 lub 1. Domyślnie False.

    Returns:
        Union[int, float]: Liczba przekonwertowana do typu int albo float.
    """
    try:
        value = 0

        if(num_type=='int'):
            value = int(dictionary[key].get_text().replace(',', '.'))
        elif(num_type=='float'):
            value = float(dictionary[key].get_text().replace(',', '.'))
        else:
            raise Exception('Parametr num_type nie jest prawidłowy - jego dostępne wartości to "int" oraz "float"')

        if boolean:
            if value not in [0,1]:
                raise Exception(f'Wartość klucza "{key}" powinna wynosić 0 albo 1 - inne wartości nie są prawidłowe')
            else:
                pass

        if nonnegative:
            if value < 0:
                raise Exception(f'Wartość klucza "{key}" ({value}) jest mniejsza od zera')
            else:
                pass
        
        return value
    
    except ValueError:
        raise ValueError(f'Wartość klucza "{key}" nie została podana lub jest nieprawidłowa')
    except Exception as e:
        raise e


def validate_category(dictionary: dict,
                      key: str) -> str:
    """Funkcja walidująca wartości w postaci ciągu znaków

    Args:
        dictionary (dict): Słownik, w którym znajduje się dana wartość.
        key (str): Klucz, pod którym w słowniku znajduje się dana wartość.

    Returns:
        str: Zwalidowana wartość.
    """
    try:
        
        value = str(dictionary[key].get_text())
        
        if value not in ENCODER_VALID_VALUES:
            raise Exception(f'Kod pocztowy o tej wartości nie jest znany enkoderowi ani modelowi')
        
        return value
    
    except ValueError:
        raise ValueError(f'Wartość klucza "{key}" nie została podana lub jest nieprawidłowa')
    except Exception as e:
        raise e
    
def create_df_from_dict(dct: dict) -> pd.DataFrame:
    """Funkcja zwracająca ramkę danych, utworzoną na podstawie słownika

    Args:
        dct (dict): Słownik na podstawie którego zostanie utworzona ramka danych.

    Returns:
        pd.DataFrame: Ramka danych utworzona na podstawie słownika.
    """
    return pd.DataFrame.from_dict(data=[dct])

def process_data(df: pd.DataFrame) -> csr_matrix:
    """Funkcja, która zakodowuje wartości kategoryczne, standaryzuje wartości numeryczne
       oraz przekształca ramkę danych do postaci macierzy CSR

    Args:
        df (pd.DataFrame): Ramka danych.

    Returns:
        csr_matrix: Macierz CSR utworzona na podstawie ramki danych.
    """
    df = encode_category_attributes(df)
    standarize_numeric_attributes(df)
    return conver_to_csr(df)

def encode_category_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Funkcja zakodowywująca wartości kategoryczne

    Args:
        df (pd.DataFrame): Ramka danych.

    Returns:
        pd.DataFrame: Ramka danych z zakodowanymi wartościami kategorycznymi.
    """
    category_values = ENCODER.transform(df[['zipcode']])
    category_names = ENCODER.get_feature_names_out(['zipcode'])

    df = pd.concat([
        df.select_dtypes(exclude='object'),
        pd.DataFrame(category_values, columns=category_names).astype(int)],
        axis=1)
    
    return df

def standarize_numeric_attributes(df: pd.DataFrame) -> None:
    """Funkcja standaryzująca wartości numeryczne

    Args:
        df (pd.DataFrame): Ramka danych.
    """
    df[COLS_TO_STANDARIZE] = SCALER.transform(df[COLS_TO_STANDARIZE])

def conver_to_csr(df: pd.DataFrame) -> csr_matrix:
    """Funkcja przekształcająca ramkę danych do postaci macierzy CSR

    Args:
        df (pd.DataFrame): Ramka danych.

    Returns:
        csr_matrix: Macierz CSR.
    """
    return csr_matrix(df)

def predict_value(csr: csr_matrix) -> int:
    """Funkcja zwracająca wynik predykcji dokonanej przez model

    Args:
        csr (csr_matrix): Macierz CSR.

    Returns:
        int: Wynik predykcji.
    """
    return int(MODEL.predict(csr)[0])
