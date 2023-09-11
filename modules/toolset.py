import pandas as pd
import numpy as np

from scipy.stats import gaussian_kde, stats

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import seaborn as sns

def get_iqr_info(df: pd.DataFrame,
                 col: str,
                 mulitplier: float = 1.5) -> dict:
    
    """Funkcja obliczająca odstające wartości wartości (outliery) za pomocą kwartyli

    Args:
        df (pd.Dataframe): Ramka danych.
        col (str): Nazwa kolumny.
        threshold (float, optional): Próg. Domyślna wartość to 1.5.

    Returns:
        dict: Słownik zawierający wybrane informacje o odstających wartościach danej kolumny.
    """

    first_quartile = df[col].quantile(0.25)
    third_quartile = df[col].quantile(0.75)

    cutoff = (third_quartile - first_quartile) * mulitplier
    lower_limit = first_quartile - cutoff
    upper_limit = first_quartile + cutoff

    outliers_indexes = []

    lower = df[df[col] < lower_limit]
    upper = df[df[col] > upper_limit]

    lower_outliers_num = lower.shape[0]
    upper_outliers_num = upper.shape[0]

    outliers_indexes.extend(lower.index)
    outliers_indexes.extend(upper.index)

    return {
        'lower_limit': lower_limit,
        'upper_limit': upper_limit,
        'lower_outliers_num': lower_outliers_num,
        'upper_outliers_num': upper_outliers_num,
        'outliers_indexes': outliers_indexes
    }

def delete_outliers(df: pd.DataFrame,
                    col: str,
                    method: str = 'interquartile') -> None:
    
    """Funkcja usuwająca odstające wartości na podstawie z_score lub IQR

    Args:
        df (pd.DataFrame): Ramka danych z wartościami do usunięcia.
        col (str): Nazwa kolumny z której zostaną usunięte odstające wartości.
        method (str): Metoda - 'interquartile' albo 'zscore'. Domyślnie 'interquartile'.
    Returns:
        pd.DataFrame: Ramka danych zawierająca usunięte outliery.
    """
    df.reset_index(drop=True, inplace=True)

    outliers_interquartile_indexes = get_iqr_info(df=df, col=col)['outliers_indexes']
    outliers_zscore_indexes = get_z_info(df=df, col=col)['outliers_indexes']

    if method == 'interquartile':
        df.drop(outliers_interquartile_indexes, inplace=True)
        df.reset_index(drop=True, inplace=True)

    elif method == 'zscore':
        df.drop(outliers_zscore_indexes, inplace=True)
        df.reset_index(drop=True, inplace=True)



def plot_box(df: pd.DataFrame,
             col: str,
             ax,
             title: str = "PLACEHOLDER",
             labels: tuple[str, str] = ('x_label', 'y_label')) -> None:
    
    """Funkcja rysująca wykres pudełkowy

    Args:
        df (pd.DataFrame): Ramka danych.
        col (str): Nazwa kolumny.
        ax: Oś do narysowania wykresu.
        title (str, optional): Tytuł wykresu. Domyślnie "PLACEHOLDER".
        labels (tuple[str, str], optional): Nazwy osi w formacie (nazwa_x, nazwa_y).
    """

    ax.ticklabel_format(useOffset=False, style='plain')

    sns.boxplot(y=df[col], ax=ax)

    xlabel, ylabel = labels
    ax.set_ylabel(xlabel, fontsize=12)
    ax.set_xlabel(ylabel, fontsize=12)

    for label_x in ax.get_xticklabels():
        label_x.set_fontsize(10)
    for label_y in ax.get_yticklabels():
        label_y.set_fontsize(10)

    ax.set_title(title, fontsize=15)

    ax.grid(axis='both', visible=True, color='k', linestyle='--')
    ax.autoscale(tight=True)
    ax.set_ylim(auto=True)
    ax.set_xlim(auto=True)

def plot_dist(df: pd.DataFrame,
              col: str,
              ax,
              title: str = "PLACEHOLDER",
              labels: tuple[str, str] = ('x_label', 'y_label'),
              span_bottom: tuple[float, float] = (0, 0),
              span_upper: tuple[float, float] = (0, 0)) -> None:
   
    """Funkcja tworząca wykres dystrybucji wartości danej kolumny ramki wraz z limitami dotyczącymi
    odstających wartości

    Args:
        df (pd.DataFrame): Ramka danych.
        col (str): Kolumna.
        ax: Oś do narysowania wykresu.
        title (str, optional): Tytuł wykresu. Domyślnie "PLACEHOLDER".
        labels (tuple[str, str], optional): Nazwy osi w formacie (nazwa_x, nazwa_y).
        span_bottom (tuple[float, float], optional): Zakres dolnego przedziału.
        span_bottom (tuple[float, float], optional): Zakres górnego przedziału.
    """

    ax.ticklabel_format(useOffset=False, style='plain')

    sns.distplot(df[col], kde=False, ax=ax, color='green')

    ax.axvspan(xmin=span_bottom[0], xmax=span_bottom[1], alpha=0.5, color='blue', label='Dolne odstające wartości')
    ax.axvspan(xmin=span_upper[0], xmax=span_upper[1], alpha=0.5, color='red', label='Górne odstające wartości')

    ax.legend(fontsize = 12)

    xlabel, ylabel = labels
    ax.set_ylabel(xlabel, fontsize=12)
    ax.set_xlabel(ylabel, fontsize=12)
    
    for label_x in ax.get_xticklabels():
        label_x.set_fontsize(10)
    for label_y in ax.get_yticklabels():
        label_y.set_fontsize(10)

    ax.set_title(title, fontsize=15)
    ax.grid(axis='both', visible=True, color='k', linestyle='--')
    ax.autoscale(tight=True)
    ax.set_ylim(auto=True)
    ax.set_xlim(auto=True)

def get_z_info(df: pd.DataFrame,
                col: str,
                threshold: int = 3) -> dict:
    """Funkcja zwracająca informacje o wartościach odstających określonych za pomocą z-score

    Args:
        df (pd.DataFrame): Ramka danych.
        col (str): Nazwa kolumny.
        threshold (int, optional): Próg. Domyślnie to 3.

    Returns:
        dict: Słownik zawierający informacje o wartościach odstających.
    """
    
    df.reset_index(drop=True, inplace=True)

    z_score = []
    outliers_indexes = []

    mean = np.mean(df[col])
    std = np.std(df[col])

    lower_limit = mean - (threshold * std)
    upper_limit = mean + (threshold * std)

    for index, row in df.iterrows():

        z = (row[col]-mean)/std
        z_score.append(z)

        if np.abs(z) > threshold:
            outliers_indexes.append(index)

    return {
        'z_score': z_score,
        'outliers_indexes': outliers_indexes,
        'mean': mean,
        'std': std,
        'lower_limit': lower_limit,
        'upper_limit': upper_limit
    }


def analyze_and_remove_outliers(df: pd.DataFrame,
                                col: str,
                                labels_box: tuple[str, str] = ('x_label', 'y_label'),
                                labels_dist: tuple[str, str] = ('x_label', 'y_label'),
                                method: str = 'interquartile',
                                title: str = 'placeholder') -> None:
    """Funkcja analizująca wartości odstające oraz usuwająca je

    Args:
        df (pd.DataFrame): Ramka danych.
        col (str): Nazwa kolumny.
        labels_box (tuple[str, str], optional): Etykiety osi wykresów pudełkowych. Domyślnie ('x_label', 'y_label').
        labels_dist (tuple[str, str], optional): Etykiety osi wykresów dystrybucji. Domyślnie ('x_label', 'y_label').
        method (str, optional): Metoda usuwania wartości odstających. Domyślnie 'interquartile'.
        title (str, optional): Tytuł figury na której znajdują się wykresy. Domyślnie 'placeholder'.
    """
    
    plt.rcParams['font.size'] = 5


    fig, ax = plt.subplot_mosaic("AB;CC;DD")
    fig.set_size_inches((12,18))
    fig.suptitle(title, fontsize=25)

    span_bottom, span_upper = None, None

    if method == 'interquartile':
        outliers_info = get_iqr_info(df, col)

        span_bottom = (outliers_info['lower_limit'], df[col].min()) 
        span_upper = (outliers_info['upper_limit'], df[col].max())

        print(f'Dolna granica: {outliers_info["lower_limit"]}')
        print(f'Górna granica: {outliers_info["upper_limit"]}')
    
    elif method == 'zscore':
        outliers_info = get_z_info(df, col)

        span_bottom = (outliers_info['lower_limit'], df[col].min()) 
        span_upper = (outliers_info['upper_limit'], df[col].max())

        print(f'Dolna granica: {outliers_info["lower_limit"]}')
        print(f'Górna granica: {outliers_info["upper_limit"]}')

    plot_box(df, col, ax=ax['A'], title=f'Rozkład z odstającymi wartościami', labels=labels_box)
    plot_dist(df, col, ax=ax['C'], title=f'Rozkład z odstającymi wartościami', labels=labels_dist, span_bottom=span_bottom, span_upper=span_upper)

    delete_outliers(df=df, col=col, method=method)

    plot_box(df=df, col=col, ax=ax['B'], title=f'Rozkład bez odstających wartości', labels=labels_box)
    plot_dist(df=df, col=col, ax=ax['D'], title=f'Rozkład bez odstających wartości', labels=labels_dist, span_bottom=span_bottom, span_upper=span_upper)

def plot_heatmap(df: pd.DataFrame, threshold: float, fontsize: int, annotsize: int) -> None:
    """Funkcja rysująca macierz korelacji

    Args:
        df (pd.DataFrame): Ramka danych.
        threshold (float): Próg.
        fontsize (int): Rozmiar czcionki.
        annotsize (int): Rozmiar czcionki etykiet poszczególnych wartości korelacji.
    """

    ATTRIBUTES_COLUMNS = ['bedrooms', 'bathrooms', 'm2_living', 'm2_lot', 'floors', 'waterfront', 'view',
                      'condition', 'grade', 'm2_above', 'm2_basement', 'lat', 'long', 'm2_living15',
                      'm2_lot15', 'age']
    
    corr_matrix = df[['price'] + ATTRIBUTES_COLUMNS].corr()
    corr_matrix[corr_matrix < threshold] = np.nan

    fig, ax = plt.subplots()
    fig.set_size_inches((10, 10))

    sns.set(font_scale=2)
    sns.heatmap(corr_matrix, yticklabels=['price'] + ATTRIBUTES_COLUMNS, xticklabels= ['price'] + ATTRIBUTES_COLUMNS, fmt='.2f', annot=True, ax=ax, annot_kws={"size": annotsize})
        
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize) 

    ax.get_xaxis().tick_top()
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize, rotation=90)

def plot_correlations(df: pd.DataFrame, samples_num: int = 10000, figsize: tuple = (15,15), fontsize: int = 10) -> None:
    """Funkcja tworząca wykresy zależności ceny od poszczególnych cech

    Args:
        df (pd.DataFrame): Ramka danych.
        samples_num (int, optional): Ilość próbek. Domyślnie 10000.
        figsize (tuple, optional): Rozmiar figury wykresów w calach. Domyślnie (15,15).
        fontsize (int, optional): Rozmiar czcionki. Domyślnie 10.
    """
    
    columns = list(df.columns.copy())

    columns.remove('price')
    columns.remove('zipcode')

    LABELS = ['Ilość sypialni', 'Ilość łazienek', 'Powierzchnia mieszkalna (m\N{SUPERSCRIPT TWO})', 'Powierzchnia działki (m\N{SUPERSCRIPT TWO})',
              'Ilość pięter', 'Nad nabrzeżem', 'Ocena widoku', 'Ocena kondycji',
              'Ocena konstrukcji', 'Powierzchnia piwnicy (m\N{SUPERSCRIPT TWO})', 'Szerokość geo.', 'Długość geo.',
              'Powierzchnia mieszkalna - 15 sąsiadów (m\N{SUPERSCRIPT TWO})', 'Powierzchnia działki - 15 sąsiadów (m\N{SUPERSCRIPT TWO})', 'Wiek (lata)']

    LETTERS = ['A', 'B', 'C',
               'D', 'E', 'F',
               'G', 'H', 'I',
               'J', 'K', 'L',
               'M', 'N', 'O']

    fig, ax = plt.subplot_mosaic("ABC;DEF;GHI;JKL;MNO")
    fig.set_size_inches(figsize)
    
    for letter, column, label in zip(LETTERS, columns, LABELS):

        ax[letter].ticklabel_format(useOffset=False, style='plain')
        
        samples = df.sample(n=samples_num)     
        x=samples[column]
        y=samples['price']

        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        equation = f'y = {slope:.2f}x + {intercept:.2f}'

        sns.scatterplot(x=x, y=y, marker='p', c=z, s=100, ax=ax[letter])
        sns.regplot(x=x, y=y, scatter=False, color='red', ax=ax[letter], label=equation)
        
        
        ax[letter].set_ylabel('Cena (dolary amerykańskie)', fontsize=fontsize)
        ax[letter].set_xlabel(label, fontsize=fontsize)
        ax[letter].set_ylim(bottom=0)
        ax[letter].legend(loc='upper right', fontsize=fontsize)
        ax[letter].grid(axis='both', visible=True, color='k', linestyle='--')
        ax[letter].set_yticklabels(ax[letter].get_yticklabels(), fontsize=fontsize) 
        ax[letter].set_xticklabels(ax[letter].get_xticklabels(), fontsize=fontsize)
  

        fig.tight_layout()

def calculate_metrics(y_true, y_pred) -> dict:
    """Funkcja sprawdzająca wybrane metryki modeli

    Args:
        y_true: Wartości rzeczywiste.
        y_pred: Wartości przewidziane.

    Returns:
        dict: Słownik zawierający informacje.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "Średni błąd absolutny": mae,
        "Średni błąd kwadratowy": mse,
        "Współczynnik determinacji R2": r2
    }