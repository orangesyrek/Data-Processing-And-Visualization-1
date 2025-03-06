#!/usr/bin/env python3
"""
IZV cast1 projektu
Autor: xpauli08

Detailni zadani projektu je v samostatnem projektu e-learningu.
Nezapomente na to, ze python soubory maji dane formatovani.

Muzete pouzit libovolnou vestavenou knihovnu a knihovny predstavene na prednasce
"""

from bs4 import BeautifulSoup
import requests
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import List, Callable, Dict, Any


def integrate(f: Callable[[NDArray], NDArray], a: float, b: float, steps=1000) -> float:
    x = np.linspace(a, b, steps)
    y = f(x)
    dx = (b - a) / (steps - 1) # steps - 1, jelikoz zaciname od i=1
    return np.sum((dx * (y[1:] + y[:-1]) / 2))


def generate_graph(a: List[float], show_figure: bool = False, save_path: str | None = None):
    x = np.linspace(-3, 3, 200)
    a = np.array(a).reshape(-1, 1) # zmena a na numpy pole, abychom mohli provest broadcasting
    y = a**2 * x**3 * np.sin(x)

    integrals = np.trapz(y, x) # vypocet jednotlivych integralu

    plt.figure().set_figwidth(10) # nastaveni vetsi velikosti x osy

    # vykresleni jednotlivych krivek i s popisky
    for i in range(len(a)):
        plt.plot(x, y[i], label=f'$y_{{{a[i][0]}}}(x)$')
        plt.fill_between(x, y[i], alpha=0.1)
        plt.text(x[-1], y[i][-1], f'$\\int f_{{{a[i][0]}}}(x)dx = {round(integrals[i], 2)}$')

    # prodlouzeni osy x, aby se vesly popisky
    plt.xlim(-3, 4.5)
    plt.ylim(0, 40)

    # nastaveni labels a legendy
    plt.xlabel('$x$')
    plt.ylabel('$f_a(x)$')
    plt.legend(loc="upper center", ncol=len(a), bbox_to_anchor=(0.5, 1.15))

    # zobrazeni roszahu pouze od -3 do 3
    plt.xticks(np.arange(-3, 4, 1))

    if save_path is not None:
        plt.savefig(save_path)

    if show_figure:
        plt.show()


def generate_sinus(show_figure: bool = False, save_path: str | None = None):

    # tvorba funkci
    t = np.linspace(0, 100, 15000)
    f1 = 0.5 * np.cos(1/50 * np.pi * t)
    f2 = 0.25 * (np.sin(np.pi * t) + np.sin(3/2 * np.pi * t))
    f_sum = f1 + f2

    # tvorba podgrafu
    fig, axs = plt.subplots(3, figsize=(8,10))

    # podgraf 1
    axs[0].plot(t, f1)
    axs[0].set_ylabel('$f_1(t)$')

    # podgraf 2
    axs[1].plot(t, f2)
    axs[1].set_ylabel('$f_2(t)$')

    # podgraf 3
    filter = f_sum > f1 # zjisteni, ktere hodnoty f_sum jsou vyssi
    f_sum[~filter] = np.nan # vynulovani hodnot, kde je f_sum nizsi (abychom predesli prekryti v miste, kde se hodnoty rovnaji)
    axs[2].plot(t, f_sum, color="g") # vykresleni 1. casti grafu

    f_sum = f1 + f2 # obnoveni puvodni funkce f_sum
    f_sum[filter] = np.nan # vynulovani hodnot, kde je f_sum nizsi (abychom predesli prekryti v miste, kde se hodnoty rovnaji)
    axs[2].plot(t, f_sum, color="r") # vykresleni 2. casti grafu
    
    axs[2].set_ylabel('$f_1(t) + f_2(t)$')

    # nastaveni spolecna pro vsechny podgrafy
    for ax in axs:
        # rozsah y osy
        ax.set_ylim([-0.8, 0.8])
        ax.set_yticks(np.arange(-0.8, 1.2, 0.4))

        # rozsah x osy
        ax.set_xlim([0,100])
        ax.set_xticks(np.arange(0, 120, 20))

        # popisky x osy
        ax.set_xlabel("t")

    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    if show_figure:
        plt.show()


def download_data() -> List[Dict[str, Any]]:
    url = 'https://ehw.fit.vutbr.cz/izv/st_zemepis_cz.html' # na teto url je tabulka
    response = requests.get(url)
    response.encoding = 'utf-8' # nastaveni encodingu
    soup = BeautifulSoup(response.text, 'html.parser')

    table = soup.find_all('table')[-1] # zajima nas posledni tabulka
    rows = table.find_all('tr')


    data = []
    i = 2
    for row in rows[1:]:  # preskocime hlavicku
        cols = row.find_all('td')
        record = {
            'position': cols[0].text.strip(),
            'lat': float(cols[2].text.strip()[:-1].replace(",", ".")), # odstraneni symbolu stupnu, nahrazeni carky teckou
            'long': float(cols[4].text.strip()[:-1].replace(",", ".")), # stejne jako vyse
            'height': float(cols[6].text.strip().replace(",", ".")) # obdobny postup
        }
        data.append(record)

    return data
