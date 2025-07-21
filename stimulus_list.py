#!/usr/bin/env python3
"""
build_stimulus_list.py
----------------------

Crea StimulusList.csv con un número arbitrario de radicales.

Pasos:
1) Lee Unihan para radical, pinyin y frecuencia SUBTLEX-CH.
2) Cruza con CC-CEDICT para obtener traducción 1-a-1 (sense disambig heurística).
3) Aplica filtros opcionales (--zipf-min, --n-per-radical).
4) Devuelve CSV listo para el notebook + un log detallado.
"""

import argparse, json, re, random
from pathlib import Path
import pandas as pd
from opencc import OpenCC
from pypinyin import pinyin, Style

cc = OpenCC("t2s")

SUBTLEX_PATH = Path("subtlex_ch_zipf.txt")      # hanzi\tzipf
CEDICT_PATH  = Path("cedict_ts.u8")             # CC-CEDICT
UNIHAN_RAD_PATH = Path("Unihan_DictionaryLikeData.txt")
UNIHAN_READ_PATH = Path("Unihan_Readings.txt")

def load_zipf(path: Path) -> pd.DataFrame:
    """
    Lee SUBTLEX-CH (carácter, frecuencia) con detección flexible de encoding.
    """
    tried = ("utf-8", "utf-16", "utf-16le", "gb18030", "latin1")
    for enc in tried:
        try:
            df = pd.read_csv(path, sep=r"[\t,]",  # tab o coma
                             names=["hanzi", "zipf"],
                             engine="python",
                             encoding=enc,
                             encoding_errors="replace")
            # asegúrate de que la primera columna son caracteres chinos
            if df["hanzi"].str.len().lt(5).all():
                print(f"✔  Leído con encoding='{enc}'  ({len(df)} filas)")
                return df.set_index("hanzi")
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "Fallo al decodificar SUBTLEX-CH. Prueba otra codificación o revisa el archivo."
    )

def load_unihan_radicals() -> dict[str, str]:
    rad = {}
    with open(UNIHAN_RAD_PATH, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("U+"):  # format: U+4E00 kRSUnicode 1.0
                code, tag, data = line.strip().split(None, 2)
                if tag == "kRSUnicode":
                    char = chr(int(code[2:], 16))
                    radical = data.split(".")[0]  # before .alt
                    rad[char] = radical
    return rad

def load_cedict() -> dict[str, str]:
    eng = {}
    pattern = re.compile(r"^(\S+)\s+(\S+)\s+\[(.+?)\]\s+/(.+)/")
    with open(CEDICT_PATH, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"): continue
            m = pattern.match(line)
            if m:
                trad, simp, p, gloss = m.groups()
                glosses = gloss.split("/")
                word = cc.convert(simp)
                if len(word) == 1 and re.match(r"[a-zA-Z]", glosses[0]):
                    eng[word] = glosses[0]
    return eng

def build_dataset(n_per_radical: int, zipf_min: float):
    zipf = load_zipf()
    radicals = load_unihan_radicals()
    english = load_cedict()

    rows = []
    grouped = {}
    for char, rad in radicals.items():
        if char not in zipf.index or zipf.at[char, "zipf"] < zipf_min:
            continue
        if char not in english:  # skip if no simple translation
            continue
        grouped.setdefault(rad, []).append(char)

    for rad, chars in grouped.items():
        random.shuffle(chars)
        sample = chars[:n_per_radical]
        for ch in sample:
            pin = "".join(s[0] for s in pinyin(ch, style=Style.TONE3))
            rows.append({
                "hanzi": ch,
                "pinyin": pin,
                "radical": rad,
                "english_consensus": english[ch],
                "zipf_cn": zipf.at[ch, "zipf"],
                "zipf_en": None,                 # rellenar con SUBTLEX-US si se desea
                "concreteness_en": None          # rellenar con Brysbaert si se desea
            })

    df = pd.DataFrame(rows)
    return df.sort_values(["radical", "hanzi"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=80,
        help="número de caracteres por radical (default 80)")
    parser.add_argument("--zipf-min", type=float, default=4.0,
        help="umbral mínimo Zipf SUBTLEX-CH (default 4.0)")
    parser.add_argument("--out", type=Path, default=Path("StimulusList.csv"))
    args = parser.parse_args()

    df = build_dataset(args.n, args.zipf_min)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"✅  Guardado {len(df)} filas ({len(df['radical'].unique())} radicales) en {args.out}")

if __name__ == "__main__":
    main()

