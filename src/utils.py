"""Rutas del proyecto y funciones de carga y limpieza que se repiten en los notebooks."""

import urllib.request
from pathlib import Path

import pandas as pd

# Raíz del proyecto: así las rutas funcionan igual desde cualquier carpeta
ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "10.000_Empresas_mas_Grandes_del_País.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "empresas_limpias.parquet"
MODEL_PATH = ROOT / "data" / "processed" / "dataset_modelo.parquet"
FIGURES_DIR = ROOT / "figures"

# Descarga directa del conjunto de datos en datos.gov.co
RAW_URL = "https://www.datos.gov.co/api/views/6cat-2gcs/rows.csv?accessType=DOWNLOAD"

# Nombres en inglés, después de pasar las columnas a snake_case
COLUMN_NAMES = {
    "razon_social": "company_name",
    "departamento_domicilio": "department",
    "ciudad_domicilio": "city",
    "ingresos_operacionales": "operating_revenue",
    "ganancia_perdida": "net_income",
    "total_activos": "total_assets",
    "total_pasivos": "total_liabilities",
    "total_patrimonio": "total_equity",
    "ano_de_corte": "year",
}

# Variables categóricas
CAT_COLS = ["supervisor", "region", "department", "city", "ciiu", "macrosector"]

# Variables financieras (en billones de COP)
FINANCIAL_COLS = [
    "operating_revenue",
    "net_income",
    "total_assets",
    "total_liabilities",
    "total_equity",
]


def load_raw() -> pd.DataFrame:
    """Lee el CSV original. Si no está en data/raw/, primero lo descarga de datos.gov.co."""
    if not RAW_PATH.exists():
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(RAW_URL, RAW_PATH)
    return pd.read_csv(RAW_PATH)


def load_clean() -> pd.DataFrame:
    """Lee los datos limpios que genera 02_limpieza.ipynb."""
    return pd.read_parquet(CLEAN_PATH)


def save(df: pd.DataFrame, path: Path) -> None:
    """Guarda en parquet, que conserva los tipos (el NIT y el CIIU siguen siendo texto)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def strip_accents(s):
    """Quita tildes y diéresis (á -> a, ü -> u, ñ -> n). Sirve para Series e Index."""
    return (
        # Separar cada letra de su tilde (á -> a + ´)
        s.str.normalize("NFKD")
        # Pasar a ASCII descartando las tildes sueltas y volver a texto
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
    )


def normalize_text(s: pd.Series) -> pd.Series:
    """Texto sin espacios en los bordes, sin tildes y en mayúsculas."""
    return strip_accents(s.astype(str).str.strip()).str.upper()


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Pasa las columnas a snake_case, sin tildes ni signos, y las traduce al inglés."""
    snake_case = (
        strip_accents(df.columns.str.strip().str.lower())
        # Eliminar signos de puntuación y reemplazar espacios por guiones bajos
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
    )
    return df.set_axis(snake_case, axis=1).rename(columns=COLUMN_NAMES)


def parse_amounts(s: pd.Series) -> pd.Series:
    """Convierte cifras en texto ('$ 1,234.56', '-$ 0.82') a número. Lo que no se pueda queda vacío."""
    # Quitar el signo de pesos, las comas de miles y los espacios
    clean_text = s.astype(str).str.replace(r"[$,\s]", "", regex=True)
    return pd.to_numeric(clean_text, errors="coerce")


def explore_prefix_variants(series: pd.Series, prefix_len: int = 3, min_items: int = 2) -> dict:
    """Agrupa los valores únicos por sus primeras letras para detectar variantes del mismo nombre."""
    unique_vals = pd.Series(series.dropna().unique())
    variants = (
        unique_vals.groupby(unique_vals.str[:prefix_len])
        .agg(list)
        .loc[lambda s: s.str.len() >= min_items]
    )
    return variants.to_dict()


def is_distressed(net_income: pd.Series, equity: pd.Series) -> pd.Series:
    """1 si la empresa está en dificultad financiera: pérdidas netas o patrimonio negativo."""
    return ((net_income < 0) | (equity < 0)).astype(int)
