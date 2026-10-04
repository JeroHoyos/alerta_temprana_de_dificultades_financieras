# Crecimiento Empresas Colombianas


## Requisitos

- Python ≥ 3.13 
- [uv](https://docs.astral.sh/uv/) 
- Opcional: GPU NVIDIA con driver compatible con CUDA 13

## Uso

```bash
# Instalación de librerías
uv sync
```

En Windows y Linux se instala PyTorch con CUDA 13.0; sin GPU NVIDIA funciona igual en CPU, solo descarga más. En macOS se instala la versión CPU. Para verificar que PyTorch ve la GPU:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Si imprime una versión terminada en `+cpu`, volver a correr `uv sync`.

## Datos

[10.000 empresas más Grandes del País](https://www.datos.gov.co/Comercio-Industria-y-Turismo/10-000-Empresas-mas-Grandes-del-Pa-s/6cat-2gcs/about_data)

El notebook necesita la versión con los años 2021–2025 (50.000 filas):

```bash
curl -L -o "data/raw/10.000_Empresas_mas_Grandes_del_País.csv" "https://www.datos.gov.co/api/views/6cat-2gcs/rows.csv?accessType=DOWNLOAD"
```
