"""
API FastAPI — Riesgo de crédito (Give Me Some Credit, Kaggle).

Expone el modelo @champion `mlops-credito-default` servido en Databricks Model Serving.
La API no carga el modelo: valida los datos de entrada y llama al endpoint de Databricks.
Si el modelo se reentrena y se promueve en Databricks, esta API no se modifica.

Credenciales: nunca se escriben en el código. Azure Container Apps las inyecta como
variables de entorno (DATABRICKS_URL y DATABRICKS_TOKEN) desde secretos.
"""
import os
from typing import List

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

VERSION = "1.0.0"
DATABRICKS_URL = os.environ.get("DATABRICKS_URL")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")
TIMEOUT_SEGUNDOS = 220  # margen para el cold start de Databricks (el ingress de Azure corta a los 240 s)

# Las 14 variables exactas del modelo (NB02), mismo nombre y tipo que su firma.
FEATURES = [
    "uso_credito_rotativo", "edad", "atrasos_30_59", "ratio_deuda",
    "ingreso_mensual", "lineas_credito_abiertas", "atrasos_90_mas",
    "creditos_hipotecarios", "atrasos_60_89", "dependientes",
    "ingreso_faltante", "total_atrasos", "ingreso_per_capita", "sobreendeudado",
]

app = FastAPI(
    title="API MLOps — Riesgo de crédito",
    description=(
        "Predice si un cliente caerá en mora grave (90+ días) en los próximos 2 años. "
        "Modelo @champion entrenado con Give Me Some Credit (Kaggle) y servido en Databricks."
    ),
    version=VERSION,
)


class Cliente(BaseModel):
    """Un cliente. Rangos tomados de la limpieza del NB01. Rechaza variables que el modelo no usa."""
    model_config = ConfigDict(extra="forbid")

    uso_credito_rotativo: float = Field(..., ge=0, le=2, description="Saldo / cupo de tarjetas y líneas rotativas", examples=[0.95])
    edad: int = Field(..., ge=18, le=110, examples=[28])
    atrasos_30_59: int = Field(..., ge=0, le=20, description="Veces con atraso de 30-59 días", examples=[3])
    ratio_deuda: float = Field(..., ge=0, description="Pagos mensuales de deuda / ingreso mensual", examples=[0.85])
    ingreso_mensual: float = Field(..., ge=0, examples=[2500.0])
    lineas_credito_abiertas: int = Field(..., ge=0, examples=[4])
    atrasos_90_mas: int = Field(..., ge=0, le=20, description="Veces con atraso de 90 días o más", examples=[2])
    creditos_hipotecarios: int = Field(..., ge=0, examples=[0])
    atrasos_60_89: int = Field(..., ge=0, le=20, description="Veces con atraso de 60-89 días", examples=[1])
    dependientes: int = Field(..., ge=0, examples=[3])
    ingreso_faltante: int = Field(..., ge=0, le=1, description="1 si el ingreso original no estaba reportado", examples=[0])
    total_atrasos: int = Field(..., ge=0, le=60, description="atrasos_30_59 + atrasos_60_89 + atrasos_90_mas", examples=[6])
    ingreso_per_capita: float = Field(..., ge=0, description="ingreso_mensual / (dependientes + 1)", examples=[625.0])
    sobreendeudado: int = Field(..., ge=0, le=1, description="1 si uso rotativo > 0.8 o ratio de deuda > 1", examples=[1])


class Lote(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clientes: List[Cliente] = Field(..., min_length=1, max_length=500)


def _llamar_databricks(registros: list) -> list:
    """Envía los registros al endpoint de Databricks y devuelve la lista de predicciones."""
    if not DATABRICKS_URL or not DATABRICKS_TOKEN:
        raise HTTPException(status_code=503, detail="Faltan DATABRICKS_URL o DATABRICKS_TOKEN en el entorno.")
    try:
        resp = requests.post(
            DATABRICKS_URL,
            headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"},
            json={"dataframe_records": registros},
            timeout=TIMEOUT_SEGUNDOS,
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"No se pudo contactar el endpoint de Databricks: {e}")
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Databricks respondió {resp.status_code}: {resp.text[:300]}")
    return [int(p) for p in resp.json().get("predictions", [])]


def _etiqueta(pred: int) -> str:
    return "Mora (alto riesgo)" if pred == 1 else "Al día (bajo riesgo)"


# 1) Bienvenida
@app.get("/")
def inicio():
    return {"mensaje": "API de riesgo de crédito activa", "documentacion": "/docs", "version": VERSION}


# 2) Estado del servicio (no llama a Databricks: responde rápido)
@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": VERSION,
        "modelo": "mlops-credito-default@champion",
        "databricks_configurado": bool(DATABRICKS_URL and DATABRICKS_TOKEN),
        "endpoint_databricks": DATABRICKS_URL.split("/serving-endpoints/")[-1].split("/")[0] if DATABRICKS_URL else None,
    }


# 3) Variables que espera el modelo
@app.get("/features")
def features():
    return {"features": FEATURES, "n_features": len(FEATURES), "objetivo": "mora_2anios (1 = mora 90+ días en 2 años)"}


# 4) Predicción para un cliente
@app.post("/predecir")
def predecir(cliente: Cliente):
    preds = _llamar_databricks([cliente.model_dump()])
    return {"predictions": preds, "resultado": _etiqueta(preds[0])}


# 5) Predicción por lote
@app.post("/predecir/lote")
def predecir_lote(lote: Lote):
    preds = _llamar_databricks([c.model_dump() for c in lote.clientes])
    return {"predictions": preds, "resultados": [_etiqueta(p) for p in preds]}
