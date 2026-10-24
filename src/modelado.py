"""Métricas, búsqueda de hiperparámetros y red neuronal del sistema de alerta temprana."""

from functools import partial

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import PredefinedSplit
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optimizadores que Optuna puede elegir para la red
OPTIMIZERS = {
    "Adam": torch.optim.Adam,
    "RMSprop": torch.optim.RMSprop,
    # Momentum para que SGD no avance demasiado lento
    "SGD": partial(torch.optim.SGD, momentum=0.9),
}


# --- Puntaje de riesgo de cada tipo de modelo -------------------------------------------

def positive_proba(model, X):
    """Probabilidad de dificultades (columna 1 de predict_proba)."""
    return model.predict_proba(X)[:, 1]


def margin(model, X):
    """SVM no da probabilidades. La distancia a la frontera sirve igual para ordenar."""
    return model.decision_function(X)


def log_odds(model, X):
    """Naive Bayes da probabilidades pegadas a 0 o a 1, con muchos empates.
    La diferencia de log-probabilidades entre las dos clases ordena igual y sin empates."""
    log_prob = model.predict_joint_log_proba(X)
    return log_prob[:, 1] - log_prob[:, 0]


# --- Regla de alerta y métricas ---------------------------------------------------------

def flag_at_risk(prob, share: float = 0.1):
    """Marca con 1 al `share` (10 %) de empresas con mayor riesgo. Con empates marca el número exacto."""
    n_at_risk = round(len(prob) * share)
    # Posiciones de las empresas, de mayor a menor riesgo
    order = np.argsort(-prob, kind="stable")
    flags = np.zeros(len(prob), dtype=int)
    flags[order[:n_at_risk]] = 1
    return flags


def compute_metrics(prob, pred, actual) -> dict:
    return {
        # Qué tan bien ordena a las empresas por riesgo: 0.5 = azar, 1 = perfecto
        "AUC": roc_auc_score(actual, prob),
        # Proporción de empresas bien clasificadas, con y sin dificultades
        "Accuracy": accuracy_score(actual, pred),
        # De las empresas marcadas en riesgo, cuántas tuvieron dificultades
        "Precisión": precision_score(actual, pred),
        # De las empresas que tuvieron dificultades, cuántas fueron marcadas
        "Recall": recall_score(actual, pred),
    }


def evaluate(prob_by_model: dict, actual) -> pd.DataFrame:
    """Métricas de cada modelo marcando en riesgo al 10 % con mayor puntaje, de mejor a peor AUC."""
    rows = {
        name: compute_metrics(prob, flag_at_risk(prob), actual)
        for name, prob in prob_by_model.items()
    }
    return pd.DataFrame(rows).T.sort_values("AUC", ascending=False)


# --- Entrenamiento y búsqueda de hiperparámetros ----------------------------------------

def fit_and_score(model, train, X, score=positive_proba):
    """Entrena con `train` = (X, y) y devuelve el puntaje de riesgo de las filas de X."""
    model.fit(*train)
    return score(model, X)


def validation_auc(model, train, val, score=positive_proba) -> float:
    """Entrena con `train` y mide el AUC en `val`, ambos como (X, y)."""
    return roc_auc_score(val[1], fit_and_score(model, train, val[0], score))


def tune(make_model, values, train, val, score=positive_proba):
    """Entrena un modelo por cada valor y se queda con el de mayor AUC en validación.

    Devuelve el mejor valor y el AUC de cada uno.
    """
    aucs = {value: validation_auc(make_model(value), train, val, score) for value in values}
    return max(aucs, key=aucs.get), aucs


def fixed_validation(train, val):
    """Une entrenamiento y validación para GridSearchCV con un único corte fijo (-1 = entrenar, 0 = validar)."""
    X = np.vstack([train[0], val[0]])
    y = np.concatenate([train[1], val[1]])
    fold = np.concatenate([np.full(len(train[1]), -1), np.zeros(len(val[1]))])
    return X, y, PredefinedSplit(fold)


def run_study(objective, n_trials: int):
    """Búsqueda bayesiana (TPE) que maximiza el valor que devuelve `objective`."""
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


def tuning_summary(aucs_by_model: dict) -> pd.DataFrame:
    """Peor y mejor AUC de validación de cada búsqueda, y lo que se gana al elegir bien."""
    rows = {
        name: {"Configuraciones": len(aucs), "Peor AUC": min(aucs), "Mejor AUC": max(aucs)}
        for name, aucs in aucs_by_model.items()
    }
    summary = pd.DataFrame.from_dict(rows, orient="index")
    summary["Ganancia"] = summary["Mejor AUC"] - summary["Peor AUC"]
    return summary.sort_values("Ganancia", ascending=False)


# --- Red neuronal -------------------------------------------------------------------------

class NeuralNet(nn.Module):
    def __init__(self, n_inputs, n_layers, n_neurons, dropout):
        super().__init__()
        layers = []
        # Cada capa oculta: lineal + ReLU + dropout. La siguiente recibe la salida de esta
        for _ in range(n_layers):
            layers += [nn.Linear(n_inputs, n_neurons), nn.ReLU(), nn.Dropout(dropout)]
            n_inputs = n_neurons
        # Capa de salida con un único valor
        layers.append(nn.Linear(n_inputs, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def to_tensor(array):
    return torch.tensor(array, dtype=torch.float32, device=DEVICE)


def train_epoch(net, optimizer, loss_fn, X_t, y_t, batch_size: int = 256) -> None:
    """Una pasada por todos los datos, en lotes de filas en orden aleatorio."""
    for batch in torch.randperm(len(X_t), device=DEVICE).split(batch_size):
        optimizer.zero_grad()
        loss = loss_fn(net(X_t[batch]), y_t[batch])
        # Calcular los gradientes y actualizar los pesos
        loss.backward()
        optimizer.step()


def train_net(params: dict, X_np, y_np) -> NeuralNet:
    """Entrena la red con los hiperparámetros elegidos por Optuna."""
    X_t = to_tensor(X_np)
    y_t = to_tensor(y_np).unsqueeze(1)

    net = NeuralNet(
        n_inputs=X_t.shape[1],
        n_layers=params["n_layers"],
        n_neurons=params["n_neurons"],
        dropout=params["dropout"],
    ).to(DEVICE)
    optimizer = OPTIMIZERS[params["optimizer"]](net.parameters(), lr=params["lr"])
    # Pérdida de clasificación binaria (sí / no)
    loss_fn = nn.BCEWithLogitsLoss()

    net.train()
    for _ in range(params["epochs"]):
        train_epoch(net, optimizer, loss_fn, X_t, y_t)
    return net


def predict_net(net: NeuralNet, X_np):
    """Probabilidad de dificultades según la red."""
    # Modo evaluación (apaga el dropout) y sin gradientes, porque solo se predice
    net.eval()
    with torch.no_grad():
        output = net(to_tensor(X_np))
    # sigmoid convierte la salida en una probabilidad entre 0 y 1
    return torch.sigmoid(output).cpu().numpy().ravel()
