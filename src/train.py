from pathlib import Path
import joblib
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
from fontTools.misc.cython import returns
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
  average_precision_score, classification_report, roc_auc_score, confusion_matrix, f1_score,
  precision_recall_curve, ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

# Import des modules internes du projet
from data_loader import load_raw_data
from features import engineer_features

# Définition de la constante aléatoire
RANDOM_SEED = 42

def evaluate_model(
    model, X_test, y_test, model_name="Modèle", threshold=0.5
):
  #Récupération des probabilités continues
  y_proba = model.predict_proba(X_test)[:, 1]

  #Application du seuil personnalisé pour obtenir les prédictions binaires
  y_pred = (y_proba >= threshold).astype(int)

  #Calcul des métriques
  f1 = f1_score(y_test, y_pred)
  pr_auc = average_precision_score(
      y_test, y_proba
  )
  #Le PR-AUC reste global
  roc_auc = roc_auc_score(
      y_test, y_proba
  )
  #Le ROC-AUC reste global

  print(f"Résultats pour {model_name} (Seuil = {threshold:.4f})")
  print(classification_report(y_test, y_pred))
  print(f"F1-Score (Classe 1) : {f1:.4f}")
  print(f"AUC-PR              : {pr_auc:.4f}")
  print(f"AUC-ROC             : {roc_auc:.4f}\n")

  print("\nRapport de classification détaillé :")
  print(classification_report(y_test, y_pred))

  #Visualisons les indicateurs
  fig, axes = plt.subplots(1, 3, figsize=(18, 5))

  #Matrice de confusion
  cm = confusion_matrix(y_test, y_pred)
  disp_cm = ConfusionMatrixDisplay(
      confusion_matrix=cm, display_labels=["Légitime (0)", "Fraude (1)"]
  )
  disp_cm.plot(ax=axes[0], cmap="Blues", values_format="d", colorbar=False)
  axes[0].set_title(f"Matrice de confusion - {model_name}\n(Seuil = {threshold:.4f})")

  #Visualisons la précision du Recall
  PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=axes[1])
  axes[1].set_title(f"Precision-Recall (AP = {pr_auc:.3f})")

  #Visualisons le ROC
  RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axes[2])
  axes[2].set_title(f"Courbe ROC (AUC = {roc_auc:.3f})")

  plt.tight_layout()
  plt.show()


def train_fraud_model():
  #Cette fonction charge les données, entraîne le modèle XGBoost et évalue ses performances avec le PR-AUC à l'aide de la fonction evaluate_model.

  print("Préparation des données pour l'entraînement")
  #chargement et traitement des données
  try:
    df_raw = load_raw_data()
    df = engineer_features(df_raw)
  except Exception as e:
    # Fallback si le fichier pré-transformé existe déjà
    base_dir = Path(__file__).resolve().parent.parent
    processed_path = base_dir / "data" / "processed" / "clean_transactions.csv"
    if processed_path.exists():
      print(f"Chargement direct depuis le fichier transformé : {processed_path}")
      df = pd.read_csv(processed_path)
    else:
      raise FileNotFoundError(f"Impossible de charger les données. Erreur d'origine : {e}")

  #Séparation de la variable cible et des variables indépendantes
  target_col = "is_fraud"
  if target_col not in df.columns:
    raise ValueError(f"La colonne cible '{target_col}' est introuvable dans le jeu de données.")

  drop_cols = [target_col]
  if "transaction_id" in df.columns:
    drop_cols.append("transaction_id")

  X = df.drop(columns=drop_cols)
  y = df[target_col]

  print(f" Dimensions des features (X) : {X.shape}")
  print(f"Taux de fraude global : {y.mean():.4f} ({y.sum()} cas frauduleux sur {len(y)} transactions)")

  #Séparation des données d'entrainement et de test
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
  )
  #On rattrape le préprocessing des colonnes numériques non fait dans le fichier features
  # Liste des colonnes numériques
  num_cols=X_train.select_dtypes(include="number").columns.tolist()
  preprocessor = ColumnTransformer(
    transformers=[
      ("num", RobustScaler(), num_cols),
    ],
    remainder="passthrough",
  )
  # Application de la transformation
  X_train = preprocessor.fit_transform(X_train)
  X_test = preprocessor.transform(X_test)
  print (f"Dimensions des features d'entrainement (X_train) : {X_train.shape}")
  print(f"Dimensions des features de Test (X_test) : {X_test.shape}")

  #Gestion du déséquilibre de classes à l'aide de la pondération scale_pos_weight
  neg_count = (y_train == 0).sum()
  pos_count = (y_train == 1).sum()
  scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
  print(f"scale_pos_weight calculé pour XGBoost : {scale_pos_weight:.2f}")

  #Initialisation et entraînement du modèle XGBoost
  xgb_base = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    eval_metric='aucpr',
    random_state=RANDOM_SEED,
    n_jobs=-1,
  )
  param_distributions = {
    'max_depth': [2, 3, 4, 5],
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.85, 1.0],
    'colsample_bytree': [0.7, 0.85, 1.0],
    'reg_alpha': [0, 0.5, 1, 5],
    'reg_lambda': [1, 5, 10], }

  random_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_distributions,
    n_iter=50,
    scoring='average_precision',
    cv=3,
    verbose=1,
    random_state=RANDOM_SEED,
    n_jobs=-1,
  )
  random_search.fit(X_train, y_train)
  print(f'\nMeilleurs paramètres trouvés : {random_search.best_params_}')
  print(f'Meilleur PR-AUC en validation croisée : {random_search.best_score_:.4f}\n')
  model = random_search.best_estimator_
  evaluate_model(model, X_test, y_test, model_name='XGBoost Optimisé (Seuil 0.5)')

  #Sauvegarde du modèle entraîné
  base_dir = Path(__file__).resolve().parent.parent
  models_dir = base_dir / "models"
  models_dir.mkdir(parents=True, exist_ok=True)
  model_path = models_dir / "xgboost_fraud_model.joblib"

  joblib.dump(model, model_path)
  print(f"Modèle sauvegardé avec succès dans : {model_path}")

  return model


if __name__ == "__main__":
  train_fraud_model()