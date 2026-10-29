from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Import des modules internes du projet
from data_loader import load_raw_data
from features import engineer_features


def train_fraud_model():
  """Charge les données, entraîne un modèle XGBoost adapté aux données

  déséquilibrées et évalue ses performances avec le PR-AUC.
  """
  print("[INFO] Préparation des données pour l'entraînement...")

  # 1. Chargement et traitement des données
  try:
    df_raw = load_raw_data()
    df = engineer_features(df_raw)
  except Exception as e:
    # Fallback si le fichier pré-transformé existe déjà
    base_dir = Path(__file__).resolve().parent.parent
    processed_path = base_dir / "data" / "processed" / "clean_transactions.csv"
    if processed_path.exists():
      print(f"[INFO] Chargement direct depuis le fichier transformé : {processed_path}")
      df = pd.read_csv(processed_path)
    else:
      raise FileNotFoundError(f"Impossible de charger les données. Erreur d'origine : {e}")

  # 2. Séparation Features (X) / Cible (y)
  target_col = "is_fraud"
  if target_col not in df.columns:
    raise ValueError(f"La colonne cible '{target_col}' est introuvable dans le jeu de données.")

  drop_cols = [target_col]
  if "transaction_id" in df.columns:
    drop_cols.append("transaction_id")

  X = df.drop(columns=drop_cols)
  y = df[target_col]

  print(f"[INFO] Dimensions des features (X) : {X.shape}")
  print(f"[INFO] Taux de fraude global : {y.mean():.4f} ({y.sum()} cas frauduleux sur {len(y)} transactions)")

  # 3. Division Train/Test stratifiée
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )

  # 4. Gestion du déséquilibre de classes (scale_pos_weight pour XGBoost)
  neg_count = (y_train == 0).sum()
  pos_count = (y_train == 1).sum()
  scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
  print(f"[INFO] scale_pos_weight calculé pour XGBoost : {scale_pos_weight:.2f}")

  # 5. Initialisation et entraînement du modèle XGBoost
  model = XGBClassifier(
      n_estimators=100,
      learning_rate=0.1,
      max_depth=5,
      scale_pos_weight=scale_pos_weight,
      random_state=42,
      eval_metric="aucpr"
  )

  print("[INFO] Entraînement du modèle XGBoost en cours...")
  model.fit(X_train, y_train)

  # 6. Évaluation sur l'ensemble de test
  y_pred_proba = model.predict_proba(X_test)[:, 1]
  y_pred = model.predict(X_test)

  pr_auc = average_precision_score(y_test, y_pred_proba)
  roc_auc = roc_auc_score(y_test, y_pred_proba)

  print("\n--- RÉSULTATS DE L'ÉVALUATION ---")
  print(f"PR-AUC (Precision-Recall AUC) : {pr_auc:.4f}")
  print(f"ROC-AUC                      : {roc_auc:.4f}")
  print("\nRapport de classification détaillé :")
  print(classification_report(y_test, y_pred))

  # 7. Sauvegarde du modèle entraîné
  base_dir = Path(__file__).resolve().parent.parent
  models_dir = base_dir / "models"
  models_dir.mkdir(parents=True, exist_ok=True)
  model_path = models_dir / "xgboost_fraud_model.joblib"

  joblib.dump(model, model_path)
  print(f"[INFO] Modèle sauvegardé avec succès dans : {model_path}")

  return model, pr_auc


if __name__ == "__main__":
  train_fraud_model()