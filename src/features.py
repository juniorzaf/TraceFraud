from pathlib import Path
import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
  """Transforme le DataFrame brut en encodant les variables catégorielles

  et en créant de nouvelles interactions pertinentes pour la détection de fraude.

  Args:
      df (pd.DataFrame): Le DataFrame brut chargé.

  Returns:
      pd.DataFrame: Le DataFrame enrichi et prêt pour l'entraînement.
  """
  data = df.copy()

  print("[INFO] Début du Feature Engineering sur les colonnes spécifiques...")

  # 1. Transformation logarithmique du montant pour réduire l'asymétrie (skewness)
  if "amount_usd" in data.columns:
    data["log_amount"] = np.log1p(data["amount_usd"])
    print("-> Variable 'log_amount' créée.")

  # 2. Encodage des variables booléennes (True/False en 1/0)
  bool_cols = [
      "is_foreign_transaction",
      "is_new_merchant",
      "used_vpn",
      "ip_country_mismatch",
      "billing_shipping_mismatch",
      "is_ai_generated_scam_attempt",
  ]
  for col in bool_cols:
    if col in data.columns:
      data[col] = data[col].astype(int)
  print("-> Variables booléennes converties en entiers (0/1).")

  # 3. Encodage One-Hot pour les variables catégorielles (merchant_category, card_type, auth_method, channel, device_type)
  cat_cols = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
  existing_cats = [c for c in cat_cols if c in data.columns]

  if existing_cats:
    data = pd.get_dummies(data, columns=existing_cats, drop_first=True, dtype=int)
    print(f"-> Encodage One-Hot appliqué sur : {existing_cats}")

  # 4. Suppression de l'identifiant unique s'il est présent (inutile pour le modèle)
  if "transaction_id" in data.columns:
    # On peut le stocker temporairement ou le dropper pour l'entraînement
    # data = data.drop(columns=['transaction_id'])
    pass

  print(f"[INFO] Feature Engineering terminé. Nombre total de colonnes finales : {data.shape[1]}")
  return data


if __name__ == "__main__":
  from data_loader import load_raw_data

  try:
    df_raw = load_raw_data()
    df_processed = engineer_features(df_raw)

    # Sauvegarde dans data/processed
    base_dir = Path(__file__).resolve().parent.parent
    processed_path = base_dir / "data" / "processed" / "clean_transactions.csv"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(processed_path, index=False)
    print(f"[INFO] Données transformées sauvegardées avec succès à : {processed_path}")

  except Exception as e:
    print(f"[ERREUR] {e}")