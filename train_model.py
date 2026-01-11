# train_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# -----------------------------
# 1️⃣ Datasetni o‘qish
# -----------------------------
df = pd.read_csv("data/loan.csv")

# Kolonka nomlarini tozalash
df.columns = df.columns.str.strip()

# -----------------------------
# 2️⃣ Target tozalash
# -----------------------------
df = df.dropna(subset=["loan_status"])
df["loan_status"] = df["loan_status"].map(
    {"Approved": 1, "Rejected": 0}
).fillna(df["loan_status"])

# -----------------------------
# 3️⃣ Categorical ustunlar
# -----------------------------
categorical_cols = ["education", "self_employed"]

for col in categorical_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.lower()   # 🔥 ENG MUHIM
    )
    df[col] = df[col].fillna(df[col].mode()[0])

le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# -----------------------------
# 4️⃣ Numeric ustunlar
# -----------------------------
numeric_cols = [
    "no_of_dependents", "income_annum", "loan_amount",
    "loan_term", "cibil_score", "residential_assets_value",
    "commercial_assets_value", "luxury_assets_value",
    "bank_asset_value"
]

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# -----------------------------
# 5️⃣ X va y
# -----------------------------
X = df.drop(columns=["loan_id", "loan_status"])
y = df["loan_status"]

# -----------------------------
# 6️⃣ Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 7️⃣ Model
# -----------------------------
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_scaled, y)

# -----------------------------
# 8️⃣ Saqlash
# -----------------------------
joblib.dump(knn_model, "models/loan_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(le_dict, "models/le_dict.pkl")

print("✅ Model, scaler va encoder yangilandi")
