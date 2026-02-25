# =========================================
# 1️⃣ Import thư viện
# =========================================
import pandas as pd
from sqlalchemy import create_engine
from pycaret.regression import *

# =========================================
# 2️⃣ Kết nối database
# =========================================
engine = create_engine("sqlite:///../data/sample_strategy/samples.db")

time = 2011   # ví dụ

# =========================================
# 3️⃣ Load dữ liệu từ SQL
# =========================================
df = pd.read_sql(
    f"""
    SELECT 
        x1, x2, x3, x4, x5, x6, x7, x8, x9, y
    FROM NearsestSample
    WHERE TimeDim = {time}
    """,
    engine
)

df = df.dropna()

print("Dataset shape:", df.shape)

# =========================================
# 4️⃣ Setup PyCaret
# =========================================
exp = setup(
    data=df,
    target='y',
    session_id=131006,
    train_size=0.8,
    fold=5,
    normalize=True,
    verbose=True
)

# =========================================
# 5️⃣ Tự động chọn model tốt nhất
# =========================================
best_model = compare_models()

print("Best Model:")
print(best_model)

# =========================================
# 6️⃣ Tuning model (tối ưu hyperparameter)
# =========================================
tuned_model = tune_model(best_model)

# =========================================
# 7️⃣ Đánh giá model
# =========================================
evaluate_model(tuned_model)

# Hoặc xem riêng metric
final_model = finalize_model(tuned_model)

# =========================================
# 8️⃣ Dự đoán trên toàn bộ dữ liệu
# =========================================
predictions = predict_model(final_model)

print(predictions.head())

# =========================================
# 9️⃣ Lưu model
# =========================================
save_model(final_model, 'best_regression_model')

print("Model saved successfully.")