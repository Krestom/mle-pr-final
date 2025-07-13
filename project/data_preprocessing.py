# project/data_processing.py
# Полный скрипт с учётом требований:
#   • для отладки используем только DEBUG_FRACTION (по умолчанию 5 %)
#   • ALS обучается на всей (отфильтрованной) выборке без train/test разделения

import os
import pandas as pd
import numpy as np
import pickle
from implicit.als import AlternatingLeastSquares
import lightgbm as lgb

from scipy.sparse import csr_matrix, save_npz
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def run_full_pipeline():
    # ─────────────────────────────────────────────────────────────
    # 1. Параметры отладки / пути к данным
    # ─────────────────────────────────────────────────────────────
    DEBUG_FRACTION = float(os.getenv("DEBUG_FRACTION", "0.05"))  # 5 % по умолчанию
    print(f"⚡️ DEBUG_FRACTION = {DEBUG_FRACTION:.0%}")

    RAW_EVENTS_PATH = "../data/events.csv"
    RAW_ITEMS_PATH  = "../data/items.parquet"

    # ─────────────────────────────────────────────────────────────
    # 2. Загрузка и первичная обработка данных
    # ─────────────────────────────────────────────────────────────
    items  = pd.read_parquet(RAW_ITEMS_PATH)
    events = pd.read_csv(RAW_EVENTS_PATH)

    # Приводим timestamp → datetime
    events["dt"] = pd.to_datetime(events["timestamp"], unit="ms")
    items["dt"]  = pd.to_datetime(items["timestamp"],  unit="ms")
    events.drop(columns=["timestamp"], inplace=True)
    items.drop(columns=["timestamp"],  inplace=True)

    # Для быстрой отладки берём только часть событий
    if DEBUG_FRACTION < 1.0:
        events = (
            events.sample(frac=DEBUG_FRACTION, random_state=42)
                .reset_index(drop=True)
        )
        print(f"   …оставлено {len(events):,} строк событий для отладки")

    # ─────────────────────────────────────────────────────────────
    # 3. Взвешиваем события и кодируем id
    # ─────────────────────────────────────────────────────────────
    event_weight = {"view": 1, "addtocart": 10, "transaction": 5}
    events["weights"] = events["event"].map(event_weight)
    events = events.dropna(subset=["weights"])

    item_popularity = (
        events.groupby("itemid")["weights"]
            .sum()
            .reset_index()
            .sort_values(by="weights", ascending=False)
    )

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    events["user_idx"] = user_encoder.fit_transform(events["visitorid"])
    events["item_idx"] = item_encoder.fit_transform(events["itemid"])

    # ─────────────────────────────────────────────────────────────
    # 4. Формируем матрицу взаимодействий (train = все данные)
    # ─────────────────────────────────────────────────────────────
    events = events.sort_values("dt")
    train_events = events.copy()

    user_item_train = csr_matrix(
        (train_events["weights"],
        (train_events["user_idx"], train_events["item_idx"]))
    )

    # ─────────────────────────────────────────────────────────────
    # 5. Обучаем ALS на всей выборке
    # ─────────────────────────────────────────────────────────────
    als_params = {
        "factors": 64,
        "regularization": 0.05,
        "iterations": 50,
        "random_state": 42
    }

    als_model = AlternatingLeastSquares(**als_params)
    als_model.fit(user_item_train)

    # ─────────────────────────────────────────────────────────────
    # 6. Генерируем персональные рекомендации (5 штук для скорости)
    # ─────────────────────────────────────────────────────────────
    user_ids_encoded = range(len(user_encoder.classes_))
    als_recommendations = als_model.recommend(
        user_ids_encoded,
        user_item_train[user_ids_encoded],
        filter_already_liked_items=False,
        N=5
    )

    item_ids_enc = als_recommendations[0]
    als_scores   = als_recommendations[1]

    als_recs_df = (
        pd.DataFrame({
            "user_idx": user_ids_encoded,
            "item_idx": item_ids_enc.tolist(),
            "score":    als_scores.tolist()
        })
        .explode(["item_idx", "score"], ignore_index=True)
        .astype({"item_idx": "int", "score": "float"})
    )

    als_recs_df["user_id"] = user_encoder.inverse_transform(als_recs_df["user_idx"])
    als_recs_df["item_id"] = item_encoder.inverse_transform(als_recs_df["item_idx"])
    personal_als = als_recs_df[["user_id", "item_id", "score"]]

    # ─────────────────────────────────────────────────────────────
    # 7.Считаем похожие товары (I → I)
    # ─────────────────────────────────────────────────────────────
    train_item_idx_enc = train_events["item_idx"].unique()
    similar_items_res  = als_model.similar_items(train_item_idx_enc, N=5)

    sim_items_idx_enc  = similar_items_res[0]
    sim_item_scores    = similar_items_res[1]

    similar_items = (
        pd.DataFrame({
            "item_idx_enc":         train_item_idx_enc,
            "similar_item_idx_enc": sim_items_idx_enc.tolist(),
            "score":                sim_item_scores.tolist()
        })
        .explode(["similar_item_idx_enc", "score"], ignore_index=True)
        .astype({"similar_item_idx_enc": "int", "score": "float"})
    )

    similar_items["item_idx_1"] = item_encoder.inverse_transform(similar_items["item_idx_enc"])
    similar_items["item_idx_2"] = item_encoder.inverse_transform(similar_items["similar_item_idx_enc"])
    similar_items = (similar_items
                    .drop(columns=["item_idx_enc", "similar_item_idx_enc"])
                    .query("item_idx_1 != item_idx_2"))

    # ─────────────────────────────────────────────────────────────
    # 8. Фичи + LightGBM-ранжирование (как было)
    # ─────────────────────────────────────────────────────────────
    train_events["is_addtocart"] = (train_events["event"] == "addtocart").astype(int)

    # популярность товара
    train_events = train_events.merge(
        item_popularity.rename(columns={"weights": "item_popularity"}),
        on="itemid", how="left"
    )

    # персональные баллы ALS
    train_events = train_events.merge(
        personal_als,
        left_on=["visitorid", "itemid"],
        right_on=["user_id", "item_id"],
        how="left"
    )

    # средний score похожих товаров
    sim_avg = (similar_items
            .rename(columns={"item_idx_1": "item_idx"})
            .groupby("item_idx")["score"]
            .mean()
            .reset_index()
            .rename(columns={"score": "avg_similar_score"}))

    train_events = train_events.merge(sim_avg, on="item_idx", how="left")
    train_events.fillna(0, inplace=True)

    features_df = train_events[[
        "user_idx", "itemid", "item_idx",
        "item_popularity",
        "score",
        "avg_similar_score",
        "is_addtocart"
    ]].rename(columns={"user_idx": "group_id", "is_addtocart": "target"})

    X = features_df.drop(columns=["target", "group_id", "itemid", "item_idx"])
    y = features_df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.1,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "verbose": -1,
        "num_boost_round": 50
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data  = lgb.Dataset(X_test,  label=y_test, reference=train_data)

    lgb_model = lgb.train(
        lgb_params,
        train_data,
        valid_sets=[train_data, test_data]
    )

    # Ранжируем рекомендации
    train_events["predicted_relevance"] = lgb_model.predict(X)
    ranked_recs = (
        train_events
        .sort_values(by=["visitorid", "predicted_relevance"], ascending=[True, False])
        .sort_values(by=["itemid", "weights"], ascending=[True, False])
    )

    ranked_recs["rank"] = ranked_recs.groupby("itemid").cumcount() + 1

    # ─────────────────────────────────────────────────────────────
    # 9. Сохранение всех артефактов (только нужное сервису)
    # ─────────────────────────────────────────────────────────────

    item_popularity.to_parquet("../data/item_popularity.parquet", index=False)
    similar_items.to_parquet("../data/similar_items.parquet",     index=False)
    ranked_recs[["visitorid", "itemid", "rank"]].to_parquet(
        "../data/recommendations.parquet", index=False
    )

    save_npz("../models/user_item_train.npz", user_item_train)

    with open("../models/user_encoder.pkl", "wb") as f:
        pickle.dump(user_encoder, f)
    with open("../models/item_encoder.pkl", "wb") as f:
        pickle.dump(item_encoder, f)
    with open("../models/als_model.pkl", "wb") as f:
        pickle.dump(als_model, f)
    with open("../models/lgb_model.pkl", "wb") as f:
        pickle.dump(lgb_model, f)

    print("✅ Артефакты успешно сохранены; скрипт завершён")