import logging
import os
import pickle
from contextlib import asynccontextmanager
from typing import Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from scipy.sparse import load_npz

from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator

logger = logging.getLogger("uvicorn.error")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

PERSONAL_RECS_COUNTER = Counter(
    "personal_recommendations_total", "Personal recommendation responses"
)
DEFAULT_RECS_COUNTER = Counter(
    "default_recommendations_total", "Default (popular) recommendation responses"
)
SESSION_RECS_COUNTER = Counter(
    "session_recommendations_total", "Session-based recommendation responses"
)

class Recommendations:
    def __init__(self):
        self._recs = {"personal": None, "default": None}
        self._stats = {
            "request_personal_count": 0,
            "request_default_count": 0,
            "request_session_count": 0,
        }

    def load(self, rec_type: str, path: str, **kwargs):
        self._recs[rec_type] = pd.read_parquet(path, **kwargs)

        if rec_type == "personal":
            self._recs[rec_type] = self._recs[rec_type].set_index("visitorid")
        elif rec_type == "default":
            self._recs[rec_type] = self._recs[rec_type].sort_values(
                "weights", ascending=False
            )
        logger.info(f"Загружены {rec_type} рекоммендации")

    def get(self, user_id: int, k: int = 10) -> List[int]:
        try:
            recs_data = self._recs["personal"].loc[user_id]

            if isinstance(recs_data, pd.Series):
                recs = [int(recs_data["itemid"])]
            else:
                recs = recs_data["itemid"].astype(int).tolist()

            recs = recs[:k]
            self._stats["request_personal_count"] += 1
            PERSONAL_RECS_COUNTER.inc()
            logger.info(f"Персональные рекомендации для пользователя {user_id}")
            return recs
        except KeyError:
            recs = (
                self._recs["default"]["itemid"].head(k).astype(int).tolist()
            )
            self._stats["request_default_count"] += 1
            DEFAULT_RECS_COUNTER.inc()
            logger.info(f"Дефолтные рекомендации для пользователя {user_id}")
            return recs
        except Exception as e:
            logger.error(f"Ошибка рекоммендаций: {str(e)}")
            return []

    def get_popular(self, k: int = 10) -> List[int]:
        return (
            self._recs["default"]["itemid"].head(k).astype(int).tolist()
        )

    def stats(self):
        logger.info("Статистика:")
        for name, value in self._stats.items():
            logger.info(f"{name:<30} {value}")


class SimilarItems:
    def __init__(self):
        self._similar_items = None

    def load(self, path: str, **kwargs):
        self._similar_items = pd.read_parquet(path, **kwargs)
        self._similar_items = self._similar_items.set_index("item_idx_1")
        logger.info("Похожие загружены")

    def get(self, item_id: int, k: int = 10) -> Dict[str, List]:
        try:
            item_id = int(item_id)
            i2i = self._similar_items.loc[[item_id]].head(k)
            return {
                "similar_items": i2i["item_idx_2"].astype(int).tolist(),
                "scores": i2i["score"].astype(float).tolist(),
            }
        except KeyError:
            logger.warning(f"Нет похожих для item {item_id}")
            return {"similar_items": [], "scores": []}
        except Exception as e:
            logger.error(f"Ошибка в похожих: {str(e)}")
            return {"similar_items": [], "scores": []}

    def get_batch(self, item_ids: List[int], k_per_item: int = 3
    ) -> Dict[int, List[int]]:
        results = {}
        for item_id in item_ids:
            try:
                item_id_int = int(item_id)
                similar = self.get(item_id_int, k_per_item)
                results[item_id] = similar["similar_items"]
            except Exception as e:
                logger.warning(f"Ошибка айтема {item_id}: {str(e)}")
                results[item_id] = []
        return results


class EventStore:
    def __init__(self, max_events_per_user: int = 20, max_total_events: int = 100_000):
        self.events = {}
        self.max_events_per_user = max_events_per_user
        self.max_total_events = max_total_events
        self.total_events = 0
        logger.info("Хранилище событий готово")

    def put(self, user_id: int, item_id: int):
        try:
            user_id = int(user_id)
            item_id = int(item_id)
        except Exception as e:
            logger.error(f"Неправильный формат: {str(e)}")
            return

        if self.total_events >= self.max_total_events:
            self.events.clear()
            self.total_events = 0
            logger.warning("Хранилище очищено по достижении лимита")

        if user_id not in self.events:
            self.events[user_id] = []

        self.events[user_id].insert(
            0, {"item_id": item_id, "timestamp": pd.Timestamp.now()}
        )

        if len(self.events[user_id]) > self.max_events_per_user:
            self.events[user_id] = self.events[user_id][: self.max_events_per_user]

        self.total_events += 1

    def get(self, user_id: int, k: int) -> List[int]:
        try:
            user_id = int(user_id)
        except Exception:
            return []

        if user_id not in self.events:
            return []
        return [int(e["item_id"]) for e in self.events[user_id][:k]]

    def get_full_events(self, user_id: int, k: int) -> List[Dict]:
        try:
            user_id = int(user_id)
        except Exception:
            return []

        if user_id not in self.events:
            return []
        return self.events[user_id][:k]


# без этого нифига не работало, спасибо ChatGPT
class RecommendationRequest(BaseModel):
    user_id: int
    k: int = 10


class SimilarItemsRequest(BaseModel):
    item_id: int
    k: int = 10


class EventPutRequest(BaseModel):
    user_id: int
    item_id: int


class EventGetRequest(BaseModel):
    user_id: int
    k: int = 10


class SessionRecommendationRequest(BaseModel):
    user_id: int
    k: int = 10
    session_length: int = 5
    similar_per_item: int = 3


rec_store = Recommendations()
sim_items_store = SimilarItems()
events_store = EventStore()
model = None
user_encoder = None
item_encoder = None
user_item_matrix_train = None
reverse_item_map = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Начинаем вечеринку")

    try:
        required_files = {
            "personal_recs": "data/recommendations.parquet",
            "default_recs": "data/item_popularity.parquet",
            "similar_items": "data/similar_items.parquet",
            "als_model": "models/als_model.pkl",
            "user_encoder": "models/user_encoder.pkl",
            "item_encoder": "models/item_encoder.pkl",
            "user_item_matrix": "models/user_item_train.npz",
        }

        for path in required_files.values():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Нет обязательного файла: {path}")

        rec_store.load(
            "personal",
            "data/recommendations.parquet",
            columns=["visitorid", "itemid", "rank"],
        )
        rec_store.load(
            "default",
            "data/item_popularity.parquet",
            columns=["itemid", "weights"],
        )

        sim_items_store.load(
            path="data/similar_items.parquet",
            columns=["item_idx_1", "item_idx_2", "score"],
        )

        global model, user_encoder, item_encoder
        
        with open("models/als_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("models/user_encoder.pkl", "rb") as f:
            user_encoder = pickle.load(f)
        with open("models/item_encoder.pkl", "rb") as f:
            item_encoder = pickle.load(f)

        global user_item_matrix_train, reverse_item_map
        user_item_matrix_train = load_npz("models/user_item_train.npz")
        reverse_item_map = {
            i: int(item_id) for i, item_id in enumerate(item_encoder.classes_)
        }

        logger.info("Сервис успешно запущен")
    except Exception as e:
        logger.critical(f"Сервис лежит: {str(e)}")
        raise RuntimeError(f"Не получилось загрузить: {str(e)}")

    yield

    logger.info("Вырубаем")
    rec_store.stats()
    logger.info(f"Всего событий записано: {events_store.total_events}")
    logger.info(f"Уникальные юзеы с событиями: {len(events_store.events)}")


app = FastAPI(title="Unified Recommendation Service", lifespan=lifespan)

# Прометеус
Instrumentator().instrument(app).expose(app)

# Навигация
@app.get("/")
def root():
    return {
        "message": "Unified Recommendation Service",
        "endpoints": [
            "/recommendations",
            "/online_recommendations",
            "/similar_items",
            "/session_recommendations",
            "/events/put",
            "/events/get",
            "/metrics",
        ],
    }


@app.post("/recommendations")
async def recommendations(request: RecommendationRequest):
    return {"recs": rec_store.get(user_id=request.user_id, k=request.k)}


@app.get("/online_recommendations")
async def online_recommendations(user_id: int, k: int = 10):
    try:
        user_idx = user_encoder.transform([user_id])[0]
    except ValueError:
        recent_events = events_store.get(user_id, 5)
        if recent_events:
            return {"recs": recent_events[:k]}
        return {"recs": rec_store.get_popular(k)}

    try:
        user_items = user_item_matrix_train[user_idx]
        item_ids, _ = model.recommend(user_idx, user_items, N=k)
        recommend_ids = [int(reverse_item_map[item_id]) for item_id in item_ids]
        return {"recs": recommend_ids}
    except Exception as e:
        logger.error(f"ALS ошибка: {str(e)}")
        return {"recs": rec_store.get(user_id=user_id, k=k)}


@app.post("/similar_items")
async def similar_items(request: SimilarItemsRequest):
    return sim_items_store.get(item_id=request.item_id, k=request.k)


@app.post("/events/put")
async def put_event(request: EventPutRequest):
    events_store.put(request.user_id, request.item_id)
    return {"status": "event recorded", "user_id": request.user_id, "item_id": request.item_id}


@app.post("/events/get")
async def get_events(request: EventGetRequest):
    events = events_store.get(request.user_id, request.k)
    return {"user_id": request.user_id, "events": events}


@app.post("/session_recommendations")
async def session_recommendations(request: SessionRecommendationRequest):
    recent_events = events_store.get(request.user_id, request.session_length)
    if not recent_events:
        DEFAULT_RECS_COUNTER.inc()
        return {"recs": rec_store.get_popular(request.k)}

    similar_items = sim_items_store.get_batch(recent_events, request.similar_per_item)

    all_recs = []
    for similar_list in similar_items.values():
        all_recs.extend(similar_list)

    seen_items = set(recent_events)
    unique_recs = [int(item) for item in all_recs if item not in seen_items]

    rec_counts = {}
    for item in unique_recs:
        rec_counts[item] = rec_counts.get(item, 0) + 1

    sorted_recs = sorted(rec_counts.items(), key=lambda x: x[1], reverse=True)
    top_recs = [int(item) for item, _ in sorted_recs[: request.k]]

    if len(top_recs) < request.k:
        additional = request.k - len(top_recs)
        popular_recs = rec_store.get_popular(additional)
        popular_recs = [
            int(item) for item in popular_recs if item not in top_recs and item not in seen_items
        ]
        top_recs.extend(popular_recs[:additional])

    SESSION_RECS_COUNTER.inc()
    return {"recs": top_recs}