import logging
import pandas as pd

from fastapi import FastAPI
from contextlib import asynccontextmanager

from recommendation_class import Recommendations
from similar_items import SimilarItems

from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

rec_store = Recommendations()
sim_items_store = SimilarItems()

class RecommendationRequest(BaseModel):
    user_id: int
    k: int = 10
    
class SimilarItemsRequest(BaseModel):
    item_id: int
    k: int = 10

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading models and data...")

    # Загрузка персональных рекомендаций
    rec_store.load(
        'personal',
        'data/recommendations.parquet',
        columns=['visitorid', 'itemid', 'rank']
    )
    rec_store.load(
        'default',
        'data/item_popularity.parquet',
        columns=['itemid', 'weights']
    )
    # Загрузка похожих товаров
    sim_items_store.load(
        path='data/similar_items.parquet',
        columns=['item_idx_1', 'item_idx_2', 'score']
    )

    logger.info("Data is ready")

    yield

    logger.info("Stop")
    rec_store.stats()
    
    
app = FastAPI(title="Recs service", lifespan=lifespan)

@app.get("/")
def root():
    return {"message": "Сервис запущен, используй эндпоинты /recommendations или /similar_items"}

@app.post("/recommendations")
async def recommendations(request: RecommendationRequest):
    """
    Возвращает список рекомендаций длиной k для пользователя user_id.
    """
    recs = rec_store.get(user_id=request.user_id, k=request.k)
    return {"recs": recs}

@app.post("/similar_items")
async def similar_items(request: SimilarItemsRequest):
    """
    Возвращает список похожих товаров для item_id.
    """
    i2i = sim_items_store.get(item_id=request.item_id, k=request.k)
    return i2i