import pandas as pd
from fastapi import FastAPI
from contextlib import asynccontextmanager

import logging
logger = logging.getLogger("uvicorn.error")

class SimilarItems():
    
    def __init__(self):
        self._similar_items = None
    
    def load(self, path, **kwargs):
        self._similar_items = pd.read_parquet(path, **kwargs)
        self._similar_items = self._similar_items.set_index('item_idx_1')
        
        logger.info(f'Loaded')
        
    def get(self, item_id: int, k: int=10):
        try:
            i2i = self._similar_items.loc[[item_id]].head(k)
            i2i = i2i[['item_idx_2', 'score']].to_dict(orient='list')
            
        except:
            logger.error("No recommendations found")
            i2i = {'item_id': [], 'score': {}}
            
        return i2i
    
sim_items_store = SimilarItems()

@asynccontextmanager
async def lifespan(app: FastAPI):
    sim_items_store.load(
        path = 'data/similar_items.parquet',
        columns = ['item_idx_1', 'item_idx_2', 'score']
    )
    yield
    
app = FastAPI(title='features', lifespan=lifespan)

@app.post('/similar_items')
async def recommendations(item_id: int, k: int=10):
    i2i = sim_items_store.get(item_id, k)
    
    return i2i



sim_items_store = SimilarItems()
sim_items_store.load(
    path='data/similar_items.parquet',
    columns=['item_idx_1', 'item_idx_2', 'score']
)

item_id = 421640
k = 2
result = sim_items_store.get(item_id=item_id, k=k)
print(result)