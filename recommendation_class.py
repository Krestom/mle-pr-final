import logging as logger
import pandas as pd

class Recommendations:
    
    def __init__(self):
        self._recs = {'personal': None, 'default': None}
        self._stats = {
            'request_personal_count': 0,
            'request_default_count': 0
        }
        
    def load(self, type, path, **kwargs):
        logger.info(f"Loading recommendations, type: {type}")
        self._recs[type] = pd.read_parquet(path, **kwargs)
        
        if type == 'personal':
            self._recs[type] = self._recs[type].set_index('visitorid')
        logger.info(f'Loaded')
    
    def get(self, user_id: int, k: int=3):
        
        try:
            recs = self._recs['personal'].loc[user_id]
            recs = recs['itemid'].to_list()[:k]
            self._stats['request_personal_count'] += 1
        except KeyError:
            # Рекомендуем k самых популярных айтемов
            recs = self._recs['default']['itemid'].head(k).to_list()
            self._stats['request_default_count'] += 1
        except:
            logger.error("No recommendations found")
            recs = []
            
        return recs
            
    def stats(self):
        logger.info('Stats for recommendations')
        for name, value in self._stats.items():
            logger.info(f'{name:<30} {value}')
            


rec_store = Recommendations()

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


rec_store.get(user_id=2, k=5)