import requests

BASE = 'http://localhost:8000'

print(requests.get(BASE).json())

print(
    requests.post(
        f'{BASE}/recommendations',
        json={'user_id': 2, 'k': 5},
        timeout=5
    ).json()
)

print(
    requests.post(
        f'{BASE}/similar_items',
        json={'item_id': 421640, 'k': 5},
        timeout=5
    ).json()
)