import requests
import random
import time
import json

BASE_URL = "http://localhost:8000"

def test_endpoint(endpoint, method="GET", payload=None):
    try:
        url = f"{BASE_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, params=payload)
        else:
            response = requests.post(url, json=payload)
        
        response.raise_for_status()
        print(f"✅ {endpoint} - Success")
        return response.json()
    
    except requests.exceptions.HTTPError as err:
        print(f"❌ {endpoint} - Error: {err.response.status_code}")
        if err.response.text:
            try:
                print(json.dumps(err.response.json(), indent=2))
            except:
                print(err.response.text)
    except Exception as err:
        print(f"❌ {endpoint} - Unexpected error: {str(err)}")
    
    return None

def run_full_test():
    print("🚀 Starting comprehensive service test\n")
    
    # 1. Тест корневого эндпоинта
    test_endpoint("/")
    
    # 2. Тест предрассчитанных рекомендаций
    print("\nTesting precomputed recommendations:")
    for user_id in [64, 1407479, 123123123123]:  # Последний - несуществующий
        payload = {"user_id": user_id, "k": 5}
        result = test_endpoint("/recommendations", "POST", payload)
        if result:
            print(f"  User {user_id}: {result['recs']}")
    
    # 3. Тест онлайн-рекомендаций
    print("\nTesting online recommendations:")
    for user_id in [64, 1407479, 123123123123]:  # Последний - несуществующий
        result = test_endpoint(f"/online_recommendations?user_id={user_id}&k=3")
        if result:
            print(f"  User {user_id}: {result['recs']}")
    
    # 4. Тест похожих товаров
    print("\nTesting similar items:")
    for item_id in [421640, 44122, 123123123123]:  # Последний - несуществующий
        payload = {"item_id": item_id, "k": 3}
        result = test_endpoint("/similar_items", "POST", payload)
        if result:
            print(f"  Item {item_id}: {result['similar_items']}")
    
    # 5. Тест системы событий
    user_id = 64
    print(f"\nTesting event system for user {user_id}")
    
    # Записываем события
    for i in range(1, 6):
        item_id = random.randint(100000, 999999)
        payload = {"user_id": user_id, "item_id": item_id}
        result = test_endpoint("/events/put", "POST", payload)
        print(f"  Recorded event {i}: user={user_id}, item={item_id}")
        time.sleep(0.1)
    
    # Получаем события
    payload = {"user_id": user_id, "k": 3}
    events = test_endpoint("/events/get", "POST", payload)
    if events:
        print(f"  Retrieved events: {events['events']}")
    
    # 6. Тест сессионных рекомендаций
    print("\nTesting session-based recommendations:")
    
    # Сначала без событий
    payload = {"user_id": 0, "k": 5}
    result = test_endpoint("/session_recommendations", "POST", payload)
    if result:
        print(f"  New user recs: {result['recs']}")
    
    # Теперь для пользователя с событиями
    payload = {"user_id": user_id, "k": 5}
    result = test_endpoint("/session_recommendations", "POST", payload)
    if result:
        print(f"  Session recs for user {user_id}: {result['recs']}")
    
    # 7. Комплексный тест: события + рекомендации
    print("\nTesting full workflow:")
    new_user = random.randint(2000, 2999)
    
    # Записываем 3 события
    for item_id in [342816, 72028, 325215]:
        payload = {"user_id": new_user, "item_id": item_id}
        test_endpoint("/events/put", "POST", payload)
        print(f"  Recorded event: user={new_user}, item={item_id}")
    
    # Получаем сессионные рекомендации
    payload = {"user_id": new_user, "k": 5}
    result = test_endpoint("/session_recommendations", "POST", payload)
    if result:
        print(f"  Session recs: {result['recs']}")
    
    # Получаем обычные рекомендации
    payload = {"user_id": new_user, "k": 5}
    result = test_endpoint("/recommendations", "POST", payload)
    if result:
        print(f"  Standard recs: {result['recs']}")
    
    print("\n✅ All tests completed")

if __name__ == "__main__":
    run_full_test()