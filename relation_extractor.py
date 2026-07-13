import json
from openai import OpenAI

# Настройка подключения к локальному vLLM
client = OpenAI(
    base_url="http://localhost:8010/v1",
    api_key="EMPTY"
)

# Список допустимых связей
allowed_relations = ["RUNS_ON", "STORES", "IS_PART_OF", "CREATED_BY", "NONE"]

def validate_relation(text, ent_a, ent_b):
    """Отправляет пару сущностей в LLM и получает JSON-ответ"""

    prompt = f"""
    Текст: "{text}"
    Субъект: {ent_a['name']}
    Объект: {ent_b['name']}

    Найди связь между ними из списка: {allowed_relations}.
    Если связи нет, верни 'NONE'.
    """

    response = client.chat.completions.create(
        model="google/gemma-4-E4B",
        messages=[
            {"role": "system", "content": "Ты строгий валидатор. Отвечай только JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}  # Заставляем модель выдавать JSON
    )

    return json.loads(response.choices[0].message.content)


def process_chunk(chunk_data):
    """Перебирает все пары сущностей в чанке и собирает триплеты"""
    text = chunk_data["text"]
    entities = chunk_data.get("entities", [])
    triplets = []

    print(f"Обработка чанка: {len(entities)} сущностей...")

    for i in range(len(entities)):
        for j in range(len(entities)):
            if i == j: continue

            # Получаем результат от LLM
            result = validate_relation(text, entities[i], entities[j])

            # Сохраняем, если связь найдена
            if result.get("relation") not in ["NONE", None]:
                triplets.append({
                    "subject_id": entities[i]["id"],
                    "predicate": result["relation"],
                    "object_id": entities[j]["id"],
                    "evidence": result.get("evidence", "")
                })

    return triplets

test_chunk = {
    "text": "vLLM работает на GPU и хранит данные в Qdrant.",
    "entities": [
        {"id": "1", "name": "vLLM"},
        {"id": "2", "name": "GPU"},
        {"id": "3", "name": "Qdrant"}
    ]
}

found_triplets = process_chunk(test_chunk)
print(json.dumps(found_triplets, indent=2, ensure_ascii=False))