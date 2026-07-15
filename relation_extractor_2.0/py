import json
import uuid
from openai import OpenAI

# 1. Настройка подключения к локальному vLLM
client = OpenAI(
    base_url="http://localhost:8010/v1",
    api_key="EMPTY"
)

# 2. Онтология связей (должна совпадать с тем, что ты хочешь видеть в графе)
ALLOWED_RELATIONS = ["RUNS_ON", "STORES", "IS_PART_OF", "CREATED_BY", "NONE"]

# 3. Жесткая JSON Schema для Structured Output
RELATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "relation_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "relation": {"type": "string", "enum": ALLOWED_RELATIONS},
                "evidence": {"type": "string"}
            },
            "required": ["relation", "evidence"],
            "additionalProperties": False
        }
    }
}

def validate_relation(text: str, ent_a: dict, ent_b: dict) -> dict:
    """Отправляет пару сущностей в LLM и получает строго структурированный JSON"""
    
    prompt = f"""
    Текст: "{text}"
    Субъект: {ent_a['name']}
    Объект: {ent_b['name']}

    Определи связь между ними ИСКЛЮЧИТЕЛЬНО из списка: {ALLOWED_RELATIONS}.
    Если явной связи нет, верни 'NONE'.
    В поле 'evidence' укажи точную цитату из текста, подтверждающую эту связь.
    """

    try:
        response = client.chat.completions.create(
            model="google/gemma-4-E4B", # Или твоя модель
            messages=[
                {"role": "system", "content": "Ты строгий валидатор связей. Отвечай ТОЛЬКО валидным JSON по заданной схеме."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format=RELATION_SCHEMA # Магия: модель физически не может сломать схему
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"[ERROR] Ошибка валидации пары ({ent_a['name']}, {ent_b['name']}): {e}")
        return {"relation": "ERROR", "evidence": ""}


def convert_entities_to_triplet_format(chunk_data: dict) -> dict:
    """
    АДАПТЕР: Конвертирует строки из process_entities_pipeline 
    в формат словарей с ID, понятный этому модулю.
    """
    entities_str = chunk_data.get("final_entities", "")
    if not entities_str:
        return {"text": chunk_data.get("chunk_text", ""), "entities": []}
    
    parsed_entities = []
    for item in entities_str.split(","):
        item = item.strip()
        if ":" in item:
            name, tag = item.rsplit(":", 1)
            # Генерируем стабильный канонический ID для Memgraph
            entity_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{name.strip()}:{tag.strip()}"))
            parsed_entities.append({
                "id": entity_id,
                "name": name.strip(),
                "type": tag.strip()
            })
            
    return {
        "text": chunk_data.get("chunk_text", ""),
        "entities": parsed_entities
    }


def process_chunk(chunk_data: dict) -> list:
    """Перебирает все пары сущностей в чанке и собирает триплеты"""
    text = chunk_data["text"]
    entities = chunk_data.get("entities", [])
    triplets = []

    print(f"Обработка чанка: {len(entities)} сущностей...")

    for i in range(len(entities)):
        for j in range(len(entities)):
            if i == j: continue
            
            result = validate_relation(text, entities[i], entities[j])
            
            # Сохраняем только если связь найдена и это не ошибка
            if result.get("relation") not in ["NONE", "ERROR", None]:
                triplets.append({
                    "subject_id": entities[i]["id"],
                    "predicate": result["relation"],
                    "object_id": entities[j]["id"],
                    "evidence": result.get("evidence", "")
                })
                
    return triplets
