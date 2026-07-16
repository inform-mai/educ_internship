    # Симуляция данных
    raw_output_from_stage_1 = {
        "chunk_text": "vLLM работает на GPU и хранит данные в Qdrant.",
        "final_entities": "vLLM:TEC, GPU:TEC, Qdrant:ORG"
    }
    
    # ШАГ 1: Конвертация формата
    formatted_chunk = convert_entities_to_triplet_format(raw_output_from_stage_1)
    
    # ШАГ 2: Извлечение связей
    found_triplets = process_chunk(formatted_chunk)
    
    # ШАГ 3: Вывод результата
    print(json.dumps(found_triplets, indent=2, ensure_ascii=False))
