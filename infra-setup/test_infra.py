import requests
from neo4j import GraphDatabase
def test_vllm():
    response = requests.post(
    "http://localhost:8010/v1/chat/completions",
    json={
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": "Hi"}]
        }
    )
    assert response.status_code == 200
    print('vllm is working')

def test_qdrant():
    try:
        response = requests.get("http://localhost:6343/collections", timeout=5)
        assert response.status_code == 200
        print('qdrant is working')
    except Exception as e:
        print(f'qdrant error: {e}')

def test_memgraph():
    driver = GraphDatabase.driver("bolt://localhost:7697", auth=("", ""))
    with driver.session() as session:
        result = session.run('RETURN 1 AS test')
        record = result.single()
        if record["test"] == 1:
            print('Memgraph is working (with driver neo4j)')
            driver.close()
            return True
        else:
            print('Memgraph: something is wrong')
            driver.close()
            return False


if __name__ == "__main__":
    test_vllm()
    test_qdrant()
    test_memgraph()