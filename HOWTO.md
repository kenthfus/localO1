# MongoDB Vector DB Test Results

## 🧪 Test Execution Command
```bash
python -m pytest vector_db/test-mongovector.py -v

platform win32 -- Python 3.10.11, pytest-8.3.5, pluggy-1.5.0
cachedir: .pytest_cache
rootdir: C:\Users\thf679\Documents\projects\eHRSS\ehrss-eaif-common-lib\src
plugins: anyio-4.9.0, langsmith-0.3.42
collected 8 items

vector_db/test-mongovector.py::test_add_documents PASSED                                                         [ 12%]
vector_db/test-mongovector.py::test_index_creation PASSED                                                        [ 25%]
vector_db/test-mongovector.py::test_similarity_search PASSED                                                     [ 37%]
vector_db/test-mongovector.py::test_check_documents_exist PASSED                                                 [ 50%]
vector_db/test-mongovector.py::test_query_search PASSED                                                          [ 62%]
vector_db/test-mongovector.py::test_query_search_02 PASSED                                                       [ 75%]
vector_db/test-mongovector.py::test_remove_embeddings PASSED                                                     [ 87%]
vector_db/test-mongovector.py::test_delete_collection PASSED                                                     [100%]

vector_db\test-mongovector.py:17
  C:\Users\thf679\...\src\vector_db\test-mongovector.py:17: PytestCollectionWarning:
  cannot collect test class 'TestEmbeddings' because it has a __init__ constructor
    class TestEmbeddings:

8 passed, 1 warning in 125.28s (0:02:05)
