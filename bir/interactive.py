from bir.index import get_docs, create_index
from bir.query import evaluate_boolean_query

docs = get_docs("../en.txt")
inverted_index = create_index(docs)

def query_interface():
    while True:
        user_query = input("\nZadejte boolean dotaz (nebo 'exit' pro ukončení): ")
        if user_query.lower() == 'exit':
            break
        res = evaluate_boolean_query(user_query, inverted_index)
        if res:
            print("Dokumenty odpovídající dotazu:", res)
            # Například můžeme zobrazit první větu dokumentu:
            for doc_id in res:
                first_sentence = docs[doc_id].split('.')[0]
                print(f"Dokument {doc_id}: {first_sentence}")
        else:
            print("Žádné dokumenty neodpovídají dotazu.")

query_interface()