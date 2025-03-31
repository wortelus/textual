from bir.index import create_index
from bir.query import evaluate_boolean_query
from bir.index import get_docs

docs = get_docs("../en.txt")
inverted_index = create_index(docs)

query = "masatoshi AND koshiba"
results = evaluate_boolean_query(query, inverted_index)
print("\nVýsledky pro dotaz '{}':".format(query))
print(results)

# analýza efektivity indexu
total_tokens = len(inverted_index)
total_postings = sum(len(posting) for posting in inverted_index.values())
avg_postings = total_postings / total_tokens if total_tokens > 0 else 0

print(f"\nCelkový počet unikátních tokenů: {total_tokens}")
print(f"Celkový počet záznamů (postings): {total_postings}")
print(f"Průměrná délka seznamu: {avg_postings:.2f}")
