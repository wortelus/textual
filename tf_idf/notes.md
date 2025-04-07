### Uveďte příklad oblasti, témata, kde by častá slova mohla být navzdory vysoké frekvenci velmi důležitá.

#### Recenze filmů na ČSFD
- přídavné jména, chválící, kritické slova budou velmi
časté v daném korpusu, avšak budou mít vliv na celkové hodnocení filmu...

příklad:
- měli bychom korpus dokumentů, kde každý dokument by obsahoval N recenzí pro daný film
- pokud bychom to proběhli přes `cos_sim.py`, dostali bychom pravděpodobnosti mezi režiséry/podobnými herci... 
jednoduše podobnosti mezi vzácnými jmény herců, tvůrců místo např. chválicích nebo naopak kritických slov


- pokud bychom chtěli podobnost mezi dokumenty s recenzemi zároveň na základě kvalit filmů s podobnou:
  - **kvalitou** - častá slova hodnocení - dobrý/špatný
  - **tématikou/žánrem** - častá slova žánru - akční, romantický, nudný
- místo např.
    - jména tvůrců - **_Tarantino_** bychom dostali např. **_Pulp Fiction_** a **_Kill Bill_**

--> museli bychom IDF buď upravit nebo vyhodit, aby se to nezaměřovalo např. na jméno **_Tarantino_**

IDF se dá zmírnit, ale **je třeba myslet na dobrou filtraci stopwords**, aby na sebe nepoutávaly pozornost

místo
- `idf(t) = log( N / df(t) )` které pro slovo vyskytující se v celém korpusu dá váhu 0

můžeme 
- `idf(t) = log( 1 + N / df(t) )` které ranking slov vyváží...

idf nebude začínat od 0 pro 20/20, ale od 0.69 pro 20/20
idf bude obecně "víc sploštělé"

### Navrhněte váhovací schéma pro krátké texty (např. tweety), které by lépe zachytilo význam slov než klasické tf-idf.
- **tf** bude mít malý rozsah hodnot, čili bych ho vypustil úplně... co se týče tf-idf

- kvalitní odstranění stop slov
- použil bych přístup na základě **GloVe** předtrénovaných embeddingů
- slova bych převedl na vektory
- provedl bych clustering, např DBSCAN nebo HDBSCAN, jelikož
nemusí všechny slova dávat do svého clusteru (slova s nízkou inf. hodnotou)
a taky nepotřebujeme znám předem daný počet clusterů
- spočetl aritmetický průměr pro každý cluster

- teď pár nápadů:
1. pro největší cluster bych spočítal aritmetický průměr a ten bych použil jako vektor
který bych následně použil pro výpočet cosine similarity mezi tweety
2. ne pouze největší cluster, ale váha `w` by byla dána počtem slov v clusteru... to by poté
bylo použito pro vážení aritmetického průměru mezi jednotlivými clustery