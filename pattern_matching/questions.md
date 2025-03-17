### Kdy je KMP výhodné ?
- KMP je výhodné ->
  - malá abeceda
  - malé vzory
- zde prefixový set nám dokáže lépe dát najevo, zda přeskočit část
textu dřív, než BMH, jelikož pro malé abecedy BMH dělá malé skoky

### Kdy je BMH výhodnější než bruteforce ?
prakticky vždy, pokud má BMH dost rozmanitou abecedu
a ještě bonusově k tomu velký rozmanitý vzor, pak ho velmi rychle
přeskočí

### Kdy je KMP nevýhodné používat
Slabost KMP je v naší lidské přirozené abecedě a chuti vyhledávat delší vzory
takže v těchto případěch... příklady lze vidět v `benchmark.py`

### Opakující se vzory ?
Záleží na abecedě... dlouhá/krátká
- Brute force -> nulová heurestika, čili špatný performance
- BMH - pokud je malá abeceda, stane se, že se posuneme dle `shift_table` pouze o malý kousek
- KMP - pokud je malá abeceda, je o něco lepší než BMH... síla je v prefixovém setu
