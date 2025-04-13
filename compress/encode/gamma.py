def gamma_encode(n: int) -> str:
    if n < 1:
        raise ValueError("Eliasovo gamma kódování je definováno pouze pro n >= 1")

    # binární zápis čísla bez prefixu '0b'
    binary = bin(n)[2:]

    # délka binárního zápisu čísla
    L = len(binary) - 1

    # L nul + binární zápis čísla
    return "0" * L + binary

def gamma_decode(code: str) -> (int, str):
    L = 0
    i = 0
    while i < len(code) and code[i] == "0":
        L += 1
        i += 1
    # nyní musí následovat L+1 bitů (včetně prvního '1')
    if i + L >= len(code):
        raise ValueError("Chybný gamma kód – nedostatek bitů.")
    num_bits = code[i:i+L+1]
    n = int(num_bits, 2)
    return n, code[i+L+1:]
