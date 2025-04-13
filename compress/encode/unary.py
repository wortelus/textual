def unary_encode(n: int) -> str:
    if n < 1:
        raise ValueError("Unární kódování je definováno pouze pro n >= 1")
    return "1" * (n - 1) + "0"


def unary_decode(code: str) -> (int, str):
    # počet znaků '1' před prvním znakem '0'
    count = 0

    i = 0
    length = len(code)
    while i < length and code[i] == "1":
        count += 1
        i += 1

    # očekáváme, že další znak je '0'
    if i >= len(code) or code[i] != "0":
        raise ValueError("Chybný unární kód, chybí zakončující nula.")
    return count + 1, code[i + 1:]
