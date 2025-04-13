def fib_sequence_by_max(upto: int) -> list:
    # F_2 a F_3
    fibs = [1, 2]

    # pokračujeme v generování Fibonacciho čísel, dokud nedosáhneme 'upto'
    while fibs[-1] <= upto:
        fibs.append(fibs[-1] + fibs[-2])

    # pokud poslední číslo přesáhne 'upto', odstraníme ho...
    if fibs[-1] > upto:
        fibs.pop()
    return fibs


def fib_sequence_by_length(length: int) -> list:
    fibs = []
    # F_2 a F_3
    a, b = 1, 2
    fibs.append(a)
    for _ in range(1, length):
        fibs.append(b)
        a, b = b, a + b

    return fibs


def fibonacci_encode(n: int) -> str:
    if n < 1:
        raise ValueError("Fibonacciho kódování je definováno pouze pro n >= 1")

    # vypočítáme Fibonacciho posloupnost
    fibs = fib_sequence_by_max(n)
    length = len(fibs)

    # iterujeme od největšího Fibonacciho čísla
    # od F_k k nejmenšímu F_2
    code_bits = list("0" * length)
    for i, fib in enumerate(reversed(fibs)):
        # pokud je zbytek 0 -> break
        if n == 0:
            break

        # narazili jsme na první Fibonacciho číslo, které je menší nebo rovno n
        if fib <= n:
            # na indexu se fib využilo -> set 1
            code_bits[-i - 1] = "1"
            n -= fib
            # print(f"fib: {fib}, code_bits: {"".join(code_bits)}")
        else:
            # na indexu se fib nevyužilo -> set 0
            code_bits[-i - 1] = "0"
            # print(f"fib: {fib}")

    # odebráním případných úvodních nul (ne nutné, ale kód by měl začínat "1")
    code_str = "".join(code_bits)

    # pokud se vyřezáním všech nul vyprázdní řetězec, znamená to, že číslo bylo 0 (což nemá být)
    if not code_str:
        code_str = "0"

    # připojíme terminátor "1", aby na konci byly 2x"1"
    return code_str + "1"


def fibonacci_decode(code: str) -> (int, str):
    # dle Zeckendorfovy věty je každé číslo jedinečně reprezentovatelné jako součet nesousedních Fibonacciho čísel.
    i = 0
    end_index = None
    for i in range(1, len(code)):
        if code[i - 1] == "1" == "1" and code[i] == "1":
            end_index = i  # ukončující "1" je právě na tomto místě
            break
        i += 1
    if end_index is None:
        raise ValueError("Chybný Fibonacci kód, chybí terminátor.")

    # kód bez ukončujícího bitu
    fib_code = code[:end_index]
    # if end_index != len(code) - 1:
        # jelikož kód nemůže obsahovat 2x'1' kromě úplného konce s terminátorem
        # print("Pozor, dvojitá posloupnost '1' není na úplném konci kódu. Buď následuje další kód, nebo je kód poškozený.")

    code_length = len(fib_code)
    fibs = list(fib_sequence_by_length(code_length + 1))

    n = 0
    for i, (bit, fib) in enumerate(zip(fib_code, fibs)):
        if bit == "1":
            n += fib

    # číslo z kódu + zbytek, pokud by vstup obsahoval více kódů
    return n, code[end_index + 1:]
