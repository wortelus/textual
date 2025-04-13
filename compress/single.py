from encode.unary import unary_encode, unary_decode
from encode.gamma import gamma_encode, gamma_decode
from encode.fibonacci import fibonacci_encode, fibonacci_decode

def main():
    num = 6
    encoded = unary_encode(num)
    decoded, remainder = unary_decode(encoded)
    assert num == decoded, f"Chyba: {num} != {decoded}"
    print(f"encoded unary: {encoded}")
    print(f"decoded unary: {decoded} se zbytkem {remainder if len(remainder) != 0 else '(Žádný zbytek)'}")

    num = 15
    encoded = gamma_encode(num)
    decoded, remainder = gamma_decode(encoded)
    assert num == decoded, f"Chyba: {num} != {decoded}"
    print(f"encoded gamma: {encoded}")
    print(f"decoded gamma: {decoded} se zbytkem {remainder if len(remainder) != 0 else '(Žádný zbytek)'}")

    num = 1591
    encoded = fibonacci_encode(num)
    decoded, remainder = fibonacci_decode(encoded)
    assert num == decoded, f"Chyba: {num} != {decoded}"
    print(f"encoded fibonacci: {encoded}")
    print(f"decoded fibonacci: {decoded} se zbytkem {remainder if len(remainder) != 0 else '(Žádný zbytek)'}")

if __name__ == "__main__":
    main()