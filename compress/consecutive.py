from compress.encode.fibonacci import fibonacci_encode, fibonacci_decode
from compress.encode.gamma import gamma_encode, gamma_decode
from compress.encode.unary import unary_encode, unary_decode

# ----------------------
# UNÁRNÍ KÓDOVÁNÍ (Unary)
# ----------------------
num1 = 6
num2 = 9
encoded1 = unary_encode(num1)
encoded2 = unary_encode(num2)
combined_unary = encoded1 + encoded2

decoded1, remainder = unary_decode(combined_unary)
decoded2, remainder = unary_decode(remainder)

assert num1 == decoded1, f"Chyba: {num1} != {decoded1}"
assert num2 == decoded2, f"Chyba: {num2} != {decoded2}"

print("Unární kódování:")
print(f"  Zakódováno společně: {combined_unary}")
print(f"  Dekódované první číslo: {decoded1}")
print(f"  Dekódované druhé číslo: {decoded2}")
print(f"  Zbytek: {remainder if remainder != '' else '(Žádný zbytek)'}\n")


# ----------------------
# ELIASOVO GAMMA KÓDOVÁNÍ (Gamma)
# ----------------------
num1 = 15
num2 = 20
encoded1 = gamma_encode(num1)
encoded2 = gamma_encode(num2)
combined_gamma = encoded1 + encoded2

decoded1, remainder = gamma_decode(combined_gamma)
decoded2, remainder = gamma_decode(remainder)

assert num1 == decoded1, f"Chyba: {num1} != {decoded1}"
assert num2 == decoded2, f"Chyba: {num2} != {decoded2}"

print("Eliasovo gamma kódování:")
print(f"  Zakódováno společně: {combined_gamma}")
print(f"  Dekódované první číslo: {decoded1}")
print(f"  Dekódované druhé číslo: {decoded2}")
print(f"  Zbytek: {remainder if remainder != '' else '(Žádný zbytek)'}\n")


# ----------------------
# FIBONACCIHO KÓDOVÁNÍ (Fibonacci)
# ----------------------
num1 = 1591
num2 = 42
encoded1 = fibonacci_encode(num1)
encoded2 = fibonacci_encode(num2)
combined_fib = encoded1 + encoded2

decoded1, remainder = fibonacci_decode(combined_fib)
decoded2, remainder = fibonacci_decode(remainder)

assert num1 == decoded1, f"Chyba: {num1} != {decoded1}"
assert num2 == decoded2, f"Chyba: {num2} != {decoded2}"

print("Fibonacciho kódování:")
print(f"  Zakódováno společně: {combined_fib}")
print(f"  Dekódované první číslo: {decoded1}")
print(f"  Dekódované druhé číslo: {decoded2}")
print(f"  Zbytek: {remainder if remainder != '' else '(Žádný zbytek)'}")
