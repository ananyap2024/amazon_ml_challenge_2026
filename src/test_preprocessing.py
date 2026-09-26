from preprocessing import (
    normalize_business_name,
    normalize_address,
    tokenize,
)


name_examples = [
    "Delta Tetlecommunication Inc.",
    "Prime Realty Ventures Public Limited",
    "Shri Supreme Consulting Private (Limited)",
]

address_examples = [
    "G.t. Karnal Road, Industrial Area, New Delhi, null, A-68, दिल्ली",
    "105 ELM ST, MORGANTOWN, NC",
]


print("=" * 70)
print("BUSINESS NAME NORMALIZATION")
print("=" * 70)

for value in name_examples:

    print("\nOriginal:")
    print(value)

    print("Normalized:")
    print(normalize_business_name(value))

    print("Tokens:")
    print(tokenize(value))


print("\n" + "=" * 70)
print("ADDRESS NORMALIZATION")
print("=" * 70)

for value in address_examples:

    print("\nOriginal:")
    print(value)

    print("Normalized:")
    print(normalize_address(value))

    print("Tokens:")
    print(tokenize(value))