
input_lines = [
    "1abc2",
    "pqr3stu8vwx",
    "a1b2c3d4e5f",
    "treb7uchet"
]

FST = None
SND = None
sum_value = 0

for line in input_lines:
    FST = None
    SND = None
    for char in line:
        if char.isdigit():
            if FST is None:
                FST = char
            SND = char
    if FST is not None and SND is not None:
        sum_value += int(FST + SND)

print(sum_value)
