FST = None
SND = None
sum_value = 0

while True:
    try:
        line = input()
        if not line:
            break
        FST = None
        SND = None
        for char in line:
            if char.isdigit():
                if FST is None:
                    FST = char
                SND = char
        if FST is not None and SND is not None:
            sum_value += int(FST + SND)
    except EOFError:
        break

print(sum_value)
