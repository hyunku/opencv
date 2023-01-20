# opencv 사용에 최소로 필요한 파이썬 문법

v1 = 1.2 + 3.4j
v2 = 100
v3 = float(v2)
print(f'v1 type: {v1}, {type(v1)}')
print(f'v2 type: {v2}, {type(v2)}')
print(f'v3 type: {v3}, {type(v3)}')

li1 = [1,2,3,4]
tu1 = (1,1.5,'a')
di1 = {"name":"강현구"}
se1 = set(li1)

print(f'li1 type: {li1}, {type(li1)}')
print(f'tu1 type: {tu1}, {type(tu1)}')
print(f'di1 type: {di1}, {type(di1)}')
print(f'se1 type: {se1}, {type(se1)}')


title = '서기' \
    '오늘' \
    '까지'
year, month = 2020, 1
day = 7; ratio = 365.42
print(day)

a = [0,1,2,3,4,5,6,7,8,9]
print(a[:2])
print(a[2::2])
print(a[1::-1])
print(a[8:1:-2])

kor = [7,8,9,4,5]
eng = [9,8,7,7,6]
fdsf = []
a2 = []

for idx, val in enumerate(kor):
    fdsf.append(idx)
    a2.append(val)

print(fdsf)
print(a2)

print("#####")

for k, e in zip(kor, eng):
    print(k)
    print(e)

print("#####")




