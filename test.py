from collections.abc import Generator

def averager() -> Generator[float, float, None]:
    total = 0.0
    count = 0
    average = 0.0
    while True:
        term = yield average
        total += term
        count += 1
        average = total / count
        
coro_avg = averager()
print(next(coro_avg))

print(coro_avg.send(10))
print(coro_avg.send(30))
print(coro_avg.send(5))