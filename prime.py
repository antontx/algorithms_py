import math
import time

def isPrime(num):
    for i in range(2,int(math.sqrt(num))-1):
        if num % i == 0:
            return False
    return True

def main():
    numbers = 0
    for number in range(numbers):
        print(f"{number}: {isPrime(number)}")

    testzahlen = [
        11,
        101,
        1009,
        10007,
        100003,
        1000003,
        10000019,
        100000007,
        1000000007,
        10000000019,
        100000000003,
        1000000000039,
        10000000000037,
        100000000000031,
        1000000000000037,
        10000000000000061,
        100000000000000003,
        1000000000000000003,
        10000000000000000051,
        100000000000000000039,
        1000000000000000000117,
        10000000000000000000009,
        100000000000000000000117,
        1000000000000000000000007,
        10000000000000000000000013,
        100000000000000000000000067,
        1000000000000000000000000103,
        10000000000000000000000000331,
        100000000000000000000000000319]
    start_time = time.time()

    for number in testzahlen:
        print(f"{number}: {isPrime(number)} - {time.time()-start_time: .2f}s")

if __name__ == '__main__':
    main()