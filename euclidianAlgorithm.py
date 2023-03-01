def main():
    print(gcd(192,35))


def gcd(a: int, b:int) -> int:
    r = b

    while True:
        q = a//b
        if a%b == 0:
            return r
        r = a % b


        print(a,q,b,r)

        a = b
        b = r




if __name__ == '__main__':
    main()