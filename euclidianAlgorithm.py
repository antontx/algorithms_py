def main():
    print(gcd(192,35))


def gcd(a: int, b:int) -> int:
    r = b

    while True:
        q = a//b
        if a%b == 0:
            return r
        r = a % b

        a = b
        b = r


def extgcd(a: int, b:int) -> int:
    r = b
    steps = []
    
    #euclidean 
    while True:
        q = a//b
        if a%b == 0:
            break
        r = a % b


        steps.append([a,q,b,r])

        a = b
        b = r
    
    #extension
    x = 1
    a = steps[-1][0]
    y = steps[-1][1]*-1
    b = steps[-1][2]
    
    for i in range(len(steps)-1):
        b = a
        a = steps[-2-i][0]
        _y = y
        y = x + (steps[-2-i][1]*-1)*y
        x = _y
        
    return(r,x,y)



if __name__ == '__main__':
    main()