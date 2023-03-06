def main():
    print(str_extgcd(37,60))


def gcd(a: int, b:int) -> int:
    r = b

    while True:
        q = a//b
        if a%b == 0:
            return r
        r = a % b

        a = b
        b = r


def extgcd(a: int, b:int) -> tuple[int, int, int]:
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
def str_extgcd(a: int, b:int) -> str:
    r = b
    steps = []
    out = []

    #euclidean
    while True:
        q = a//b
        if a%b == 0:
            break
        r = a % b


        steps.append([a,q,b,r])
        out.append(f"{a}={q}*{b}+{r}")

        a = b
        b = r

    out.append(f"{a}={q}*{b}+0")
    out.append("")

    #extension
    x = 1
    a = steps[-1][0]
    y = steps[-1][1]*-1v

    out.append(f"{r}={x}*{a}+{y}*{b}")

    for i in range(len(steps)-1):
        b = a
        a = steps[-2-i][0]
        _y = y
        y = x + (steps[-2-i][1]*-1)*y
        x = _y
        out.append(f"{r}={x}*{a}+{y}*{b}")


    out = "\n".join(out).replace("+-","-").replace("*","\u2022")

    return out



if __name__ == '__main__':
    main()