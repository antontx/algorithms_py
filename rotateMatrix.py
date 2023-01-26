from typing import List


def main():
    inp = [[1,2,3],[4,5,6],[7,8,9]]
    out = [[7,4,1],[8,5,2],[9,6,3]]

    inp2 = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]

    rotate(inp)
    print("---------")
    rotate(inp2)





def rotate(matrix: List[List[int]]) -> None:
    size = len(matrix)**2
    ring_width = len(matrix)
    ring = 0
    x,y = 0,0

    for ring in range(2):
        print(f"---{ring=} {ring_width=}")


        for j in range(0,ring_width-1):
            x,y = ring+j,ring
            for i in range(4):
                nX, nY = ring_width-1-y+(ring*2),x
                print(matrix[y][x])
                x,y = nX,nY

        ring_width -= 2





if __name__ == '__main__':
    main()
