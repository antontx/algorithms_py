from typing import List
import numpy as np


def main():
    example_matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
    print(np.matrix(example_matrix))

    rotate(example_matrix)
    print(np.matrix(example_matrix))


def rotate(matrix: List[List[int]]) -> None:
    ring_width = len(matrix)-1

    for ring in range(2):
        for j in range(0,ring_width):
            x,y = ring+j,ring
            store = matrix[y][x]

            for i in range(4):
                nX, nY = ring_width-y+(ring*2),x
                store,matrix[nY][nX] = matrix[nY][nX],store
                x,y = nX,nY

        ring_width -= 2


if __name__ == '__main__':
    main()
