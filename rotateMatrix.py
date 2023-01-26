from typing import List
import numpy as np


def main():
    example_matrix = [[2,29,20,26,16,28],[12,27,9,25,13,21],[32,33,32,2,28,14],[13,14,32,27,22,26],[33,1,20,7,21,7],[4,24,1,6,32,34]]
    print(np.matrix(example_matrix))

    rotate(example_matrix)
    print(np.matrix(example_matrix))


def rotate(matrix: List[List[int]]) -> None:
    ring_width = len(matrix)-1
    ring_count = int (len(matrix) / 2) + len(matrix) % 2

    for ring in range(ring_count):
        print(f"{ring=} {ring_width=}")
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
