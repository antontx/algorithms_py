from collections import deque

def isValid(s: str):
    stack = deque()

    pairs = {
        "(":")",
        "[":"]",
        "{":"}"
    }

    for bracket in s:
        if bracket in ["(","[","{"]:
            stack.append(bracket)
        elif len(stack) == 0 or bracket != pairs[stack.pop()]:
            return False

    if len(stack) == 0:
        return True
    return False

