from typing import List


def longestCommonPrefix(strs: List[str]) -> str:
    prefix = []
    minStr = min(strs, key=len)
    for index, letter in enumerate(minStr):
        for word in strs:
            if letter != word[index]:
                return "".join(prefix)
        prefix.append(letter)

    return "".join(prefix)


strs = ["flower", "flow", "flight"]
print(longestCommonPrefix(strs))
