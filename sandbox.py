
from typing import Tuple

# Recursive solution without DP
def longest_palindrome_subsequence(s: str) -> str:

    def aux(s: str, begin: int, end: int) -> Tuple[str, int]:

        # Base case: string has one character in it
        if begin == end:
            return (s[begin], 1)

        # Recursive case
        if s[begin] == s[end]:
            # if the characters on the end are identical, then we can add them to the longest palindrome between them
            middle, middle_length = aux(s, begin+1, end-1)
            return (s[begin] + middle + s[end], middle_length + 2)
        else:
            # if they aren't identical, we need to figure out which is the longest:
            # (1) the left (ignoring the right character) or (2) the right (ignoring the left character)
            left, left_length = aux(s, begin+1, end)
            right, right_length = aux(s, begin, end-1)

            return (left, left_length) if left_length > right_length else (right, right_length)

    return aux(s, 0, len(s)-1)[0]

# Recursive solution without DP
def longest_palindrome_subsequence_dp(s: str) -> str:

    def aux(s: str, begin: int, end: int, memo: dict) -> Tuple[str, int]:

        # Don't recompute something we've memoized!
        if s[begin:end] in memo:
            return (s[begin:end], memo[s[begin:end]])

        # Base case: string has one character in it
        if end == begin:
            memo[s[begin]] = 1
            return (s[begin], 1)

        # Recursive case
        if s[begin] == s[end]:
            # if the characters on the end are identical, then we can add them to the longest palindrome between them
            middle, middle_length = aux(s, begin+1, end-1, memo)
            val = s[begin] + middle + s[end]
            memo[s[begin:end]] = middle_length + 2
            return (s[begin] + middle + s[end], middle_length + 2)
        else:
            # if they aren't identical, we need to figure out which is the longest:
            # (1) the left (ignoring the right character) or (2) the right (ignoring the left character)
            left, left_length = aux(s, begin+1, end, memo)
            right, right_length = aux(s, begin, end-1, memo)

            val, length = (left, left_length) if left_length > right_length else (right, right_length)
            memo[s[begin:end]] = length
            return (val, length)

    return aux(s, 0, len(s)-1, {})[0]

s = 'character'

p1 = longest_palindrome_subsequence(s)
p2 = longest_palindrome_subsequence_dp(s)
print(p1)
print(p2)
