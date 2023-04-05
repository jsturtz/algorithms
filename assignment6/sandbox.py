
# Recursive solution without DP
def longest_palindrome_subsequence(s):

    # Base case: string has one character in it
    if len(s) == 1:
        return s

    # Recursive case
    if s[0] == s[-1]:
        middle = longest_palindrome_subsequence(s[1:-1])
        return s[0] + middle + s[-1]
    else:
        left = longest_palindrome_subsequence(s[1:])
        right = longest_palindrome_subsequence(s[:-1])

        val = left if len(left) > len(right) else right
        return val

def longest_palindrome_subsequence_memo(s):

    def recursive_helper(s, memo):

        # Don't recompute something we've memoized!
        if s in memo:
            return memo[s]

        if len(s) == 1:
            return s

        if s[0] == s[-1]:
            middle = recursive_helper(s[1:-1], memo)
            val = s[0] + middle + s[-1]
            memo[s] = val
            return val
        else:
            left = recursive_helper(s[1:], memo)
            right = recursive_helper(s[:-1], memo)

            val = left if len(left) > len(right) else right
            memo[s] = val
            return val

    return recursive_helper(s, {})

def longest_palindrome_subsequence_bottom_up(s):

    # the longest palindrome for every c in s is c itself
    memo = {c: c for c in s}
    memo[""] = ""

    # we slide over s with window to compute subproblems in bottom-up fashion
    for window in range(2, len(s) + 1):
        for i in range(len(s) - window + 1):
            subproblem = s[i:i+window]

            if subproblem[0] == subproblem[-1]:
                palindrome = subproblem[0] + memo[subproblem[1:-1]] + subproblem[-1]
            else:
                left = memo[subproblem[1:]]
                right = memo[subproblem[:-1]]
                palindrome = left if len(left) > len(right) else right

            memo[subproblem] = palindrome

    return memo[s]

s = 'ccharacter'

print(longest_palindrome_subsequence(s))
print(longest_palindrome_subsequence_memo(s))
print(longest_palindrome_subsequence_bottom_up(s))



def print_neatly(text, max_len):

    rows = []
    i = j = 0
    while words_printed < len(text):
        linesum=0
        while max_len - j + i - linesum > 0:
            # print(text[words_printed], end=" ")
            linesum+=len(text[words_printed])
            words_printed += 1
            words_on_line += 1
            if words_printed >= len(text): break
        rows.append(words_on_line)
    return rows

text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
text = text.split(" ")
rows=print_neatly(text, 50)

start=0
for row in rows:
    print(text[start:row])
