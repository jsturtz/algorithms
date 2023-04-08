
# # Recursive solution without DP
# def longest_palindrome_subsequence(s):

#     # Base case: string has one character in it
#     if len(s) == 1:
#         return s

#     # Recursive case
#     if s[0] == s[-1]:
#         middle = longest_palindrome_subsequence(s[1:-1])
#         return s[0] + middle + s[-1]
#     else:
#         left = longest_palindrome_subsequence(s[1:])
#         right = longest_palindrome_subsequence(s[:-1])

#         val = left if len(left) > len(right) else right
#         return val

# def longest_palindrome_subsequence_memo(s):

#     def recursive_helper(s, memo):

#         # Don't recompute something we've memoized!
#         if s in memo:
#             return memo[s]

#         if len(s) == 1:
#             return s

#         if s[0] == s[-1]:
#             middle = recursive_helper(s[1:-1], memo)
#             val = s[0] + middle + s[-1]
#             memo[s] = val
#             return val
#         else:
#             left = recursive_helper(s[1:], memo)
#             right = recursive_helper(s[:-1], memo)

#             val = left if len(left) > len(right) else right
#             memo[s] = val
#             return val

#     return recursive_helper(s, {})

# def longest_palindrome_subsequence_bottom_up(s):

#     # the longest palindrome for every c in s is c itself
#     memo = {c: c for c in s}
#     memo[""] = ""

#     # we slide over s with window to compute subproblems in bottom-up fashion
#     for window in range(2, len(s) + 1):
#         for i in range(len(s) - window + 1):
#             subproblem = s[i:i+window]

#             if subproblem[0] == subproblem[-1]:
#                 palindrome = subproblem[0] + memo[subproblem[1:-1]] + subproblem[-1]
#             else:
#                 left = memo[subproblem[1:]]
#                 right = memo[subproblem[:-1]]
#                 palindrome = left if len(left) > len(right) else right

#             memo[subproblem] = palindrome

#     return memo[s]

# s = 'ccharacter'

# print(longest_palindrome_subsequence(s))
# print(longest_palindrome_subsequence_memo(s))
# print(longest_palindrome_subsequence_bottom_up(s))

def print_neatly(text, max_len):

    n = len(text)

    # costs[k] is holding the cost of printing neatly from k through n
    costs = {n: 0} # cost of last word guaranteed to be zero

    # positions[k] holds the position of the last words that should be on the kth line
    positions = {}

    # the "base case" in this bottom-up implementation is the last word to print
    # so we start there, working our way to cost[0], which holds the final cost
    for k in list(reversed(range(n))):

        # the words at the end that sum together to be less than max_len start with a zero cost
        charactersum = sum(len(text[i]) for i in range(k, n))
        whitespacesum = n - 1 - k
        
        if charactersum + whitespacesum < max_len:
            costs[k] = 0

        # at this point, we don't know which option yields the mincost
        # so we iterate over all the possibilities, using the lookup table costs
        mincost=float('inf')

        # j is a value we add to k, so j represents how many words we are considering
        for j in range(n-k):

            # space left computes the total character count from k to k+j plus whitespace between
            space_left = sum(len(text[k+i]) + i for i in range(j+1))
            if space_left > max_len:
                # no point in continuing this loop since more looping will only add to space_left
                break 

            # candidate cost computes the cube of that space left plus the (already computed) cost from k+j+1 to n
            candidate_cost = pow(max_len - space_left, 3) + costs[k + j + 1]
            if candidate_cost < mincost:
                mincost = candidate_cost
                positions[k] = k + j

        # once we've found the smallest cost option, that is the cost of printing neatly from k to n
        costs[k] = mincost

    return positions, costs

# text = "ABCD E FG HIJKLM".split(" ")
# positions, costs = print_neatly(text, 7)

# start=0
# while start < len(text):
#     end = positions.get(start)
#     print(" ".join(text[start:end+1]))
#     start = end+1
# print(f"FINAL COST OF SOLUTION: {costs[0]}")

def edit_distance(x, y):

    operations = {
        "copy": 1,
        "twiddle": 2, # should be cheap
        "replace": 3,
        "insert": 4,
        "delete": 4,
        "kill": 0,
    }

    def recursive_helper(x, y, memo, operations):

        # memoize
        if (x, y) in memo:
            return memo[(x, y)]

        # base case: nothing else to transform
        if len(y) == 0:
            return (operations["kill"], ["kill"])

        # decide on the valid operations
        valid_operations = ["insert", "delete"]
        if len(x) == 0 and len(y) > 0:
            # There's only one thing to do, which is insert
            valid_operations = ["insert"]
        elif x[0] == y[0]:
            # copy only valid if the current leading characters match
            valid_operations.append("copy")
        else:
            # and replace is only valid otherwise
            valid_operations.append("replace")

        # twiddle can only be used if the next two characters match
        if len(x) > 1 and len(y) > 1 and x[0] == y[1] and x[1] == y[0]:
            valid_operations.append("twiddle")

        mincost = float('inf')
        ops = []
        for candidate_op in valid_operations:
            opcost = operations[candidate_op]
            if candidate_op == "copy":
                nextop = f"copy {x[0]}"
                cost_after_op, candidate_ops = recursive_helper(x[1:], y[1:], memo, operations)
            elif candidate_op == "twiddle":
                nextop = f"twiddle {x[0]}, {x[1]} to {x[1]}, {x[0]}"
                cost_after_op, candidate_ops = recursive_helper(x[2:], y[2:], memo, operations)
            elif candidate_op == "replace":
                nextop = f"replace {y[0]} with {x[0]}"
                cost_after_op, candidate_ops = recursive_helper(x[1:], y[1:], memo, operations)
            elif candidate_op == "delete":
                nextop = f"delete {y[0]}"
                cost_after_op, candidate_ops = recursive_helper(x[1:], y, memo, operations)
            elif candidate_op == "insert":
                nextop = f"insert {y[0]}"
                cost_after_op, candidate_ops = recursive_helper(x, y[1:], memo, operations)
            
            if opcost + cost_after_op < mincost:
                mincost = opcost + cost_after_op
                ops = [nextop] + candidate_ops

        # memoize
        memo[(x, y)] = (mincost, ops)
        return (mincost, ops)

    return recursive_helper(x, y, {}, operations)

def edit_distance_bottomup(x, y, operations):

    operations = {
        "copy": 1,
        "twiddle": 2, # should be cheap
        "replace": 3,
        "insert": 4,
        "delete": 4,
        "kill": 0,
    }

    # regardless of size of x, if y is empty, we just kill
    memo = {(x[i:], ""): (operations["kill"], ["kill"]) for i in list(range(len(x)+1))}

    for j in list(reversed(range(len(y)))):
        for i in list(reversed(range(len(x)+1))):

            suby = y[j:]
            subx = x[i:]

            # decide on the valid operations
            valid_operations = ["insert", "delete"]
            if len(subx) == 0 and len(suby) > 0:
                # There's only one thing to do, which is insert
                valid_operations = ["insert"]
            elif subx[0] == suby[0]:
                # copy only valid if the current leading characters match
                valid_operations.append("copy")
            else:
                # and replace is only valid otherwise
                valid_operations.append("replace")

            # twiddle can only be used if the next two characters match
            if len(subx) > 1 and len(suby) > 1 and subx[0] == suby[1] and subx[1] == suby[0]:
                valid_operations.append("twiddle")

            mincost = float('inf')
            ops = []
            for candidate_op in valid_operations:
                opcost = operations[candidate_op]
                if candidate_op == "copy":
                    nextop = f"copy {subx[0]}"
                    cost_after_op, candidate_ops = memo[(subx[1:], suby[1:])]
                elif candidate_op == "twiddle":
                    nextop = f"twiddle {subx[0]}, {subx[1]} to {subx[1]}, {subx[0]}"
                    cost_after_op, candidate_ops = memo[(subx[2:], suby[2:])]
                elif candidate_op == "replace":
                    nextop = f"replace {suby[0]} with {subx[0]}"
                    cost_after_op, candidate_ops = memo[(subx[1:], suby[1:])]
                elif candidate_op == "delete":
                    nextop = f"delete {suby[0]}"
                    cost_after_op, candidate_ops = memo[(subx[1:], suby)]
                elif candidate_op == "insert":
                    nextop = f"insert {suby[0]}"
                    cost_after_op, candidate_ops = memo[(subx, suby[1:])]
                
                if opcost + cost_after_op < mincost:
                    mincost = opcost + cost_after_op
                    ops = [nextop] + candidate_ops
            memo[(subx, suby)] = (mincost, ops)

    return memo[(x, y)]

# cost, ops = edit_distance_bottomup("algorithm", "altrustic")

# for op in ops:
#     print(op)
# print(f"final cost: {cost}")

class Node():

    def __init__(self, 
                 name,
                 conviv,
                 parent=None, 
                 leftchild=None, 
                 rightsib=None) -> None:
        self.name = name
        self.conviv = conviv
        self.parent=parent
        self.leftchild=leftchild
        self.rightsib=rightsib

    def addchild(self, newchild):

        old_left = self.leftchild
        self.leftchild = newchild
        self.leftchild.rightsib = old_left
        return self

    def addchildren(self, children):
        for child in children:
            self.addchild(child)
        return self

def optimal_company_party(root):
    """
    Assumes root is a Node class, defined as such:

    class Node():

        def __init__(self, name, conviv, parent=None, leftchild=None, rightsib=None):
            self.name = name
            self.conviv = conviv
            self.parent=parent
            self.leftchild=leftchild
            self.rightsib=rightsib
    """
    # base case
    if root.leftchild is None:
        return {
            "optimal_cost": root.conviv,
            "optimal_cost_without": 0,
            "optimal_party": [root],
            "optimal_party_without": [],
        }

    # recursive cases
    cost_with_root = root.conviv
    party_with_root = [root]

    cost_without_root = 0
    party_without_root = []

    child = root.leftchild
    while child is not None:
        results = optimal_company_party(child)

        cost_with_root += results["optimal_cost_without"]
        party_with_root.extend(results["optimal_party_without"])

        cost_without_root += results["optimal_cost"]
        party_without_root.extend(results["optimal_party"])

        child = child.rightsib

    if cost_with_root > cost_without_root:
        optimal_cost = cost_with_root
        optimal_party = party_with_root
    else:
        optimal_cost = cost_without_root
        optimal_party = party_without_root

    return {
        "optimal_cost": optimal_cost,
        "optimal_party": optimal_party,
        "optimal_cost_without": cost_without_root,
        "optimal_party_without": party_without_root,
    }

root = Node("Dr. Professor Jordan, Esq", 100).addchildren([
    Node("Guy1", 10).addchildren([
        Node("Guy1Child1", 20).addchildren([
            Node("Guy1Child1Child1", 30),
            Node("Guy1Child1Child2", 40),
            Node("Guy1Child1Child3", 50),
        ]),
        Node("Guy1Child2", 60).addchildren([
            Node("Guy1Child2Child1", 70),
            Node("Guy1Child2Child2", 80),
            Node("Guy1Child2Child3", 90),
        ]),
        Node("Guy1Child3", 100).addchildren([
            Node("Guy1Child3Child1", 10),
            Node("Guy1Child3Child2", 20),
            Node("Guy1Child3Child3", 30),
        ]),
    ]),
    Node("Guy2", 50).addchildren([
        Node("Guy2Child1", 10).addchildren([
            Node("Guy2Child1Child1", 50),
            Node("Guy2Child1Child2", 20),
            Node("Guy2Child1Child3", 30),
        ]),
        Node("Guy2Child2", 80).addchildren([
            Node("Guy2Child2Child1", 20),
            Node("Guy2Child2Child2", 20),
            Node("Guy2Child2Child3", 10),
        ]),
        Node("Guy2Child3", 100).addchildren([
            Node("Guy2Child3Child1", 100),
            Node("Guy2Child3Child2", 100),
            Node("Guy2Child3Child3", 10),
        ]),
    ]),
    Node("Guy3", 10).addchildren([
        Node("Guy3Child1", 10).addchildren([
            Node("Guy3Child1Child1", 50),
            Node("Guy3Child1Child2", 100),
            Node("Guy3Child1Child3", 80),
        ]),
        Node("Guy3Child2", 100).addchildren([
            Node("Guy3Child2Child1", 40),
            Node("Guy3Child2Child2", 10),
            Node("Guy3Child2Child3", 10),
        ]),
        Node("Guy3Child3", 10).addchildren([
            Node("Guy3Child3Child1", 100),
            Node("Guy3Child3Child2", 50),
            Node("Guy3Child3Child3", 40),
        ]),
    ]),
])

# guy1 = Node("Guy1", 80)

# guy1.addchild(Node("Guy1Child1", 100))
# guy1.addchild(Node("Guy1Child2", 80))
# guy1.addchild(Node("Guy1Child3", 70))

# guy2 = Node("Guy2", 10)
# guy3 = Node("Guy3", 60)
# guy2.addchild(Node("Guy2Child1", 80))
# guy2.addchild(Node("Guy2Child2", 60))
# guy2.addchild(Node("Guy2Child3", 10))
# guy3.addchild(Node("Guy3Child1", 20))
# guy3.addchild(Node("Guy3Child2", 60))
# guy3.addchild(Node("Guy3Child3", 90))
# root.addchild(guy1)
# root.addchild(guy2)
# root.addchild(guy3)

results = optimal_company_party(root)
print(f"Total cost: {results['optimal_cost']}")
for node in results['optimal_party']:
    print(node.name)
# def longest_sub_alphabetical(s):

#     def recurse_aux(s, memo):

#         # memoize
#         if s in memo:
#             return memo[s]

#         # base case
#         if len(s) == 1:
#             return s

#         # recursive case
#         left = recurse_aux(s[1:], memo)
#         right = recurse_aux(s[:-1], memo)
#         left = s[0] + left if s[0] < left[0] else left
#         right = right + s[-1] if s[-1] > right[-1] else right

#         best = left if len(left) > len(right) else right
#         memo[s] = best
#         return best

#     return recurse_aux(s, {})

# def longest_sub_alphabetical_bottom_up(s):

#     memo = {c:c for c in s}
#     memo[""] = ""

#     for window in range(2, len(s)+1):

#         for i in range(len(s) - window + 1):

#             substring = s[i:i+window]

#             left = memo[substring[1:]]
#             right = memo[substring[:-1]]

#             left = substring[0] + left if substring[0] < left[0] else left
#             right = right + substring[-1] if substring[-1] > right[-1] else right

#             best = left if len(left) > len(right) else right
#             memo[substring] = best

#     return memo[s]

# print(longest_sub_alphabetical("algorithm"))
# print(longest_sub_alphabetical_bottom_up("algorithm"))
