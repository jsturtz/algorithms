

# def find_missing(lst):

#     min = lst[0]
#     max = lst[-1]

#     # This is the index of median for odd case, larger index for even case
#     mid = int(len(lst) / 2)

#     # if lst is even, then median is average between middle two
#     actual_median = (lst[mid] + lst[mid-1]) / 2 if len(lst) % 2 == 0 else lst[mid]
#     expected_median = (min + max) / 2

#     # the missing value must be on the right, if at all
#     if expected_median > actual_median:
#         return find_missing(lst[mid+1:])
#     elif expected_median < actual_median:
#         return find_missing(lst[: mid-1])
#     else:
#         # Base case. The median we computed is correct, so that's what's missing
#         return expected_median

# lst = [5, 6, 7, 8, 10]
# print(find_missing(lst))
# print(find_missing([-8, -6]))

def get_minimums(A):

    evens_minimums = get_minimum_even_cols(A)
    odds_minimums = get_odds(A, evens_minimums)

    is_odd = len(A) % 2 == 1
    total_mins = [evens_minimums.pop(0)] if is_odd else []
    for odd, even in zip(odds_minimums, evens_minimums):
        total_mins.extend([odd, even])
    return total_mins

def get_odds(A, even_mins):

    odds = list(filter(None, [row if i % 2 == 1 else None for i, row in enumerate(A)]))
    odd_mins = []
    for i, row in enumerate(odds):
        colstart = even_mins[i]
        colend = even_mins[i+1]
        mincol = colstart
        min = row[mincol]
        for i in range(colstart, colend):
            if row[i] < min:
                mincol = i
                min = odds[0][mincol]
        odd_mins.append(mincol)
    return odd_mins

def get_minimum_even_cols(A):
    evens = list(filter(None, [row if i % 2 == 0 else None for i, row in enumerate(A)]))
    numcols = len(A[0])
    return get_minimums_by_column(evens, colstart=0, colend=numcols)

def get_minimums_by_column(rows, colstart, colend):

    numrows = len(rows)
    numcols = len(rows[0])

    # Base Case
    if numrows == 1:
        mincol = colstart
        min = rows[0][colstart]
        for i in range(colstart, colend):
            if rows[0][i] < min:
                mincol = i
                min = rows[0][mincol]
        return [mincol]

    # Recursive Case
    midrow = int(numrows / 2)
    abovemins = get_minimums_by_column(rows[:midrow], colstart=colstart, colend=numcols)
    belowmins = get_minimums_by_column(rows[midrow:], colstart=abovemins[-1], colend=numcols)
    return abovemins + belowmins

# def get_evens(A):
#     evens = list(filter(None, [row if i % 2 == 0 else None for i, row in enumerate(A)]))
#     return get_minimums(evens, 0, len(A[0]))

# def get_odds(A, even_minimums):

#     # Base Case
#     # if there's only one row to check
#     if len(even_minimums) == 2:
#         return get_minimums(A, even_minimums[0], even_minimums[1])
    
#     else:


#     evens = list(filter(None, [row if i % 2 == 0 else None for i, row in enumerate(A)]))
#     return get_minimums(evens, 0, len(A[0]))
    
A = [
    [1, 2, 8],
    [2, 1, 3],
    [9, 3, 2],
]

print(get_minimums(A))
# evens = get_evens(A)
# odds = get_odds(A)

