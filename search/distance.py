"""Unicode-aware Levenshtein demonstrator, separate from unchanged HW1 statistics."""
import unicodedata

def normalize(text, form='NFC', fold=True):
    value=unicodedata.normalize(form,text) if form in {'NFC','NFKC'} else text
    return unicodedata.normalize(form,value.casefold()) if fold and form in {'NFC','NFKC'} else value.casefold() if fold else value

def edit_distance(left,right):
    previous=list(range(len(right)+1))
    for i,a in enumerate(left,1):
        current=[i]
        for j,b in enumerate(right,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(a!=b)))
        previous=current
    return previous[-1]


def edit_matrix(left, right):
    """Return the full Levenshtein DP grid and one optimal backtrace path."""
    matrix = [list(range(len(right) + 1))]
    for i, left_char in enumerate(left, 1):
        row = [i]
        for j, right_char in enumerate(right, 1):
            row.append(min(
                row[-1] + 1,
                matrix[i - 1][j] + 1,
                matrix[i - 1][j - 1] + (left_char != right_char),
            ))
        matrix.append(row)

    path = set()
    i, j = len(left), len(right)
    while True:
        path.add((i, j))
        if i == 0 and j == 0:
            break
        value = matrix[i][j]
        if i and j and matrix[i - 1][j - 1] + (left[i - 1] != right[j - 1]) == value:
            i, j = i - 1, j - 1
        elif i and matrix[i - 1][j] + 1 == value:
            i -= 1
        else:
            j -= 1

    return {
        'columns': ['∅', *right],
        'rows': [
            {
                'label': '∅' if i == 0 else left[i - 1],
                'cells': [
                    {'value': value, 'on_path': (i, j) in path}
                    for j, value in enumerate(row)
                ],
            }
            for i, row in enumerate(matrix)
        ],
        'distance': matrix[-1][-1],
    }
