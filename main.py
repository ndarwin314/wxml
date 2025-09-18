import numpy as np
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Execution time: {round(time.time()-start, 4)}")
        return result
    return wrapper

def power_method(matrix, size, prec=10**(-4)):
    x = np.ones((size, ))
    previous = np.zeros((size, ))
    while np.linalg.norm((previous - x)) > prec:
        previous = x
        x = matrix@x
        x /= np.linalg.norm(x)
    return x, x.T@matrix@x


@timer
def main(n=100, m=None):
    if m is None:
        m = 3 * n ** (1 / 3)
    complete = np.ones((n, n)) - np.identity(n)
    distance = np.zeros((n + m, n + m))
    distance[:n, :n] = complete
    for i in range(n, n + m):
        for j in range(n, n + m):
            distance[i][j] = abs(i - j)
    for i in range(m):
        distance[0:n - 1, i + n] = i + 2
        distance[i + n, 0:n - 1] = i + 2
        distance[n - 1, i + n] = i + 1
        distance[i + n, n - 1] = i + 1
    normalized, l1 = power_method(distance, m+n)
    print(round(normalized.sum() / np.sqrt(n + m), 4))

@timer
def comet(n=100, m=None):
    if m is None:
        m = 3*n ** (1/3)
    distance = np.zeros((n + m, n + m))
    distance[:n, :n] = 2*(np.ones((n, n)) - np.identity(n))
    distance[n-1] = 1
    distance[:, n-1] = 1
    distance[n-1, n-1] = 0
    for i in range(n, n + m):
        for j in range(n, n + m):
            distance[i][j] = abs(i - j)
    for i in range(m):
        distance[0:n - 1, i + n] = i + 2
        distance[i + n, 0:n - 1] = i + 2
        distance[n - 1, i + n] = i + 1
        distance[i + n, n - 1] = i + 1
    normalized, l1 = power_method(distance, m+n, )
    print(round(normalized.sum() / np.sqrt(n + m), 4))

def row_compute(n: int, m: int, i: int):
    row: np.ndarray = np.ones((m+n,))
    if i < n-1:
        row[i] = 0
        row[n:] = np.arange(2, m+2)
    elif i == n-1:
        row[n-1] = 0
        row[n:] = np.arange(1, m+1)
    else:
        row[:n-1] = i - n + 2
        for j in range(n-1, n+m):
            row[j] = abs(i-j)
    return row


def power_low_mem(n: int, m: int, prec=10**(-4)):
    current = np.ones((n + m,))
    #current[n:] = np.arange(1,m+1)
    previous = np.zeros((n + m,))
    while np.linalg.norm(current-previous) > prec:
        previous = current.copy()
        for i in range(n+m):
            current[i] = np.dot(row_compute(n, m, i), previous)
        current /= np.linalg.norm(current)
        #print(current)
    return current, np.dot(row_compute(n, m, i), current) / current[0]

@timer
def complete_low_mem(n: int, m: int, prec=10**(-4)):
    vector, value = power_low_mem(n, m, prec)
    return vector.sum()/np.sqrt(n+m), value

@timer
def complete_reduction(m: int, prec=10**(-5)):
    n = m**2
    distance = np.zeros((2 + m, 2 + m))
    distance[0, 0] = n-2
    distance[0, 1] = 1
    distance[1, 0] = n-1
    for i in range(2, m+2):
        distance[i, 0] = i*(n-1)
        distance[i, 1] = i-1
        for j in range(2, m+2):
            distance[i][j] = abs(i - j)
    distance[2][2] = 0
    for i in range(m):
        distance[0:2 - 1, i + 2] = i + 2
        distance[2 - 1, i + 2] = i + 1
    pre, l1 = power_method(distance, m+2, prec)
    corrected = np.zeros((n+m,))
    corrected[n-2:] = pre
    corrected[:n-2] = pre[0]
    corrected /= np.linalg.norm(corrected)
    return round(corrected.sum() / np.sqrt(n + m), 4), l1

print(complete_reduction(10000))