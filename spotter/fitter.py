def piecewise_const(x, y, edges):
    result_x = []
    result_y = []

    for a, b in edges:
        mask = (x >= a) & (x <= b)
        result_x.append(x[mask].min())
        result_y.append(y[mask].mean())
        result_x.append(x[mask].max())
        result_y.append(y[mask].mean())

    return result_x, result_y
