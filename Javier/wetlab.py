def test_gene(x):
    x1 = x[:, 0]
    x2 = x[:, 1]*2
    x3 = x[:, 2]*2
    x4 = x[:, 3]/2
    
    term1 = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2 ** 2) * x2 ** 2
    term4 = x3 * x4
    y = term1 + term2 + term3 + term4 - 5
    return y[:,None]
    