import math

# helper functions 
def get_distance(A, B, metric='euclidean'):
    assert len(A) == len(B), "Choose Points of same dimension"

    # euclidean metric 
    if metric=='euclidean':
        return math.sqrt(sum([((A[i]-B[i]))**2 for i in range(len(A))])) 

    elif metric=='markorov':
        pass    

    else:
        raise KeyError(f"Invalid Metric '{metric}' (Valid: ['euclidean','markorov']")


def test():
    x = get_distance([0,0,1], [1,1])
    print(x)


