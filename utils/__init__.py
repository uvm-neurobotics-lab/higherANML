
def unzip(l):
    """
    Transpose a list of lists.
    """
    return list(zip(*l))


def divide_chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    """
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i : i + n]
