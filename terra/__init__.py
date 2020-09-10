from joblib import Memory

cache_dir = "/afs/cs.stanford.edu/u/sabrieyuboglu/code/terra/_cache"
memory = Memory(cache_dir, verbose=0)


@memory.cache
def test_function(x="sabri"):
    return f"{x} eyuboglu 2"