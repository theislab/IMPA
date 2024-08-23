import numpy as np
import scipy.sparse
import argparse

def load_sparse(filename):
    """Loads sparse from Matrix market or Numpy .npy file."""
    if filename is None:
        return None
    if filename.endswith('.mtx'):
        return scipy.io.mmread(filename).tocsr()
    elif filename.endswith('.npy'):
        return np.load(filename, allow_pickle=True).item().tocsr()
    raise ValueError(f"Loading '{filename}' failed. It must have a suffix '.mtx' or '.npy'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sphere exclusion on the given X")
    parser.add_argument("--x", help="Molecule descriptor file, e.g., ECFPs (matrix market or numpy)", type=str, required=True)
    parser.add_argument("--out", help="Output file for the clusters (.npy)", type=str, required=True)
    parser.add_argument("--dists", nargs="+", help="Distances", type=float, required=True)

    args = parser.parse_args()
    print(args)

    print(f"Loading '{args.x}'.")
    X = load_sparse(args.x).tocsr()

    print("Clustering.")
    # two step clustering, first at 0.5, then at 0.6
    cl_hier = hierarchical_clustering(X, dists=args.dists)

    np.save(args.out, cl_hier)
    print(f"Saved clustering into '{args.out}'.")