import csr
import scipy

def sparse_print(matrix):
    print(matrix.row_ptrs)
    print(matrix.col_indices)
    print(matrix.data)

mat_foo = csr.MyCSR(3,3,4)
mat_foo.row_ptrs = scipy.array([0,1,3,4])
mat_foo.col_indices = scipy.array([0,1,2,1])
mat_foo.data = scipy.ones(4)
dmat_foo = scipy.array([[1,0,0],
                        [0,1,1],
                        [0,1,0]], dtype=scipy.float_)
assert((dmat_foo == mat_foo.to_dense()).all())
print("Left matrix")
sparse_print(mat_foo)
print(mat_foo.to_dense())

mat_bar = csr.MyCSR(3,3,5)
mat_bar.row_ptrs = scipy.array([0,1,3,5])
mat_bar.col_indices = scipy.array([1,0,2,1,2])
mat_bar.data = scipy.ones(5)
dmat_bar = scipy.array([[0,1,0],
                        [1,0,1],
                        [0,1,1]], dtype=scipy.float_)
assert((dmat_bar == mat_bar.to_dense()).all())
print("Right matrix")
sparse_print(mat_bar)
print(mat_bar.to_dense())

mat_baz = mat_foo @ mat_bar
dmat_baz = dmat_foo @ dmat_bar
sparse_print(mat_baz)
print(mat_baz.to_dense())
print(dmat_baz)
assert((dmat_baz == mat_baz.to_dense()).all())
