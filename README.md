# Compressed Sensing Image Recovery
Machine Learning class project, which use LASSO regression with a DCT-inspired design to recover corrupted images.

This project provides a practical (and interesting!) application of regularized sparse regression. 

If pixels are missing from an image, either by design (compressed sensing) or by accident (a corrupted image), we can estimate the values of the missing pixels via regression.  The regression problem, however, may be ill-posed because we have fewer observations than parameters (pixels) to estimate.  We can address the ill-posed nature of the problem by imposing one or more constraints.  A common constraint is smoothness.  A limitation of a smoothness constraint for recovery of natural images is natural images do not tend to exhibit smoothness.  They do, however, tend to exhibit sparsity.  We can leverage this domain knowledge (that natural images tend to be sparse) to design our constraint to impose sparsity on the model from which we estimate the missing pixels.
