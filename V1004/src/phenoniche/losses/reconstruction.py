from phenoniche.utils.validation import matrix, same_context


def squared_frobenius(prediction, target):
    matrix(prediction, "prediction", nonnegative=False)
    matrix(target, "target")
    same_context((prediction, target))
    if prediction.shape != target.shape:
        raise ValueError("Reconstruction prediction and target shapes must match")
    return (prediction - target).square().sum()
