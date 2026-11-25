#' define_initial_weight: define initial weights for a list of matrices
#'
#' @description
#' This function computes an initial weight for each matrix in a list
#' based on its Frobenius norm and matrix size. The weights are designed
#' to balance the contribution of matrices with different scales and
#' dimensions in downstream joint or multi-modal analyses.
#' @param data.list A list of numeric matrices.
#'
#' @return A numeric vector of weights, with length equal to length(data.list).
#' @keywords internal
define_initial_weight = function(data.list){

  mean_matrix_norm_v = c()
  element_number_v = c()

  for (i in 1:length(data.list)) {
    one_matrix=data.list[[i]]
    matrix_norm <- norm(one_matrix, "F")

    element_num = dim(one_matrix)[1] * dim(one_matrix)[2]
    element_number_v = append(element_number_v,element_num)

    mean_matrix_norm = matrix_norm / element_num
    mean_matrix_norm_v = append(mean_matrix_norm_v,mean_matrix_norm)
  }

  theta_v = (max(mean_matrix_norm_v) / mean_matrix_norm_v) *
    (max(element_number_v) / element_number_v)
  theta_v
}
