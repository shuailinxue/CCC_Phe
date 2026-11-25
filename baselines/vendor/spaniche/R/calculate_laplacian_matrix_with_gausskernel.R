#' Calculate graph Laplacian matrix using a Gaussian kernel
#'
#' @description
#' This function constructs a weighted adjacency matrix from spatial
#' coordinates using a Gaussian kernel, and then computes the graph Laplacian matrix.
#' @param spa_mat A numeric matrix of spatial coordinates. Rows correspond to observations (e.g. cells or spots), and columns correspond to spatial dimensions.
#' @param sigma A positive numeric value specifying the bandwidth of the Gaussian kernel. Larger values lead to smoother, more global connectivity.
#'
#' @return A list with the following components:
#' Graph Laplacian matrix;
#' Degree matrix;
#' Gaussian kernel-based adjacency matrix.
calculate_laplacian_matrix_with_gausskernel = function(
    spa_mat,
    sigma = 1
  ){

  # ------------------------------------------------------------------
  # 1. Compute pairwise squared Euclidean distances between spatial points
  # spa_mat: rows are points (cells/spots), columns are spatial coordinates
  # dist() returns Euclidean distances, then square them
  # ------------------------------------------------------------------
  dst2=as.matrix(dist(spa_mat)^2)


  # ------------------------------------------------------------------
  # 2. Construct weighted adjacency matrix using Gaussian kernel
  # sigma controls the spatial smoothness / neighborhood size
  # ------------------------------------------------------------------
  adj_matrix = exp(-1 * dst2 / (2*sigma^2))


  # ------------------------------------------------------------------
  # 3. Compute degree matrix
  # Degree of each node = sum of weights connected to that node
  # Degree matrix is a diagonal matrix
  # ------------------------------------------------------------------
  degree_matrix <- diag(rowSums(adj_matrix))


  # ------------------------------------------------------------------
  # 4. Compute graph Laplacian
  # L = D - A
  # ------------------------------------------------------------------
  laplacian_matrix <- degree_matrix - adj_matrix


  # ------------------------------------------------------------------
  # 5. Return Laplacian, degree matrix, and adjacency matrix
  # ------------------------------------------------------------------
  return(list(laplacian = laplacian_matrix,
              degree = degree_matrix,
              adj = adj_matrix))
}
