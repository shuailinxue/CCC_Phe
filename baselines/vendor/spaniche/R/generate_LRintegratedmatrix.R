#' generate an integrated ligand-receptor interaction matrix
#'
#' @description
#' For each spatial view (e.g., view1 and view2), this function is executed twice
#' to capture bidirectional ligand–receptor signaling between center spots and
#' their neighboring spots.
#' Specifically, center spots and neighbor spots can alternately act as signal
#' senders and receivers:
#' \itemize{
#'   \item If the center spot expresses ligands, neighboring spots express receptors.
#'   \item Conversely, if the center spot expresses receptors, neighboring spots express ligands.
#' }
#' @param whoisneighbor
#' A square numeric matrix indicating spatial neighborhood relationships.
#' Rows and columns correspond to spatial spots (barcodes), where non-zero
#' values indicate neighboring relationships. The row and column names must be
#' identical and ordered consistently with the gene expression matrices.
#' @param neighbor_geneexp A matrix with rows representing spatial spots and columns representing genes.
#' @param center_geneexp A matrix with rows representing spatial spots and columns representing genes.
#' @param center_is_ligand
#' A logical value specifying the signaling role of the center spots.
#' \itemize{
#'   \item \code{TRUE}: center spots are treated as ligand-expressing cells,
#'   and neighboring spots as receptor-expressing cells.
#'   \item \code{FALSE}: center spots are treated as receptor-expressing cells,
#'   and neighboring spots as ligand-expressing cells.
#' }
#'
#' @return
#' A numeric matrix with rows representing micro-regions (defined by a center spot
#' and its neighbors) and columns representing ligand–receptor gene pairs.
#' Each entry corresponds to the element-wise product of mean neighbor gene
#' expression and center spot gene expression for a given ligand–receptor pair.
#'
generate_LRintegratedmatrix=function(
    whoisneighbor,
    neighbor_geneexp,
    center_geneexp,
    center_is_ligand
){
  print("generating L-R integrated matrix:")

  if(is.matrix(whoisneighbor) &
     is.matrix(neighbor_geneexp) &
     is.matrix(center_geneexp)){
    print("All data is matrix. ok!")
  } else {
    stop("All inputs should be matrices!")
  }

  if (identical(rownames(whoisneighbor),colnames(whoisneighbor)) &
      identical(rownames(whoisneighbor),rownames(neighbor_geneexp)) &
      identical(rownames(whoisneighbor),rownames(center_geneexp))) {
    print("The order of spatial barcodes is identical. ok!")
  } else {
    stop("The order of spatial barcodes is not identical across matrices!")
  }

  neighborspots_by_gene.Meanexp=whoisneighbor %*% neighbor_geneexp # Matrix multiplication
  centerspot_by_gene=center_geneexp
  region_by_LR=neighborspots_by_gene.Meanexp * centerspot_by_gene # Element-wise integration of center and neighbor expression

  if(center_is_ligand){
    LR_name=paste(colnames(center_geneexp),colnames(neighbor_geneexp),sep = "->")
    LR_name=paste0(LR_name," (CisL, NisR)")
  } else {
    LR_name=paste(colnames(center_geneexp),colnames(neighbor_geneexp),sep = "<-")
    LR_name=paste0(LR_name," (CisR, NisL)")
  }

  colnames(region_by_LR) = LR_name

  print("Finished.")
  print("------------------------------------")
  return(region_by_LR)
}
