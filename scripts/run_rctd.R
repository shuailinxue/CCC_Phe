#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(Matrix)
  library(spacexr)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6) {
  stop("usage: run_rctd.R <reference_dir> <spatial_dir> <output_dir> <cores> <seed> <sample...>")
}
reference_dir <- args[[1]]
spatial_dir <- args[[2]]
output_dir <- args[[3]]
cores <- as.integer(args[[4]])
seed <- as.integer(args[[5]])
samples <- args[6:length(args)]
set.seed(seed)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

read_mtx_gz <- function(path) as(readMM(gzfile(path)), "CsparseMatrix")
genes <- fread(file.path(reference_dir, "genes.tsv"), header = FALSE)[[1]]
cells <- fread(file.path(reference_dir, "cells.tsv"), data.table = FALSE)
counts <- read_mtx_gz(file.path(reference_dir, "counts_genes_by_cells.mtx.gz"))
rownames(counts) <- genes
colnames(counts) <- cells$cell_id
cell_types <- factor(cells$cell_type)
names(cell_types) <- cells$cell_id
n_umi <- setNames(cells$nUMI, cells$cell_id)
reference <- Reference(counts, cell_types, n_umi, n_max_cells = 500, min_UMI = 100)
rm(counts); gc()

for (sample in samples) {
  message("RCTD: ", sample)
  source <- file.path(spatial_dir, sample)
  spot_genes <- fread(file.path(source, "genes.tsv"), header = FALSE)[[1]]
  spots <- fread(file.path(source, "spots.tsv"), data.table = FALSE)
  spot_counts <- read_mtx_gz(file.path(source, "counts_genes_by_spots.mtx.gz"))
  rownames(spot_counts) <- spot_genes
  colnames(spot_counts) <- spots$spot_id
  coords <- as.data.frame(spots[, c("x", "y")])
  rownames(coords) <- spots$spot_id
  puck <- SpatialRNA(coords, spot_counts, setNames(spots$nUMI, spots$spot_id))
  fit <- create.RCTD(
    puck, reference, max_cores = cores, UMI_min = 1000,
    UMI_min_sigma = 1000, CELL_MIN_INSTANCE = 25, keep_reference = TRUE
  )
  fit <- run.RCTD(fit, doublet_mode = "full")
  weights <- as.matrix(normalize_weights(fit@results$weights))
  if (is.null(rownames(weights)) || anyDuplicated(rownames(weights))) {
    stop("RCTD returned missing or duplicate spot IDs for ", sample)
  }
  unknown <- setdiff(rownames(weights), spots$spot_id)
  if (length(unknown)) {
    stop("RCTD returned spot IDs absent from the input for ", sample, ": ",
         paste(head(unknown, 3), collapse = ", "))
  }
  # RCTD intentionally removes low-UMI spots. Keep only fitted spots, in the
  # original Visium order; downstream code intersects these with the h5ad.
  retained <- spots$spot_id[spots$spot_id %in% rownames(weights)]
  weights <- weights[retained, , drop = FALSE]
  if (!nrow(weights) || any(!is.finite(weights)) || any(rowSums(weights) <= 0)) {
    stop("RCTD returned invalid normalized weights for ", sample)
  }
  destination <- file.path(output_dir, sample)
  dir.create(destination, recursive = TRUE, showWarnings = FALSE)
  temporary <- file.path(destination, paste0(".tmp-", Sys.getpid()))
  dir.create(temporary, showWarnings = FALSE)
  fwrite(data.frame(spot_id = rownames(weights), weights, check.names = FALSE),
         file.path(temporary, "weights.tsv.gz"), sep = "\t")
  saveRDS(fit, file.path(temporary, "rctd.rds"), compress = "gzip")
  fwrite(data.frame(sample_id = sample, input_spots = nrow(spots),
                    fitted_spots = nrow(weights), excluded_low_umi = nrow(spots) - nrow(weights),
                    row_sum_min = min(rowSums(weights)),
                    row_sum_max = max(rowSums(weights))),
         file.path(temporary, "audit.tsv"), sep = "\t")
  publish <- function(name) {
    target <- file.path(destination, name)
    if (file.exists(target)) unlink(target)
    if (!file.rename(file.path(temporary, name), target)) {
      stop("Failed to publish ", name, " for ", sample)
    }
  }
  publish("rctd.rds")
  publish("audit.tsv")
  # Write the notebook's completion marker last.
  publish("weights.tsv.gz")
  unlink(temporary, recursive = TRUE)
}
