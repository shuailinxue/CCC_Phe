#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(BayesPrism)
  library(data.table)
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 9) {
  stop(paste("usage: run_bayesprism.R <reference_dir> <bulk_dir> <output_dir>",
             "<lr_genes.txt> <malignant_label> <cores> <chain_length> <burn_in> <seed>"))
}
reference_dir <- args[[1]]; bulk_dir <- args[[2]]; output_dir <- args[[3]]
lr_file <- args[[4]]; malignant_label <- args[[5]]; cores <- as.integer(args[[6]])
chain_length <- as.integer(args[[7]]); burn_in <- as.integer(args[[8]]); seed <- as.integer(args[[9]])
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
success_file <- file.path(output_dir, "_SUCCESS")
unlink(success_file)

read_mtx_gz <- function(path) as(readMM(gzfile(path)), "CsparseMatrix")
genes <- fread(file.path(reference_dir, "genes.tsv"), header = FALSE)[[1]]
cells <- fread(file.path(reference_dir, "cells.tsv"), data.table = FALSE)
reference <- t(read_mtx_gz(file.path(reference_dir, "counts_genes_by_cells.mtx.gz")))
rownames(reference) <- cells$cell_id; colnames(reference) <- genes
bulk_genes <- fread(file.path(bulk_dir, "genes.tsv"), header = FALSE)[[1]]
samples <- fread(file.path(bulk_dir, "samples.tsv"), header = FALSE)[[1]]
mixture <- as.matrix(read_mtx_gz(file.path(bulk_dir, "counts_samples_by_genes.mtx.gz")))
rownames(mixture) <- samples; colnames(mixture) <- bulk_genes
common <- intersect(colnames(reference), colnames(mixture))
reference <- reference[, common, drop = FALSE]
mixture <- mixture[, common, drop = FALSE]

prism <- new.prism(
  reference = reference, input.type = "count.matrix",
  cell.type.labels = cells$cell_type, cell.state.labels = cells$cell_state,
  key = malignant_label, mixture = mixture, outlier.cut = 0.01,
  outlier.fraction = 0.1
)
fit <- run.prism(
  prism = prism, n.cores = cores, update.gibbs = TRUE,
  gibbs.control = list(chain.length = chain_length, burn.in = burn_in,
                       thinning = 2, seed = seed),
  opt.control = list(optimizer = "MAP", n.cores = cores)
)
saveRDS(fit, file.path(output_dir, "bayesprism.rds"), compress = "gzip")
fractions <- get.fraction(fit, which.theta = "final", state.or.type = "type")
fwrite(as.data.table(fractions, keep.rownames = "Sample_ID"),
       file.path(output_dir, "fractions_final.tsv.gz"), sep = "\t")

lr_genes <- unique(readLines(lr_file))
mapping <- list()
for (cell_type in sort(unique(cells$cell_type))) {
  z <- get.exp(fit, state.or.type = "type", cell.name = cell_type)
  denominator <- rowSums(z); denominator[denominator == 0] <- 1
  z <- z / denominator * 1e6
  z <- z[, intersect(colnames(z), lr_genes), drop = FALSE]
  safe <- gsub("[^A-Za-z0-9]+", "_", cell_type)
  fwrite(as.data.table(z, keep.rownames = "Sample_ID"),
         file.path(output_dir, paste0(safe, ".tsv.gz")), sep = "\t")
  mapping[[length(mapping) + 1]] <- data.frame(cell_type = cell_type,
                                                file = paste0(safe, ".tsv.gz"))
}
fwrite(rbindlist(mapping), file.path(output_dir, "cell_type_files.tsv"), sep = "\t")
writeLines(capture.output(sessionInfo()), file.path(output_dir, "sessionInfo.txt"))
writeLines("complete", success_file)
