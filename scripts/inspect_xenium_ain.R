# Inspect the author's deposited Seurat object before constructing features.
suppressPackageStartupMessages(library(SeuratObject))
root <- '/home/nas3/biod/xueshuailin/CCC_Phe/workspace/data/xenium_ain_kidney'
out <- '/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/xenium_ain_kidney'
dir.create(out, showWarnings = FALSE, recursive = TRUE)
obj <- readRDS(file.path(root, 'xenium_AIN_for_deposit.rds'))
meta <- obj[[]]
capture.output({
  cat('Object:', class(obj), '\nCells:', ncol(obj), '\n')
  cat('Assays:', paste(Assays(obj), collapse = ', '), '\n')
  for (a in Assays(obj)) {
    cat('Assay', a, 'features:', nrow(obj[[a]]), 'layers:', paste(Layers(obj[[a]]), collapse = ', '), '\n')
  }
  cat('Images:', paste(Images(obj), collapse = ', '), '\n')
  cat('Metadata columns:', paste(colnames(meta), collapse = ', '), '\n')
  for (nm in names(meta)) {
    z <- meta[[nm]]
    if ((is.character(z) || is.factor(z)) && length(unique(z)) < 120) {
      cat('\n', nm, '\n'); print(table(z, useNA = 'ifany'))
    }
  }
  for (im in head(Images(obj), 3)) {
    cat('\nCoordinates for', im, '\n')
    print(tryCatch(head(GetTissueCoordinates(obj, image = im)), error = function(e) conditionMessage(e)))
  }
}, file = file.path(out, 'object_audit.txt'))
write.csv(head(meta, 100), file.path(out, 'metadata_preview.csv'))
cat('Audit complete:', file.path(out, 'object_audit.txt'), '\n')
