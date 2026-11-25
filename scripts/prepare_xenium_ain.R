# Export one cortex sample per group, selected by comparable detection depth.
suppressPackageStartupMessages(library(SeuratObject))
suppressPackageStartupMessages(library(Matrix))
root <- '/home/nas3/biod/xueshuailin/CCC_Phe/workspace/data/xenium_ain_kidney'
obj <- readRDS(file.path(root, 'xenium_AIN_for_deposit.rds'))
meta <- obj[[]]
required <- c('base_patient_id','patient_category','tissue_region','final_annotations2','x_centroid','y_centroid')
stopifnot(all(required %in% colnames(meta)))
meta <- meta[meta$tissue_region == 'Cortex' & !is.na(meta$final_annotations2) & !is.na(meta$x_centroid) & !is.na(meta$y_centroid),,drop=FALSE]
groups <- split(meta, meta$base_patient_id, drop=TRUE)
tab <- do.call(rbind, lapply(groups, function(z) data.frame(Var1=as.character(z$base_patient_id[1]),Var2=as.character(z$patient_category[1]),Freq=nrow(z),median_genes=median(z$nFeature_RNA))))
tab <- tab[tab$Freq >= 5000 & tab$Var2 %in% c('AIN','REF'),]
ref <- tab[tab$Var2 == 'REF',,drop=FALSE]
ain <- tab[tab$Var2 == 'AIN',,drop=FALSE]
pairs <- expand.grid(r=seq_len(nrow(ref)),a=seq_len(nrow(ain)))
pairs$diff <- abs(log1p(ref$median_genes[pairs$r])-log1p(ain$median_genes[pairs$a]))
pairs <- pairs[order(pairs$diff,ref$Var1[pairs$r],ain$Var1[pairs$a]),,drop=FALSE]
chosen <- rbind(ref[pairs$r[1],,drop=FALSE],ain[pairs$a[1],,drop=FALSE])
write.csv(chosen, file.path(root, 'selected_patients.csv'), row.names=FALSE)
for (i in seq_len(nrow(chosen))) {
  patient <- chosen$Var1[i]
  m <- meta[meta$base_patient_id == patient,,drop=FALSE]
  x <- GetAssayData(obj[['RNA']], layer='counts')[,rownames(m),drop=FALSE]
  sub <- file.path(root, as.character(patient)); dir.create(sub, showWarnings=FALSE)
  writeMM(x, file.path(sub,'counts.mtx'))
  writeLines(rownames(x), file.path(sub,'genes.txt'))
  write.csv(m, file.path(sub,'metadata.csv'))
  cat('Exported', patient, ncol(x), 'cells', nrow(x), 'genes\n', flush=TRUE)
}
