# Preserve native counts, cell order and author-provided annotations; do not infer spatial units.
suppressPackageStartupMessages(library(Matrix))
p <- '/data1/xueshuailin/CCC_Phe/workspace/data/nsclc_zhencaiwei/raw/ST1_Cellist_scRNAHVG_SeuratObj.rds'
out <- '/data1/xueshuailin/CCC_Phe/workspace/data/nsclc_zhencaiwei/processed/ST1'
dir.create(out,recursive=TRUE,showWarnings=FALSE)
x <- readRDS(p); a <- attributes(x); ass <- attributes(a$assays$RNA); counts <- ass$counts; meta <- a$meta.data
stopifnot(identical(colnames(counts),rownames(meta)))
meta$spatial_coordinates_valid <- is.finite(meta$x) & is.finite(meta$y)
meta$cell_id <- rownames(meta);write.csv(meta,file.path(out,'metadata.csv'),row.names=FALSE,na='')
writeLines(rownames(counts),file.path(out,'genes.txt'));writeLines(colnames(counts),file.path(out,'cells.txt'))
writeMM(counts,file.path(out,'counts.mtx'))
cat('Saved',nrow(counts),'genes',ncol(counts),'cells',length(counts@x),'nonzero counts\n')
