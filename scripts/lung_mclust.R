# Full-cell native mclust EEE; BIC chooses K without our niche count or labels.
args <- commandArgs(trailingOnly=TRUE)
folder <- args[1]
set.seed(as.integer(args[2]))
lib <- Sys.getenv('R_LIBS_USER')
dir.create(lib, recursive=TRUE, showWarnings=FALSE)
.libPaths(c(lib,.libPaths()))
if (!requireNamespace('mclust',quietly=TRUE)) {
  install.packages('mclust',repos='https://cloud.r-project.org',lib=lib)
}
library(mclust)
x <- as.matrix(read.table(file.path(folder,'mclust_input.tsv'),header=FALSE,sep='\t'))
fit <- Mclust(x,G=2:20,modelNames='EEE',verbose=TRUE)
write.table(fit$classification,file.path(folder,'mclust_labels.txt'),row.names=FALSE,col.names=FALSE)
write.csv(data.frame(K=2:20,BIC=as.numeric(fit$BIC)),file.path(folder,'BIC.csv'),row.names=FALSE)
writeLines(sprintf('Selected K=%d; boundary optimum=%s',fit$G,fit$G %in% c(2,20)),file.path(folder,'selection.txt'))
saveRDS(fit,file.path(folder,'mclust.rds'))
