# Thin official-solver entry. Only held-out W inference is added locally;
# neither held-out matrices nor W enter estimation of H/modality weights.
args <- commandArgs(trailingOnly=TRUE)
output <- normalizePath(args[1]); set.seed(as.integer(args[2]))
rank <- as.integer(args[3]); lambda <- as.numeric(args[4]); sigma <- as.numeric(args[5])
maxiter <- as.integer(args[6]); init_iter <- as.integer(args[7]); restarts <- as.integer(args[8])
script <- sub('^--file=', '', commandArgs()[grep('^--file=',commandArgs())])
root <- dirname(normalizePath(script))
library(MASS)
for (name in c('IntNMF-internal.R','adj.cssls.R','H.fcnnls.R','W.fcnnls.R','nmf.mnnals.R')) {
  source(file.path(root,'vendor','intnmf','R',name))
}
for (name in c('calculate_laplacian_matrix_with_gausskernel.R','define_initial_weight.R','nmf_modified.R')) {
  source(file.path(root,'vendor','spaniche','R',name))
}
read_matrix <- function(name) as.matrix(read.csv(file.path(output,name),check.names=FALSE))
x1 <- read_matrix('abundance.csv'); x2 <- read_matrix('LR_views.csv')
xy <- read_matrix('coordinates.csv'); owner <- read_matrix('physical_owner.csv')[,1]
train <- which(owner==0); dat <- list(x1[train,,drop=FALSE],x2[train,,drop=FALSE])
weights <- define_initial_weight(dat)
initial <- nmf.mnnals(dat=dat,k=rank,maxiter=init_iter,st.count=10,n.ini=restarts,
                     ini.nndsvd=TRUE,seed=FALSE,wt=weights)
fit <- nmf_modified(dat=dat,Wzero=initial$W,Hzero=initial$H,wt=weights,k=rank,
                    lambda=lambda,sigma=sigma,maxiter=maxiter,st.count=20,
                    spatialdf=xy[train,,drop=FALSE],epsilon=1e-4)
saveRDS(list(fit=fit,initial=initial,weights=weights,train=train),file.path(output,'checkpoint.rds'))
write.csv(fit$convergence,file.path(output,'training.csv'),row.names=FALSE)
write.csv(fit$H[[1]],file.path(output,'H_abundance.csv'),row.names=FALSE)
write.csv(fit$H[[2]],file.path(output,'H_LR.csv'),row.names=FALSE)
W <- matrix(0,nrow=nrow(x1),ncol=rank); W[train,] <- fit$W
for (fold in setdiff(sort(unique(owner)),0)) {
  ids <- which(owner==fold); inputs <- list(x1[ids,,drop=FALSE],x2[ids,,drop=FALSE])
  # Extend the fitted factorization without fitting H on test observations.
  local <- t(W.fcnnls(x=fit$H,y=inputs,weight=weights)$coef)
  kernel <- calculate_laplacian_matrix_with_gausskernel(xy[ids,,drop=FALSE],sigma)
  numerator <- matrix(0,nrow=length(ids),ncol=rank); gram <- matrix(0,rank,rank)
  for (i in seq_along(inputs)) {
    numerator <- numerator + 2*weights[i]*inputs[[i]] %*% t(fit$H[[i]])
    gram <- gram + 2*weights[i]*fit$H[[i]] %*% t(fit$H[[i]])
  }
  for (iter in seq_len(maxiter)) {
    previous <- local
    local <- local * ((numerator+2*lambda*kernel$adj %*% local) /
                      pmax(local %*% gram+2*lambda*kernel$degree %*% local,.Machine$double.eps))
    if (sum(abs(local-previous))/max(sum(abs(previous)),.Machine$double.eps)<1e-4) break
  }
  W[ids,] <- local
}
if (any(!is.finite(W)) || any(W<0)) stop('Invalid SpaNiche factors')
write.csv(W,file.path(output,'W.csv'),row.names=FALSE)
write.csv(data.frame(native_topic=apply(W,1,which.max)-1),file.path(output,'native_topic_assignments.csv'),row.names=FALSE)
