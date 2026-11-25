# All-cell native IntNMF/SpaNiche execution. No statistical holdout.
args <- commandArgs(trailingOnly=TRUE)
output <- normalizePath(args[1]); set.seed(as.integer(args[2]))
rank <- as.integer(args[3]); lambda <- as.numeric(args[4]); sigma <- as.numeric(args[5])
maxiter <- as.integer(args[6]); init_iter <- as.integer(args[7]); restarts <- as.integer(args[8])
script <- sub('^--file=', '', commandArgs()[grep('^--file=',commandArgs())])
root <- dirname(normalizePath(script))
library(MASS); library(Matrix); library(irlba); library(jsonlite)
for (name in c('IntNMF-internal.R','adj.cssls.R','H.fcnnls.R','W.fcnnls.R','nmf.mnnals.R')) {
  source(file.path(root,'vendor','intnmf','R',name))
}
for (name in c('define_initial_weight.R','nmf_modified.R')) source(file.path(root,'vendor','spaniche','R',name))

# R integer dimensions overflow above 2^31 matrix entries. Promote just the
# element-count arithmetic; the native modality-weight formula is unchanged.
weight_body <- paste(deparse(body(define_initial_weight)),collapse='\n')
weight_old <- 'dim(one_matrix)[1] * dim(one_matrix)[2]'
if (!grepl(weight_old,weight_body,fixed=TRUE)) stop('Native weighting interface changed')
body(define_initial_weight) <- parse(text=gsub(weight_old,
    'as.double(dim(one_matrix)[1]) * as.double(dim(one_matrix)[2])',weight_body,fixed=TRUE))[[1]]

# The native initialization only requests k singular vectors. irlba avoids
# computing the other thousands of singular vectors; all rows enter the SVD.
svd <- function(x,nu=min(n,p),nv=min(n,p),LINPACK=FALSE) {
  n <- nrow(x); p <- ncol(x)
  if (min(n,p) <= max(nu,nv)+1) return(base::svd(x,nu=nu,nv=nv))
  irlba::irlba(x,nu=nu,nv=nv,tol=1e-6,maxit=1000)
}

# Native connectivity matrices are used only for partition stability and an
# unused consensus export. Canonical partition IDs give exactly the same
# equality check under arbitrary cluster renumbering, in O(N) memory.
body_text <- paste(deparse(body(nmf.mnnals)),collapse='\n')
replace_required <- function(old,new) {
  if (!grepl(old,body_text,fixed=TRUE)) stop(paste('Native IntNMF interface changed:',old))
  body_text <<- gsub(old,new,body_text,fixed=TRUE)
}
replace_required('consensus <- matrix(0, n, n)','consensus <- NULL')
replace_required('connect.old <- matrix(0, nrow = n, ncol = n)','connect.old <- rep(0L,n)')
replace_required('tmp1 <- matrix(rep(clust.mem, n), nrow = n, byrow = T)','tmp1 <- NULL')
replace_required('tmp2 <- matrix(rep(clust.mem, n), nrow = n, byrow = F)','tmp2 <- NULL')
replace_required('connect.new <- ifelse(tmp1 == tmp2, 1, 0)','connect.new <- match(clust.mem, unique(clust.mem))')
replace_required('dimnames(consensus) <- list(rownames(W), rownames(W))','consensus <- NULL')
body(nmf.mnnals) <- parse(text=body_text)[[1]]

shapes <- jsonlite::fromJSON(file.path(output,'input_shapes.json'))
read_binary <- function(name,shape) {
  connection <- file(file.path(output,paste0(name,'.bin')),'rb')
  on.exit(close(connection))
  value <- readBin(connection,what='numeric',n=prod(shape),size=8,endian='little')
  if (length(value)!=prod(shape)) stop('Incomplete binary input')
  matrix(value,nrow=shape[1],ncol=shape[2],byrow=TRUE)
}
dat <- list(read_binary('abundance',shapes$abundance),read_binary('LR_views',shapes$LR_views))
kernel <- Matrix::readMM(file.path(output,'gaussian.mtx'))
degree <- Matrix::Diagonal(x=Matrix::rowSums(kernel))
calculate_laplacian_matrix_with_gausskernel <- function(spa_mat,sigma=1) {
  list(adj=kernel,degree=degree,laplacian=degree-kernel)
}
weights <- define_initial_weight(dat)
if (any(!is.finite(weights)) || any(weights<=0)) stop('Invalid modality weights')
write.csv(data.frame(modality=names(shapes),weight=weights),file.path(output,'modality_weights.csv'),row.names=FALSE)
cat('Native all-cell IntNMF initialization:',nrow(dat[[1]]),'cells\n');flush.console()
initial <- nmf.mnnals(dat=dat,k=rank,maxiter=init_iter,st.count=10,n.ini=restarts,
                     ini.nndsvd=TRUE,seed=FALSE,wt=weights)
cat('Native all-cell spatial joint NMF\n');flush.console()
fit <- nmf_modified(dat=dat,Wzero=initial$W,Hzero=initial$H,wt=weights,k=rank,
    lambda=lambda,sigma=sigma,maxiter=maxiter,st.count=20,spatialdf=matrix(0,nrow=nrow(dat[[1]]),ncol=2),epsilon=1e-4)
if (any(!is.finite(fit$W)) || any(fit$W<0)) stop('Invalid native SpaNiche factors')
saveRDS(list(fit=fit,initial=initial,weights=weights),file.path(output,'checkpoint.rds'))
write.csv(as.matrix(fit$W),file.path(output,'W.csv'),row.names=FALSE)
write.csv(fit$convergence,file.path(output,'training.csv'),row.names=FALSE)
write.csv(as.matrix(fit$H[[1]]),file.path(output,'H_abundance.csv'),row.names=FALSE)
write.csv(as.matrix(fit$H[[2]]),file.path(output,'H_LR.csv'),row.names=FALSE)
