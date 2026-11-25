# Non-visual data conversion only. Read serialized RNA counts without Seurat accessors.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2L)
suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(jsonlite))
out <- args[2]; dir.create(out,recursive=TRUE,showWarnings=FALSE)
obj <- readRDS(args[1])
# slotNames() on an unknown Seurat S4 class tries to load SeuratObject. Serialized
# attributes are sufficient here; no package installation or normalization is needed.
assays <- attr(obj,'assays')
if(!'RNA'%in%names(assays)) stop('Expected an explicit RNA assay')
a <- assays[['RNA']]; fields <- names(attributes(a))
if('counts'%in%fields) {
  m <- attr(a,'counts')
} else if('layers'%in%fields) {
  layers <- attr(a,'layers'); keys <- names(layers)[grepl('^counts($|\\.)',names(layers))]
  if(length(keys)!=1L) stop('Expected one native RNA counts layer; refuse ambiguous layers')
  m <- layers[[keys[1]]]
  fm <- attr(a,'features'); cm <- attr(a,'cells')
  attr(fm,'class') <- NULL; attr(cm,'class') <- NULL
  if(is.null(rownames(m))) rownames(m) <- rownames(fm)[fm[,keys[1]]]
  if(is.null(colnames(m))) colnames(m) <- rownames(cm)[cm[,keys[1]]]
} else stop('No native counts found in RNA assay')
if(!inherits(m,'dgCMatrix')) stop('Expected native sparse dgCMatrix')
stopifnot(!is.null(rownames(m)),!is.null(colnames(m)))
if(any(!is.finite(m@x))||any(m@x<0)||any(m@x!=round(m@x))||any(m@x>2^24))
  stop('Native nonnegative integer counts exactly representable as float32 required')
# CSC(genes x cells) is CSR(cells x genes) with the identical sparse buffers.
writeBin(m@x,file.path(out,'values.f32'),size=4,endian='little')
writeBin(m@i,file.path(out,'indices.i32'),size=4,endian='little')
writeBin(m@p,file.path(out,'indptr.i32'),size=4,endian='little')
writeLines(rownames(m),file.path(out,'genes.txt'))
writeLines(colnames(m),file.path(out,'cells.txt'))
report <- list(cells=ncol(m),genes=nrow(m),nnz=length(m@x),assay='RNA',
  source_slot='RNA counts only',source_class=class(a),buffer_layout='CSR cells x genes',
  normalized_values_used=FALSE,embeddings_or_niche_labels_read=FALSE)
write_json(report,file.path(out,'complete.json'),pretty=TRUE,auto_unbox=TRUE)
cat('Exported native RNA counts:',ncol(m),'cells x',nrow(m),'genes;',length(m@x),'nonzeros\n')
