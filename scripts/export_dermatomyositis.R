# Data conversion only; figures are produced by the existing Python workflow.
# Read original serialized slots directly: no Seurat normalization or imputation.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2L)
suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(jsonlite))
obj <- readRDS(args[1]); out <- args[2]; dir.create(out,recursive=TRUE,showWarnings=FALSE)
meta <- slot(obj,'meta.data')
# Polygon/list columns are unnecessary for centroid CCC and cannot be written
# as rectangular CSV; exporting them would introduce spurious text rows.
meta <- meta[,!vapply(meta,is.list,logical(1)),drop=FALSE]
meta$cell_id <- rownames(meta)
write.csv(meta,file.path(out,'metadata.csv'),row.names=FALSE)
cat('METADATA',paste(colnames(meta),collapse=', '),'\n');flush.console()
assays <- slot(obj,'assays');cat('ASSAYS',names(assays),'\n')
audit <- list(assays=list(),images=list())
export_matrix <- function(m,key,features=NULL,cells=NULL) {
  stopifnot(inherits(m,'sparseMatrix'))
  if(is.null(rownames(m))) rownames(m)<-features
  if(is.null(colnames(m))) colnames(m)<-cells
  stopifnot(!is.null(rownames(m)),!is.null(colnames(m)))
  if(any(!is.finite(m@x))||any(m@x<0)||any(abs(m@x-round(m@x))>1e-5))stop('Non-count matrix')
  writeMM(m,file.path(out,paste0(key,'.mtx')))
  writeLines(rownames(m),file.path(out,paste0(key,'.genes.txt')))
  writeLines(colnames(m),file.path(out,paste0(key,'.cells.txt')))
  cat('COUNTS',key,dim(m),length(m@x),'\n');flush.console()
}
for(n in names(assays)) {
  a<-assays[[n]];sn<-setdiff(names(attributes(a)),'class');audit$assays[[n]]<-sn
  if('counts'%in%sn) {
    m<-slot(a,'counts');if(nrow(m)>1000L)export_matrix(m,paste0('counts_',n))
  } else if('layers'%in%sn) {
    layers<-slot(a,'layers')
    for(l in names(layers)[grepl('^counts($|\\.)',names(layers))]) {
      # LogMap is a serialized logical matrix. Strip its class attribute instead
      # of invoking SeuratObject's S4 accessors; values and dimnames are unchanged.
      fm<-slot(a,'features');cm<-slot(a,'cells')
      attr(fm,'class')<-NULL;attr(cm,'class')<-NULL
      features<-rownames(fm)[fm[,l]];cells<-rownames(cm)[cm[,l]]
      if(length(features)>1000L)export_matrix(layers[[l]],paste0('counts_',n,'_',l),features,cells)
    }
  }
}
images<-slot(obj,'images');coordinates<-list()
if(all(c('x','y','sample_paper')%in%colnames(meta))) {
  coordinates[['published_metadata']]<-data.frame(cell_id=meta$cell_id,x=meta$x,y=meta$y,image=meta$sample_paper)
  images<-list()
}
for(n in names(images)) {
  im<-images[[n]];sn<-setdiff(names(attributes(im)),'class');audit$images[[n]]<-sn
  if('boundaries'%in%sn) {
    boundaries<-slot(im,'boundaries')
    for(b in names(boundaries)) {
      cent<-boundaries[[b]];bs<-setdiff(names(attributes(cent)),'class')
      if(all(c('coords','cells')%in%bs)) {
        xy<-slot(cent,'coords');ids<-slot(cent,'cells')
        if(nrow(xy)==length(ids)) {
          coordinates[[n]]<-data.frame(cell_id=ids,x=xy[,1],y=xy[,2],image=n)
          break
        }
      }
    }
  } else if('coordinates'%in%sn) {
    xy<-slot(im,'coordinates')
    if(all(c('x','y')%in%colnames(xy)))coordinates[[n]]<-data.frame(cell_id=rownames(xy),x=xy$x,y=xy$y,image=n)
  }
}
if(length(coordinates))write.csv(do.call(rbind,coordinates),file.path(out,'coordinates.csv'),row.names=FALSE)
write_json(audit,file.path(out,'r_object_slots.json'),pretty=TRUE,auto_unbox=TRUE)
cat('EXPORTED',nrow(meta),'cells;',length(images),'images;',length(coordinates),'coordinate blocks\n')
