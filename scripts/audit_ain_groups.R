suppressPackageStartupMessages(library(SeuratObject))
root <- '/home/nas3/biod/xueshuailin/CCC_Phe/workspace'
obj <- readRDS(file.path(root, 'data/xenium_ain_kidney/xenium_AIN_for_deposit.rds'))
m <- obj[[]]
groups <- split(m, interaction(m$base_patient_id,m$tissue_region,drop=TRUE))
stats <- do.call(rbind, lapply(groups, function(z) data.frame(patient=as.character(z$base_patient_id[1]), condition=as.character(z$patient_category[1]), region=as.character(z$tissue_region[1]), cells=nrow(z), umi_median=median(z$nCount_RNA,na.rm=TRUE), genes_median=median(z$nFeature_RNA,na.rm=TRUE), types=length(unique(z$final_annotations2)), x_min=min(z$x_centroid), x_max=max(z$x_centroid), y_min=min(z$y_centroid),y_max=max(z$y_centroid))))
write.csv(stats,file.path(root,'outputs/xenium_ain_kidney/sample_qc.csv'),row.names=FALSE)
print(stats)
