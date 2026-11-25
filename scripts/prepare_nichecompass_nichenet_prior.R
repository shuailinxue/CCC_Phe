# NicheCompass NicheNet v2 ligand-receptor-target programs; panel restriction
# is applied only after ranking target genes on the complete v2 matrix.
root <- '/data1/xueshuailin/CCC_Phe/workspace/data/priors/nichenet_v2'
x <- readRDS(file.path(root,'ligand_target_matrix_nsga2r_final.rds'))
lr <- readRDS(file.path(root,'lr_network_human_21122021.rds'))
panel <- unique(toupper(readLines(file.path(root,'simulation_panel.txt'))))
ligands <- intersect(intersect(colnames(x),unique(lr$from)),panel)
all_min <- min(x,na.rm=TRUE); gps <- list(); stats <- list()
for (ligand in ligands) {
  scores <- x[,ligand]; cutoff <- sort(scores,partial=length(scores)-250)[length(scores)-250]
  # With ascending order statistics, the (n-250)th value is the 251st highest.
  targets <- intersect(rownames(x)[scores>cutoff & scores>all_min],panel)
  receptors <- intersect(unique(as.character(lr$to[lr$from==ligand])),panel)
  targets <- setdiff(targets,receptors)
  combined <- c(receptors,targets)
  if (!length(combined)) next
  name <- paste0('NICHENET_',ligand,'_ligand_receptor_target_gene_GP')
  gps[[name]] <- list(sources=c(ligand),sources_categories=c('ligand'),targets=combined,targets_categories=c(rep('receptor',length(receptors)),rep('target_gene',length(targets))))
  stats[[length(stats)+1L]] <- data.frame(ligand=ligand,receptors=length(receptors),downstream_targets=length(targets))
}
jsonlite::write_json(gps,file.path(root,'nichecompass_lrt_programs.pending.json'),auto_unbox=FALSE)
file.rename(file.path(root,'nichecompass_lrt_programs.pending.json'),file.path(root,'nichecompass_lrt_programs.json'))
write.csv(do.call(rbind,stats),file.path(root,'nichecompass_lrt_program_coverage.csv'),row.names=FALSE)
cat('NicheNet GP programs:',length(gps),'with downstream target genes:',sum(vapply(stats,function(z) z$downstream_targets>0,logical(1))),'\n')
