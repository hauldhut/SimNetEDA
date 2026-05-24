Method = "SimNetRLEDA"
GWASdb = "gwasrapidd_PubMedID"# PhenoScanner_Full|ieugwasr|gwasrapidd_PubMedID|CAUSALdb


# maxK = 20
# start_time <- Sys.time()

library(RandomWalkRestartMH)
library(igraph)
library(foreach)
library(doParallel)

setwd("~/Manuscripts/130GNN4EDA/Code")


#Extract DOID2TraitMap
library(hash)
# Disease2EnhancerFile = "~/Manuscripts/98RWDisEnhPlus/Data/Disease2Enhancers.txt"
# Disease2Enhancer = read.delim(Disease2EnhancerFile, sep = "\t", header = FALSE)
# DOID2TraitMap = hash()
# 
# for(i in 1:nrow(Disease2Enhancer)){
#   did = Disease2Enhancer[i,1]
#   trait = Disease2Enhancer[i,2]
#   DOID2TraitMap[[did]] = trait
# }
# length(DOID2TraitMap)

## ===============================
## Inputs
## ===============================
DO2MeSH_file <- "/Users/hauldhut/Data/DO/doid.obo_DOID2MeSHID.txt"
enh2target_file <- "/Users/hauldhut/Manuscripts/130GNN4EDA/Data/enh2target-1.0.2-Tab.txt"
# Evid_file    <- "/Users/hauldhut/Data/GWAS/AllDisease_Evid_PhenoScanner.txt"
# Evid_file    <- "/Users/hauldhut/Data/GWAS/AllDisease_Evid_ieugwasr.txt"
# Evid_file    <- "/Users/hauldhut/Data/GWAS/AllDisease_Evid_gwasrapidd.txt"
Evid_file    <- paste0("/Users/hauldhut/Data/GWAS/AllDisease_Evid_",GWASdb,".txt")



## ===============================
## Read files
## ===============================
do2mesh_df <- read.table(
  DO2MeSH_file,
  header = TRUE,
  sep = "\t",
  stringsAsFactors = FALSE
)
dim(do2mesh_df)


enh2target_df <- read.table(
  enh2target_file,
  header = TRUE,
  sep = "\t",
  stringsAsFactors = FALSE
)
dim(enh2target_df)
enh2target_df = unique(enh2target_df)
dim(enh2target_df)

evid_df <- read.table(
  Evid_file,
  header = TRUE,
  sep = "\t",
  stringsAsFactors = FALSE,
  quote = "",          # <- CRITICAL
  fill = TRUE,         # <- CRITICAL
  comment.char = "",
  check.names = FALSE
)

dim(evid_df)


## ===============================
## Build DOID -> MeSH hash
## ===============================
DOID2MeSH <- hash()

for (i in seq_len(nrow(do2mesh_df))) {
  DOID2MeSH[[do2mesh_df$DOID[i]]] <- do2mesh_df$MeSH[i]
}


#####################################################
#Summarize for each k
k=10

# h.K2TotalEvidencedNewAssoc = hash()
# h.K2topKEnhEvidence = hash()

# for(k in seq(10, maxK, by = 10)){
  cat("k =",k,"\n")
  
  topK_file = paste0("../Prediction/DODisSimNet_gat_d_128_e_100_EnhNetG_EnhEmbS_initVec_DNABERT-2-max_gat_d_128_e_100_Balanced_XGB_top_10000000_predictions_group_top",k,".txt")
  # topK_final_file = paste0("../Prediction/",Method,"_predict_top",k,".txt")
  topK_Evid_file <- paste0("../Prediction/", Method, "_predict_top", k, "_Evid_",GWASdb,".txt")
  
  ## read topK file
  topK_df <- read.table(
    topK_file,
    header = TRUE,
    sep = "\t",
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  
  # ## keep only diseases present in DOID2TraitMap
  # common_doids <- intersect(topK_df$disease, hash::keys(DOID2TraitMap))
  # df.topKEnh <- topK_df[topK_df$disease %in% common_doids, , drop = FALSE]
  # 
  # ## map DOID -> disease name
  # df.topKEnh$diseaseName <- sapply(df.topKEnh$disease, function(x) DOID2TraitMap[[x]])
  # 
  # ## reorder columns: disease, diseaseName, Enh_1 ... Enh_10
  # df.topKEnh <- df.topKEnh[, c("disease", "diseaseName",
  #                        paste0("Enh_", 1:k))]
  # 
  # ## write output WITHOUT header
  # write.table(
  #   df.topKEnh,
  #   file = topK_final_file,
  #   sep = "\t",
  #   quote = FALSE,
  #   row.names = FALSE,
  #   col.names = FALSE
  # )
  
  # library(biomaRt)
  
  ## collect all enhancer columns
  enh_cols <- grep("^Enh_", colnames(topK_df), value = TRUE)
  
  ## flatten, remove NA, and get unique enhancers
  enhancers <- unique(na.omit(unlist(topK_df[, enh_cols])))
  
  length(enhancers)
  
  enh_df <- do.call(
    rbind,
    lapply(enhancers, function(x) {
      parts <- sub("^chr", "", x)
      chr <- sub(":.*", "", parts)
      start <- as.numeric(sub(".*:(\\d+)-.*", "\\1", parts))
      end <- as.numeric(sub(".*-(\\d+)$", "\\1", parts))
      
      data.frame(
        enhancer = x,
        chr = chr,
        start = start,
        end = end,
        stringsAsFactors = FALSE
      )
    })
  )
  

  
  library(GenomicRanges)
  
  enh_gr <- GRanges(
    seqnames = paste0("chr", enh_df$chr),
    ranges = IRanges(
      start = enh_df$start,
      end = enh_df$end
    ),
    enhancer = enh_df$enhancer
  )
  

  Enh2SNP_file <- paste0("../Prediction/enhancer_rsids", k, ".txt")
  
  if (!file.exists(Enh2SNP_file)) {
    library(GenomicRanges)
    library(SNPlocs.Hsapiens.dbSNP155.GRCh38)
    
    ## 1. Harmonize chromosome names FIRST
    seqlevels(enh_gr) <- sub("^chr", "", seqlevels(enh_gr))
    
    ## sanity check
    unique(seqnames(enh_gr))
    # should be: "1" "2" ... "22" "X"
    
    ## 2. Now split by chromosome
    enh_by_chr <- split(enh_gr, seqnames(enh_gr))
    
    ## 3. Process chromosome-by-chromosome
    snp_list <- vector("list", length(enh_by_chr))
    names(snp_list) <- names(enh_by_chr)
    
    for (chr in names(enh_by_chr)) {
      
      message("Processing chr ", chr)
      
      snp_chr <- snpsBySeqname(
        SNPlocs.Hsapiens.dbSNP155.GRCh38,
        seqnames = chr
      )
      
      hits <- findOverlaps(enh_by_chr[[chr]], snp_chr)
      
      if (length(hits) == 0) {
        rm(snp_chr); gc()
        next
      }
      
      snp_list[[chr]] <- data.frame(
        enhancer = mcols(enh_by_chr[[chr]])$enhancer[queryHits(hits)],
        refsnp_id = mcols(snp_chr)$RefSNP_id[subjectHits(hits)],
        chr = chr,
        pos = start(snp_chr)[subjectHits(hits)],
        stringsAsFactors = FALSE
      )
      
      rm(snp_chr)
      gc()
    }
    
    snp_df <- unique(do.call(rbind, snp_list))
    
    write.table(
      snp_df,
      file = Enh2SNP_file,
      sep = "\t",
      quote = FALSE,
      row.names = FALSE
    )
    
  } else {
    
    snp_df <- read.table(
      Enh2SNP_file,
      header = TRUE,
      sep = "\t",
      stringsAsFactors = FALSE
    )
  }
  dim(snp_df)
  
  library(hash)
  
  ## enhancer columns
  enh_cols <- grep("^Enh_", colnames(topK_df), value = TRUE)
  
  ## initialize hash
  Disease2SNP <- hash()
  
  ## loop over diseases
  for (i in seq_len(nrow(topK_df))) {
    
    disease_id <- topK_df$disease[i]
    
    ## enhancers for this disease
    enhancers <- unique(na.omit(as.character(
      unlist(topK_df[i, enh_cols])
    )))
    
    ## rsIDs linked to these enhancers
    rsids <- unique(na.omit(
      snp_df$refsnp_id[snp_df$enhancer %in% enhancers]
    ))
    
    ## store in hash
    Disease2SNP[[disease_id]] <- rsids
  }
  
  ## check
  head(hash::keys(Disease2SNP))
  length(hash::keys(Disease2SNP))
  
  ## ===============================
  ## Collect evidence
  ## ===============================
  res_list <- list()
  idx <- 1
  
  mesh_c = 0
  for (doid in hash::keys(Disease2SNP)) {
    
    ## skip if DOID has no MeSH mapping
    if (!has.key(doid, DOID2MeSH)) next
    
    mesh_id <- DOID2MeSH[[doid]]
    snp_set <- unique(Disease2SNP[[doid]])
    
    ## subset evidence by MeSH
    evid_sub <- evid_df[evid_df$Disease_ID == mesh_id, ]
    if (nrow(evid_sub) == 0) next
    
    cat(doid, "->", mesh_id,":", length(snp_set),"\n")
    mesh_c=mesh_c+1
    
    ## SNP overlap
    evid_hit <- evid_sub[evid_sub$SNP_ID %in% snp_set, ]
    if (nrow(evid_hit) == 0) next
    
    rsid = evid_hit$SNP_ID
    cat(rsid, "\n")
    
    ## store results
    res_list[[idx]] <- data.frame(
      DOID         = doid,
      Disease_ID   = evid_hit$Disease_ID,
      Disease_Name = evid_hit$Disease_Name,
      Enhancer     = "",
      target_genes = "",
      SNP_ID       = rsid,
      P_value      = evid_hit$P_value,
      Evidence     = evid_hit$Evidence,
      stringsAsFactors = FALSE
    )
    idx <- idx + 1
  }
  
  cat("mesh_c: ",mesh_c, "/",length(Disease2SNP), "\n")
  
  ## ===============================
  ## Final result
  ## ===============================
  if (length(res_list) > 0) {
    topK_Evid_df <- unique(do.call(rbind, res_list))
    for (i in seq_len(nrow(topK_Evid_df))) {
      rsid <- topK_Evid_df$SNP_ID[i]
      enh = snp_df[snp_df$refsnp_id==rsid,]$enhancer
      topK_Evid_df$Enhancer[i] = enh
      # cat(rsid, "->", enh,"\n")
      tg <- enh2target_df$target_genes[
        enh2target_df$enhancer == enh
      ]
      
      if (length(tg) == 0) {
        topK_Evid_df$target_genes[i] <- NA
      } else {
        tg_vec = strsplit(paste(unique(tg), collapse = ", "),", ")[[1]]
        topK_Evid_df$target_genes[i] <- paste(unique(tg_vec), collapse = ", ")
      }
      # topK_Evid_df$target_genes[i] = enh2target_df[enh2target_df$enhancer==enh,]$target_genes
    }
      
  } else {
    topK_Evid_df <- data.frame(
      DOID = character(),
      Disease_ID = character(),
      Disease_Name = character(),
      Enhancer = character(),
      target_genes = character(),
      SNP_ID = character(),
      P_value = character(),
      Evidence = character(),
      stringsAsFactors = FALSE
    )
  }
  
  dim(topK_Evid_df)
  ## ===============================
  ## Write output
  ## ===============================
  write.table(
    topK_Evid_df,
    file = topK_Evid_file,
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
  )
