#Step 1: Load Required Libraries and Data
# Install packages if not already installed
# install.packages(c("ggplot2", "dplyr", "tidyr", "ggrepel", "pheatmap"))
library(ggplot2)
library(dplyr)
library(tidyr)
library(ggrepel)
# library(pheatmap)

Method = "SimNetRLEDA"

# Set working directory to where your files are stored (adjust path as needed)
setwd("~/Manuscripts/130GNN4EDA/Code")

topk=10
GWASdb = "All"# All|PhenoScanner_Full|ieugwasr|gwasrapidd|CAUSALdb

topk_file <- paste0("../Prediction/",Method,"_predict_top", topk, ".txt")
evid_file <- paste0("../Prediction/",Method,"_predict_top", topk, "_Evid_",GWASdb,".txt")
enh2Entrez_file <- paste0("../Data/enh2target-1.0.2-Entrez.txt")

library(hash)


# Read the file topk_file
df_topk <- read.table(topk_file, sep = "\t", header = FALSE,
                 stringsAsFactors = FALSE, quote = "")

# Create a new hash
disease2enh <- hash()

# Fill the hash
for (i in seq_len(nrow(df_topk))) {
  key <- df_topk[i, 1]                       # disease name
  val <- as.character(df_topk[i, 3:ncol(df_topk)]) # enhancer regions (vector)
  val <- val[!is.na(val)]                 # remove empty columns if any
  disease2enh[[key]] <- val
}

# Example: retrieve enhancer list for "autism spectrum disorder"
disease2enh[["DOID:0050567"]]
# disease2enh[["bronchial asthma"]]
# disease2enh[["cardiovascular disease"]]

# Read the file evid_file
df_evid <- read.table(evid_file, sep = "\t", header = TRUE,
                 stringsAsFactors = FALSE, quote = "")
# evidEnh = unique(df_evid$eid)
# evidEnh

disease2evidEnh <- hash()

for (d in unique(df_evid$DOID)) {
  eids <- unique(df_evid[df_evid$DOID == d,]$Enhancer)
  disease2evidEnh[[d]] <- eids
}

# view all keys
hash::keys(disease2evidEnh)

# example: access the eids for "bronchial asthma"
disease2evidEnh[["DOID:2841"]]

evidDisease = unique(df_evid$DOID)
evidDisease


# Read the file enh2EntrezID_file
df_e2g <- read.table(enh2Entrez_file, sep = "\t", header = TRUE,
                 stringsAsFactors = FALSE, quote = "")
# Create a new hash
enh2EntrezID <- hash()

# Fill the hash: split the value string back into a vector
for (i in seq_len(nrow(df_e2g))) {
  key <- df_e2g$enhancer[i]
  val <- strsplit(df_e2g$target_Entrez[i], ",")[[1]]
  enh2EntrezID[[key]] <- val
}

# Test: retrieve one element
enh2EntrezID[["chr10:6074002-6104800"]]

disease2UnEvidEntrez = hash()
disease2UnEvidEnh = hash()
for(d in hash::keys(disease2enh)){
  pEnh = disease2enh[[d]]
  evidEnh = disease2evidEnh[[d]]
  
  unevidEnh = setdiff(pEnh, evidEnh)
  
  disease2UnEvidEnh[[d]] = unevidEnh
  
  unevidEntrezGene <- character(0)
  for(ue in unevidEnh){
    unevidEntrezGene <- c(unevidEntrezGene,enh2EntrezID[[ue]])
  }
  disease2UnEvidEntrez[[d]] = unique(unevidEntrezGene)
  
} 
print(length(disease2UnEvidEntrez))
print(length(disease2UnEvidEnh))

#######################
library(clusterProfiler)

# #Test with one disease
# d = keys(disease2Entrez)[2]
# gene_entrez = disease2Entrez[[d]]
# print(paste(d, toString(gene_entrez)))
# # Perform KEGG pathway enrichment
# kegg_enrich <- enrichKEGG(gene = gene_entrez,
#                           organism = "hsa",  # Human (Homo sapiens)
#                           pvalueCutoff = 0.05,  # Adjust p-value threshold as needed
#                           qvalueCutoff = 0.2,   # Adjust q-value threshold as needed
#                           minGSSize = 10,       # Minimum gene set size
#                           maxGSSize = 500)      # Maximum gene set size
# 
# # View the results
# dEnrichment = as.data.frame(kegg_enrich)
# dEnrichment
# nrow(dEnrichment)

disease2Enrichment = hash()

for(d in hash::keys(disease2UnEvidEnh)){
  enhancer2Enrichment = hash()
  for(e in disease2UnEvidEnh[[d]]){
    gene_entrez = enh2EntrezID[[e]]
    cat(d,"->",e, ":", toString(gene_entrez), "\n")
    # Perform KEGG pathway enrichment
    kegg_enrich <- enrichKEGG(gene = gene_entrez,
                              organism = "hsa",  # Human (Homo sapiens)
                              pvalueCutoff = 0.05,  # Adjust p-value threshold as needed
                              qvalueCutoff = 0.2,   # Adjust q-value threshold as needed
                              minGSSize = 10,       # Minimum gene set size
                              maxGSSize = 500)      # Maximum gene set size
    
    # View the results
    if(is.null(kegg_enrich)) next
    
    dEnrichment = kegg_enrich@result
    
    kegg_sig <- dEnrichment %>%
      filter(p.adjust <= 0.05)
    
    if (nrow(kegg_sig)>0){
      countSig = nrow(kegg_sig)
      
      print(paste(e, nrow(kegg_sig)))
      
      kegg_sig$PathwayID <- rownames(kegg_sig)   # add rownames as a new column
      rownames(kegg_sig) <- NULL               # remove rownames
      
      enhancer2Enrichment[[e]] = kegg_sig
      
    }
  }
  if(length(enhancer2Enrichment)>0){
    disease2Enrichment[[d]] = enhancer2Enrichment  
  }
  
}
length(disease2Enrichment)

# ---- Flatten to one data frame ----
all_diseases <- hash::keys(disease2Enrichment)

df_list <- lapply(all_diseases, function(disease) {
  
  enhancer_hash <- disease2Enrichment[[disease]]
  enhancers <- hash::keys(enhancer_hash)
  
  lapply(enhancers, function(enh) {
    df <- enhancer_hash[[enh]]
    df$Disease  <- disease
    df$Enhancer <- enh
    df
  })
})

# flatten the nested lists
final_df <- do.call(
  rbind,
  unlist(df_list, recursive = FALSE)
)

final_df <- final_df[,c("Disease","Enhancer" ,"PathwayID","Description", "p.adjust", "GeneRatio", "BgRatio", "geneID", "Count")]
dim(final_df)


# Read the file enh2target_file
enh2target_file <- paste0("../Data/enh2target-1.0.2-Tab.txt")
df_e2target <- read.table(enh2target_file, sep = "\t", header = TRUE,
                     stringsAsFactors = FALSE, quote = "")
# Create a new hash
enh2target <- hash()

# Fill the hash: split the value string back into a vector
for (i in seq_len(nrow(df_e2target))) {
  key <- df_e2target$enhancer[i]
  val <- strsplit(df_e2target$target_genes[i], ",")[[1]]
  enh2target[[key]] <- val
}


final_df$Gene = ""
for(i in 1:nrow(final_df)){
  disease = final_df$Disease[i]
  enh <- final_df$Enhancer[i]
  final_df$Gene[i] = enh2target[[enh]]
}
head(final_df)

final_df_All = final_df[,c("Disease","Enhancer","Gene", "PathwayID","Description", "p.adjust", "GeneRatio", "BgRatio", "geneID", "Count")]
write.table(final_df_All, file = paste0("../Prediction/",Method,"_predict_top",topk,"_UnEvidence_forAllDisease_ForEachEnhancer_Final.txt"), sep = "\t", row.names = FALSE)

final_df_EvidDisease = final_df_All[final_df_All$Disease %in% evidDisease,]
write.table(final_df_EvidDisease, file = paste0("../Prediction/",Method,"_predict_top",topk,"_UnEvidence_forEvidDisease_ForEachEnhancer_Final.txt"), sep = "\t", row.names = FALSE)


################################
#Generate Table 6 (Pathway enrichment for GWAS-unsupported enhancers)
DOID2Name_file = "/Users/hauldhut/Data/DO/doid.obo_DOID2Name.txt"

# Read the file evid_file
df_DOID2Name <- read.table(DOID2Name_file, sep = "\t", header = TRUE,
                           stringsAsFactors = FALSE, quote = "")
dim(df_DOID2Name)

# DOID2Name = hash()
## write final integrated file
#"Disease"	"Enhancer"	"Gene"	"PathwayID"	"Description"	"p.adjust"	"GeneRatio"	"BgRatio"	"geneID"	"Count"
TablePathwayEnrich_df <- final_df_EvidDisease
TablePathwayEnrich_df$DOTerm = ""
TablePathwayEnrich_df$DesPathwayID = ""
for(i in 1:nrow(TablePathwayEnrich_df)){
  doid = TablePathwayEnrich_df$Disease[i]
  TablePathwayEnrich_df$DOTerm[i] = df_DOID2Name[df_DOID2Name$DOID == doid,]$Name  
  TablePathwayEnrich_df$DesPathwayID[i] = paste0(TablePathwayEnrich_df$Description[i], " (", TablePathwayEnrich_df$PathwayID[i],")")
}

TablePathwayEnrich_file <- paste0("../Prediction/",Method, "_predict_top", topk, "_TablePathwayEnrich_ForEachEnhancer_Final.txt")

TablePathwayEnrich_df = TablePathwayEnrich_df[,c("DOTerm", "Enhancer", "Gene", "DesPathwayID")]
TablePathwayEnrich_df = TablePathwayEnrich_df[order(TablePathwayEnrich_df$DOTerm, TablePathwayEnrich_df$Enhancer),]
# write.table(
#   TablePathwayEnrich_file,
#   file = TableGWASEvidence_file,
#   sep = "\t",
#   quote = FALSE,
#   row.names = FALSE
# )

library(data.table)

setDT(TablePathwayEnrich_df)
group_cols <- c("DOTerm", "Enhancer", "Gene")
collapsed_df <- TablePathwayEnrich_df[
  ,
  lapply(.SD, function(x) {
    x <- unique(na.omit(as.character(x)))
    paste(x, collapse = ", ")
  }),
  by = group_cols
]
write.table(
  collapsed_df,
  file = TablePathwayEnrich_file,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)
