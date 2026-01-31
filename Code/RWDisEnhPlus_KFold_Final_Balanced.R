Method = "MH12_Balanced"#H1_Balanced|MH12_Balanced
gamma = 0.5

start_time <- Sys.time()

# Load required library
library(dplyr)
library(RandomWalkRestartMH)
library(igraph)

library(ROCR)

setwd("~/Manuscripts/130GNN4EDA/Code")

EnhSimNet1 <- read.delim("../Data/EnhNetG.txt",header = FALSE)
# EnhSimNet1 <- read.delim("../Data/EnhNetG_Unweighted.txt",header = FALSE)
EnhSimNet1.frame <- data.frame(EnhSimNet1[[1]], EnhSimNet1[[3]])
EnhSimNet1.g <- graph.data.frame(d = EnhSimNet1.frame, directed = FALSE)
EnhSimNet1.weight = EnhSimNet1[[2]]
E(EnhSimNet1.g)$weight <- EnhSimNet1.weight

EnhSimNet2 <- read.delim("../Data/EnhNetS.txt",header = FALSE)
EnhSimNet2.frame <- data.frame(EnhSimNet2[[1]], EnhSimNet2[[3]])
EnhSimNet2.g <- graph.data.frame(d = EnhSimNet2.frame, directed = FALSE)
EnhSimNet2.weight = EnhSimNet2[[2]]
E(EnhSimNet2.g)$weight <- EnhSimNet2.weight


if(Method == "MH12_Balanced"){
  Enh_MultiplexObject <- create.multiplex(list(EnhSimNet1.g,EnhSimNet2.g),Layers_Name = c("EnhSimNet1","EnhSimNet2"))
  tau1 = 1
  tau2 = 1
  tau <- c(tau1, tau2)
}else if(Method == "H1_Balanced"){
  Enh_MultiplexObject <- create.multiplex(list(EnhSimNet1.g),Layers_Name = c("EnhSimNet1"))
  tau <- c(1)
}

DiseaseSimNet = "DODisSimNet"
if(DiseaseSimNet == "DODisSimNet"){
  DiSimNet <- read.delim("../Data/DODisSimNet.txt",header = FALSE)
}else{
  DiSimNet <- read.delim(paste0("/Users/hauldhut/Manuscripts/124GNN4PiDA/Data/",DiseaseSimNet,".txt"),header = FALSE)  
}

DiSimNet.frame <- data.frame(DiSimNet[[1]], DiSimNet[[3]])
DiSimNet.weight = DiSimNet[[2]]

DiSimNet.g <- graph.data.frame(d = DiSimNet.frame, directed = FALSE)
E(DiSimNet.g)$weight <- DiSimNet.weight

disease_MultiplexObject <- create.multiplex(list(DiSimNet.g),
                                         Layers_Name = c("DiSimNet"))

#Add EnhDiRelation
EnhDi.frame <- read.csv("../Data/EDRelation.csv", header = TRUE)
EnhDi.frame <- EnhDi.frame[which(EnhDi.frame$Enh %in% Enh_MultiplexObject$Pool_of_Nodes),]
EnhDi.frame <- EnhDi.frame[which(EnhDi.frame$disease %in% disease_MultiplexObject$Pool_of_Nodes),]

#add func for RWR on multiplex-heter nw
do_something <- function(Enh_MultiplexObject,disease_MultiplexObject,
                         EnhDiRelation,SeedEnh, SeedDisease, prd_enhs) {
  
  # #Create multiplex-heterosgenous nw
  # EnhDiRelation_enh <- EnhDiRelation[which(EnhDiRelation$Enh %in% Enh_MultiplexObject$Pool_of_Nodes),]
  
  #Create multiplex-heterosgenous nw
  Enh_disease_Net <- create.multiplexHet(Enh_MultiplexObject, disease_MultiplexObject, 
                                         EnhDiRelation)
  
  Enh_disease_Net_TranMatrix <- compute.transition.matrix(Enh_disease_Net)
  
  #compute 
  Ranking_Results <- Random.Walk.Restart.MultiplexHet(Enh_disease_Net_TranMatrix,
                                                      Enh_disease_Net,
                                                      SeedEnh,
                                                      SeedDisease, r = gamma)
  
  #create labels for ranking results
  tf = Ranking_Results$RWRMH_Multiplex1
  
  tf$labels <- ifelse(tf$NodeNames %in% prd_enhs, 1, 0)
  cat(dim(tf),colnames(tf),"\n")
  
  
  # Select all nodes with label=1 and equal number of random nodes with label=0
  label_1_indices <- which(tf$labels == 1)
  label_0_indices <- which(tf$labels == 0)
  n_label_1 <- length(label_1_indices)

  if (n_label_1 > 0 && length(label_0_indices) >= n_label_1) {
    sampled_label_0_indices <- sample(label_0_indices, n_label_1)
    selected_indices <- c(label_1_indices, sampled_label_0_indices)
    tf <- tf[selected_indices, ]
  }
  
  # calculating AUC
  resultspred = prediction(tf$Score, tf$labels)
  
  auc.perf = performance(resultspred, measure = "auc")
  auroc = auc.perf@y.values[[1]]
  
  aupr.perf = performance(resultspred, measure = "aucpr")
  auprc = aupr.perf@y.values[[1]]
  
  return(list(auroc, auprc,data.frame(Scores=tf$Score, Labels=tf$labels)))
}

set.seed(123)

k <- 5  # CV folds

# Assign fold IDs at association level
EnhDi.frame$fold <- sample(rep(1:k, length.out = nrow(EnhDi.frame)))

#loop through
res <- vector("list", k)

for (i in 1:k) {
  
  cat("Fold", i, "\n")
  
  test_assoc  <- EnhDi.frame[EnhDi.frame$fold == i, ]
  train_assoc <- EnhDi.frame[EnhDi.frame$fold != i, ]
  
  prd_enhs <- unique(test_assoc$Enh)
  
  SeedEnh <- unique(train_assoc$Enh)
  SeedDisease <- unique(train_assoc$disease)
  
  EnhDiRelation <- train_assoc[, c("Enh", "disease")]
  
  tmp <- do_something(
    Enh_MultiplexObject,
    disease_MultiplexObject,
    EnhDiRelation,
    SeedEnh,
    SeedDisease,
    prd_enhs
  )
  
  res[[i]] <- list(
    auc  = tmp[[1]],
    aupr = tmp[[2]],
    obj  = tmp[[3]]
  )
  
  # ---- Save Scores & Labels for this fold ----
  fold_score_file <- paste0("../Results/",Method, "_MetaDisease_", DiseaseSimNet,"_ScoresLabels_Fold", i, "_KFold", k, ".csv")
  
  write.csv(tmp[[3]][, c("Scores", "Labels")],fold_score_file,row.names = FALSE,quote = FALSE)
  
}

length(res)
# stopCluster(cl)

df.res <- data.frame(
  trial = 1:k,
  auc  = sapply(res, `[[`, "auc"),
  aupr = sapply(res, `[[`, "aupr")
)

Result_byTrialFile = paste0("../Results/",Method,"_MetaDisease_",DiseaseSimNet,"_ROC_KFold",k,".csv")
write.csv(df.res, Result_byTrialFile, row.names = FALSE, quote = FALSE)

aucavgbyTrial     <- round(mean(df.res$auc), 3)
aucavgbyTrial.sd  <- round(sd(df.res$auc), 3)
aupravgbyTrial    <- round(mean(df.res$aupr), 3)
aupravgbyTrial.sd <- round(sd(df.res$aupr), 3)

cat("Meta-Disease CV\n")
cat("auc =", aucavgbyTrial, "(+-", aucavgbyTrial.sd, ")\n")
cat("aupr=", aupravgbyTrial, "(+-", aupravgbyTrial.sd, ")\n")


res.final <- do.call(rbind, lapply(res, function(x) x$obj))

# ---- Save Scores & Labels for entire CV ----
Result_AllScoresLabels_File <- paste0("../Results/",Method, "_MetaDisease_", DiseaseSimNet,"_ScoresLabels_All_KFold", k, ".csv")

write.csv(res.final[, c("Scores", "Labels")],Result_AllScoresLabels_File,row.names = FALSE,quote = FALSE)


resultspred <- prediction(res.final$Scores, res.final$Labels)

aucavgbyAll  <- round(performance(resultspred, "auc")@y.values[[1]], 3)
aupravgbyAll <- round(performance(resultspred, "aucpr")@y.values[[1]], 3)

cat("Summarize All (Meta-Disease)\n")
cat("aucavgbyAll =", aucavgbyAll, "\n")
cat("aupravgbyAll=", aupravgbyAll, "\n")


