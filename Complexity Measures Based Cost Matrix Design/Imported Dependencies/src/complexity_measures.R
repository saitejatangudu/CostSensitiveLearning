if (!require("devtools")) {
  install.packages("devtools")
}
devtools::install_github("victorhb/ImbCoL")
#install.packages("ECoL")
#library("ImbCoL")
ImbCoL::complexity(Species ~., iris)


# These datasets are the small subselection for the preliminary results.
# dataset_names = c("02a","02b","03subcl5","04clover5","04clover5-noise",
#                   "paw3-2d-border-center","paw3-2d-border-dense-center","paw3-2d-only-border","paw3-2d-very-dense-center",
#                   "gaussian_overlap_0.83_0.17_1000_1_1","gaussian_overlap_0.83_0.17_1000_1_2","gaussian_overlap_0.83_0.17_1000_1_3","gaussian_overlap_0.83_0.17_1000_1_4",
#                   "local_imbalance_degree_0.83_0.17_0.05_1000","local_imbalance_degree_0.83_0.17_0.1_1000","local_imbalance_degree_0.83_0.17_0.2_1000","local_imbalance_degree_0.83_0.17_0.5_1000",
#                   "uniform_overlap_0.83_0.17_10_1000","uniform_overlap_0.83_0.17_20_1000","uniform_overlap_0.83_0.17_40_1000","uniform_overlap_0.83_0.17_60_1000","uniform_overlap_0.83_0.17_80_1000",
#                  "uniform_only_boundary_no_overlap_0.83_0.17_1000","multi_modal_no_overlap_0.83_0.17_1000","multi_modal_overlap_0.83_0.17_1000")
#
# removed coil2000 since it's too memory intensive


#dataset_names = c("appendicitis", "australian","bands","bupa","haberman","heart","hepatitis","ionosphere","monk-2","pima","saheart","sonar","spectfheart","titanic","USPS_PCA","wdbc","wisconsin")

dataset_names = c("adversary_n1","adversary_n3")
#dataset_names = c("02a")
N1_values = list()
N2_values = list()
N3_values = list()
N4_values = list()
T1_values = list()
F1_values = list()
F2_values = list()
F3_values = list()
F4_values = list()
L1_values = list()
L2_values = list()
L3_values = list()

for (dataset_name in dataset_names){
  start_time <- Sys.time()
  print(dataset_name)
  dataset <- read.csv2(sprintf("/home/goettcke/PhD/class_imbalance_measure/src/datasets/%s.csv", dataset_name), sep = ",", header = FALSE, stringsAsFactors = FALSE, dec=".")
  X <- dataset[,1:ncol(dataset)-1]
  y <- dataset[,ncol(dataset)]
  
  N1_values <- c(N1_values,ECoL::neighborhood(X,y,measures="N1")$N1[1])
  cat("..N1")
  N2_values <- c(N2_values,ECoL::neighborhood(X,y,measures="N2")$N2[1])
  cat("..N2")
  N3_values <- c(N3_values,ECoL::neighborhood(X,y,measures="N3")$N3[1])
  cat("..N3")
  N4_values <- c(N4_values,ECoL::neighborhood(X,y,measures="N4")$N4[1])
  cat("..N4")
  
  F1_values <- c(F1_values,ECoL::overlapping(X,y,measures="F1")$F1[1])
  cat("..F1")
  F2_values <- c(F2_values,ECoL::overlapping(X,y,measures="F2")$F2[1])
  cat("..F2")
  F3_values <- c(F3_values,ECoL::overlapping(X,y,measures="F3")$F3[1])
  cat("..F3")
  F4_values <- c(F4_values,ECoL::overlapping(X,y,measures="F4")$F4[1])
  cat("..F4")
  
  T1_values <- c(T1_values,ECoL::neighborhood(X,y,measures="T1")$T1[1])
  cat("..T1")
  
  L1_values <- c(L1_values,ECoL::linearity(X,y,measures="L1")$L1[1])
  cat("..L1")
  L2_values <- c(L2_values,ECoL::linearity(X,y,measures="L2")$L2[1])
  cat("..L2")
  L3_values <- c(L3_values,ECoL::linearity(X,y,measures="L3")$L3[1])
  cat("..L3, time:")

  end_time <- Sys.time()
  cat(end_time - start_time)
}

complexity_df <- data.frame(dataset=dataset_names, N1 = unlist(N1_values), N2 = unlist(N2_values), N3=unlist(N3_values),
                            N4 = unlist(N4_values), T1 = unlist(T1_values), F1=unlist(F1_values), 
                            F2=unlist(F2_values),F3 = unlist(F3_values), F4 = unlist(F4_values), 
                            L1=unlist(L1_values), L2=unlist(L2_values), L3=unlist(L3_values))
#complexity_df <- complexity_df[1:25,] #Slice off all the std dev. (since it's only over 1 value )
# REMEMBER TO CHANGE THE FILE NAME DEPENDENT ON THE DATASET TYPES
write.csv(complexity_df, file = "/home/goettcke/PhD/class_imbalance_measure/results/complexity_measures_adversary_datasets.csv", row.names = FALSE)

N1_Imb_values = list()
N2_Imb_values = list()
N3_Imb_values = list()
N4_Imb_values = list()
T1_Imb_values = list()
F2_Imb_values = list()
F3_Imb_values = list()
F4_Imb_values = list()
L1_Imb_values = list()
L2_Imb_values = list()
L3_Imb_values = list()

for (dataset_name in dataset_names){
  start_time <- Sys.time()
  print(dataset_name)
  dataset <- read.csv2(sprintf("/home/goettcke/PhD/class_imbalance_measure/src/datasets/%s.csv", dataset_name), sep = ",", header = FALSE, stringsAsFactors = FALSE, dec=".")
  X <- dataset[,1:ncol(dataset)-1]
  y <- dataset[,ncol(dataset)]
  
  n <- ImbCoL::neighborhood(X,y,measures="all")
  N1_Imb_values <- c(N1_Imb_values,mean(n$N1_partial)[1])
  cat("..N1")
  N2_Imb_values <- c(N2_Imb_values,mean(n$N2_partial)[1])
  cat("..N2")
  N3_Imb_values <- c(N3_Imb_values,mean(n$N3_partial)[1])
  cat("..N3")
  N4_Imb_values <- c(N4_Imb_values,mean(n$N4_partial)[1])
  cat("..N4")

  T1_Imb_values <- c(F4_Imb_values,mean(n$T1_partial)[1])
  cat("..T1")

  f <- ImbCoL::overlapping(X,y,measures="all")
  F2_Imb_values <- c(F2_Imb_values,mean(f$F2_partial)[1])
  cat("..F1")
  F3_Imb_values <- c(F3_Imb_values,mean(f$F3_partial)[1])
  cat("..F3")
  F4_Imb_values <- c(F4_Imb_values,mean(f$F4_partial)[1])
  cat("..F4")

  l <- ImbCoL::linearity.class(X,y,measures="all")
  L1_Imb_values <- c(L1_Imb_values,mean(l$L1_partial)[1])
  cat("..L1")
  L2_Imb_values <- c(L2_Imb_values,mean(l$L2_partial)[1])
  cat("..L2")
  L3_Imb_values <- c(L3_Imb_values,mean(l$L3_partial)[1])
  cat("..L3, time:")
  #neighborhood_measures <- c(N1_values,ECoL::neighborhood(X,y,measures="all"))
  #overlapping_measures <- c()
  end_time <- Sys.time()
  cat(end_time - start_time)
}


imb_complexity <- data.frame(dataset=dataset_names, N1 = unlist(N1_Imb_values), N2 = unlist(N2_Imb_values), N3=unlist(N3_Imb_values),
                            N4 = unlist(N4_Imb_values), T1 = unlist(T1_Imb_values), F2 = unlist(F2_Imb_values),
                            F3 = unlist(F3_Imb_values), F4 = unlist(F4_Imb_values),
                            L1 = unlist(L1_Imb_values), L2 = unlist(L2_Imb_values), L3 = unlist(L3_Imb_values))

# REMEMBER TO CHANGE THE FILE NAME DEPENDENT ON THE DATASET TYPES
write.csv(imb_complexity, file = "/home/goettcke/PhD/class_imbalance_measure/results/imbalance_complexity_measures_adversary_datasets.csv", row.names = FALSE)

dataset <- read.csv2("/home/goettcke/PhD/class_imbalance_measure/src/datasets/adversary_n3.csv", sep = ",", header = TRUE, stringsAsFactors = TRUE, dec=".")
dataset <- dataset[c(1,2,8,9),]
X = dataset[,1:2]
y = dataset[,3]
ImbCoL::neighborhood(class~., dataset,"N3",metric)

