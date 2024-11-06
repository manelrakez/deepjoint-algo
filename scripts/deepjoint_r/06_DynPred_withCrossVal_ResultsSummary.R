  ###########################################################################################################
  ###########################################################################################################
  ######         Project: The DeepJoint algorithm: Breast cancer risk prediction (with cross-val)      ######
  ######                  Part 4 : Summarize AUC and BS results                                        ######
  ######         Programmer: M.RAKEZ                                                                   ######
  ######         Creation date: Friday, 12th of April 2024                                             ######
  ######         Last updated: 12APR2024                                                               ######
  ###########################################################################################################
  ###########################################################################################################
  
  ####################################
  # Start with a clean environment
  rm(list=ls(all=TRUE))
  ####################################
  
  ####################################
  # Set your work directory
  directory <- "~/work_directory"
  setwd(directory)
  ####################################
  
  ####################################
  # Model's predictive performance
  ###############
  
  #############
  # # Here, an example with prediction window of 2 years
  w2 <- 2
  #w5 <- 5
  #w10 <- 10
  #############
  
  #############
  # AUC
  
  #####
  # Extract AUC results for all landmark times and a given prediction window (here w = 2)
  AUC_list_w2 <- list()
  AUC_all_w2 <- NULL
  HCI_AUC_all_w2 <- NULL
  LCI_AUC_all_w2 <- NULL
  i <- 1
  for (t0 in seq (1, 25, by = 1)){
    if (t0 == 1){
      i <- 1
    } else {
      i <- i + 1
    }
    load(c(paste0("AUC_CVCS_t0_",t0,"_w",w2,".RData")))
    AUC_list_w2[[i]] <- AUC_t0
    AUC_all_w2 <- c(AUC_all_w2, round(unname(AUC_t0$AUC[2]),4))

    load(c(paste0("HCI_AUC_CVCS_t0_",t0,"_w",w2,".RData")))
    HCI_AUC_all_w2 <- c(HCI_AUC_all_w2, round(HCI_AUC_t0/100,4))

    load(c(paste0("LCI_AUC_CVCS_t0_",t0,"_w",w2,".RData")))
    LCI_AUC_all_w2 <- c(LCI_AUC_all_w2, round(LCI_AUC_t0/100,4))
  }
  mat_res_AUC_w2 <-as.data.frame(cbind(seq(1, 25, by = 1)+40, AUC_all_w2, c(paste0("[", LCI_AUC_all_w2, " ", HCI_AUC_all_w2, "]"))))
  colnames(mat_res_AUC_w2) <- c("Landmark time","AUC","95%CI")
  print(mat_res_AUC_w2)
  #####
  
  #####
  #Plot dynamic AUC results
  # png(file = c(paste0("Dynamic_AUC_window_", w2, "_yrs.png")))
  plot(seq(1, 25, by = 1),
       AUC_all_w2,
       ylim=c(0, 1),
       ylab="AUC", xlab="Age (years)",
       cex.lab = 1.2, 
       xaxt="n",
       cex.axis = 1.2, pch=19,
       main = "Dynamic AUC (landmark time in [41 - 65], w = 2)")
  axis(1, at = seq(1, 25, by = 1),
       labels = 40+seq(1, 25, by = 1),
       cex.axis = 1.2)
  lines(seq(1, 25, by = 1), AUC_all_w2)
  abline(h=50, lty=2)
  #CI
  points(seq(1, 25, by = 1), LCI_AUC_all_w2)
  lines(seq(1, 25, by = 1), LCI_AUC_all_w2, lty = 3)
  points(seq(1, 25, by = 1), HCI_AUC_all_w2)
  lines(seq(1, 25, by = 1), HCI_AUC_all_w2, lty = 3)
  legend("bottomright",lty = c(1,3), legend=c("mean AUC", "95%CI"))
  # dev.off()
  #####
  #############
  
  #############
  # BS
  
  #####
  # Extract BS results for all landmark times and a given prediction window (here w = 2)
  BS_list_w2 <- list()
  BS_all_w2 <- NULL
  HCI_BS_all_w2 <- NULL
  LCI_BS_all_w2 <- NULL
  i <- 1
  for (t0 in seq (1, 25, by = 1)){
    if (t0 == 1){
      i <- 1
    } else {
      i <- i + 1
    }
    load(c(paste0("BS_CVCS_t0_",t0,"_w",w2,".RData")))
    BS_list_w2[[i]] <- BS_t0
    BS_all_w2 <- c(BS_all_w2, round(unlist(unname(BS_t0$Brier$score[1,3])),5))

    load(c(paste0("HCI_BS_CVCS_t0_",t0,"_w",w2,".RData")))
    HCI_BS_all_w2 <- c(HCI_BS_all_w2, round(HCI_BS_t0,5))

    load(c(paste0("LCI_BS_CVCS_t0_",t0,"_w",w2,".RData")))
    LCI_BS_all_w2 <- c(LCI_BS_all_w2, round(LCI_BS_t0,5))
  }
  mat_res_BS_w2 <-as.data.frame(cbind(seq(1, 25, by = 1)+40, BS_all_w2, c(paste0("[", LCI_BS_all_w2, " ", HCI_BS_all_w2, "]"))))
  colnames(mat_res_BS_w2) <- c("Landmark time","BS","95%CI")
  print(mat_res_BS_w2)
  #####
  
  #####
  #Plot dynamic BS results
  # png(file = c(paste0("Dynamic_BS_window_", w2, "_yrs.png")))
  plot(seq(1, 25, by = 1),
       BS_all_w2,
       ylim=c(0, 1),
       ylab="BS", xlab="Age (years)",
       cex.lab = 1.2, 
       xaxt="n",
       cex.axis = 1.2, pch=19,
       main = "Dynamic BS (landmark time in [41 - 65], w = 2)")
  axis(1, at = seq(1, 25, by = 1),
       labels = 40+seq(1, 25, by = 1),
       cex.axis = 1.2)
  lines(seq(1, 25, by = 1), BS_all_w2)
  #CI
  points(seq(1, 25, by = 1), LCI_BS_all_w2)
  lines(seq(1, 25, by = 1), LCI_BS_all_w2, lty = 3)
  points(seq(1, 25, by = 1), HCI_BS_all_w2)
  lines(seq(1, 25, by = 1), HCI_BS_all_w2, lty = 3)
  legend("bottomright",lty = c(1,3), legend=c("mean BS", "95%CI"))
  # dev.off()
  #####
  #############
  ####################################
  print("End of program")