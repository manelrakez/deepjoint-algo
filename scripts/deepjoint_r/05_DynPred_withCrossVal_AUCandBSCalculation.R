  ###########################################################################################################
  ###########################################################################################################
  ######         Project: The DeepJoint algorithm: Breast cancer risk prediction (with cross-val)      ######
  ######                  Part 3 : Compute AUC and BS to evaluate the model's predictive performance   ######
  ######         Programmer: M.RAKEZ                                                                   ######
  ######         Creation date: Thursday, 11th of April 2024                                           ######
  ######         Last updated: 11APR2024                                                               ######
  ###########################################################################################################
  ###########################################################################################################
  
  ####################################
  # Start with a clean environment
  rm(list=ls(all=TRUE))
  ####################################
  
  ####################################
  # Load packages:
  pcks <- c("JMbayes2", "dplyr", "timeROC", "riskRegression")
  invisible(lapply(pcks, require, character.only = TRUE))
  ####################################
  
  ####################################
  # Set your work directory
  directory <- "~/work_directory"
  setwd(directory)
  ####################################
  
  ####################################
  # Import the dataset
  data <- read.csv2("~/data.csv")
  # data is a long format dataset that should include at least the following variables
  # pts_id: women's ID (in numeric format)
  # pts_age: continuous variable for a woman's age at each screening visit (years)
  # bl_age: continuous variable for baseline age (bl_age in [40-74]) corresponding to the woman's age at first screening visit (years)
  # manuf: binary variable for manufacturer (GE vs. Hologic)
  # sqrt_da_cm2: continuous variable for sqrt dense area values (cm²)
  # sqrt_pd: continuous variable for sqrt percent density values (%)
  # event: event indicator (1: event, 0: censored) (should be numeric)
  # visit_date: date of each screening visit  
  
  data <- data %>%
    arrange(pts_id, visit_date)
  data$pts_age_modif <- round(data$pts_age-40,2)
  data <- data  %>%
    group_by(pts_id)  %>%
    mutate(start = first(pts_age_modif),
           end = last(pts_age_modif)) %>%
    ungroup()
  data <- as.data.frame(data)
  ####################################
  
  ####################################
  # Create folds
  print("Create folds")
  num_folds <- 5
  Folds_CV <- create_folds(data, V = num_folds, id_var = "pts_id", seed = 16694)
  data_train <- Folds_CV$training
  data_test <- Folds_CV$test
  ####################################
  
  ####################################
  ########
  #Define prediction windows
  windows <- c(2, 5, 10)
  ########

  ########
  #Prepare data/results for metrics calculation
  for (k in 1:length(windows)){
    w <- windows[k]
    for (t0 in seq(1, 25, by = 1)){
      tend <- t0 + w
      
      print(c(paste0("Load predictions for t0 = ", t0)))
      load(c(paste0("pred_SR_all_folds_t0_",t0,"_w",w,".RData")))
      pred_SR_all_folds <- pred_SR_all_folds[,c("pts_id","pred_CIF","low_CIF","upp_CIF")]
      
      ########
      # Define the test data from all folds for a given t0
      print(c(paste0("Prepare test data from all folds for t0 = ", t0)))
      test_data_allfolds <- data.frame()
      for(i in 1:num_folds){
        test_data <- data_test[[i]]
        pat_id <- unique(factor(test_data$pts_id[which(test_data$end > t0)]))
        test_data <- test_data[test_data$pts_id %in% pat_id, ] # Only women who are BC-free at t0
        test_data <- test_data[test_data$pts_age_modif < t0, ] # Only the biomarker's repeated measurements up to t0
        test_data$event <- NULL # These women are BC-free, so their event equals to zero before t0
        test_data$end <- t0
        test_data <- test_data %>% 
          group_by(pts_id) %>%
          slice(n()) %>%
          arrange(pts_id) %>%
          ungroup()
        
        test_data_allfolds <- rbind(test_data_allfolds, test_data)
      }
      ########
      
      ########
      # Create data for metrics evaluation (test data where we replace end and event with the true values from the original dataset) 
      pat_id <- unique(factor(test_data_allfolds$pts_id))
      landmark_t0 <- data[data$pts_id %in% pat_id, ]
      landmark_t0 <- landmark_t0 %>%
        group_by(pts_id) %>%
        slice(n()) %>%
        arrange(pts_id) %>%
        ungroup()
      landmark_t0 <- landmark_t0[,c("pts_id","end","event")]
      data_eval <- merge(landmark_t0, pred_SR_all_folds, by="pts_id")
      data_eval <- data_eval %>%
        arrange(pts_id) %>%
        ungroup()
      ########
      
      ########
      # AUC computation
      print(c(paste0("Compute AUC for t0 = ", t0)))
      AUC_t0 <- timeROC(data_eval$end,
                        data_eval$event,
                        data_eval$pred_CIF,
                        weighting = "marginal",
                        cause = 1,
                        times = c(t0, tend),
                        iid = TRUE)
      LCI_AUC_t0 <- confint(AUC_t0, parm=NULL,
                            level = 0.95, n.sim = 2000)$CI_AUC[1] #2.5% boundary
      HCI_AUC_t0 <- confint(AUC_t0, parm=NULL,
                            level = 0.95, n.sim = 2000)$CI_AUC[2] #97.5% boundary
      
      save(AUC_t0, file=c(paste0("AUC_CVCS_t0_",t0,"_w",w,".RData")))
      save(LCI_AUC_t0, file=c(paste0("LCI_AUC_CVCS_t0_",t0,"_w",w,".RData")))
      save(HCI_AUC_t0, file=c(paste0("HCI_AUC_CVCS_t0_",t0,"_w",w,".RData")))
      ########
      
      ########
      # BS computation
      print(c(paste0("Compute BS for t0 = ", t0)))
      pred_mat <- data.matrix(data_eval[,"pred_CIF"]) 
      BS_t0 <- Score(object = list(pred_mat),
                     formula = Hist(end, event)~1,
                     data = data_eval,
                     conf.int = T,
                     times = tend,
                     cens.method = "ipcw",
                     metrics = "brier",
                     null.model = F,
                     plots = "calibration")
      LCI_BS_t0 <- unlist(unname(BS_t0$Brier$score[1,5])) #2.5% boundary
      HCI_BS_t0 <- unlist(unname(BS_t0$Brier$score[1,6])) #97.5% boundary
      
      save(BS_t0, file=c(paste0("BS_CVCS_t0_",t0,"_w",w,".RData")))
      save(LCI_BS_t0, file=c(paste0("LCI_BS_CVCS_t0_",t0,"_w",w,".RData")))
      save(HCI_BS_t0, file=c(paste0("HCI_BS_CVCS_t0_",t0,"_w",w,".RData")))
      ########
    }
    
  }
  ####################################
  print("End of program")