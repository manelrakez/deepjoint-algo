  ###########################################################################################################
  ###########################################################################################################
  ######         Project: The DeepJoint algorithm: Breast cancer risk prediction                       ######     
  ######                   with cross-validation for a subset of 2501 women (test data)                ######
  ######                  NB: Results are solely for illustration purposes                             ######
  ######         Programmer: M.RAKEZ                                                                   ######
  ######         Creation date: Wednesday, 23rd of October 2024                                        ######
  ######         Last updated: 30OCT2024                                                               ######
  ###########################################################################################################
  ###########################################################################################################
  
  ####################################
  # Start with a clean environment
  rm(list=ls(all=TRUE))
  ####################################
  
  ####################################
  # Source for jm_consensus
  source("~/deepjoint-algo/src/deepjoint_r/jm_consensus_source_script_CV.R")
  ####################################
  
  ####################################
  # Load packages:
  pcks <- c("JMbayes2", "dplyr", "timeROC", "riskRegression")
  invisible(lapply(pcks, require, character.only = TRUE))
  ####################################
  
  ####################################
  # Define your work directory
  directory <- "~/deepjoint-algo/tests/deepjoint_r"
  setwd(directory)
  ####################################
  
  ####################################
  # Import the dataset
  data <- read.csv2("~/deepjoint-algo/test_data/deepjoint_r/data.csv")
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
  # Step 1: Fit the model
  
  ########
  # Initialize variables
  n_slices <- 3
  n_chains <- 3
  n_iter <- 3500L
  n_burnin <- 500L
  n_iter_net <- n_iter-n_burnin

  args_long <- list(fixed = sqrt_da_cm2 ~ pts_age_modif + bl_age + manuf, 
                    random = ~ pts_age_modif | pts_id, #Random intercept and slope
                    control = lmeControl(opt = "optim"))
  args_surv <- list(formula = survival::Surv(start, end, event) ~ 1)
  args_jm_CVCS <- list(time_var = "pts_age_modif", 
                       n_iter = n_iter, n_burnin = n_burnin, n_thin = 1L, 
                       #parallel = 'multicore', #multicore is not available on windows
                       functional_forms = ~ value(sqrt_da_cm2) + slope(sqrt_da_cm2))
  ########
  
  ########
  for (i in 1:num_folds){
    print(c(paste0("Fit the model with parallel run for fold number: ", i)))
    data_app <- data_train[[i]]
    
    #Create Surv dataset
    data_mean_mean_surv <- data_app  %>%
      group_by(pts_id)  %>%
      mutate(start = first(pts_age_modif),
             end = last(pts_age_modif)) %>%
      slice(n())
    data_mean_mean_surv <- as.data.frame(data_mean_mean_surv %>% ungroup())
    
    #######
    # 1/ Fit the model with parallel run using the consensus algorithm
    mod_split_i <- jm_parallel(args_jm_CVCS,
                               id_var = "pts_id",
                               data_long = data_app,
                               data_surv = data_mean_mean_surv, 
                               args_long = args_long,
                               args_surv = args_surv,
                               n_slices = n_slices,
                               CV_pred = 1,
                               event_var = "event")      
    save(mod_split_i, file=c(paste0("mod_split_",i,".RData")))
    #######
    
    #######
    # 2/ Create consensus model  
    jm_cons_i <- jm_consensus(mod_split_i$jm,combine_chains = FALSE)
    save(jm_cons_i, file=c(paste0("jm_cons_",i,".RData")))
    #######
    
    #######
    # 3/ Save consensus chains in a jm_class() object  
    output_jmcons <- jm_class_func(jm_split_obj=mod_split_i$jm,
                                   jm_cons_obj=jm_cons_i,
                                   n_chains=n_chains,
                                   n_iter= n_iter_net)
    save(output_jmcons, file=c(paste0("output_jmcons_",i,".RData")))
    #######
  }
  ####################################
  
  ####################################
  # Step 2: Compute breast cancer risk probabilities
  first_run <- 1 # if 1st time running this code
  # first_run <- NULL #if it's not the first time running this code
  
  if (!is.null(first_run)){
    # Create the model_fits object from the output_jmcons of each fold
    print("Create the model_fits object")
    model_fits <- vector("list", 5)
    for (i in 1:num_folds){
      load(c(paste0("output_jmcons_",i,".RData")))
      model_fits[[i]] <- output_jmcons
    }
    save(model_fits, file="model_fits_CVCS_PredCrossVal.RData")
    
  } else {
    #Load model fits
    print("Load model_fits")
    load("model_fits_CVCS_PredCrossVal.RData")
  }
  ########
  
  ########
  #Define prediction windows
  windows <- c(2, 5, 10)
  ########
  
  ########
  #Calculate predictions for landmark times [5, 10, 15, 20, 25] corresponding to ages between 41 ans 65 years old
  for (k in 1:length(windows)){
    w <- windows[k]
    
    for (t0 in seq(5, 25, by = 5)){
      tend <- t0 + w
      pred_SR_all_folds <- data.frame()
      
      for(i in 1: num_folds){
        test_data <- data_test[[i]]
        model_fit <- model_fits[[i]]
        
        pat_id <- unique(factor(test_data$pts_id[which(test_data$end > t0)]))
        test_data <- test_data[test_data$pts_id %in% pat_id, ] # Only women who are BC-free at t0
        test_data <- test_data[test_data$pts_age_modif < t0, ] # Only the biomarker's repeated measurements up to t0
        test_data$event <- NULL # These women are BC-free, so their event equals to zero before t0
        test_data$end <- t0
        
        print(c(paste0("Compute predictions for t0 =", t0," on fold N°: ", i)))
        pred <- predict(model_fit, newdata = test_data, process = "event",
                        times = c(t0, tend),
                        return_newdata = TRUE)
        pred_SR <- pred %>% 
          group_by(pts_id) %>% 
          slice(n()) %>%
          arrange(pts_id) %>%
          ungroup()
        
        #save(pred_SR, file=c(paste0("pred_fold",i,"_t0_",t0,"_w",w,".RData")))
        pred_SR_all_folds <- rbind(pred_SR_all_folds, pred_SR)
        
      }
      
      #Save prediction results for a given t0
      save(pred_SR_all_folds, file=c(paste0("pred_SR_all_folds_t0_",t0,"_w",w,".RData")))
      
    }
  }
  ####################################
  
  ####################################
  # Step 3: Compute AUC and BS metrics
  
  ########
  #Prepare data/results for metrics calculation
  for (k in 1:length(windows)){
    w <- windows[k]
    for (t0 in seq(5, 25, by = 5)){
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
  
  ####################################
  # Step 4: Summarize the model's predictive performance
  # Here, an example with prediction window of 2 years
  w2 <- 2
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
  for (t0 in seq(5, 25, by = 5)){
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
  mat_res_AUC_w2 <-as.data.frame(cbind(seq(5, 25, by = 5)+40, AUC_all_w2, c(paste0("[", LCI_AUC_all_w2, " ", HCI_AUC_all_w2, "]"))))
  colnames(mat_res_AUC_w2) <- c("Landmark time","AUC","95%CI")
  print(mat_res_AUC_w2)
  #####
  
  #####
  #Plot dynamic AUC results
  # png(file = c(paste0("Dynamic_AUC_window_", w2, "_yrs.png")))
  plot(seq(5, 25, by = 5),
       AUC_all_w2,
       ylim=c(0, 1),
       ylab="AUC", xlab="Age (years)",
       cex.lab = 1.2, 
       xaxt="n",
       cex.axis = 1.2, pch=19,
       main = "Dynamic AUC (landmark time in [45 - 65], w = 2)")
  axis(1, at = seq(5, 25, by = 5),
       labels = 40+seq(5, 25, by = 5),
       cex.axis = 1.2)
  lines(seq(5, 25, by = 5), AUC_all_w2)
  abline(h=50, lty=2)
  #CI
  points(seq(5, 25, by = 5), LCI_AUC_all_w2)
  lines(seq(5, 25, by = 5), LCI_AUC_all_w2, lty = 3)
  points(seq(5, 25, by = 5), HCI_AUC_all_w2)
  lines(seq(5, 25, by = 5), HCI_AUC_all_w2, lty = 3)
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
  for (t0 in seq(5, 25, by = 5)){
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
  mat_res_BS_w2 <-as.data.frame(cbind(seq(5, 25, by = 5)+40, BS_all_w2, c(paste0("[", LCI_BS_all_w2, " ", HCI_BS_all_w2, "]"))))
  colnames(mat_res_BS_w2) <- c("Landmark time","BS","95%CI")
  print(mat_res_BS_w2)
  #####
  
  #####
  #Plot dynamic BS results
  # png(file = c(paste0("Dynamic_BS_window_", w2, "_yrs.png")))
  plot(seq(5, 25, by = 5),
       BS_all_w2,
       ylim=c(0, 1),
       ylab="BS", xlab="Age (years)",
       cex.lab = 1.2, 
       xaxt="n",
       cex.axis = 1.2, pch=19,
       main = "Dynamic BS (landmark time in [45 - 65], w = 2)")
  axis(1, at = seq(5, 25, by = 5),
       labels = 40+seq(5, 25, by = 5),
       cex.axis = 1.2)
  lines(seq(5, 25, by = 5), BS_all_w2)
  #CI
  points(seq(5, 25, by = 5), LCI_BS_all_w2)
  lines(seq(5, 25, by = 5), LCI_BS_all_w2, lty = 3)
  points(seq(5, 25, by = 5), HCI_BS_all_w2)
  lines(seq(5, 25, by = 5), HCI_BS_all_w2, lty = 3)
  legend("topright",lty = c(1,3), legend=c("mean BS", "95%CI"))
  # dev.off()
  #####
  #############
  ####################################
  print("End of program")