  ###########################################################################################################
  ###########################################################################################################
  ######         Project: The DeepJoint algorithm: Breast cancer risk prediction (with cross-val)      ######
  ######                  Part 2 : Compute risk propabilities in the test set of each fold             ######
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
  pcks <- c("JMbayes2", "dplyr")
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
  #Predict breast cancer risk
  
  ########
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
  #Calculate predictions for landmark times [1, ..., 25] corresponding to ages between 41 ans 65 years old
  for (k in 1:length(windows)){
    w <- windows[k]
    
    for (t0 in seq(1, 25, by = 1)){
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
  #########
  ######################################
  print("End of program")