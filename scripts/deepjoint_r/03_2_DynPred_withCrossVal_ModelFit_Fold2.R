  ########################################################################
  # Copyright © 2024 INSERM U1219, Therapixel SA
  # Contributors: Manel Rakez, Julien Guillaumin
  # All rights reserved.
  # This file is subject to the terms and conditions described in the
  # LICENSE file distributed in this package.
  ########################################################################
  
  ####################################
  # Project: The DeepJoint algorithm: Breast cancer risk prediction (with cross-val)
  # Part 1-2 : Fit the joint model to data from the 2nd fold using jm_consensus
  # NB: Use this code to decompose the cross-validation process when the dataset size is too big (>= 10k subjects) 
  ####################################
  
  ####################################
  # Start with a clean environment
  rm(list=ls(all=TRUE))
  ####################################
  
  ####################################
  # Source for jm_consensus
  source("~/jm_consensus_source_script_CV.R")
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
  #Fit the model
  # Example with dense area values (sqrt_da_cm2). Just replace sqrt_da_cm2 by sqrt_pd to run model with percent density values
  
  ########
  # Initialize variables
  n_chains <- 3
  n_iter <- 8500
  n_burnin <- 3500
  n_iter_net <- n_iter-n_burnin
  n_slices <- 10 #To adapt according to the size of your data (see Afonso et al., arXiv:2310.03351)

  args_long <- list(fixed = sqrt_da_cm2 ~ pts_age_modif + bl_age + manuf, 
                    random = ~ pts_age_modif | pts_id,
                    control = lmeControl(opt = "optim")
                    )
  args_surv <- list(formula = survival::Surv(start, end, event) ~ 1)
  args_jm_CVCS <- list(time_var = "pts_age_modif", 
                       n_iter = n_iter, n_burnin = n_burnin, n_thin = 1L,
                       parallel = 'multicore',
                       functional_forms = ~ value(sqrt_da_cm2) + slope(sqrt_da_cm2))
  ########
  
  ########
  i <- 2
  print(c(paste0("Fit the model with parallel run for fold number: ", i)))
  data_app <- data_train[[i]]

  #Create Surv dataset
  data_mean_mean_surv <- data_app  %>%
      group_by(pts_id)  %>%
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
  ####################################
  print("End of program")