  ###########################################################################################################
  ###########################################################################################################
  ######         Project: The DeepJoint algorithm: Joint Model fit using consensus algorithm           ######
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
  # pts_id: women's ID
  # pts_age: continuous variable for a woman's age at each screening visit (years)
  # bl_age: continuous variable for baseline age (bl_age in [40-74]) corresponding to the woman's age at first screening visit (years)
  # manuf: binary variable for manufacturer (GE vs. Hologic)
  # sqrt_da_cm2: continuous variable for sqrt dense area values (cm²)
  # sqrt_pd: continuous variable for sqrt percent density values (%)
  # event: event indicator (1: event, 0: censored) (should be numeric)
  # visit_date: date of each screening visit
  ####################################
  
  ####################################
  # Prepare data for fit
  data <- data %>%
    arrange(pts_id, visit_date) %>%
    ungroup()
  data$pts_age_modif <- round(data$pts_age-40,2)
  data <- as.data.frame(data)
  
  # Create surv data (keep only the last row per patient) + calculate start/end times
  data_surv <- data
  data_surv <- data_surv  %>%
    group_by(pts_id)  %>%
    mutate(start = first(pts_age_modif),
           end = last(pts_age_modif)) %>%
    slice(n()) %>%
    ungroup()
  data_surv <- as.data.frame(data_surv)
  ####################################
  
  ####################################
  # Model specifications
  # Example with dense area values (sqrt_da_cm2). Just replace sqrt_da_cm2 by sqrt_pd to run model with percent density values
  n_slices <- 12 #To adapt in accordance with the size of your data (see Afonso et al., arXiv:2310.03351)
  n_chains <- 3
  n_iter <- 8500L
  n_burnin <- 3500L
  n_iter_net <- n_iter-n_burnin
  #######
  args_long <- list(fixed = sqrt_da_cm2 ~ pts_age_modif + bl_age + manuf, 
                    random = ~ pts_age_modif | pts_id, #Random intercept and slope
                    control = lmeControl(opt = "optim"))
  #######
  
  #######
  args_surv <- list(formula = survival::Surv(start, end, event) ~ 1)
  #######
  
  #######
  #JM with current value (CV)
  args_jm_CV <- list(time_var = "pts_age_modif", 
                  n_iter = n_iter, n_burnin = n_burnin, n_thin = 1L, 
                  parallel = 'multicore',
                  functional_forms = ~ value(sqrt_da_cm2))
  #######
  
  #######
  #JM with current slope (CS)
  args_jm_CS <- list(time_var = "pts_age_modif", 
                  n_iter = n_iter, n_burnin = n_burnin, n_thin = 1L, 
                  parallel = 'multicore',
                  functional_forms = ~ slope(sqrt_da_cm2))
  #######
  
  #######
  #JM with CVCS
  args_jm_CVCS <- list(time_var = "pts_age_modif", 
                  n_iter = n_iter, n_burnin = n_burnin, n_thin = 1L, 
                  parallel = 'multicore',
                  functional_forms = ~ value(sqrt_da_cm2) + slope(sqrt_da_cm2))
  #######
  
  #######
  #JM with cumulative effect (area)
  args_jm_area <- list(time_var = "pts_age_modif", 
                       n_iter = n_iter, n_burnin = n_burnin, n_thin = 1L, 
                       parallel = 'multicore',
                       functional_forms = ~ area(sqrt_da_cm2))
  #######
  ####################################
  
  ####################################
  #Example employing args_jm_CVCS. Adapt the code to use another association type
  
  #######
  # 1/ Fit the model with parallel run using the consensus algorithm
  mod_split_CVCS <- jm_parallel(args_jm_CVCS,
                                id_var="pts_id",
                                data_long = data,
                                data_surv = data_surv,
                                args_long = args_long,
                                args_surv = args_surv,
                                n_slices = n_slices,
                                CV_pred = 1, # 1 if you don't want to save split data.
                                event_var = "event")
  save(mod_split_CVCS, file = "mod_split_CVCS.RData")
  #######
  
  #######
  # 2/ Create consensus model
  jm_cons_CVCS <- jm_consensus(mod_split_CVCS$jm, combine_chains = FALSE)
  save(jm_cons_CVCS, file = "jm_cons_CVCS.RData")
  #######
  
  #######
  # 3/ Save consensus chains in a jm_class() object
  output_jmcons_CVCS <- jm_class_func(jm_split_obj = mod_split_CVCS$jm,
                                 jm_cons_obj = jm_cons_CVCS,
                                 n_chains = n_chains,
                                 n_iter = n_iter_net)
  save(output_jmcons_CVCS, file = "output_jmcons_CVCS.RData")
  #######
  ####################################
  print("End of program")