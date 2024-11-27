  ########################################################################
  # Copyright © 2024 INSERM U1219, Therapixel SA
  # Contributors: Manel Rakez, Julien Guillaumin
  # All rights reserved.
  # This file is subject to the terms and conditions described in the
  # LICENSE file distributed in this package.
  ########################################################################
  
  ####################################
  # Project: The DeepJoint algorithm: Joint Model fit using consensus algorithm on a subset of 2501 women (test data)
  # NB: Results are solely for illustration purposes
  ####################################
  
  ####################################
  # Start with a clean environment
  rm(list=ls(all=TRUE))
  ####################################

  ####################################
  source("~/deepjoint-algo/src/deepjoint_r/jm_consensus_source_script_CV.R")
  pcks <- c("JMbayes2", "dplyr")
  invisible(lapply(pcks, require, character.only = TRUE)) #Load packages
  directory <- "~/deepjoint-algo/tests/deepjoint_r"
  setwd(directory) #Define your directory
  ####################################
  
  ####################################
  # Import datasets
  data <- read.csv2("~/deepjoint-algo/test_data/deepjoint_r/data.csv")
  data_surv <- read.csv2("~/deepjoint-algo/test_data/deepjoint_r/data_surv.csv")
  ####################################
  
  ####################################
  # Example for Model 3 with dense area as the longitudinal biomarker and CVCS as the link function. For Model 3 with percent density replace "sqrt_da_cm2" by "sqrt_pd"
  # 1/ Model specification
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
  
  # 2/ Fit the model with parallel run using the consensus algorithm
  mod_split_CVCS <- jm_parallel(args_jm_CVCS,
                                id_var="pts_id",
                                data_long = data,
                                data_surv = data_surv,
                                args_long = args_long,
                                args_surv = args_surv,
                                n_slices = n_slices,
                                CV_pred = 1, # 1 if you don't want to save split data
                                event_var = "event")
  
  # 3/ Create consensus model
  jm_cons_CVCS <- jm_consensus(mod_split_CVCS$jm, combine_chains = FALSE)
  
  # 4/Summarize model parameters in (mean, lower, and upper quantiles)
  summary_mcmc <- function(mcmc_list, param_type) {
    summarize_mcmc <- function(x) {
      res_mean <- round(apply(x, 2, mean), 3)
      res_lower <- round(apply(x, 2, quantile, probs = 0.025), 3)
      res_upper <- round(apply(x, 2, quantile, probs = 0.975), 3)
      return(data.frame(mean = res_mean, lower_95CI = res_lower, upper_95CI = res_upper))
    }
    
    if (param_type == "betas") {
      res_param <- summarize_mcmc(mcmc_list$betas1)
    } else if (param_type == "sigmas") {
      res_param <- summarize_mcmc(mcmc_list$sigmas)
    } else if (param_type == "gammas") {
      res_param <- summarize_mcmc(mcmc_list$gammas)
    } else if (param_type == "alphas") {
      res_param <- summarize_mcmc(mcmc_list$alphas)
    } else {
      stop("Invalid parameter type. Please specify 'betas', 'sigmas', 'gammas', or 'alphas'.")
    }
    return(res_param)
  }
  # 4.a/ Calculate summary for 'betas'
  betas_summary <- summary_mcmc(jm_cons_CVCS$mcmc, "betas")
  # 4.b/ Calculate summary for 'sigmas'
  sigmas_summary <- summary_mcmc(jm_cons_CVCS$mcmc, "sigmas")
  # 4.c/ Calculate summary for 'alphas'
  alphas_summary <- summary_mcmc(jm_cons_CVCS$mcmc, "alphas")
  # 4.d/ All results
  All_param_summary <- rbind(betas_summary, sigmas_summary, alphas_summary)
  print(All_param_summary)
  ####################################
  print("End of program")