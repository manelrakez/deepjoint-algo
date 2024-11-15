  ########################################################################
  # Copyright © 2024 INSERM U1219, Therapixel SA
  # Contributors: Manel Rakez, Julien Guillaumin
  # All rights reserved.
  # This file is subject to the terms and conditions described in the
  # LICENSE file distributed in this package.
  ########################################################################
  
  ####################################
  # Project: The DeepJoint algorithm: Joint model fit results summary
  ####################################
  
  ####################################
  # Start with a clean environment
  rm(list=ls(all=TRUE))
  ####################################
  
  ####################################
  # Load packages:
  pcks <- c("JMbayes2", "coda")
  invisible(lapply(pcks, require, character.only = TRUE))
  ####################################
  
  ####################################
  # Set your work directory
  directory <- "~/work_directory"
  setwd(directory)
  ####################################
  
  ####################################
  #Load model fit results
  load(file="jm_cons_CVCS.RData")
  load(file="output_jmcons_CVCS.RData")
  load(file="mod_split_CVCS.RData")
  ####################################
  
  ####################################
  # Summarize model parameters
  
  ###############
  summary_mcmc <- function(mcmc_list, param_type) {
    # Function to summarize MCMC results (mean, lower, and upper quantiles)
    summarize_mcmc <- function(x) {
      res_mean <- round(apply(x, 2, mean), 3)
      res_lower <- round(apply(x, 2, quantile, probs = 0.025), 3)
      res_upper <- round(apply(x, 2, quantile, probs = 0.975), 3)
      return(data.frame(mean = res_mean, lower_95CI = res_lower, upper_95CI = res_upper))
    }
    
    # Extract and summarize the specified parameter type
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
  ###############
  
  ###############
  # 1/ Calculate summary for 'betas'
  betas_summary <- summary_mcmc(jm_cons_CVCS$mcmc, "betas")
  # print(betas_summary)
  
  # 2/ Calculate summary for 'sigmas'
  sigmas_summary <- summary_mcmc(jm_cons_CVCS$mcmc, "sigmas")
  # print(sigmas_summary)
  
  # 3/ Calculate summary for 'alphas'
  alphas_summary <- summary_mcmc(jm_cons_CVCS$mcmc, "alphas")
  # print(alphas_summary)

  # 4/ All results
  All_param_summary <- rbind(betas_summary, sigmas_summary, alphas_summary)
  print(All_param_summary)
  ###############
  ####################################
  
  ####################################
  # Check the model's convergence
  
  # 1/ Calculate R_hat
  Rhat_betas <- gelman.diag(output_jmcons_CVCS$mcmc$betas1, autoburnin = FALSE)$psrf
  Rhat_sigmas <- gelman.diag(output_jmcons_CVCS$mcmc$sigmas, autoburnin = FALSE)$psrf
  Rhat_alphas <- gelman.diag(output_jmcons_CVCS$mcmc$alphas, autoburnin = FALSE)$psrf
  Rhat_D <- gelman.diag(output_jmcons_CVCS$mcmc$D, autoburnin = FALSE)$psrf
  
  # 2/ Plot trace and density plots
  # ggtraceplot(output_jmcons_CVCS, "alphas", grid = TRUE)
  # ggdensityplot(output_jmcons_CVCS, "alphas", grid = TRUE)
  ####################################
  
  ####################################
  # Check the model's goodness of fit with averaged WAIC, DIC, and LPML
  params <- c("DIC", "WAIC","LPML")
  DIC <- NULL
  WAIC <- NULL
  LPML <- NULL
  for (i in 1:mod_split_CVCS$n_slices){
    for (param in params){
      if (param == "DIC"){
        DIC <- c(DIC, mod_split_CVCS$jm[[i]]$fit_stats$marginal[[param]])
      } else if(param == "WAIC") {
        WAIC <- c(WAIC, mod_split_CVCS$jm[[i]]$fit_stats$marginal[[param]])
      } else {
        LPML <- c(LPML, mod_split_CVCS$jm[[i]]$fit_stats$marginal[[param]])
      }
    }
  }
  
  average_DIC <- round(mean(DIC),2)
  average_WAIC <- round(mean(WAIC),2)
  average_LPML <- round(mean(LPML),2)
  print(list(paste0("average_DIC = ", average_DIC),
             paste0("average_WAIC = ", average_WAIC),
             paste0("average_LPML = ", average_LPML)))
  ####################################
  print("End of program")