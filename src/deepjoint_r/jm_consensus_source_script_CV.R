  ########################################################################
  # Copyright © 2024 INSERM U1219, Therapixel SA
  # Contributors: Manel Rakez, Julien Guillaumin
  # All rights reserved.
  # This file is subject to the terms and conditions described in the
  # LICENSE file distributed in this package.
  ########################################################################
  
  ####################################
  # Project: The DeepJoint algorithm: Source script for the JM consensus method and other functions
  # Source code adapted by M. RAKEZ from Miranda Afonso et al (2023)
  ####################################
  
  ####################################
  # Packages:
  library("JMbayes2")
  ####################################
  
  ####################################
  ###############
  #Functions for jm_consensus
  # Slice data
  slicer <- function(n_slices = 3, id_var = NULL, 
                     data_long = NULL, data_surv= NULL, 
                     seed = 1234L) {
    data_long_event <- data_long %>%
      group_by(.data[[id_var]]) %>%
      filter(any(.data[[event_var]] == 1))
    data_surv_event <- subset(data_surv, data_surv[[event_var]] == 1)
    data_long_noevent <- data_long %>%
      group_by(.data[[id_var]]) %>%
      filter(all(.data[[event_var]] == 0))
    data_surv_noevent <- subset(data_surv, data_surv[[event_var]] == 0)
    
    ids_unq_event <- unique(c(data_long_event[[id_var]], data_surv_event[[id_var]]))
    ids_unq_noevent <- unique(c(data_long_noevent[[id_var]], data_surv_noevent[[id_var]]))
    ids_slc_event <- split(sample(ids_unq_event), (seq_along(ids_unq_event) %% n_slices) + 1)
    ids_slc_noevent <- split(sample(ids_unq_noevent), (seq_along(ids_unq_noevent) %% n_slices) + 1)
    ids_slc <- lapply(seq_along(ids_slc_event), function(i) c(ids_slc_noevent[[i]], ids_slc_event[[i]]))
    
    long <- lapply(ids_slc, function(ids) 
      data_long[data_long[[id_var]] %in% ids, ])
                   
    surv <- lapply(ids_slc, function(ids) 
      data_surv[data_surv[[id_var]] %in% ids, ])
    
    class(long) <- class(surv) <- "sliced_data"
    
    output <- list(long = long,
                   surv = surv)
    output
    
  }
  ###############
                                   
   ###############
  #' Run JMbayes2 on sliced data
  #'
  #' This function runs a joint model on sliced data.
  #' 
  #' @param args_jm List of arguments to pass to `jm` function
  #' @param data_long,data_surv Longitudinal and survival data to pass to `nlme::lme`
  #'      and `survival:coxph`.
  #' @param args_long,args_surv Lists of arguments to pass to longitudinal and survival models
  #' @param model_long,model_surv If split models have already been fit, they can be
  #'      provided here to save time on fitting them. The `data_*` and `args_*` functions 
  #'      are not used in this case.
  #' @param n_slices Number of slices to split data into
  #' @param n_cores Maximum number of cores to run model on. Should be between 1 and `n_slices`. Probably no 
  #'      reason to change this from default of `n_slices`. 
  #' @param seed Random seed
  #' @param CV_pred indicates whether wee're performing cross-validation or not
  #' @param event_var column name for the event indicator
  #' 
  #' @value A list with values `long`, `surv`, and `jm`, each of which is a list containing `n_slices` fitted models.
  jm_parallel <- function(args_jm, id_var=NULL, 
                          data_long = NULL, data_surv = NULL, 
                          args_long = NULL, args_surv = NULL, 
                          model_long = NULL, model_surv= NULL,
                          n_slices = 3, n_cores = n_slices, seed = 1234L,
                          CV_pred = NULL,
                          event_var = NULL) {
    set.seed(seed)
    # Haven't set up for Windows yet, run serially instead and warn user
    if (.Platform$OS.type == 'windows'){
      n_cores <- 1
      warning('Multiple cores not supported on Windows, defaulting to 1 core.')
    }
    
    # Slice data
    slicer <- function(n_slices, id_var = id_var, data_long, data_surv, seed = seed) {
        data_long_event <- data_long %>%
            group_by(.data[[id_var]]) %>%
            filter(any(.data[[event_var]] == 1))
        data_surv_event <- subset(data_surv, data_surv[[event_var]] == 1)
        data_long_noevent <- data_long %>%
            group_by(.data[[id_var]]) %>%
            filter(all(.data[[event_var]] == 0))
        data_surv_noevent <- subset(data_surv, data_surv[[event_var]] == 0)
    
        ids_unq_event <- unique(c(data_long_event[[id_var]], data_surv_event[[id_var]]))
        ids_unq_noevent <- unique(c(data_long_noevent[[id_var]], data_surv_noevent[[id_var]]))
        ids_slc_event <- split(sample(ids_unq_event), (seq_along(ids_unq_event) %% n_slices) + 1)
        ids_slc_noevent <- split(sample(ids_unq_noevent), (seq_along(ids_unq_noevent) %% n_slices) + 1)
        ids_slc <- lapply(seq_along(ids_slc_event), function(i) c(ids_slc_noevent[[i]], ids_slc_event[[i]]))
                       
        long <- lapply(ids_slc, function(ids)
                      data_long[data_long[[id_var]] %in% ids, ])
      if (is.null(CV_pred)) {
        for (i in 1:n_slices) {
          split_data_i <- as.data.frame(long[[i]])
          write.csv2(split_data_i,
                     paste0("Splits_data/split_",i,".csv"),
                     row.names = FALSE)
        }
      }
      surv <- lapply(ids_slc, function(ids) 
        data_surv[data_surv[[id_var]] %in% ids, ])
      class(long) <- class(surv) <- "sliced_data"
      list(long = long, surv = surv)
      
    }
    if (is.null(model_long)) {
      sliced_data <- slicer(n_slices, id_var, data_long, data_surv)
    }
    
    # Run longitudinal models
    if (is.null(model_long)) {
      f <- function(i) do.call(nlme::lme.formula, c(list(data = sliced_data$long[[i]]), args_long))
      out_long <- parallel::mclapply(1:n_slices, f, mc.cores = n_cores)
    } else {
      out_long <- model_long
    }
    
    # Run survival models
    if (is.null(model_surv)) {
      f <- function(i) do.call(survival::coxph, c(list(data = sliced_data$surv[[i]]), args_surv))
      out_surv <- parallel::mclapply(1:n_slices, f, mc.cores = n_cores)
    } else {
      out_surv <- model_surv
    }
    
    # Run joint model
    f <- function(i) do.call(JMbayes2::jm, c(list(Surv_object = out_surv[[i]],
                                                  Mixed_objects = out_long[[i]]),
                                             args_jm))
    out_jm <- parallel::mclapply(1:n_slices, f, mc.cores = n_cores, mc.allow.recursive = TRUE) 
    
    res <- list(long = out_long,
                surv = out_surv,
                jm = out_jm,
                n_slices = n_slices)
    res
  }
  ###############
                 
  ###############
  #' Create consensus MCMC chains from split JM
  #' 
  #' Merge the MCMC chains from a list of joint models fit on split data. Weighting is applied 
  #' to each chain 
  #' 
  #' @param jm_split A list of joint models to combine
  #' @param combine_chains If true, after creating the merged model, all MCMC 
  #'      chains will be stacked together
  #'      
  #' Notes: 
  #'      Need to add special handling for 'b', which as a different structure
  #'      Maybe need handling for case when all vars are 0 (eg in 'alphaF' and 'frailty')
  jm_consensus <- function(jm_split, combine_chains = FALSE) {
    
    n_slices <- length(jm_split)
    n_chains <- jm_split[[1]]$control$n_chains
    consensus_mcmc_chains <- function(jm_split, par_name, n_slices, n_chains, 
                                      method = c('union', 'equal_weight', 'var_weight'),
                                      combine_chains = FALSE) {
      # Create 4 dimensional array with dims: iteration, parameter, chain, model
      x <- lapply(1:n_slices, function(i) jm_split[[i]]$mcmc[[par_name]])
      x <- array(unlist(x), dim = c(dim(x[[1]][[1]]), n_chains, n_slices),
                 dimnames = list(NULL, colnames(x[[1]][[1]]), NULL, NULL))
      n_iter <- dim(x)[[1]]
      
      # Only implemented variance weights so far
      if (method != 'var_weight') stop('Method not implemented')
      w <- 1 / apply(x, c(2,3,4), var)
      w <- w / array(rep(apply(w, c(1,2), sum), n_slices), dim = dim(w))
      w <- array(rep(w, each = n_iter), dim = c(n_iter, dim(w)))
      
      # Final matrix: iteration, parameter, model
      x <- apply(x * w, c(1, 2, 3), sum)
      x
    }
    
    if ('gammas' %in% names(jm_split[[1]]$mcmc)){
      mcmc_parnames <- c( 'bs_gammas', 'tau_bs_gammas', 'gammas', 'alphas', 
                          'W_std_gammas', 'Wlong_std_alphas', 'D', 'betas1', 
                          'sigmas', 'alphaF', 'frailty', 'sigmaF'
      )
    } else {
      mcmc_parnames <- c( 'bs_gammas', 'tau_bs_gammas', 'alphas', 
                          'W_std_gammas', 'Wlong_std_alphas', 'D', 'betas1', 
                          'sigmas', 'alphaF', 'frailty', 'sigmaF'
      )
    }
    
    mcmc <- sapply(mcmc_parnames, function(par) {
      consensus_mcmc_chains(jm_split, par_name = par, n_slices = n_slices, 
                            n_chains = n_chains, method = 'var_weight',
                            combine_chains = )},
      USE.NAMES = TRUE)
    
    if (combine_chains){
      unfold <- function(x, n) do.call(rbind, lapply(1:n_chains, function(i) x[,,i]))
      mcmc <- lapply(mcmc, unfold)
    }
    
    mcmc[['b']] <- list() # Add later
    
    # Return object (giving structure similar to JM object)
    output <- list(mcmc = mcmc)
    class(output) <- 'jm.consensus' # Giving different name since currently will break if dispatched to jm generics
    output
  }                                                                                                      
  ###############
  ####################################
  
  ####################################
  #' Transform jm_cons_obj to a jm class by borrowing the structure from one of the jm splits
  #' 
  #' @param jm_split_obj A list of joint models
  #' @param jm_cons_obj The combined mcmc chains
  #' @param n_chains number of chains
  #' @param n_iter number of iterations (equals to n_iter - nburnin from the jm() function)
  #' 
  ###############
  jm_class_func <- function(jm_split_obj, jm_cons_obj, n_chains, n_iter) {
    # 1/ Pick one random jm_object from jm_split_obj
    init <- jm_split_obj[[1]]
    jm_obj_modif <- jm_split_obj[[1]]
    
    #2/ Modify the structure of jm_consens mcmc to match the jm's one
    # List of variable names
    if ('gammas' %in% names(jm_split_obj[[1]]$mcmc)){
      mcmc_parnames <- c( 'bs_gammas', 'tau_bs_gammas', 'gammas', 'alphas', 
                          'W_std_gammas', 'Wlong_std_alphas', 'D', 'betas1', 
                          'sigmas', 'alphaF', 'frailty', 'sigmaF'
      )
    } else {
      mcmc_parnames <- c( 'bs_gammas', 'tau_bs_gammas', 'alphas', 
                          'W_std_gammas', 'Wlong_std_alphas', 'D', 'betas1', 
                          'sigmas', 'alphaF', 'frailty', 'sigmaF'
      )
    }
    
    # Initialize the 'jm_cons_obj_modif' list
    jm_cons_obj_modif <- list(mcmc = list())
    
    # Iterate through each variable name
    for (par_name in mcmc_parnames) {
      jm_cons_obj_modif$mcmc[[par_name]] <- list()
      
      # Iterate through the third dimension of jm_cons_obj$mcmc$par_name
      for (i in 1:dim(jm_cons_obj$mcmc[[par_name]])[n_chains]) {
        jm_cons_obj_modif$mcmc[[par_name]][[i]] <- jm_cons_obj$mcmc[[par_name]][,,i]
        if (length(class(jm_cons_obj_modif$mcmc[[par_name]][[i]])) == 1){
          jm_cons_obj_modif$mcmc[[par_name]][[i]] <- matrix(jm_cons_obj_modif$mcmc[[par_name]][[i]],
                                                                               ncol=1)
        }
      }
      class(jm_cons_obj_modif$mcmc[[par_name]]) <- "mcmc.list"
    }
    
    # 3/ Replace the mcmc samples inside the new object
    jm_obj_modif$mcmc[1:length(mcmc_parnames)] <- jm_cons_obj_modif$mcmc
    jm_obj_modif$mcmc[['b']] <- init$mcmc[['b']]
    if (('gammas' %in% names(jm_split_obj[[1]]$mcmc)) == FALSE){
      for (i in 1:n_chains){
        for (j in 1:n_iter){
          jm_obj_modif$mcmc[['W_std_gammas']][[i]][[j]] <- 0
        } 
      }
    }
    
    jm_obj_modif
    
    }
  ####################################