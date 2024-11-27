# Copyright © 2024 INSERM U1219, Therapixel SA
# Contributors: Manel Rakez, Julien Guillaumin
# All rights reserved.
# This file is subject to the terms and conditions described in the
# LICENSE file distributed in this package.

# install required packages :
install.packages('IRkernel')
install.packages('dplyr')
install.packages('utils')
install.packages('JMbayes2')
install.packages('timeROC')
install.packages('riskRegression')
install.packages('ggplot2')
install.packages('RColorBrewer')
install.packages('parallel')
install.packages('tidyverse')
install.packages('survival')

# Setup R kernel in notebook
library("IRkernel")
IRkernel::installspec(user = FALSE)
