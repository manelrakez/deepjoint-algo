# install required packages :
install.packages('IRkernel')
install.packages('dplyr')
install.packages('utils')
install.packages('JMbayes2')
install.packages('timeROC')
install.packages('ggplot2')
install.packages('RColorBrewer')
install.packages('parallel')
install.packages('tidyverse')
install.packages('survival')

# Setup R kernel in notebook
library("IRkernel")
IRkernel::installspec(user = FALSE)
