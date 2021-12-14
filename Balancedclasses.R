setwd("C:/your/directory/here") #Set directory to file path where datasets are locally stored
cleaned_data <- read.csv("cleaned_data.csv") #import clean data

#Balanced Classes
data1 <- subset(data_cleaned, data_cleaned$TARGET == 1)

data0 <- subset(data_cleaned, data_cleaned$TARGET == 0)

data0.2 <- data0[sample(nrow(data0), 24825), ]

data_balanced <- rbind(data1, data0.2)