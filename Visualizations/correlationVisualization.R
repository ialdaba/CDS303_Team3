#####################################
## Name: Bharath Maniraj
## Assignment: CDS 303 Correlation Visualizations
#####################################

library(tidyverse)
library(ggplot2)

CDSData <- read.csv("C:\\Users\\bhara\\OneDrive\\Documents\\CDS\\CDS 303\\Untitled Folder\\cleaned_data.csv")


#Variables to test: AMNT_INCOME_TOTAL, AMT_CREDIT, TARGET

reqDat <- data.frame(CDSData$AMT_INCOME_TOTAL, CDSData$AMT_CREDIT, CDSData$TARGET)

#Get 50 random rows
graphData <- reqDat[sample(nrow(reqDat), 50), ]

#graph correlation between the two variables
graphData %>%
  ggplot(aes(x = CDSData.AMT_INCOME_TOTAL, 
             y = CDSData.AMT_CREDIT,
             color = CDSData.TARGET)) + 
  geom_point(alpha = 0.5) + 
  labs(x = "Total Income", y = "Credit") + 
  geom_smooth(method = lm)




