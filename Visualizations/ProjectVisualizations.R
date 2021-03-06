#####################################
## Name: David Ricks
## Assignment: CDS 303 Project Visualizations
#####################################

setwd("C:/Users/David/Documents/SR-YR-3S/CDS-303") #Set directory to file path where datasets are locally stored
datatrain <- read.csv("application_train.csv") #import dirty data
data_cleaned <- read.csv("cleaned_data.csv") #import clean data
library(ggplot2) #import visualization library
library(tidyverse)
library(naniar)

#Income Distribution
datatrain2 <- subset(datatrain, datatrain$AMT_INCOME_TOTAL < 400000) #removing income outliers for better visualization
datatrain2$TARGET <-factor(datatrain2$TARGET, labels = c("Difficulty in Repayment", "Repaid Loan")) #creating Target variable labels
  
ggplot(datatrain2, #calling dataset
       aes(x = AMT_INCOME_TOTAL)) + #calling income variable
  geom_histogram(bins = 18) + #defining visualization as histogram and bin count
  facet_grid(~ TARGET) + #faceting over the target variable
  scale_x_continuous(labels = function(x) format(x, scientific = FALSE)) + #Telling R to interpret income as discrete rather than continuous variable so it will plot properly
  geom_vline(aes(xintercept = mean(AMT_INCOME_TOTAL)),color = "red") + #Creating red line that displays the mean income of each group
  labs(title = "Income Distribution", #Setting title
       x = "Total Income") + #labeling x axis
  theme(plot.title = element_text(hjust = 0.5, #centering title
                                  face = "bold", #bolding title
                                  size = 18), #adjusting font size
        panel.grid.major = element_blank(), #removing visual clutter
        panel.grid.minor = element_blank(),
        plot.background = element_rect(fill = "white"),
        panel.background = element_rect(fill = "white"),
        strip.background = element_blank())

#Age Distribution
datatrain2$ageindays <- datatrain2$DAYS_BIRTH * -1 #Making age value positive

datatrain2$ageyears <- datatrain2$ageindays / 365 #converting age to years

datatrain2$TARGET <-factor(datatrain2$TARGET, labels = c("Difficulty in Repayment", "Repaid Loan")) #creating target variable labels

ggplot(datatrain2, #calling dataset
       aes(x = ageyears)) + #calling age variable
  geom_density() + #defining visual as density plot
  facet_wrap(~TARGET) + #faceting over target variable
  scale_x_continuous(labels = function(x) format(x, scientific = FALSE)) + #Telling R to interpret age as discrete
  geom_vline(aes(xintercept = mean(ageyears)), color = "red") + #creating red mean line
  labs(title = "Age Distribution", #Setting title
       x = "Age") + #labeling x axis
  theme(plot.title = element_text(hjust = 0.5, #centering title
                                  face = "bold", #bolding title
                                  size = 18), #Adjusting title font size
        panel.grid.major = element_blank(), #removing visual clutter
        panel.grid.minor = element_blank(),
        plot.background = element_rect(fill = "white"),
        panel.background = element_rect(fill = "white"),
        axis.title.y = element_blank(),
        axis.ticks.y = element_blank(),
        axis.text.y = element_blank(),
        strip.background = element_blank())

#Gender Distribution
sum(datatrain$CODE_GENDER == "M") / 307511 #computing percentage of males

data <- data.frame( #creating new dataset
  group = c("Male", "Female"), #with a variable called group and rows of male and female
  value=c(34, 66)) #defining female and male distribution

ggplot(data, #calling data
       aes(x = "", #no x axis
                 y = value, #calling value
                 fill = group)) + #filling by gender
  geom_bar(stat="identity", width=1) + #defining visual as histogram
  labs(title = "Gender Distribution", #Setting title
       fill = "Gender") + #Setting legend title as "x"
  coord_polar("y", start=0) + #converting histogram to pie chart
  theme_void() + #removing visual clutter
  theme(legend.position = "bottom", #moving legend to bottom
        plot.title = element_text(hjust = 0.5, #Centering title
                                  face = "bold", #Bolding title
                                  size = 18)) #Adjusting title font size

#Cleaned Gender Distribution
                     data_cleaned3 <- data.frame( #creating new dataset
  group = c("Male", "Female"), #with a variable called group and rows of male and female
  value=c(49, 51)) #defining female and male distribution

ggplot(data_cleaned3, aes(x = "", #No x value
                          y = value, #percentages
                          fill = group)) + #labels
  geom_bar(stat="identity", width=1) + #defining visual as histogram
  labs(title = "Cleaned Gender Distribution", fill = "Gender") + # creating title
  coord_polar("y", start=0) + #converting histogram to pie chart
  theme_void() + #removing visual clutter
  theme(legend.position = "bottom", #moving legend to bottom
        plot.title = element_text(hjust = 0.5, #Centering title
                                  face = "bold", #bolding title
                                  size = 18)) #adjusting title font size
                     
#Target Distribution
sum(datatrain$TARGET == 0) / 307511 #computing percentage that repaid on time

data <- data.frame( #creating new dataset
  group = c("Repaid Loan", "Difficulty Repaying"), #with a variable called group and rows of difficulty repaying and repaid loan
  value=c(92, 8)) #defining target distribution

ggplot(data, #calling data
       aes(x = "", #no x axis
           y = value, #calling value
           fill = group)) + #filling by gender
  geom_bar(stat="identity", width=1) + #defining visual as histogram
  labs(title = "Target Variable Distribution", #Setting title
       fill = "Target") + #Setting legend title as "x"
  coord_polar("y", start=0) + #converting histogram to pie chart
  theme_void() + #removing visual clutter
  theme(legend.position = "bottom", #moving legend to bottom
        plot.title = element_text(hjust = 0.5, #Centering title
                                  face = "bold", #Bolding title
                                  size = 18)) #Adjusting title font size

#Undersampled Target Distribution
sum(data_balanced$TARGET == 0) / 49650 #computing percentage that repaid on time

data_cleaned3 <- data.frame( #creating new dataset
  group = c("Repaid Loan", "Difficulty Repaying"), #with a variable called group and rows of difficulty repaying and repaid loan
  value=c(50, 50)) #defining target distribution

ggplot(data_cleaned3, aes(x = "", #No x value
                          y = value, #percentages
                          fill = group)) + #labels
  geom_bar(stat="identity", width=1) + #defining visual as histogram
  labs(title = "Undersampled Target Variable Distribution", fill = "Target") + # creating title
  coord_polar("y", start=0) + #converting histogram to pie chart
  theme_void() + #removing visual clutter
  theme(legend.position = "bottom", #moving legend to bottom
        plot.title = element_text(hjust = 0.5, #Centering title
                                  face = "bold", #bolding title
                                  size = 18)) #adjusting title font size
                     
#Employment Distribution
datatrain3 <- subset(datatrain, datatrain$DAYS_EMPLOYED < 0)

datatrain3$adjustedaysemployed <- datatrain3$DAYS_EMPLOYED * -1 #converting days employed to positive value

ggplot(datatrain3, #calling data
       aes(x = adjustedaysemployed)) +  #calling days employed variable
  geom_histogram() + #defining plot as histogram
  scale_x_continuous(labels = function(x) format(x, scientific = FALSE)) + #making data discrete
  geom_vline(aes(xintercept = mean(adjustedaysemployed)), color = "red") + #inserting red mean line
  labs(title = "Employment Distribution", #titling plot
       x = "Days Employed") + #labeling x axis
  theme(plot.title = element_text(hjust = 0.5, #Centering title
                                  face = "bold", #boldingtitle
                                  size = 18), #adjusting title font
        panel.grid.major = element_blank(), #removing visual clutter
        panel.grid.minor = element_blank(),
        plot.background = element_rect(fill = "white"),
        panel.background = element_rect(fill = "white"))

#Job Type Distribution
ggplot(data.frame(datatrain), #Calling data
       aes(x = OCCUPATION_TYPE)) + #Calling job type variable
  geom_bar() + # defining visual and bar graph
  labs(title = "Job Type Distribution", x = "Job Type") + #labeling plot and x axis
  coord_flip() + #moving job type to y axis so labels are horizontal
  theme(axis.title.x = element_blank(), #removing x axis title
        plot.title = element_text(hjust = 0.5, #centering title
                                  face = "bold", #bolding title
                                  size = 18), #setting title font size
        panel.grid.major = element_blank(), #removing visual clutter
        panel.grid.minor = element_blank(),
        plot.background = element_rect(fill = "white"),
        panel.background = element_rect(fill = "white"))

#Income ~ Credit
data1 <- subset(data_cleaned, data_cleaned$TARGET == 1)

data0 <- subset(data_cleaned, data_cleaned$TARGET == 0)

data0.2 <- data0[sample(nrow(data0), 24825), ]

data_cleaned <- rbind(data1, data0.2)
                     
datatrain4 <- data.frame(data_cleaned$AMT_INCOME_TOTAL, data_cleaned$AMT_CREDIT, data_cleaned$TARGET) #creating dataframe with only necessary attributes

datatrain4 <- subset(datatrain4, datatrain4$data_cleaned.AMT_INCOME_TOTAL < 400000) #removing income outliers for better visualization

datatrain4 <- datatrain4[sample(nrow(datatrain4), 50), ] #Calling 50 random rows

ggplot(datatrain4, #calling data
       aes(x = data_cleaned.AMT_INCOME_TOTAL, #calling dataset and income
           y = data_cleaned.AMT_CREDIT, #calling credit
           color = data_cleaned.TARGET)) + #Coloring over target variable
  geom_point(alpha = 0.5) + #Making points translucent
  scale_x_continuous(labels = function(x) format(x, scientific = FALSE)) + #making x axis values discrete so it will plot labels properly
  scale_y_continuous(labels = function(x) format(x, scientific = FALSE)) + #making y axis values discrete so it will plot labels properly
  labs(title = "Correlation Between Income and Credit of Potential Borrowers", #titling plot
       x = "Total Income", #labeling x axis
       y = "Credit", #labeling y axis
       color = "Repayment Ability") + #legend title
  geom_smooth(method = lm) + #Display regression line and confidence interval
  theme(plot.title = element_text(face = "bold", #bolding title
                                  size = 12), #adjusting title font size
        panel.grid.major = element_blank(), #removing visual clutter
        panel.grid.minor = element_blank(),
        plot.background = element_rect(fill = "white"),
        panel.background = element_rect(fill = "white"))
                     
#Missing data
datatrain5 <- datatrain[sample(nrow(datatrain), 1000), ] #Calling 1000 random rows

vis_miss(datatrain5) + #calling data
  labs(title = "Missing Data") + #titling plot
  coord_flip() + #flipping y and x axis so that categorical data labels/text are horizontal
  theme(plot.title = element_text(hjust = 0.5, #Centering title
                                  face = "bold", #Bolding title
                                  size = 18), #Adjusting title font size
        axis.text.y = element_text(size = 4)) #Adjusting y axis label font size

data_cleaned2 <- data_cleaned[sample(nrow(data_cleaned), 1000), ] #Calling 1000 random rows

vis_miss(data_cleaned2) + #calling data
  labs(title = "Cleaned Data") + #titling plot
  coord_flip() + #flipping y and x axis so that categorical data labels/text are horizontal
  theme(plot.title = element_text(hjust = 0.5, #Centering title
                                  face = "bold", #bolding title
                                  size = 18), #Adjusting title font size
        axis.text.y = element_text(size = 2)) #Adjusting y axis label font size
