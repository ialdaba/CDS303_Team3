setwd("C:/Users/David/Documents/SR-YR-3S/CDS-303") #Set directory to file path where datasets are locally stored
datatrain <- read.csv("application_train.csv") #import dirty data
data_cleaned <- read.csv("cleaned_data.csv") #import clean data
library(ggplot2) #import visualization library
library(tidyverse) #import tidyverse

#Income Distribution
datatrain2 <- subset(datatrain, datatrain$AMT_INCOME_TOTAL < 400000) #removing income outliers for better visualization
datatrain2$TARGET <-factor(datatrain2$TARGET, labels = c("Difficulty in Repayment", "Repaid Loan")) #creating Target variable labels
  
ggplot(datatrain2, aes(x = AMT_INCOME_TOTAL)) + #calling dataset and income variable
  geom_histogram(bins = 18) + #defining visualization as histogram and bin count
  facet_grid(~ TARGET) + #faceting over the target variable
  scale_x_continuous(labels = function(x) format(x, scientific = FALSE)) + #Telling R to interpret income as discrete rather than continuous variable so it will plot properly
  geom_vline(aes(xintercept = mean(AMT_INCOME_TOTAL)),color = "red") + #Creating red line that displays the mean income of each group
  labs(title = "Income Distribution", x = "Total Lifetime Income") + #labeling visual and x axis
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 18), #centering tile and adjsuting font
        panel.grid.major = element_blank(), #removing visual clutter
        panel.grid.minor = element_blank(),
        plot.background = element_rect(fill = "white"),
        panel.background = element_rect(fill = "white"),
        strip.background = element_blank())

#Age Distribution
datatrain2$ageindays <- datatrain2$DAYS_BIRTH * -1 #Making age value positive

datatrain2$ageyears <- datatrain2$ageindays / 365 #converting age to years

datatrain2$TARGET <-factor(datatrain2$TARGET, labels = c("Difficulty in Repayment", "Repaid Loan")) #creating target variable labels

ggplot(datatrain2, aes(x = ageyears)) + #calling dataset and age variable
  geom_density() + #defining visual as density plot
  facet_wrap(~TARGET) + #faceting over target variable
  scale_x_continuous(labels = function(x) format(x, scientific = FALSE)) + #Telling R to interpret age as discrete
  geom_vline(aes(xintercept = mean(ageyears)), color = "red") + #creating red mean line
  labs(title = "Age Distribution", x = "Age") + #labeling title and x axis
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 18), #centerign and formetting title
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

ggplot(data, aes(x = "", y = value, fill = group)) + #calling data, filling by gender
  geom_bar(stat="identity", width=1) + #defining visual as histogram
  labs(title = "Gender Distribution", fill = "Gender") + # creating title
  coord_polar("y", start=0) + #converting histogram to pie chart
  theme_void() + #removing visual clutter
  theme(legend.position = "bottom", #moving legend to bottom
        plot.title = element_text(hjust = 0.5, face = "bold", size = 18)) #centering and formatting plot title

#Employment Distribution
datatrain3 <- subset(datatrain, datatrain$DAYS_EMPLOYED < 0)

datatrain3$adjustedaysemployed <- datatrain3$DAYS_EMPLOYED * -1 #converting days employed to positive value

ggplot(datatrain3, aes(x = adjustedaysemployed)) +  #calling data and days employed variable
  geom_histogram() + #defining plot as histogram
  scale_x_continuous(labels = function(x) format(x, scientific = FALSE)) + #making data discrete
  geom_vline(aes(xintercept = mean(adjustedaysemployed)), color = "red") + #inserting red mean line
  labs(title = "Employment Distribution", x = "Days Employed") + #labeling plot and x axis
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 18), #centering and formatting title
        panel.grid.major = element_blank(), #removing visual clutter
        panel.grid.minor = element_blank(),
        plot.background = element_rect(fill = "white"),
        panel.background = element_rect(fill = "white"))


#Job Type Distribution
ggplot(data.frame(datatrain), aes(x = OCCUPATION_TYPE)) + #Calling data and job type variable
  geom_bar() + # defining visual and bar graph
  labs(title = "Job Type Distribution", x = "Job Type") + #labeling plot and x axis
  coord_flip() + #moving job type to y axis so labels are horizontal
  theme(axis.title.x = element_blank(), #removing x axis title
        plot.title = element_text(hjust = 0.5, face = "bold", size = 18), #Centering and formatting title
        panel.grid.major = element_blank(), #removing visual clutter
        panel.grid.minor = element_blank(),
        plot.background = element_rect(fill = "white"),
        panel.background = element_rect(fill = "white"))

#Income ~ Credit
datatrain4 <- data.frame(data_cleaned$AMT_INCOME_TOTAL, data_cleaned$AMT_CREDIT, data_cleaned$TARGET) #creating dataframe with only necessary attributes

datatrain4 <- subset(datatrain4, reqDat$data_cleaned.AMT_INCOME_TOTAL < 400000) #removing income outliers for better visualization

graphdata <- reqDat[sample(nrow(reqDat), 50), ] #Calling 50 random rows

ggplot(graphdata, aes(x = data_cleaned.AMT_INCOME_TOTAL, #calling dataset and income
             y = data_cleaned.AMT_CREDIT, #calling credit
             color = data_cleaned.TARGET)) + #Coloring over target variable
  geom_point(alpha = 0.5) + #Making points translucent
  scale_x_continuous(labels = function(x) format(x, scientific = FALSE)) + #making x axis values discrete so it will plot labels properly
  scale_y_continuous(labels = function(x) format(x, scientific = FALSE)) + #making y axis values discrete so it will plot labels properly
  labs(title = "Correlation Between Income and Credit of Potential Borrowers",x = "Total Income", y = "Credit", color = "Repayment Ability") +#Labeling everything
  geom_smooth(method = lm) + #Display regression line and confidence interval
  theme(plot.title = element_text(face = "bold", size = 12), #formatting title
        panel.grid.major = element_blank(), #removing visual clutter
        panel.grid.minor = element_blank(),
        plot.background = element_rect(fill = "white"),
        panel.background = element_rect(fill = "white"))
