# Raw code for the Project

################# Import data ####################
data <- read.csv("Admission_Predict.csv")

############### import packages #################
library(reshape2)
library(ggplot2)
library(rpart)
library(dplyr)
library(rpart.plot)
library(randomForest)
library(kableExtra)
library(broom)
library(tidyverse)
library(caret)
library(leaps)
library(car)
library(knitr)

################## Histogram #####################
hist(data$GRE.Score,freq=FALSE, main = 'Distribution of GRE SCORE', xlab = 'GRE SCORE', ylab ='Frequency',col='blue')
x1 <- seq(min(data$GRE.Score), max(data$GRE.Score), length = 40)
f1 <- dnorm(x1, mean = mean(data$GRE.Score), sd = sd(data$GRE.Score))
lines(x1, f1, col = "black", lwd = 2)

hist(data$TOEFL.Score,freq=FALSE, main = 'Distribution of TOEFL SCORE',xlab = 'TOEFL SCORE',ylab = 'FREQUENCY',col='green')
x2 <- seq(min(data$TOEFL.Score), max(data$TOEFL.Score), length = 40)
f2 <- dnorm(x2, mean = mean(data$TOEFL.Score), sd = sd(data$TOEFL.Score))
lines(x2, f2, col = "black", lwd = 2)

hist(data$SOP,freq=FALSE, main = 'Distribution of Statement of Purpose (SOP)', xlab = 'SOP', ylab = 'FREQUENCY',col='yellow')
x3 <- seq(min(data$SOP), max(data$SOP), length = 40)
f3 <- dnorm(x3, mean = mean(data$SOP), sd = sd(data$SOP))
lines(x3, f3, col = "black", lwd = 2)

hist(data$CGPA,freq=FALSE, main = 'Distribution of GPA', xlab = 'CGPA', ylab = 'FREQUENCY',col='pink') 
x4 <- seq(min(data$CGPA), max(data$CGPA), length = 40)
f4 <- dnorm(x4, mean = mean(data$CGPA), sd = sd(data$CGPA))
lines(x4, f4, col = "black", lwd = 2)

hist(data$LOR, freq=FALSE, main = 'Distribution for Letters of Recommendation', xlab = 'LOR', ylab='FREQUENCY',col='purple')
x5 <- seq(min(data$LOR), max(data$LOR), length = 40)
f5 <- dnorm(x5, mean = mean(data$LOR), sd = sd(data$LOR))
lines(x5, f5, col = "black", lwd = 2)

hist(data$Chance.of.Admit, freq=FALSE, main = 'Distribution for Chance of Admission', xlab = 'Chance of Admit', ylab='FREQUENCY',col='red')
x7 <- seq(min(data$Chance.of.Admit), max(data$Chance.of.Admit), length = 40)
f7 <- dnorm(x7, mean = mean(data$Chance.of.Admit), sd = sd(data$Chance.of.Admit))
lines(x7, f7, col = "black", lwd = 2)

################### Correlation matrix ###################
# Remove serial no column
data <- data[,!names(data) %in% c("Serial.No.")]
# Round the data to two decimal places
cor_data <- round(cor(data),2)
# get correlation data
melted_cor_data <- melt(cor_data)
# Plot the correlations using gg-plot
p = ggplot(data = melted_cor_data, aes(x=Var1, y=Var2, fill=value)) + geom_tile()
p + labs(title = "Corelation matrix between variables")


#################### Normal distribution #################
# plot the data to check normal distribution
qqnorm(data$Chance.of.Admit)
qqline(data$Chance.of.Admit, col = 2)


################### Machine Learning ######################

#################### Running Random Forrest ###############################

set.seed(4543)
rf.fit <- randomForest(data$Chance.of.Admit ~ ., data=data,
                       keep.forest=FALSE, importance=TRUE)
ImpData <- as.data.frame(importance(rf.fit))
ImpData$Var.Names <- row.names(ImpData)

# plot the results form the random forest
ggplot(ImpData, aes(x=Var.Names, y=`%IncMSE`)) +
geom_segment( aes(x=Var.Names, xend=Var.Names, y=0, yend=`%IncMSE`), color="skyblue") +
geom_point(aes(size = IncNodePurity), color="blue", alpha=0.6) +
theme_light() +
coord_flip()+
theme(legend.position="bottom",
      panel.grid.major.y = element_blank(),panel.border = element_blank(),
      axis.ticks.y = element_blank())+
ggtitle("Importance Plot for the Variables")
rf.fit

####################### Running Linear Regression#########################

## Sub-setting

model_reg <- regsubsets(Chance.of.Admit ~ ., data=data, nvmax=7)
best.fit <- summary(model_reg)

plot(best.fit$adjr2, xlab = "Number of Variables", ylab = "Adjusted RSq", type = "l", 
     main="Adj.R-squared by Number of Variables", font.main=3)
model_reg <- regsubsets(Chance.of.Admit ~ ., data=data, nvmax=7, intercept = FALSE)
plot(model_reg,scale = "adjr2",main="bic plot for variables")

plot(best.fit$bic, xlab = "Number of Variables", ylab = "BiC", type = "l", 
     main="Bic by Number of Variables", font.main=3)
model_reg <- regsubsets(Chance.of.Admit ~ ., data=data, nvmax=7, intercept = FALSE)
plot(model_reg,scale = "bic", main="bic plot for variables")

model <-

########### GRE SCORES ##############
data <- read.csv("Admission_Predict.csv")
data <- data[,-1]
data <- data[,-8]
data <- data[,-3]

model_reg_gre <- regsubsets(GRE.Score ~ ., data=data, nvmax=5, intercept = FALSE)
best.fit <- summary(model_reg_gre)
plot(best.fit$bic, xlab = "Number of Variables", ylab = "Adjusted RSq", type = "l")
plot(model_reg_gre,scale = "bic")

model_gre <- lm(GRE.Score ~ ., data=data)
summary(model_gre)

vif(model_gre)

res <- resid(model_gre)
plot(fitted(model_gre), res, main="Plot of Residuals")
abline(0,0)

glance(model_gre)

model_gre <- lm(GRE.Score ~ . , data=data)
glance(model_gre)

intercept_only <- lm(GRE.Score ~ 1, data=data)
all <- lm(GRE.Score ~ ., data=data)
forward <- step(intercept_only, direction='forward', scope=formula(all),trace=0)
summary(forward) 