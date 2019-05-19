# ----------------- Imports -------------
library(tidyverse)
library(corrplot)
library("data.table")
library(caret)
library(randomForest)
library('e1071')
library(dplyr)
# ---------------- Data read -------------
setwd('~/Studia/SEM2/MOW/Projekt/MOW')

dat <- read.csv("WA_Fn-UseC_-HR-Employee-Attrition.csv", header=TRUE)
dat <-subset(dat, select = -c(Over18, StandardHours, EmployeeCount, EmployeeNumber))

# -------- Data preparation --------------
incomplete_count <- toString(sum(!complete.cases(dat)))
print(paste("Incomplete cases:", incomplete_count, sep = " "))

#---- Build a random forest based on whole set for feature selection ---------------------------
feature_selection_rf <- randomForest(formula = Attrition~., data = dat)
print(summary(feature_selection_rf))
forestSummary <- varImp(feature_selection_rf)
forestSummary <- data.frame(overall = forestSummary$Overall,
                  names   = rownames(forestSummary))
forestSummarySorted <- forestSummary[order(forestSummary$overall, decreasing = T), ]
rownames(forestSummarySorted) <- NULL
varImpPlot(feature_selection_rf)

#---- Test models based on different numbers of features (Initial values: 10, 15, 20, all attributes )
#feature_tresh1 <- forestSummarySorted$names[1:10]
#feature_tresh2 <- forestSummarySorted$names[1:15] 
#feature_tresh3 <- forestSummarySorted$names[1:20]
#feature_all <- forestSummarySorted$names

feature_tresh1 <- colnames(subset(dat, select = c(MonthlyIncome,Age,OverTime,DailyRate,TotalWorkingYears,MonthlyRate,JobRole,DistanceFromHome,HourlyRate,YearsAtCompany,PercentSalaryHike)))
feature_tresh2 <- c(feature_tresh1, colnames(subset(dat, select = c(NumCompaniesWorked,EducationField,YearsWithCurrManager,EnvironmentSatisfaction,StockOptionLevel))))


training_set_size <- round(0.8*nrow(dat))
train_index  <- sample(seq_len(nrow(dat)), size = training_set_size)
#treshholds <- list(feature_tresh1)#, feature_tresh2, feature_tresh3, feature_all)
#treshholds <- list(feature_all)

#Test results for differrent feature set sizes
#for (treshold in treshholds){
# Test for 10 features
  #print(treshold)
  substet_t1 <- subset(dat, select = c(MonthlyIncome,Age,OverTime,DailyRate,TotalWorkingYears,MonthlyRate,JobRole,DistanceFromHome,HourlyRate,YearsAtCompany,PercentSalaryHike, Attrition))
  training_set1 <- substet_t1[train_index, ]
  validation_set1 <- substet_t1[-train_index, ]
  rf_for_set_size1 <- randomForest(formula = Attrition~., data = training_set1)
  pred1 <- predict(rf_for_set_size1, newdata = validation_set1, type = 'response')
  conf_matrix1 <- caret::confusionMatrix(pred1, validation_set1$Attrition)
  print(conf_matrix1)
#}
  
  substet_t2 <- subset(dat, select = c(MonthlyIncome,Age,OverTime,DailyRate,TotalWorkingYears,MonthlyRate,JobRole,DistanceFromHome,HourlyRate,YearsAtCompany,PercentSalaryHike, Attrition,NumCompaniesWorked,EducationField,YearsWithCurrManager,EnvironmentSatisfaction,StockOptionLevel))
  training_set2 <- substet_t2[train_index, ]
  validation_set2 <- substet_t2[-train_index, ]
  rf_for_set_size2 <- randomForest(formula = Attrition~., data = training_set2)
  pred2 <- predict(rf_for_set_size2, newdata = validation_set2, type = 'response')
  conf_matrix2 <- caret::confusionMatrix(pred2, validation_set2$Attrition)
  print(conf_matrix2)
  
  
  substet_t3 <- subset(dat, select = c(MonthlyIncome,Age,OverTime,DailyRate,TotalWorkingYears,MonthlyRate,JobRole,DistanceFromHome,HourlyRate,YearsAtCompany,PercentSalaryHike, Attrition,NumCompaniesWorked,EducationField,YearsWithCurrManager,EnvironmentSatisfaction,StockOptionLevel,JobSatisfaction,TrainingTimesLastYear,WorkLifeBalance,YearsInCurrentRole,YearsSinceLastPromotion))
  training_set3 <- substet_t3[train_index, ]
  validation_set3 <- substet_t3[-train_index, ]
  rf_for_set_size3 <- randomForest(formula = Attrition~., data = training_set3)
  pred3 <- predict(rf_for_set_size3, newdata = validation_set3, type = 'response')
  conf_matrix3 <- caret::confusionMatrix(pred3, validation_set3$Attrition)
  print(conf_matrix3)
  
  
  training_set4 <- dat[train_index, ]
  validation_set4 <- dat[-train_index, ]
  rf_for_set_size4 <- randomForest(formula = Attrition~., data = training_set4)
  pred4 <- predict(rf_for_set_size4, newdata = validation_set4, type = 'response')
  conf_matrix4 <- caret::confusionMatrix(pred4, validation_set4$Attrition)
  print(conf_matrix4)



  # -----------------Calculate correlation --------------------
#M <- cor(dat_shaped)
#head(round(M,2))
  
