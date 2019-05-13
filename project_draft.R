# ----------------- Imports -------------
library(tidyverse)
library(corrplot)
library("data.table")
library(caret)
library(randomForest)
library('e1071')

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
forestSummary <- as.data.frame(varImp(feature_selection_rf))
forestSummary <- data.frame(overall = forestSummary$Overall,
                  names   = rownames(forestSummary))
forestSummary <- forestSummary[order(forestSummary$overall, decreasing = T), ]
forestSummary
varImpPlot(feature_selection_rf)

#---- Test models based on different numbers of features (Initial values: 10, 15, 20, all attributes )
feature_tresh1 <- 1:10
feature_tresh2 <- 1:15
feature_tresh3 <- 1:20
feature_all <- 1:ncol(dat)

#training_set_size = 1:round(0.8*nrow(dat)) 
training_set_size <- round(0.8*nrow(dat))
train_index  <- sample(seq_len(nrow(dat)), size = training_set_size)
treshholds <- list(feature_tresh1, feature_tresh2, feature_tresh3, feature_all)


#Test results for differrent feature set sizes
for (treshold in treshholds){
  substet_t <- subset(dat, select = treshold)
  training_set <- substet_t[train_index, ]
  validation_set <- substet_t[-train_index, ]
  rf_for_set_size <- randomForest(formula = Attrition~., data = substet_t)
  pred <- predict(rf_for_set_size, newdata = validation_set, type = 'response')
  conf_matrix <- caret::confusionMatrix(pred, validation_set$Attrition)
  print(conf_matrix)
}



#data_t2 <- subset(dat, select = feature_tresh2)
#data_t3 <- subset(dat, select = feature_tresh3)

#-----------
#dat_shaped <- as.data.table(dataPreparation::shapeSet(dat, finalForm = 'numerical_matrix', verbose =TRUE))
#dat_shaped <-subset(dat_shaped, select = -c(AttritionNo))
#names(dat_shaped) <- str_replace_all(names(dat_shaped), c(" " = "_", "," = "", "&" = ""))
#Split data into 3 sets

#training_set_size = 1:round(0.6*nrow(dat)) 
#validation_set_size = round(0.6*nrow(dat))+1:round(0.2*nrow(dat))
#test_set_size = round(0.6*nrow(dat))+round(0.2*nrow(dat))+1: round(0.2*nrow(dat))

#training_set_shaped <- dat_shaped[training_set_size, ]
#validation_set_shaped <- dat_shaped[validation_set_size, ]
#test_set_shaped <- dat_shaped[test_set_size,]

#training_set <- dat[training_set_size, ]
#validation_set <- dat[validation_set_size, ]
#test_set <- dat[test_set_size,]



# Model test (irrelevant)
#pred <- predict(feature_selection_rf, newdata = validation_set_shaped, type = 'response')
#caret::confusionMatrix(pred, validation_set_shaped$AttritionYes)

# -----------------Calculate correlation --------------------
#M <- cor(dat_shaped)
#head(round(M,2))
  
