# ----------------- Imports -------------
library(tidyverse)
library(corrplot)
library("data.table")
library(caret)
library(randomForest)
library('e1071')
library(dplyr)
# ---------------- Data read -------------
setwd('C:/Users/Gabrysia/git_mow')

dat <- read.csv("WA_Fn-UseC_-HR-Employee-Attrition.csv", header=TRUE)
dat <-subset(dat, select = -c(Over18, StandardHours, EmployeeCount, EmployeeNumber))

# -------- Data preparation --------------
incomplete_count <- toString(sum(!complete.cases(dat)))
print(paste("Incomplete cases:", incomplete_count, sep = " "))

####
split_ratio <- c(0.6, 0.2, 0.2)

training_set_size = round(split_ratio[1]*nrow(dat))
validation_set_size = round(split_ratio[2]*nrow(dat))
test_set_size = round(split_ratio[3]*nrow(dat))

tr_indices  <- sample(seq_len(nrow(dat)), size = training_set_size)
training_set <- dat[tr_indices,]
temp_set <- dat[-tr_indices,]
v_indices <- sample(seq_len(nrow(temp_set)), size = validation_set_size)
validation_set  = temp_set[v_indices,]
testing_set = temp_set[-v_indices,]

#--Feature selection using random forest----

feature_selection_rf <- randomForest(formula = Attrition~., data = training_set)
print(summary(feature_selection_rf))
forest_summary <- varImp(feature_selection_rf)
forest_summary <- data.frame(overall = forest_summary$Overall,
                            names   = rownames(forest_summary))
forest_summary_sorted <- forest_summary[order(forest_summary$overall, decreasing = T), ]
rownames(forest_summary_sorted) <- NULL
varImpPlot(feature_selection_rf)


features_sorted <- as.character(forest_summary_sorted$names)
features_num <- c(10, 15, 20, 30)

#----RANDOM FOREST PREDICTION----------------
for (f_num in features_num) {
  rf_training_subset <- subset(training_set, select = c("Attrition", features_sorted[1:f_num]))
  rf_validation_subset <- subset(validation_set, select = c("Attrition", features_sorted[1:f_num]))
  rf_model <- randomForest(formula = Attrition~., data = rf_training_subset)
  rf_pred <- predict(rf_model, newdata = rf_validation_subset, type = 'response')
  rf_confm <- caret::confusionMatrix(rf_pred, rf_validation_subset$Attrition)
  cat(sprintf("Confusion matrix for random forest model with %s features\n", f_num))
  print(rf_confm)
}

  
#----wplyw wyboru atrybytow------
  
#-----wplyw parametrow-----------


for (f_num in features_num)
{
#----------Numeric data preparation-----------
numeric_tr_subset <- as.data.table(dataPreparation::shapeSet(subset(training_set, select = c("Attrition", features_sorted[1:f_num])),
                                  finalForm = 'numerical_matrix', verbose =TRUE))
numeric_tr_subset <-subset(numeric_tr_subset, select = -c(AttritionNo))
names(numeric_tr_subset) <- str_replace_all(names(numeric_tr_subset), c(" " = "_", "," = "", "&" = ""))
M <- cor(numeric_tr_subset)
head(round(M,2))

numeric_val_subset <- as.data.table(dataPreparation::shapeSet(subset(validation_set, select = c("Attrition", features_sorted[1:f_num])),
                                                             finalForm = 'numerical_matrix', verbose =TRUE))
numeric_val_subset <-subset(numeric_val_subset, select = -c(AttritionNo))
names(numeric_val_subset) <- str_replace_all(names(numeric_val_subset), c(" " = "_", "," = "", "&" = ""))
M <- cor(numeric_val_subset)
head(round(M,2))


# ------------GLM------------------------
  
glm_model = stats::glm(formula = AttritionYes~., data = numeric_tr_subset)
glm_pred <- predict(glm_model, newdata = numeric_val_subset, method = "glm.fit")
glm_pred <- round(glm_pred)
glm_confm <- caret::confusionMatrix(table(glm_pred, numeric_val_subset$AttritionYes))
print(glm_confm)

#SOME CODE GOES HERE

}

  ## ----------------- ANN --------------------
  #answer_ann <- subset(dat_shaped, select = c(AttritionYes))
  #substet_ann <- subset(dat_shaped, select = -c(AttritionYes))
  #training_set_ann <- substet_ann[training_set_size,]
  #validation_set = substet_ann[validation_set_size,]
  #ann_for_set = RSNNS::mlp(x = training_set_ann, y = answer_ann, size = c(4), maxit = 100, initFunc = "Randomize_Weights", initFuncParams = c(-0.3, 0.3),
  #                         learnFunc = "Std_Backpropagation")
  #pred_ann <- predict(ann_for_set, newdata = validation_set, type = 'response')
  #pred_ann <- round(pred_ann)
  #conf_matrix_ann <- caret::confusionMatrix(table(pred_ann, validation_set$AttritionYes))
  #print(conf_matrix_ann)
  
  
